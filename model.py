import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GINConv
from torch_geometric.utils import to_dense_batch
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, Seq_Len, Dim]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class HBAI_DTA(nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(HBAI_DTA, self).__init__()

        self.embed_dim = embed_dim

        # ==========================================
        # 1. Drug Encoder: GIN (Graph Isomorphism Network)
        # ==========================================
        self.n_output = n_output
        self.drug_in = Sequential(Linear(num_features_xd, n_filters), ReLU(), Linear(n_filters, n_filters))
        self.drug_conv1 = GINConv(self.drug_in)
        self.drug_conv2 = GINConv(Sequential(Linear(n_filters, n_filters), ReLU(), Linear(n_filters, n_filters)))
        self.drug_conv3 = GINConv(Sequential(Linear(n_filters, n_filters), ReLU(), Linear(n_filters, n_filters)))
        self.drug_fc = nn.Linear(n_filters, embed_dim)

        # ==========================================
        # 2. Protein Encoder: MTSE (Multi-Scale Topological Semantic Encoder)
        # ==========================================
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim, padding_idx=0)

        self.cnn_motif = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=5, padding=2)

        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout)
        encoder_layers = TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=512,
                                                 dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=2)

        self.mtse_norm = nn.LayerNorm(embed_dim)

        self.W_bi = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.W_bi)

        self.fc1 = nn.Linear(embed_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        x = self.drug_conv1(x, edge_index)
        x = self.relu(x)
        x = self.drug_conv2(x, edge_index)
        x = self.relu(x)
        x = self.drug_conv3(x, edge_index)
        x = self.relu(x)

        H_drug_nodes, drug_mask = to_dense_batch(x, batch)
        H_drug = self.drug_fc(H_drug_nodes)  # [B, N, embed_dim]

        prot_mask = (target != 0)  # [B, M_residues]
        embedded_xt = self.embedding_xt(target)  # [B, M, embed_dim]

        P1 = self.cnn_motif(embedded_xt.permute(0, 2, 1)).permute(0, 2, 1)
        P1 = self.relu(P1)

        P1_pos = self.pos_encoder(P1)
        P2 = self.transformer_encoder(P1_pos, src_key_padding_mask=~prot_mask)

        H_prot = self.mtse_norm(P1 + P2)  # [B, M, embed_dim]

        drug_proj = torch.matmul(H_drug, self.W_bi)
        interaction_map = torch.bmm(drug_proj, H_prot.transpose(1, 2))

        valid_mask = drug_mask.unsqueeze(2) & prot_mask.unsqueeze(1)  # [B, N, M]
        interaction_map = interaction_map.masked_fill(~valid_mask, -1e9)

        alpha = torch.softmax(interaction_map.max(dim=2)[0], dim=1)
        beta = torch.softmax(interaction_map.max(dim=1)[0], dim=1)

        V_drug = torch.bmm(alpha.unsqueeze(1), H_drug).squeeze(1)
        V_prot = torch.bmm(beta.unsqueeze(1), H_prot).squeeze(1)

        xc = torch.cat((V_drug, V_prot), 1)  # 拼接处于结合激活态的药靶特征 [B, 2*d]
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)

        return out