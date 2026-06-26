import numpy as np
import pandas as pd
import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset

from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from models.ginconv_test import GINConvNet_test
from models.new import HBAI_DTA
from utils import *
from torch_geometric.data import DataLoader


def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)

        if isinstance(output, tuple) or isinstance(output, list):
            output = output[0]

        loss = loss_fn(output, data.y.view(-1, 1).float())

        loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            print(
                f'Train epoch: {epoch} [{batch_idx * len(data.x)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\t'
                f'Total Loss (MSE): {loss.item():.6f}'
            )

def predicting(model, device, loader, dataset_name='davis'):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)

            # 兼容性处理
            if isinstance(output, tuple) or isinstance(output, list):
                output = output[0]

            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

    preds_np = total_preds.numpy().flatten()
    labels_np = total_labels.numpy().flatten()

    smiles_list = []
    sequence_list = []

    try:
        smiles_list = [d.smiles for d in loader.dataset]
        sequence_list = [d.sequence for d in loader.dataset]
        print("Successfully obtained the SMILES and sequence from the dataset object.")
    except AttributeError:
        print("There is no SMILES information in the Dataset. We are currently attempting to read it from the original CSV file...")
        possible_paths = [
            f'data/{dataset_name}/test.csv',
            f'data/{dataset_name}_test.csv',
            f'data/test.csv'
        ]

        raw_test_path = None
        for p in possible_paths:
            if os.path.exists(p):
                raw_test_path = p
                break

        if raw_test_path:
            df_raw = pd.read_csv(raw_test_path)
            if 'compound_iso_smiles' in df_raw.columns:
                smiles_list = df_raw['compound_iso_smiles'].tolist()
            elif 'smiles' in df_raw.columns:
                smiles_list = df_raw['smiles'].tolist()
            else:
                smiles_list = ["N/A"] * len(preds_np)

            if 'target_sequence' in df_raw.columns:
                sequence_list = df_raw['target_sequence'].tolist()
            elif 'sequence' in df_raw.columns:
                sequence_list = df_raw['sequence'].tolist()
            else:
                sequence_list = ["N/A"] * len(preds_np)

            smiles_list = smiles_list[:len(preds_np)]
            sequence_list = sequence_list[:len(preds_np)]
        else:
            smiles_list = ["Missing"] * len(preds_np)
            sequence_list = ["Missing"] * len(preds_np)

    df_results = pd.DataFrame({
        'drug_smiles': smiles_list,
        'protein_seq': sequence_list,
        'RealValue': labels_np,
        'Prediction': preds_np
    })


    return labels_np, preds_np, df_results

if __name__ == "__main__":
    TRAIN_BATCH_SIZE = 512
    TEST_BATCH_SIZE = 512
    LR = 0.0005
    LOG_INTERVAL = 20
    NUM_EPOCHS = 1000

    modeling = [GINConvNet_test, GATNet, GAT_GCN, GCNNet]
    datasets = ['davis', 'kiba']

    dataset = datasets[int(sys.argv[1])]
    model_idx = int(sys.argv[2])
    model = modeling[model_idx]
    cuda_name = "cuda:" + str(sys.argv[3])
    model_st = model.__name__

    print(f'Running on {model_st}_{dataset}')

    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'

    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('Please run create_data.py to prepare data first!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset + '_train')
        test_data = TestbedDataset(root='data', dataset=dataset + '_test')

        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = model().to(device)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_mse = 1000
        best_ci = 0
        model_file_name = 'model_' + model_st + '_' + dataset + '.model'
        result_file_name = 'result_' + model_st + '_' + dataset + '.csv'

        for epoch in range(NUM_EPOCHS):

            train(model, device, train_loader, optimizer, epoch + 1)

            G, P, df_results = predicting(model, device, test_loader, dataset_name=dataset)
            ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]

            if ret[1] < best_mse:
                torch.save(model.state_dict(), model_file_name)
                with open(result_file_name, 'w') as f:
                    f.write(','.join(map(str, ret)))

                best_csv_name = f'best_{model_st}_{dataset}_results.csv'
                df_results.to_csv(best_csv_name, index=False)

                best_mse = ret[1]
                best_ci = ret[-1]
                print(f'Epoch {epoch + 1}, MSE improved to {best_mse:.4f}, CI improved to {best_ci:.4f}. Model & CSV saved.')

            print(f'Current Best MSE: {best_mse:.4f}, Best CI: {best_ci:.4f}')

        print(f'Finished training. Overall Best MSE: {best_mse:.4f}, Best CI: {best_ci:.4f}')