import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

def carregar_uci_har(caminho_base='./data/UCI HAR Dataset'):
    # (Mantém igual ao anterior, sem mudanças aqui)
    try:
        X_train = pd.read_csv(f'{caminho_base}/train/X_train.txt', sep=r'\s+', header=None).values
        y_train = pd.read_csv(f'{caminho_base}/train/y_train.txt', sep=r'\s+', header=None).values.flatten() - 1 
        sub_train = pd.read_csv(f'{caminho_base}/train/subject_train.txt', sep=r'\s+', header=None).values.flatten()
        return X_train, y_train, sub_train
    except FileNotFoundError:
        print(f"ERRO: Não achei os arquivos em '{caminho_base}'.")
        exit()

def criar_clientes_federados(X, y, subjects, n_clientes=5, heterogeneidade_real=False):
    clientes_data = {}
    unique_subjects = np.unique(subjects)
    
    if n_clientes > len(unique_subjects):
        n_clientes = len(unique_subjects)
        
    selected_subjects = unique_subjects[:n_clientes]
    
    for i, sub_id in enumerate(selected_subjects):
        indices = np.where(subjects == sub_id)[0]
        
        X_numpy = X[indices]
        
        # Aqui o dado inverte se a heterogeneidade estiver ligada!
        if heterogeneidade_real and (i % 2 == 0):
            X_numpy = X_numpy * -1 
            
        X_local = torch.tensor(X_numpy, dtype=torch.float32).unsqueeze(1)
        y_local = torch.tensor(y[indices], dtype=torch.long)
        
        dataset = TensorDataset(X_local, y_local)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        clientes_data[i] = loader
        
    return clientes_data