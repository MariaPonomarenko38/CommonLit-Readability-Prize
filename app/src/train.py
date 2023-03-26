import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from model import LSTMModel
from data_preparation import *
from engine import *
from pathlib import Path

from sklearn.model_selection import train_test_split
import numpy as np
import nltk

from dataset import TextDataset
import json

import config as config

def process_data(data_path):

    text_series = pd.read_csv(data_path)['excerpt']
    targets = list(pd.read_csv(data_path)['target'])

    padded_sequences, vocab_size = transform_to_sequences(text_series)

    return padded_sequences, targets, vocab_size

BASE_DIR = Path(__file__).resolve(strict=True).parent

if __name__ == '__main__':

    padded_sequences, targets, vocab_size = process_data(f"{BASE_DIR}/../input/train.csv")
    X_train, X_val, y_train, y_val = train_test_split(padded_sequences, targets, test_size=0.2)
    y_train = torch.tensor(y_train, dtype=torch.float)
    y_val = torch.tensor(y_val, dtype=torch.float)

    train_dataset = TextDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dataset = TextDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = LSTMModel(vocab_size, config.EMB_DIM, config.HIDDEN_SIZE, config.NUM_LAYERS, config.OUTPUT_SIZE)
    optimizer = optim.Adam(model.parameters(), lr= config.LEARNING_RATE)

    criterion = nn.MSELoss()

    results = {'train_loss': [], 'val_loss': [], 'val_rmse': []}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_loss = np.inf

    for epoch in range(config.NUM_EPOCHS):
        
        total_loss, total_batches = train_loop(model, train_loader, criterion, optimizer, device)
        avg_train_loss = total_loss / total_batches

        model.eval()
        total_val_loss, total_val_rmse, total_val_batches = val_loop(model, train_loader, criterion, device)
        
        avg_val_loss = total_val_loss / total_val_batches
        avg_val_rmse = total_val_rmse / total_val_batches
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val RMSE = {avg_val_rmse:.4f}")
        
        results['train_loss'].append(avg_train_loss)
        results['val_loss'].append(avg_val_loss)
        results['val_rmse'].append(avg_val_rmse)
        
        if total_val_loss < best_loss:
            torch.save(model.state_dict(), f"{BASE_DIR}/../models/model.bin")
            best_loss = avg_val_loss

        with open(f"{BASE_DIR}/../metrics.json", 'w') as f:
            json.dump(results, f)