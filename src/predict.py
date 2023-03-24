import torch
from data_preparation import *
import pandas as pd
from model import LSTMModel

import config

def process_data(file_path):
    text_series = pd.read_csv(file_path)['excerpt']
    padded_sequences, vocab_size = transform_to_sequences(text_series, True)
    return padded_sequences, vocab_size

def process_single_entry(text):
    text_series = pd.Series([text])
    padded_sequences, vocab_size = transform_to_sequences(text_series, True)
    return padded_sequences, vocab_size

def predict(sequences, vocab_size):
    model = LSTMModel(vocab_size, config.EMB_DIM, config.HIDDEN_SIZE, config.NUM_LAYERS, config.OUTPUT_SIZE)
    model.load_state_dict(torch.load(config.OUTPUT_PATH))
    model.eval()
    outputs = model(sequences).squeeze()
    return outputs

if __name__ == '__main__':
    padded_sequences, vocab_size = process_data(config.TEST_FILE)
    print(predict(padded_sequences, vocab_size))
