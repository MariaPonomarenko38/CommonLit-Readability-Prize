from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import sys
#import torch
#from app.src.data_preparation import transform_to_sequences
#from app.src.model import LSTMModel
#import app.src.config as config
from pathlib import Path
from predict import *

app = FastAPI()

BASE_DIR = Path(__file__).resolve(strict=True).parent

'''
def process_single_entry(text):
    text_series = pd.Series([text])
    padded_sequences, vocab_size = transform_to_sequences(text_series, True)
    return padded_sequences, vocab_size

def predict(sequences, vocab_size):
    model = LSTMModel(vocab_size, config.EMB_DIM, config.HIDDEN_SIZE, config.NUM_LAYERS, config.OUTPUT_SIZE)
    model.load_state_dict(torch.load(f"{BASE_DIR}/../models/model.bin"))
    model.eval()
    outputs = model(sequences).squeeze()
    return outputs
'''
class UserRequestIn(BaseModel):
    text: str

@app.post("/prediction")
def read_classification(user_request_in: UserRequestIn):
    return predict(user_request_in.text)
    #padded_sequences, vocab_size = process_single_entry(user_request_in.text)
    #return predict(padded_sequences, vocab_size).item()

@app.get("/")
def home():
    return {"health_check": "OK"}

import sys 
print(sys.path)