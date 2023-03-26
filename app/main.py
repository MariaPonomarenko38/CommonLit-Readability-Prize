from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import sys
#import torch
#from app.src.data_preparation import transform_to_sequences
#from app.src.model import LSTMModel
#import app.src.config as config
from pathlib import Path
import pickle

app = FastAPI()

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/linear_regression_model.pkl", 'rb') as file:
    model = pickle.load(file)

with open(f"{BASE_DIR}/count_vectorizer.pkl", 'rb') as file:
    vectorizer = pickle.load(file)

def predict(text):
    X_new = vectorizer.transform([text])
    y_pred = model.predict(X_new)
    return y_pred[0]

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