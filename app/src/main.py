from fastapi import FastAPI
from pydantic import BaseModel, constr, conlist
from typing import List
import sys
import os
sys.path.append('../../src/')
from predict import predict as predict_func
from predict import process_single_entry

app = FastAPI()

class UserRequestIn(BaseModel):
    text: str

@app.post("/prediction")
def read_classification(user_request_in: UserRequestIn):
    padded_sequences, vocab_size = process_single_entry(user_request_in.text)
    return predict_func(padded_sequences, vocab_size).item()