import nltk
from nltk.tokenize import RegexpTokenizer
import re
import torch
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
from pathlib import Path
import sys
import config

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

def remove_punctuation_and_digits(text_series):
    for index, text in text_series.iteritems():
        preprocessed_string = " ".join(tokenizer.tokenize(text))
        preprocessed_string_without_digits = re.sub(r'\d+', '', preprocessed_string)
        preprocessed_string_without_digits_lowercase = preprocessed_string_without_digits.lower()
        text_series.iloc[index] = preprocessed_string_without_digits_lowercase
    return text_series

def transform_to_sequences(text_series, vocab_exists=False):
    text_series = remove_punctuation_and_digits(text_series)
    word2index = {}
    sequences = []
    if vocab_exists:
        with open(config.VOCAB_PATH, 'rb') as fp:
            word2index = pickle.load(fp)
        
        for text in text_series:
            tokens = tokenizer.tokenize(text.lower())
            sequence = [word2index[word] for word in tokens if word2index.get(word) != None]
            sequences.append(sequence)
    else:
        for text in text_series:
            tokens = tokenizer.tokenize(text.lower())
            sequence = [word2index.setdefault(word, len(word2index)) for word in tokens]
            sequences.append(sequence)
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = [torch.LongTensor(seq) for seq in sequences]
    padded_sequences = pad_sequence(padded_sequences, batch_first=True)
    if vocab_exists is False:
        with open(config.VOCAB_PATH, 'wb') as fp:
            pickle.dump(word2index, fp)
    return padded_sequences, len(word2index)