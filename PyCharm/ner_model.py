# Load model once
# Run prediction
# Convert indices → tags

import pickle
from tensorflow.keras.models import load_model
import streamlit as st
from preprocessing import tokenize, encode_sentence
import numpy as np

MAX_LEN = 104


@st.cache_resource
def load_model_and_maps():
    model = load_model("artifacts/model.h5")
    word2idx = pickle.load(open("artifacts/word2idx.pkl", "rb"))
    idx2tag = pickle.load(open("artifacts/idx2tag.pkl", "rb"))
    return model, word2idx, idx2tag


model, word2idx, idx2tag = load_model_and_maps()


def predict_entities(text):
    tokens = tokenize(text)
    encoded = encode_sentence(tokens, word2idx, MAX_LEN)
    predictions = model.predict(encoded)[0]
    # first_sentence_preds = predictions[0]
    tags = [idx2tag[np.argmax(p)] for p in predictions][:len(tokens)]
    return list(zip(tokens, tags))
