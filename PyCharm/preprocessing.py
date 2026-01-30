from tensorflow.keras.preprocessing.sequence import pad_sequences
import re


def tokenize(sentence):
    return re.findall(r"\w+|[^\w\s]", sentence)


def encode_sentence(tokens, word2idx, max_len):
    encoded = [
        word2idx[word] if word in word2idx else 0
        for word in tokens
    ]
    return pad_sequences([encoded], maxlen=max_len, padding="post")


