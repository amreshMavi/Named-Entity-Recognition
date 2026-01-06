# Named-Entity-Recognition
In Machine Learning Named Entity Recognition (NER) is a task of Natural Language Processing to identify the named entities in a certain piece of text.

![Uploading image.png…]()


# Named Entity Recognition (NER) using BiLSTM

## 📌 Overview
This project implements a Named Entity Recognition (NER) system using a deep learning approach.  
NER is a core Natural Language Processing (NLP) task that involves identifying and classifying entities such as persons, locations, organizations, and others within text.

The model treats NER as a sequence labeling problem and predicts a tag for each token in a sentence.

---

## 🧠 Model Architecture
The NER model is built using TensorFlow and Keras with the following architecture:

- **Embedding Layer**
  - Trainable word embeddings
  - Embedding dimension: 64

- **Bidirectional LSTM (BiLSTM)**
  - Captures both past and future context in a sentence
  - Suitable for sequence-based NLP tasks

- **TimeDistributed Dense Layer**
  - Applies classification at each timestep
  - Outputs a tag for every token in the sequence

This architecture is commonly used for classical NER tasks and performs well on sequential text data.

---

## 📂 Dataset & Preprocessing
- The dataset consists of tokenized sentences with corresponding NER tags.
- Words and tags are converted to integer indices.
- Sentences are padded to a fixed maximum length to allow batch processing.
- The data is split into:
  - Training set
  - Validation set
  - Test set

---

## 🚀 Training
- Framework: **TensorFlow 2.x**
- Training performed using GPU acceleration when available
- Loss function: categorical cross-entropy
- Optimizer: Adam
- The model is trained for multiple epochs with validation monitoring

---

## 📊 Results
The notebook demonstrates successful training of a BiLSTM-based NER model.
Evaluation metrics such as entity-level F1 score and precision/recall can be added as future improvements.

---

## 🛠️ How to Run
1. Clone the repository
2. Install dependencies:
