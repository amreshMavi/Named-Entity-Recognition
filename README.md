# Named-Entity-Recognition
In Machine Learning Named Entity Recognition (NER) is a task of Natural Language Processing to identify the named entities in a certain piece of text.

<img width="649" height="322" alt="delete" src="https://github.com/user-attachments/assets/0dc9ecfd-a30d-4168-9111-b8eb1bb44461" />

# Named Entity Recognition (NER) using BiLSTM

### 📌 Overview
This project implements a Named Entity Recognition (NER) system using a deep learning approach.  
NER is a core Natural Language Processing (NLP) task that involves identifying and classifying entities such as persons, locations, organizations, and others within text.

The model treats NER as a sequence labeling problem and predicts a tag for each token in a sentence.

---

### 🧠 Model Architecture
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

### 📂 Dataset & Preprocessing
- The dataset consists of tokenized sentences with corresponding NER tags.
- Words and tags are converted to integer indices.
- Sentences are padded to a fixed maximum length to allow batch processing.
- The data is split into:
  - Training set
  - Validation set
  - Test set

---

### 🚀 Training
- Framework: **TensorFlow 2.x**
- Training performed using GPU acceleration when available
- Loss function: categorical cross-entropy
- Optimizer: Adam
- The model is trained for multiple epochs with validation monitoring

---

### 📊 Results
The notebook demonstrates successful training of a BiLSTM-based NER model.
Evaluation metrics such as entity-level F1 score and precision/recall can be added as future improvements.

---

### 🛠️ How to Run
Follow these instructions to set up the environment and run the Named Entity Recognition model.

### 📋 Prerequisites
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.
### Create a virtual environment
python -m venv venv

### Activate it
### On Windows:
activate (env name eg:- e:/tf210 or just tf210)
### On Mac/Linux:
source venv/bin/activate

### ⚙️ Installation
1. Clone the repository.
2. Install dependencies:
   pip install pandas numpy tensorflow spacy
3. Download the spaCy English model:
   python -m spacy download en_core_web_sm

### 📂 Dataset Setup
Place your dataset (ner_dataset.csv) in a folder named data within the root directory

### 🏃 Running the Notebook
Start the Jupyter Notebook server
To test the frontend, you will have to save the model and put it in the "artifacts" folder in PyCharm.

### 🔮 Future Improvements
- Add CRF layer for improved tag decoding
- Compute entity-level precision, recall, and F1 score
- Experiment with pretrained embeddings (GloVe / FastText)
- Extend the model using Transformer-based architectures (BERT)

---

### 📚 Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Pandas

---

### 👤 Author
Amresh
