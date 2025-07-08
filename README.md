# 🎭 Sentiment Analysis Classifier using RNN

A deep learning project that classifies IMDB movie reviews as **Positive** or **Negative** using an RNN-based NLP model built with **Keras** and deployed using **Streamlit**.

## 🔍 Project Overview

This project applies **Natural Language Processing (NLP)** techniques and **Recurrent Neural Networks (RNNs)** to perform binary sentiment classification on textual movie reviews. The model learns patterns in text sequences and predicts whether a given review conveys a positive or negative sentiment.

### 🌐 Live Demo
👉 [Try the Web App](https://movreviewsentimentclassifier-iqhyv3fchkx5k8u9h8rkcu.streamlit.app/)

### 📂 GitHub Repository
🔗 [View Code](https://github.com/lokesh-saragadam/DL_simple_RNN_project.git)

---

## 🧠 Key Features

- Uses **Keras IMDB** dataset containing preprocessed movie reviews
- **One-Hot Encoding** applied for text representation
- **Embedding + SimpleRNN** architecture for sequence modeling
- Optimized with **Adam** and **binary_crossentropy** loss
- Implements **EarlyStopping** to prevent overfitting
- Fully **deployed using Streamlit Cloud** for easy interaction

---

## 🧰 Tech Stack

| Component        | Tool/Library      |
|------------------|-------------------|
| Language         | Python            |
| Deep Learning    | Keras, TensorFlow |
| Web Deployment   | Streamlit         |
| Dataset          | Keras IMDB (50,000 labeled reviews) |

---

## 🏗️ Model Architecture
```text
Embedding (input_dim=10000, output_dim=32, input_length=500)
→ SimpleRNN (units=32)
→ Dense (1, activation='sigmoid')
```

---
## 🤔 What I Learned
- Implementing RNNs for sequential data like text

- Using Embedding layers to represent word vectors

- Preventing overfitting with EarlyStopping

- Deploying machine learning models with Streamlit

- Improving user interactivity and visualization for ML apps

- The importance of balancing model complexity vs. performance

## 📌 Future Improvements
- Use LSTM or GRU for improved long-term dependencies

- Incorporate Word2Vec/GloVe embeddings

- Add confusion matrix and precision/recall metrics


## 📬 Contact
# If you have feedback or suggestions, feel free to connect:

- 🔗 LinkedIn – Lokesh Saragadam
- 📧 lokeshs2k6@gmail.com

