📌 IMDb Movie Review Sentiment Classifier
A deep learning project that classifies IMDb movie reviews as positive or negative using Recurrent Neural Networks (RNN). The model is trained on 50,000 reviews, automatically split into training and testing sets by the Keras dataset module. This project demonstrates the effectiveness of sequence modeling for natural language processing (NLP) tasks.

It uses word embeddings to represent the input text, applies pre-padding for uniform sequence length, and employs a Simple RNN with one hidden layer. The model is deployed as an interactive web application using Streamlit.

Unlike traditional models that treat text as unordered word collections, this project uses Recurrent Neural Networks (RNNs) to process sequences word by word, capturing the context and meaning of each review. This sequential understanding enables the model to correctly interpret phrases like 'not great' or 'really loved', demonstrating the effectiveness of sequence modeling in sentiment analysis.
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
| Visualization    | Matplotlib, Streamlit UI |
| Dataset          | Keras IMDB (50,000 labeled reviews) |

---

## 🏗️ Model Architecture

