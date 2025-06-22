📌 IMDb Movie Review Sentiment Classifier
A deep learning project that classifies IMDb movie reviews as positive or negative using Recurrent Neural Networks (RNN). The model is trained on 50,000 reviews, automatically split into training and testing sets by the Keras dataset module. This project demonstrates the effectiveness of sequence modeling for natural language processing (NLP) tasks.

It uses word embeddings to represent the input text, applies pre-padding for uniform sequence length, and employs a Simple RNN with one hidden layer. The model is deployed as an interactive web application using Streamlit.

Unlike traditional models that treat text as unordered word collections, this project uses Recurrent Neural Networks (RNNs) to process sequences word by word, capturing the context and meaning of each review. This sequential understanding enables the model to correctly interpret phrases like 'not great' or 'really loved', demonstrating the effectiveness of sequence modeling in sentiment analysis.
