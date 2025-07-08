import numpy as np
import tensorflow as tf
import keras
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model

##Load the IMDB Dataset
word_index = imdb.get_word_index()
rev_word_ind = {value:key for key,value in word_index.items()} ##Loading the IMDB dataset Word index

##Loading the model
model = load_model('simple_rnn_imdb.h5')

##Some functions to preprocess and decode if needed
def decode_review(encoded_review):
    return ' '.join([rev_word_ind.get(i-3,'?') for i in encoded_review])
    ##returning the words from numbers
def preprocess_txt(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words ]  ##converting words tonum
    padded_review = sequence.pad_sequences([encoded_review],padding='pre',maxlen=500) ##padding the input number sequence
    return np.array(padded_review)
def pred_sentiment(review):
    pre_input = preprocess_txt(review) ##pre-processing input
    
    pred = model.predict(pre_input) ##predicting from the pre-trained model
    
    sent = 'Positive' if pred[0][0] > 0.5 else 'Negative'  ##classifying
     
    return sent , pred[0][0]  #remember this model had {0.9} accuracy

import streamlit as st


# Streamlit App UI
st.set_page_config(page_title="IMDB Sentiment Classifier", page_icon="🎬", layout="centered")

st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review and let the model classify it as **Positive** or **Negative**.')

# Input box
user_input = st.text_area('📝 Write your movie review here:', height=150)

if st.button('🔍 Classify Sentiment'):
    if user_input.strip():
        sentiment, score = pred_sentiment(user_input)
        st.success(f"**Sentiment:** {sentiment}")
        st.info(f"**Prediction Confidence:** {score:.4f}")
    else:
        st.warning("⚠️ Please enter a review before clicking the button.")
else:
    st.caption("Waiting for your input...")

st.markdown("</div>", unsafe_allow_html=True)

