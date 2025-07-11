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

##Taking in input
user_input = st.text_area('📝 Write your movie review here:')

if st.button('🔍 Classify Sentiment'):
    if not user_input.strip():
        st.warning("⚠️ Please enter a review before classifying.")
    else:
        try:
            processed_inp = preprocess_txt(user_input)

            #Prediction
            prediction = model.predict(processed_inp)
            sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'
        
            st.success(f"Sentiment : {sentiment}")
            st.info(f"Prediction score: {prediction[0][0]}")
        except Exception as e:
            st.error("🚫 Prediction failed due to unexpected error.")
            st.exception(e)  # optional: show full error
else :
    st.write('Please enter a movie review')    
