import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model
import streamlit as st

# Load Word Index and Reverse Index
word_index = imdb.get_word_index()
rev_word_ind = {value: key for key, value in word_index.items()}

# Load trained model
model = load_model('simple_rnn_imdb.h5')

# Utility Functions
def decode_review(encoded_review):
    return ' '.join([rev_word_ind.get(i - 3, '?') for i in encoded_review])

def preprocess_txt(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], padding='pre', maxlen=500)
    return np.array(padded_review)

def pred_sentiment(review):
    pre_input = preprocess_txt(review)
    pred = model.predict(pre_input)
    sentiment = 'Positive' if pred[0][0] > 0.5 else 'Negative'
    return sentiment, pred[0][0]

# Streamlit App UI
st.set_page_config(page_title="IMDB Sentiment Classifier", page_icon="🎬", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .stTextArea textarea {
        background-color: #ffffff !important;
        border: 1px solid #dcdcdc;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)

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
