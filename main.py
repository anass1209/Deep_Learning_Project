# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

word_index = imdb.get_word_index()

model = load_model('rnn_imdb.h5')

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) for word in words] #! 2 parce que la valeur a comme indice 2 est and -> {2: 'and'}
    encoded_review = [min(i, 9999) for i in encoded_review]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]



st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# Entrée utilisateur
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    sentiment, score = predict_sentiment(user_input)
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {score}')
else:
    st.write('Please enter a movie review.')
