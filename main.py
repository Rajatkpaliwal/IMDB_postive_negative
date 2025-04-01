import numpy as np
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Function to decode reviews
def decoded_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return np.array(padded_review)

## Streamlit app
import streamlit as st
st.title('IMDB Movie Review Setiment analysis')
st.write('Enter a moview review to classify it as positive or negativ')

# User input
user_input = st.text_area('Moview Review')
if st.button('Classify'):
    preprocess_input = preprocess_text(user_input)
    # make prediction
    prediction = model.predict(preprocess_input)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

    # Display the result
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]}")
else:
    st.write('Please enter a movie review.')