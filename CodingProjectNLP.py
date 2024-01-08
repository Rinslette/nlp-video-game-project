import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import joblib

# Load the trained model
model = joblib.load('./svmBOW.pkl')

# Streamlit app title and description
st.title("Game Genre Prediction App")
st.write("Enter the name of the game, and I'll predict its genre!")

# User input for the game name
user_input = st.text_input("Enter the name of the game:")

# Check if the user has entered a game name
if user_input:
    # Clean the input text using the same clean_text function as in the original code
    def clean_text(text):
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [t for t in tokens if t.isalpha()]
        tokens = [t for t in tokens if t not in stopwords.words('english')]
        roman_re = r'\bM{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b'
        tokens = [t for t in tokens if not re.match(roman_re, t, flags=re.IGNORECASE).group()]
        text = ' '.join(tokens).strip()
        return text

    # Clean the user input
    cleaned_input = clean_text(user_input)

    # Make prediction using the loaded model
    prediction = model.predict([cleaned_input])[0]

    # Display the predicted genre
    st.write(f"Predicted Genre: {prediction}")
else:
    st.info("Please enter the name of the game to predict its genre.")
