import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import pickle
import streamlit.components.v1 as components

# Load the trained model and vectorizer using pickle
with open('svmBOW.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('BOWvectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Set custom background image from a local directory
st.set_page_config(
    page_title="Game Genre Prediction App",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)
background_image_style = """
    <style>
        body {
            background-image: url('');
            background-size: cover;
        }
    </style>
"""
st.markdown(background_image_style, unsafe_allow_html=True)

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

    # Transform the input using the loaded vectorizer
    input_vectorized = vectorizer.transform([cleaned_input])

    # Make prediction using the loaded model
    prediction = model.predict(input_vectorized)[0]

    # Display the predicted genre with a border shape
    st.markdown(
        f'<div style="border: 2px solid white; padding: 10px; border-radius: 10px; background-color: rgba(255, 255, 255, 0.5);">{prediction}</div>',
        unsafe_allow_html=True,
    )
else:
    st.info("Please enter the name of the game to predict its genre.")

# Embed Landbot using HTML iframe
components.html(
    """
   <iframe width="350" height="430" allow="microphone;" src="https://console.dialogflow.com/api-client/demo/embedded/124565ef-4cda-4604-8fee-c4c577e7dc55"></iframe>
    """,
    height=440,
    width=360,
)
