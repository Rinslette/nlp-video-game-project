import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import pickle
import streamlit.components.v1 as components

# Download NLTK stopwords data
nltk.download('stopwords')

# Set custom background image from a URL
st.set_page_config(
    page_title="Game Genre Prediction App",
    page_icon="🎮",
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style>
        .stApp {
        background: url("https://r4.wallpaperflare.com/wallpaper/96/92/869/game-games-2014-best-wallpaper-a94028fd717a4d2bd6c7181f7021068d.jpg");
        background-size: cover;
        }
    </style>""", unsafe_allow_html=True)

# Load the trained model and vectorizer using pickle
with st.markdown("""
    <div style="padding: 20px; max-width: 800px; background-color: rgba(255, 255, 255, 0.8); border-radius: 10px;">
        <h1>Game Genre Prediction App</h1>
        <p>Enter the name of the game, and I'll predict its genre!</p>
        <label for="game_name">Enter the name of the game:</label>
        <input type="text" id="game_name" name="game_name">
        <button onclick="predictGenre()">Predict Genre</button>
        <p id="predicted_genre"></p>
    </div>
"""):
    with open('svmBOW.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('BOWvectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    st.markdown("""
        <script>
            function predictGenre() {
                var userInput = document.getElementById("game_name").value;
                if (userInput.trim() !== "") {
                    // Clean the user input
                    // ... (rest of the cleaning and prediction logic)
                    // Display the predicted genre
                    document.getElementById("predicted_genre").innerText = "Predicted Genre: " + prediction;
                } else {
                    alert("Please enter the name of the game to predict its genre.");
                }
            }
        </script>
    """)
    
    components.html(
        """
        <script src="https://www.gstatic.com/dialogflow-console/fast/messenger/bootstrap.js?v=1"></script>
        <df-messenger
          intent="WELCOME"
          chat-title="AwleBot"
          agent-id="124565ef-4cda-4604-8fee-c4c577e7dc55"
          language-code="en"
        ></df-messenger>
        """,
        height=300,
        width=300,
)
