import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import pickle
import streamlit.components.v1 as components
from streamlit_extras.stylable_container import stylable_container

# Download NLTK stopwords data
nltk.download('stopwords')

# Set custom background image from a URL
st.set_page_config(
    page_title="Game Genre Prediction App",
    page_icon="🎮",
    initial_sidebar_state="collapsed",
)

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://r4.wallpaperflare.com/wallpaper/96/92/869/game-games-2014-best-wallpaper-a94028fd717a4d2bd6c7181f7021068d.jpg");
background-size: cover;
background-repeat: no-repeat;
background-attachment: local;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)



# Load the trained model and vectorizer using pickle
with open('svmBOW.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('BOWvectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)
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
    # Display the predicted genre
    st.write(f"Predicted Genre: {prediction}")
else:
    st.info("Please enter the name of the game to predict its genre.")

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
