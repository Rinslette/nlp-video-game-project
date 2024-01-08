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
    layout="wide",
    initial_sidebar_state="collapsed",
)

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

# Add a rounded logo chat button in the bottom-left corner
st.markdown(
    """
    <style>
        .chat-button {
            position: fixed;
            bottom: 20px;
            left: 20px;
            background-color: #007BFF;
            border-radius: 50%;
            padding: 10px;
            cursor: pointer;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }

        .chat-button img {
            width: 30px;
            height: 30px;
        }
    </style>
    <div class="chat-button" onclick="toggleChatbot()">
        <img src="https://example.com/your_logo.png" alt="Chatbot Logo">
    </div>

    <script>
        let chatbotVisible = true;

        function toggleChatbot() {
            // Toggle chatbot visibility
            chatbotVisible = !chatbotVisible;

            // Use Streamlit components to update visibility
            Streamlit.setComponentValue(chatbotVisible, 'chatbotVisibility');
        }
    </script>
    """,
    unsafe_allow_html=True,
)

# Use Streamlit components to control chatbot visibility
chatbot_visibility = st.components.v1.html('<div></div>', height=0)
if chatbot_visibility.value:
    st.components.v1.iframe("https://console.dialogflow.com/api-client/demo/embedded/124565ef-4cda-4604-8fee-c4c577e7dc55", height=430, width=350)
