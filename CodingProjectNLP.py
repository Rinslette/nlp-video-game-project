import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import joblib  # Use joblib directly for loading the .pkl file
import nltk
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from pathlib import Path

# Set NLTK data path
nltk.data.path.append(str(Path(__file__).parent.resolve()) + '/nltk_data')

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to clean text
def clean_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    roman_re = r'\bM{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b'
    tokens = [t for t in tokens if not re.match(roman_re, t, flags=re.IGNORECASE).group()]
    text = ' '.join(tokens).strip()
    return text

# Load the pre-trained model from .pkl file
model = joblib.load('./svmTFIDF_model.pkl')  # Replace 'your_model.pkl' with the actual file name

# Load the data
df = pd.read_csv('./vgsales_Clean.csv')

# Clean the 'Name' column
df['Name'] = df['Name'].apply(clean_text)

# Split the data into training and validation sets
y = df['Genre']
x = df['Name']
xtrain, xval, ytrain, yval = train_test_split(x, y, test_size=0.2)

# Create the Bag of Words features
bow_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))
xtrain_bow = bow_vectorizer.fit_transform(xtrain)
xval_bow = bow_vectorizer.transform(xval)

# Streamlit app
st.title("Video Game Genre Prediction App")

# Display information about the trained model
st.header("Trained Model Details")
st.text("Model loaded from svmTFIDF_model.pkl")

# User input for game name
user_input = st.text_input("Enter the name of a video game:")

if user_input:
    # Clean the user input
    cleaned_input = clean_text(user_input)

    # Vectorize the cleaned input
    input_vectorized = bow_vectorizer.transform([cleaned_input])

    # Make prediction using the loaded model
    prediction = model.predict(input_vectorized)[0]

    st.success(f"The predicted genre for '{user_input}' is: {prediction}")
