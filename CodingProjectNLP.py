import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import nltk
import re
from nltk.corpus import stopwords

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

# Load the pre-trained model and vectorizer
model = SVC(kernel='linear')
tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))

# Assuming df is your original dataframe
df = pd.read_csv('Balance.csv')

# Clean the 'Name' column
df['Name'] = df['Name'].apply(clean_text)

# Split the data into training and validation sets
y = df['Genre']
x = df['Name']
xtrain, xval, ytrain, yval = train_test_split(x, y, test_size=0.2)

# Create the TF-IDF features
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)

# Fit the model on the training data
model.fit(xtrain_tfidf, ytrain)

# Streamlit app
st.title("Video Game Genre Prediction App")

# User input for game name
user_input = st.text_input("Enter the name of a video game:")

if user_input:
    # Clean the user input
    cleaned_input = clean_text(user_input)

    # Vectorize the cleaned input
    input_vectorized = tfidf_vectorizer.transform([cleaned_input])

    # Make prediction using the trained model
    prediction = model.predict(input_vectorized)[0]

    st.success(f"The predicted genre for '{user_input}' is: {prediction}")
