#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib

# Load the data
df = pd.read_csv('./vgsales_Clean.csv')  # Adjust the path to your CSV file

# Assuming 'Name' is the feature you want to classify and 'Genre' is the target label
y = df['Genre']
X = df['Name']

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# BOW Vectorizer
bow_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_val_bow = bow_vectorizer.transform(X_val)

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)

# Models
mnb = MultinomialNB()
lr = LogisticRegression(max_iter=1000)
rfc = RandomForestClassifier(random_state=0)
svm = SVC(kernel='linear')
knn = KNeighborsClassifier(n_neighbors=3)

# Fit models on BOW features
mnb.fit(X_train_bow, y_train)
lr.fit(X_train_bow, y_train)
rfc.fit(X_train_bow, y_train)
svm.fit(X_train_bow, y_train)
knn.fit(X_train_bow, y_train)

# Evaluate performance on BOW features
mnb_acc_bow = metrics.accuracy_score(y_val, mnb.predict(X_val_bow))
lr_acc_bow = metrics.accuracy_score(y_val, lr.predict(X_val_bow))
rfc_acc_bow = metrics.accuracy_score(y_val, rfc.predict(X_val_bow))
svm_acc_bow = metrics.accuracy_score(y_val, svm.predict(X_val_bow))
knn_acc_bow = metrics.accuracy_score(y_val, knn.predict(X_val_bow))

# Fit models on TF-IDF features
mnb.fit(X_train_tfidf, y_train)
lr.fit(X_train_tfidf, y_train)
rfc.fit(X_train_tfidf, y_train)
svm.fit(X_train_tfidf, y_train)
knn.fit(X_train_tfidf, y_train)

# Evaluate performance on TF-IDF features
mnb_acc_tfidf = metrics.accuracy_score(y_val, mnb.predict(X_val_tfidf))
lr_acc_tfidf = metrics.accuracy_score(y_val, lr.predict(X_val_tfidf))
rfc_acc_tfidf = metrics.accuracy_score(y_val, rfc.predict(X_val_tfidf))
svm_acc_tfidf = metrics.accuracy_score(y_val, svm.predict(X_val_tfidf))
knn_acc_tfidf = metrics.accuracy_score(y_val, knn.predict(X_val_tfidf))

# Print accuracy scores
print("BOW Accuracy Scores:")
print(f"Multinomial Naive Bayes: {mnb_acc_bow}")
print(f"Logistic Regression: {lr_acc_bow}")
print(f"Random Forest Classifier: {rfc_acc_bow}")
print(f"Support Vector Machine: {svm_acc_bow}")
print(f"K Nearest Neighbourhood: {knn_acc_bow}")

print("\nTF-IDF Accuracy Scores:")
print(f"Multinomial Naive Bayes: {mnb_acc_tfidf}")
print(f"Logistic Regression: {lr_acc_tfidf}")
print(f"Random Forest Classifier: {rfc_acc_tfidf}")
print(f"Support Vector Machine: {svm_acc_tfidf}")
print(f"K Nearest Neighbourhood: {knn_acc_tfidf}")

# Save models and vectorizers
joblib.dump(mnb, 'mnb_model.pkl')
joblib.dump(lr, 'lr_model.pkl')
joblib.dump(rfc, 'rfc_model.pkl')
joblib.dump(svm, 'svm_model.pkl')
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(bow_vectorizer, 'bow_vectorizer.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
