#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
import re

from sklearn.preprocessing import MultiLabelBinarizer

# Load the data
df = pd.read_csv('/vgsales_Clean.csv')
# Reduce dataframe to feature 'Name', and label 'Genre'
df = df[['Name','Genre']].copy()
df.head()


# In[25]:


tokens = nltk.word_tokenize(' '.join(df.Name.values))
text = nltk.Text(tokens)
fdist = FreqDist(text)
print(fdist)
fdist.most_common(50)


# In[26]:


fdist.plot(50,cumulative=True,title='Top 50 Word Frequency Cumulative Plot')


# In[27]:


text.collocations()


# In[28]:


genres = list(df.Genre.unique())
d = {}
for g in genres:
    names = list(df[df.Genre == g].Name.values)
    tokens = nltk.word_tokenize(' '.join(names))
    types = set(tokens)
    lexical_diversity = round(len(types) / len(tokens),3)
    d[g] = (len(tokens), len(types), lexical_diversity)
    
    #print(f"{g}: TOKENS: {len(tokens)}, TYPES: {len(types)}, LEXICAL DIVERSITY: {lexical_diversity}")
table = pd.DataFrame.from_dict(d,orient='index',columns=['tokens','type','lexical_diversity'])
display(table.sort_values('lexical_diversity'))


# In[29]:


def clean_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    # Remove roman numerals using regex
    roman_re = r'\bM{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b'
    tokens = [t for t in tokens if not re.match(roman_re,t,flags=re.IGNORECASE).group()]
    
    text = ' '.join(tokens).strip()
    
    return text

df.Name = df.Name.apply(lambda n: clean_text(n))
df.sample(20)


# In[30]:


df.Genre.value_counts(ascending=True).plot(kind='barh')
plt.show()


# # SKIP HERE

# In[ ]:


df_grouped_by = df.groupby(['Genre'])
 
df_balanced = df_grouped_by.apply(lambda x: x.sample(df_grouped_by.size().min()).reset_index(drop=True))
 
df_balanced = df_balanced.droplevel(['Genre'])
df_balanced


# In[62]:


df_balanced.Genre.value_counts(ascending=True).plot(kind='barh')
plt.show()


# In[46]:


df_balanced.to_csv('Balancevgsales.csv')


# In[63]:


from sklearn.datasets import load_iris
from sklearn.utils import shuffle
import pandas as pd
import numpy as np


# Using scikit-learn
df2 = shuffle(df_balanced)
print("Dataset shuffled using Sklearn:")
print(df2.head())


# In[64]:


df2.to_csv('Balance.csv')


# In[36]:


from numpy.random import default_rng

arr_indices_top_drop = default_rng().choice(df.index, size=4000, replace=False)
df['Genre'].drop(index=arr_indices_top_drop)=='Action'

                

            


# In[22]:


df


# In[8]:


genres = list(df.Genre.unique())
d = {}
for g in genres:
    names = list(df[df.Genre == g].Name.values)
    tokens = nltk.word_tokenize(' '.join(names))
    types = set(tokens)
    lexical_diversity = round(len(types) / len(tokens),3)
    d[g] = (len(tokens), len(types), lexical_diversity)
    
    #print(f"{g}: TOKENS: {len(tokens)}, TYPES: {len(types)}, LEXICAL DIVERSITY: {lexical_diversity}")
table = pd.DataFrame.from_dict(d,orient='index',columns=['tokens','type','lexical_diversity'])
display(table.sort_values('lexical_diversity'))


# # SKIP END

# In[31]:


tokens = nltk.word_tokenize(' '.join(df.Name.values))
text = nltk.Text(tokens)
fdist = FreqDist(text)
print(fdist)
fdist.most_common(50)


# In[32]:


fdist.plot(50,cumulative=True,title='Top 50 Word Frequency Cumulative Plot')


# In[34]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier# Applying k = 3, default Minkowski distance metrics
from sklearn.metrics import f1_score
from sklearn import metrics

bow_vectorizer = CountVectorizer(analyzer='word',ngram_range=(1,2))

# split dataset into training and validation set
y = df.Genre
x = df.Name
xtrain, xval, ytrain, yval = train_test_split(x,y, test_size = 0.2)

# create the TF-IDF features
xtrain_bow = bow_vectorizer.fit_transform(xtrain)
xval_bow = bow_vectorizer.transform(xval)


mnb = MultinomialNB()
lr = LogisticRegression(max_iter=1000)
rfc = RandomForestClassifier(random_state=0)
svm = SVC(kernel='linear')
knn = KNeighborsClassifier(n_neighbors=3)

# fit model on train data
mnb.fit(xtrain_bow, ytrain)
lr.fit(xtrain_bow, ytrain)
rfc.fit(xtrain_bow, ytrain)
svm.fit(xtrain_bow, ytrain)
knn.fit(xtrain_bow, ytrain)

# make predictions for validation set
mnb_pred = mnb.predict(xval_bow)
lr_pred = lr.predict(xval_bow)
rfc_pred = rfc.predict(xval_bow)
svm_pred = svm.predict(xval_bow)
knn_pred = knn.predict(xval_bow)

# evaluate performance
mnb_acc = metrics.accuracy_score(yval, mnb_pred)
mnb_acc = round(mnb_acc,2)
lr_acc = metrics.accuracy_score(yval, lr_pred)
lr_acc = round(lr_acc,2)
rfc_acc = metrics.accuracy_score(yval, rfc_pred)
rfc_acc = round(rfc_acc,2)
svm_acc = metrics.accuracy_score(yval, svm_pred)
svm_acc = round(svm_acc,2)
knn_acc = metrics.accuracy_score(yval, knn_pred)
knn_acc = round(knn_acc,2)


# In[35]:


print(f"BOW Accuracy Scores:\nMultinomial Naive Bayes: {mnb_acc}")
print(f"Logistic Regression: {lr_acc}")
print(f"Random Forest Classifier: {rfc_acc}")
print(f"Support Vector Machine: {svm_acc}")
print(f"K Nearest Neighbourhood: {knn_acc}")


# In[36]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(analyzer='word',ngram_range=(1,2))

# split dataset into training and validation set
y = df.Genre
x = df.Name
xtrain, xval, ytrain, yval = train_test_split(x,y, test_size = 0.2)

# create the TF-IDF features
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)


mnb = MultinomialNB()
lr = LogisticRegression(max_iter=1000)
rfc = RandomForestClassifier(random_state=0)
svm = SVC(kernel='linear')
knn = KNeighborsClassifier(n_neighbors=3)

# fit model on train data
mnb.fit(xtrain_tfidf, ytrain)
lr.fit(xtrain_tfidf, ytrain)
rfc.fit(xtrain_tfidf, ytrain)
svm.fit(xtrain_tfidf, ytrain)
knn.fit(xtrain_tfidf, ytrain)

# make predictions for validation set
mnb_pred = mnb.predict(xval_tfidf)
lr_pred = lr.predict(xval_tfidf)
rfc_pred = rfc.predict(xval_tfidf)
svm_pred = svm.predict(xval_tfidf)
knn_pred = knn.predict(xval_tfidf)

# evaluate performance
mnb_acc = metrics.accuracy_score(yval, mnb_pred)
mnb_acc = round(mnb_acc,2)
lr_acc = metrics.accuracy_score(yval, lr_pred)
lr_acc = round(lr_acc,2)
rfc_acc = metrics.accuracy_score(yval, rfc_pred)
rfc_acc = round(rfc_acc,2)
svm_acc = metrics.accuracy_score(yval, svm_pred)
svm_acc = round(svm_acc,2)
knn_acc = metrics.accuracy_score(yval, knn_pred)
knn_acc = round(knn_acc,2)


# In[37]:


print(f"TFIDF Accuracy Scores:\nMultinomial Naive Bayes: {mnb_acc}")
print(f"Logistic Regression: {lr_acc}")
print(f"Random Forest Classifier: {rfc_acc}")
print(f"Support Vector Machine: {svm_acc}")
print(f"K Nearest Neighbourhood: {knn_acc}")


# In[38]:


pred_df = pd.DataFrame(xval)
pred_df['actual'] = yval
pred_df['prediction'] = svm_pred
pred_df.sample(30)


# In[ ]:




