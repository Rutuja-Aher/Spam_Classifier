#!/usr/bin/env python
# coding: utf-8

# # Step#1: Libraries Import

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


spam_df = pd.read_csv('emails.csv')


# In[3]:


spam_df


# In[4]:


spam_df.head(10)


# In[5]:


spam_df.tail(10)


# In[6]:


spam_df.head(5)


# In[7]:


spam_df.tail(5)


# In[8]:


spam_df.describe()


# In[ ]:


spam_df.info()


# # Step#2: Visualize Dataset

# In[ ]:


ham = spam_df[spam_df['spam'] ==0]


# In[ ]:


ham


# In[ ]:


spam = spam_df[ spam_df['spam']==1 ]


# In[ ]:


spam


# In[ ]:


print('Spam Percentage=', (len(spam)/len(spam_df) ) *100, '%')


# In[ ]:


print('Ham Percentage=', (len(ham)/len(spam_df) ) *100, '%')


# In[ ]:


sns.countplot(spam_df['spam'], label = 'count Spam vs. Ham')


# # Step#3: Create Testing And Training Dataset/Data Cleaning

# # Count Vectorizer Example

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first document.','This document is the second document.','And this is the third one','Is this the first document?']
sample_vectorizer = CountVectorizer()


# In[ ]:


x=sample_vectorizer.fit_transform(sample_data)


# In[ ]:


print(x.toarray())


# In[ ]:


print(sample_vectorizer.get_feature_names_out())


# # Lets Apply Count Vectorizer To Our Spam/Ham Example

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
spamham_countvectorizer = vectorizer.fit_transform(spam_df['text'])


# In[ ]:


print(vectorizer.get_feature_names_out())


# In[ ]:


print(spamham_countvectorizer.toarray())


# In[ ]:


spamham_countvectorizer.shape


# # Step#4: Training The Model

# In[ ]:


label = spam_df['spam'].values


# In[ ]:


label


# In[ ]:





# In[ ]:


from sklearn.naive_bayes import MultinomialNB

NBclassifier = MultinomialNB()
NBclassifier.fit(spamham_countvectorizer, label)


# In[ ]:


testing_sample = ['Free Money!!!','Hi kim,Please let me know if you need any further information.Thanks']
testing_sample_countvectorizer = vectorizer.transform(testing_sample)


# In[ ]:


test_predict=NBclassifier.predict(testing_sample_countvectorizer)
test_predict


# In[ ]:


#mini challenge!
testing_sample = ['Hello , I am Ryan , I would like to book a hotel in bali by january 24th', 'money viagara!!!']
testing_sample_countvectorizer = vectorizer.transform(testing_sample)


# In[ ]:


test_predict=NBclassifier.predict(testing_sample_countvectorizer)
test_predict


# # Step#4: Divide The Data Into Training And Testing Prior To Training

# In[ ]:


X = spamham_countvectorizer
y = label


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y , test_size = 0.2)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
NBclassifier = MultinomialNB()
NBclassifier.fit(X_train, y_train)


# # Step#5: Evaluating The Model

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


y_predict_train = NBclassifier.predict(X_train)
y_predict_train


# In[ ]:


cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot = True)


# In[ ]:


y_predict_test = NBclassifier.predict(X_test)
y_predict_test


# In[ ]:


cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot = True)


# In[ ]:


print(classification_report(y_test,y_predict_test))


# In[ ]:




