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


# In[9]:


spam_df.info()


# # Step#2: Visualize Dataset

# In[10]:


ham = spam_df[spam_df['spam'] == 0]


# In[11]:


ham


# In[12]:


spam = spam_df[ spam_df['spam']==1 ]


# In[13]:


spam


# In[14]:


spam_percentage = (len(spam) / len(spam_df)) * 100
ham_percentage = (len(ham) / len(spam_df)) * 100


# In[15]:


print('Spam Percentage =', spam_percentage, '%')
print('Ham Percentage =', ham_percentage, '%')


# In[16]:


print(sns.__version__)


# In[17]:


sns.countplot(x=spam_df['spam'])
plt.title('Spam vs. Ham Email Distribution')
plt.show()


# # Step#3: Create Testing And Training Dataset/Data Cleaning

# # Count Vectorizer Example

# In[18]:


from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first document.','This document is the second document.','And this is the third one','Is this the first document?']
sample_vectorizer = CountVectorizer()


# In[19]:


x=sample_vectorizer.fit_transform(sample_data)


# In[20]:


print(x.toarray())


# In[21]:


print(sample_vectorizer.get_feature_names_out())


# # Lets Apply Count Vectorizer To Our Spam/Ham Example

# In[22]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
spamham_countvectorizer = vectorizer.fit_transform(spam_df['text'])


# In[23]:


print(vectorizer.get_feature_names_out())


# In[24]:


print(spamham_countvectorizer.toarray())


# In[25]:


spamham_countvectorizer.shape


# # Step#4: Training The Model

# In[26]:


label = spam_df['spam'].values


# In[27]:


label


# In[28]:


from sklearn.naive_bayes import MultinomialNB

NBclassifier = MultinomialNB()
NBclassifier.fit(spamham_countvectorizer, label)


# In[29]:


testing_sample = ['Free Money!!!','Hi kim,Please let me know if you need any further information.Thanks']
testing_sample_countvectorizer = vectorizer.transform(testing_sample)


# In[30]:


test_predict=NBclassifier.predict(testing_sample_countvectorizer)
test_predict


# In[31]:


#mini challenge!
testing_sample = ['Hello , I am Ryan , I would like to book a hotel in bali by january 24th', 'money viagara!!!']
testing_sample_countvectorizer = vectorizer.transform(testing_sample)


# In[32]:


test_predict=NBclassifier.predict(testing_sample_countvectorizer)
test_predict


# # Step#4: Divide The Data Into Training And Testing Prior To Training

# In[33]:


X = spamham_countvectorizer
y = label


# In[34]:


X.shape


# In[35]:


y.shape


# In[36]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y , test_size = 0.2)


# In[37]:


from sklearn.naive_bayes import MultinomialNB
NBclassifier = MultinomialNB()
NBclassifier.fit(X_train, y_train)


# # Step#5: Evaluating The Model

# In[38]:


from sklearn.metrics import classification_report, confusion_matrix


# In[39]:


y_predict_train = NBclassifier.predict(X_train)
y_predict_train


# In[40]:


cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot = True)


# In[41]:


y_predict_test = NBclassifier.predict(X_test)
y_predict_test


# In[42]:


cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot = True)


# In[43]:


print(classification_report(y_test,y_predict_test))

