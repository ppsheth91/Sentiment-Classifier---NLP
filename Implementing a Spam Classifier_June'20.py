#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os


# In[2]:


os.chdir("E:/Data Science/DL and NLP/NLP/Data Sets/smsspamcollection")


# In[3]:


os.getcwd()


# In[4]:


# reading the file, which is a tab separated file, so sep = '\t', 
# With dependent variable = 'labels' (spam or ham)  & independent variable = 'message'

df = pd.read_csv("SMSSpamCollection",sep='\t',names=['labels','message']) 


# In[5]:


df.head()


# In[6]:


import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
ps = PorterStemmer()


# In[7]:


# Performing data preprocessijng operations #

# 1) Stopwords : Need to remove stop words such as he, I, dont, at, for.. ( stopwords)
# 2) Regular expresssions : Removing expressions such as : "!, ., ", '', @, $, %,' from this text
# 3) Lower case : converting the entire text into lower case
# 4) Split : Before applying the stemming , we need to break the sentences into words
# 5) Stemming operation : Convert all the words into root form, to reduce the complexity

corpus = []

for i in range(0, len(df)):
    rev = re.sub('[^a-zA-Z]',' ',df['message'][i])
    rev = rev.lower()
    rev = rev.split()
    rev = [ps.stem(j) for j in rev if j not in stopwords.words('english')]
    a = ' '.join(rev)
    corpus.append(a)


# In[8]:


df['message'].head()


# In[15]:


corpus


# In[23]:


len(df)


# In[8]:


# Approach 1 : Applying the stemming and then applyong the bag of words 
# Approach 2: Applying the Lematization & then applying the TFIDF vectorizer 
# Will evaluate which approach gives a better accuracy #

# Approach1
# Applying the bag of words (Document Matrix) : Count the no. of words / features in the entire document, it will identify count sentence wise #

# Max_Features : There will be large no. of features in the document, but will try to take highly frequent words #
from sklearn.feature_extraction.text import CountVectorizer 

cv = CountVectorizer(max_features=2500)

X = cv.fit_transform(corpus).toarray()
X


# In[10]:


# Optional, just for ubderstanding bag of words ouput #

m = pd.DataFrame(X)
m.shape
m.head()


# In[9]:


# Converting the dependent variable intp numerical form with the help of dummy encoding : Labels

y = pd.get_dummies(df['labels'])
y.head()

Y = y.iloc[:,1].values  # taking the spam column and converting into an array#
# We can take only 1 column Spam, and represent the other through that column, For spam : 1 its a spam mail, 0 : ham 


# In[12]:


Y # Target variable # 1 : Spam, 0 : Ham


# In[10]:


# Developing the train and test splits #

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=15)


# In[11]:


# Applying Naive baiyes classifier on this data #

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB().fit(X_train,Y_train)


# In[12]:


# Predicting the output for the X_test values #

y_pred = model.predict(X_test)

y_pred


# In[13]:


#Computing tyhe accuracy & creating the confusion_matrix for the model #

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

con = confusion_matrix(y_pred,Y_test)
con

acc = accuracy_score(y_pred,Y_test)
acc


# In[40]:


# Approach 2 : Lematization & applying TFIDF

from nltk.stem import WordNetLemmatizer

wl = WordNetLemmatizer()

net = []

for i in range(0, len(df)):
    rev = re.sub('[^a-zA-Z]',' ',df['message'][i])
    rev = rev.lower()
    rev = rev.split()
    rev = [wl.lemmatize(j) for j in rev if j not in stopwords.words('english')]
    a = ' '.join(rev)
    net.append(a)


# In[41]:


print(net[0]) # After the lemmatization, which doesnt breaks the meanring of the word
print(corpus[0]) # After the Stemming, which breaks the meanring of the word


# In[42]:


# Applyijg the TFIDF vectorizer to create array of document of the words#

# Max_Features : There will be large no. of features in the document, but will try to take highly frequent words #
from sklearn.feature_extraction.text import TfidfVectorizer 

tf = TfidfVectorizer()

X1 = tf.fit_transform(net).toarray()


# In[43]:


# Optional, just for ubderstanding TFIDF ouput #

m = pd.DataFrame(X1)
m.shape
m.head()


# In[44]:


# Need to prepare the training & test set to apply the Naive Bayes Classifier #

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X1,Y,test_size=0.2,random_state=15)


# In[45]:


# Naive Bayes Alogrithm application #

model1 = MultinomialNB().fit(X_train,Y_train)


# In[46]:


# Predicting the test data set #

y_pred = model1.predict(X_test)


# In[47]:


# Checking the accuracy of this model #

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

con = confusion_matrix(y_pred,Y_test)
print(con)

acc = accuracy_score(y_pred,Y_test)
print(acc)


# In[ ]:


## Applying the Porter Stemming & Bag of Words
# Accuracy Score of 97.48% with max_features = 500
# Accuracy Score of 98.65% with max_features = 2500
# Accuracy Score of 98.38% with max_features = 5000


## Applying the Lemmatization & TFIDf vectorizer 
# Accuracy Score = 97.13%

# So the best model applied is with bag of words & Porter Stemming, which gave an accuracy of 98.65%

