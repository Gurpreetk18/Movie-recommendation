#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies=pd.read_csv('Netflix TV Shows and Movies.csv')


# In[7]:


movies.info()


# In[6]:


movies['type'].value_counts()


# In[8]:


movies=movies[['id','title','description']]


# In[9]:


movies.head()


# In[10]:


#remove null values
movies.isnull().sum()


# In[11]:


movies.dropna(inplace=True)


# In[12]:


movies.duplicated().sum()


# In[14]:


movies.iloc[0].description


# In[17]:


movies.head()


# In[16]:


movies['description']=movies['description'].apply(lambda x:x.lower())


# In[18]:


movies['description'][0]


# In[19]:


movies['description'][1]


# In[20]:


#convert text into vectors
#remove stop words and on that do vectorization
#bag of words
#most common words extract and count their occurences in each movie and the one that is close to each other will be the movie we are looking for


# In[37]:


from sklearn.feature_extraction.text import CountVectorizer
cv = cv = CountVectorizer(max_features=5000,stop_words='english')


# In[38]:


#convert to numpy array
vector=cv.fit_transform(movies['description']).toarray()


# In[39]:


vector[0]


# In[29]:


import nltk


# In[30]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[31]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[32]:


movies['description']=movies['description'].apply(stem)


# In[33]:


movies.head()


# In[36]:


ps.stem('accompanied')


# In[35]:


ps.stem('king arthur, accompanied by his squire, recruits his knights of the round table, including sir bedevere the wise, sir lancelot the brave, sir robin the not-quite-so-brave-as-sir-lancelot and sir galahad the pure. on the way, arthur battles the black knight who, despite having had all his limbs chopped off, insists he can still fight. they reach camelot, but arthur decides not  to enter, as "it is a silly place')


# In[40]:


from sklearn.metrics.pairwise import cosine_similarity


# In[41]:


similarity = cosine_similarity(vector)


# In[45]:


similarity[0]


# In[50]:


movies[movies['title'] == 'The Exorcist'].index[0]


# In[65]:


def next_recommend(movie):
    m_index = movies[movies['title'] == movie].index[0]
    distances = similarity[m_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:2]
    
    for i in movies_list:
        print(movies.iloc[i[0]].title)


# In[66]:


next_recommend('Taxi Driver')

