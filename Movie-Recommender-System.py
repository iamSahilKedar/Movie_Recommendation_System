#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head()


# In[4]:


credits.head(1)


# In[5]:


movies = movies.merge(credits,on='title')


# In[6]:


movies.head(1)


# In[7]:


movies.head()
# budget
# homepage
# id
# original_language
# original_title
# popularity
# production_comapny
# production_countries
# release-date(not sure)


# In[8]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[9]:


movies.head()


# In[10]:


movies.isnull().sum()


# In[11]:


movies.dropna(inplace=True)


# In[12]:


movies.duplicated().sum()


# In[13]:


movies.iloc[0].genres


# In[14]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[15]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name']) 
    return L


# In[16]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[17]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[18]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[19]:


movies['cast'] = movies['cast'].apply(convert)
movies.head()


# In[20]:


movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


# In[25]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[27]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[28]:


movies.head()


# In[30]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[31]:


movies.head()


# In[32]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[33]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[34]:


movies.head()


# In[35]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[36]:


new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
#new.head()


# In[37]:


new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()


# In[38]:


new['tags'][0]


# In[39]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[40]:


vector = cv.fit_transform(new['tags']).toarray()


# In[41]:


vector[0]


# In[48]:


import nltk


# In[49]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[52]:


def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[53]:


new['tags']= new['tags'].apply(stem)


# In[54]:


vector.shape


# In[55]:


from sklearn.metrics.pairwise import cosine_similarity


# In[56]:


similarity = cosine_similarity(vector)


# In[57]:


similarity


# In[58]:


new[new['title'] == 'The Lego Movie'].index[0]


# In[59]:


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)


# In[60]:


recommend('Gandhi')


# In[61]:


import pickle


# In[63]:


pickle.dump(new,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[64]:


pickle.dump(new.to_dict(),open('movie_dict.pkl','wb'))


# In[ ]:




