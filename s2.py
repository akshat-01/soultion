#!/usr/bin/env python
# coding: utf-8

# In[11]:


import nltk
import pandas as pd
from pathlib import Path
import os
import glob


# In[4]:


p=Path('.')
path=p.resolve()
path


# In[14]:


os.chdir(r'D:/Jupyter Sketch/999.DS/kuch nhi/sramev')
myFiles = glob.glob('*.txt')


# In[15]:


myFiles


# In[16]:


len(myFiles)


# In[17]:


myFiles[0]


# In[18]:


filenames = myFiles 
files = {} 
 
for filename in filenames: 
    with open(filename, "r") as file: 
        if filename in files: 
            continue 
        files[filename] = file.read()


# In[24]:


files[myFiles[0]].split('\n')


# In[25]:


documents=[]
for key in files:
    i=key
    documents.append(((files[key]),(key)))
documents[0]


# In[29]:


type(documents),len(documents)


# In[28]:


documents[0][1]


# In[35]:


# for i in len(documents):
t=documents[0][0].split('\n\n')
t


# In[39]:


finallist_x=[]
finallist_y=[]
for i in range(len(documents)):
    t=documents[i][0].split('\n\n')
    for j in range(len(t)):
        finallist_x.append(t[j])
        finallist_y.append(documents[i][1])


# In[44]:


len(finallist_x),len(finallist_y)


# In[41]:


print(finallist_x)


# In[43]:


print(finallist_y)


# In[45]:


# make the corpus rest is already done


# In[46]:


# now corpus is ready
from sklearn.feature_extraction.text import CountVectorizer


# In[47]:


cv=CountVectorizer()


# In[48]:


vectorizedcorpus=cv.fit_transform(finallist_x)


# In[49]:


vectorizedcorpus


# In[50]:


vectorizedcorpus=vectorizedcorpus.toarray()


# In[51]:


vectorizedcorpus.shape


# In[52]:


len(vectorizedcorpus[0])


# In[53]:


cv.vocabulary_


# In[56]:


print(vectorizedcorpus[0])
print(len(vectorizedcorpus[0]))


# In[54]:


cv.inverse_transform(vectorizedcorpus[0])


# In[57]:


from sklearn.preprocessing import LabelEncoder


# In[60]:


le=LabelEncoder()
finallist_y_le= le.fit_transform(finallist_y)


# In[61]:


finallist_y_le


# In[62]:


# now data is ready
print(vectorizedcorpus)
print(finallist_y_le)


# In[63]:


from sklearn.naive_bayes import GaussianNB


# In[64]:


vectorizedcorpus.shape,finallist_y_le.shape


# In[65]:


gnb=GaussianNB()


# In[66]:


gnb.fit(vectorizedcorpus,finallist_y_le)


# In[70]:


vectorizedcorpus[:10],finallist_y_le[:10]


# In[81]:


print(gnb.score(vectorizedcorpus[:100],finallist_y_le[:100]))


# In[79]:


vectorizedcorpus[2660]


# In[82]:


vectorizedcorpus[:1,:].shape


# In[83]:


gnb.predict(vectorizedcorpus[2675:2677,:])


# In[84]:


gnb.predict(vectorizedcorpus[:3,:])


# In[ ]:




