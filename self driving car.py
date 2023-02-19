#!/usr/bin/env python
# coding: utf-8

# In[41]:


import os
from itertools import islice
import numpy as np
pi=3.14
DATA_FOLD='C:\Autopilot-TensorFlow-master\Autopilot-TensorFlow-master\driving_dataset\driving_dataset'
TRAIN_FILE=os.path.join(DATA_FOLD,'data.txt')
LIMIT=None

split=0.8
x=[]
y=[]

with open(TRAIN_FILE) as fp:
    for line in islice(fp, LIMIT):
       path, angle= line.strip().split()
       full_path= os.path.join(DATA_FOLD, path)
       x.append(full_path)
    
       y.append(float(angle)*pi/180)
    
y= np.array(y)
print("successfull")


# In[42]:



y


# In[43]:


len(y)


# In[44]:


split_index=int(len(y)*split)


# In[45]:


train_y= y[:split_index]


# In[46]:


test_y=y[split_index:]


# In[47]:


import matplotlib.pyplot as plt
plt.hist(train_y, bins= 50, color= "blue", histtype= 'step')

plt.hist(test_y, bins= 50, color= "green", histtype= 'step')


# In[48]:


#regression problem

train_mean_y=np.mean(train_y)


# In[49]:


train_mean_y


# In[50]:


np.mean(np.square(test_y - train_mean_y ))


# In[51]:


np.mean(np.square(test_y ))


# In[52]:


#baseline model
np.mean(np.square(test_y - 0.0 ))


# In[ ]:




