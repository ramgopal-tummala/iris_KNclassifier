#!/usr/bin/env python
# coding: utf-8

# # 1 ML model iris classifying species
# 

# In[51]:


from sklearn.datasets  import load_iris
import pandas as pd
import numpy as np


# # Load Data Set iris dataset

# In[61]:


iris_dataset = load_iris()


# In[15]:


iris_dataset.keys()


# In[62]:


# Info about the data set 


# In[21]:


iris_dataset['data'].shape


# In[23]:


iris_dataset['target']


# In[25]:


iris_dataset['target_names']


# In[29]:


iris_dataset['feature_names']


# In[ ]:


# dividing the data into 2 parts train and test


# In[31]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)


# In[39]:


iris_Dataframe=pd.DataFrame(x_train,columns=iris_dataset['feature_names'])


# In[40]:


iris_Dataframe


# In[ ]:


# Using ML model for Predection 


# In[42]:


from sklearn.neighbors import KNeighborsClassifier


# In[43]:


kn=KNeighborsClassifier()


# In[44]:


kn.fit(x_train,y_train)


# In[55]:


pre=np.array([[5,2,3,4]])


# In[56]:


res=kn.predict(pre)


# In[58]:


res[0]


# In[60]:


iris_dataset['target_names'][res[0]]


# In[63]:


mainpred=kn.predict(x_test)


# In[64]:


mainpred


# In[65]:


y_test


# In[67]:


kn.score(x_test,y_test)


# In[ ]:




