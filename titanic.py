#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("train.csv")
data


# In[9]:


X = data.iloc[:,1:3]
X


# In[10]:


y = data['survived']
y


# In[11]:


X.sex[X.sex == 'male'] = 1
X.sex[X.sex == 'female'] = 0
print(X)


# In[13]:


#making predictions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
predictions


# In[14]:


#checking accuracy
score = accuracy_score(y_test, predictions)
score


# In[25]:


#visualization

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = ['age', 'sex'],class_names= ['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[ ]:




