#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")


# In[2]:


df = pd.read_csv("E:\\Data Science\\8.Machine Learning Algorithms\\2.Classification\\mushrooms.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


X = pd.get_dummies(df.drop('class',axis=1),drop_first=True)


# In[7]:


X


# In[8]:


y = df["class"]


# In[9]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state= 101)


# In[10]:


from xgboost import XGBClassifier


# In[11]:


pip install xgboost


# In[12]:


xg_model  =XGBClassifier()


# In[13]:


xg_model.fit(X_train,y_train)


# In[14]:


y_pred_train = xg_model.predict(X_train)
y_pred_test = xg_model.predict(X_test)


# In[15]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred_train,y_train))


# In[16]:


print(accuracy_score(y_pred_test,y_test))


# In[17]:


from sklearn.model_selection import cross_val_score
Accuracy = cross_val_score(estimator = xg_model,X = X_train,y = y_train,cv=5)


# In[18]:


Accuracy


# In[19]:


Accuracy.mean()


# In[20]:


from sklearn.metrics import plot_confusion_matrix
print(plot_confusion_matrix(xg_model,X_test,y_test))


# In[21]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_test))


# In[22]:


from sklearn.model_selection import GridSearchCV
xgb_model = XGBClassifier()
param_grid = {"n_estimators":[1,5,10,20,40,100],'max_depth':[3,4,5,6],"gamma":[0,0.15,0.3,0.5,1]}


# In[23]:


grid = GridSearchCV(xgb_model,param_grid,cv=5,scoring="accuracy")
grid.fit(X_train,y_train) 


# In[24]:


grid.best_params_


# In[25]:


predictions = grid.predict(X_test)


# In[26]:


print(classification_report(y_test,predictions))


# In[27]:


grid.best_estimator_.feature_importances_


# In[28]:


imp_features = pd.DataFrame(index=X.columns,data=grid.best_estimator_.feature_importances_,columns=["Importance"])


# In[29]:


imp_features.sort_values("Importance",ascending = False)


# In[30]:


imp_features.describe()


# In[33]:


imp_features=  imp_features[imp_features['Importance']>0.01]


# In[34]:


plt.figure(figsize=(14,6),dpi=200)
sns.barplot(data=imp_features.sort_values("Importance"),x=imp_features.index,y="Importance")
plt.xticks(rotation=90)


# In[ ]:





# In[ ]:




