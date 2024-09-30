#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import export_graphviz
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# input data reading
d = pd.read_csv('bank-full.csv',delimiter=';')
d.head() 


# In[4]:


# target value balance depiction
fig,ax = plt.subplots()
ax.set_title('yes-no balance in target value')
sns.countplot(x='y',data=d)


# In[5]:


# statistical figures of input data
d.describe()


# In[6]:


d.info()


# In[7]:


# search for any duplicated values
np.any(d.duplicated())


# In[9]:


# search for unknown values
np.sum(d=='unknown')


# In[10]:


# removing columns with too many unknown values
d.drop(['contact','poutcome'],axis=1,inplace=True)


# In[11]:


# removing rows where their 'job' or 'education' features are unknown
d.drop(d.loc[d['job']=='unknown'].index,inplace=True)
d.drop(d.loc[d['education']=='unknown'].index,inplace=True)


# In[13]:


# encoding target column to numerical values
d['y'].replace(['yes','no'],[1,0],inplace=True)
d.head()


# In[14]:


# age and job distribution according to target value
fig,ax = plt.subplots(2,1,figsize=[17,12])
sns.histplot(x='age',hue='y', bins=50,data=d,ax=ax[0],palette='Set1')
ax[0].set_title('age distribution')
sns.countplot(x='job',hue='y', data=d,ax=ax[1],palette='Set2')
ax[1].set_title('job counts')


# In[15]:


# marriage and education status distribution according to target value
fig,ax = plt.subplots(1,2,figsize=[12,5])
sns.countplot(x='marital',hue='y', data=d,ax=ax[0],palette='Set1')
ax[0].set_title('marriage status counts')
sns.countplot(x='education',hue='y', data=d,ax=ax[1],palette='Set2')
ax[1].set_title('education status counts')


# In[16]:


# default, housing and loan status distribution according to target value
fig,ax = plt.subplots(1,3,figsize=[15,5])
sns.countplot(x='default',hue='y', data=d,ax=ax[0],palette='Set1')
ax[0].set_title('default status counts')
sns.countplot(x='housing',hue='y', data=d,ax=ax[1],palette='Set2')
ax[1].set_title('housing status counts')
sns.countplot(x='loan',hue='y', data=d,ax=ax[2],palette='Set3')
ax[2].set_title('loan status counts')


# In[17]:


# marriage-oriented age distribution
fig,ax = plt.subplots(figsize=[10,5])
sns.boxplot(x='marital',y='age', data=d,hue='y')
ax.set_title('age-marital boxplot')


# In[18]:


# education-oriented age distribution
fig,ax = plt.subplots(figsize=[10,5])
sns.boxplot(x='education',y='age', data=d,hue='y')
ax.set_title('age-education boxplot')


# In[19]:


# job-oriented age distribution
fig,ax = plt.subplots(figsize=[12,7])
sns.boxplot(x='job', y='age', data=d,hue='y')
ax.set_title('age-job boxplot')


# In[20]:


# correlation vlaues among numerical features
d_corr = d.select_dtypes(include='number').corr()
sns.heatmap(d_corr,annot=True,cmap='coolwarm')


# In[21]:


# house and load status comparison
d_mat1 = d.pivot_table(index='housing',columns='loan',values='campaign')
sns.heatmap(d_mat1,cmap='coolwarm')


# In[22]:


# education and job status comparison
d_mat1 = d.pivot_table(index='education',columns='job',values='campaign')
sns.heatmap(d_mat1,cmap='coolwarm')


# In[23]:


# convert categorical data to dummy variables
d1 = pd.get_dummies(data=d)
# splitting features and target values
X = d1.drop(columns='y')
y = d1.y


# In[24]:


# classification by random forest
# simple train and test strategy
rfc = RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print(f'total number of test cases are: {y_test.size}')
print(f'Confusion matrix is: \n {confusion_matrix(y_test,rfc_pred)}')
print('\n')
print(f'Some other metrics are evaluated as: \n {classification_report(y_test,rfc_pred)}')
print('\n')
print(f'Overal accuracy is {accuracy_score(y_test,rfc_pred)}')


# In[32]:


# random forest hyper parameters optimization

# Number of trees in random forest
n_estimators = [int(x) for x in np.arange(100,1000,100)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)


# In[33]:


# best set of parameters
rf_random.best_params_


# In[34]:


# prediction with best parameters
optimized_rfc = rf_random.best_estimator_
rfc_new_pred = optimized_rfc.predict(X_test)
print(f'Confusion matrix is: \n {confusion_matrix(y_test,rfc_new_pred)}')
print('\n')
print(f'Some other metrics are evaluated as: \n {classification_report(y_test,rfc_new_pred)}')
print('\n')
print(f'Overal accuracy is {accuracy_score(y_test,rfc_new_pred)}')


# In[ ]:




