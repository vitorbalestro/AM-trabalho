#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


obesity = pd.read_csv('raw_data.csv')


# In[17]:


gender = pd.get_dummies(obesity['Gender'], drop_first = True)
calc = pd.get_dummies(obesity['CALC'], drop_first = True)
favc = pd.get_dummies(obesity['FAVC'], drop_first = True)
scc = pd.get_dummies(obesity['SCC'], drop_first = True)
smoke = pd.get_dummies(obesity['SMOKE'], drop_first = True)
family_history = pd.get_dummies(obesity['family_history_with_overweight'], drop_first = True)
caec = pd.get_dummies(obesity['CAEC'], drop_first = True)
mtrans = pd.get_dummies(obesity['MTRANS'], drop_first = True)
obesity_level = pd.get_dummies(obesity['NObeyesdad'])
caec.rename(columns={'Frequently': 'CAEC.Frequently', 'Sometimes': 'CAEC.Sometimes','no': 'CAEC.No'}, inplace=True)
calc.rename(columns={'Frequently': 'CALC.Frequently', 'Sometimes': 'CALC.Sometimes','no': 'CALC.No'}, inplace=True)
scc.rename(columns={'yes': 'SCC.yes'}, inplace=True)
smoke.rename(columns={'yes': 'SMOKE.yes'}, inplace=True)
family_history.rename(columns={'yes': 'family_hist.yes'}, inplace=True)

obesity = pd.concat([obesity,gender,calc,favc,scc,smoke,family_history,caec,mtrans], axis=1)
obesity.drop(['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE','family_history_with_overweight','CAEC','MTRANS' ], axis=1, inplace=True)
pd.options.display.max_columns = 100


obesity.head()


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

parameters = [{'learning_rate': [0.1,0.3,0.5], 'n_estimators': [100,200], 'max_depth': [2,3,4,5], 'min_impurity_decrease': [0.0,0.1,0.3,0.5]}]

X = obesity.drop('NObeyesdad', axis = 1)
y = obesity['NObeyesdad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

grad=GradientBoostingClassifier()

GS = GridSearchCV(estimator = grad, 
                   param_grid = parameters, 
                   scoring = 'accuracy', 
                   refit = 'accuracy',
                   cv = 10,
                   verbose = 4 ,
                  error_score='raise'
                 )
GS.fit(X_train, y_train)


# In[19]:


GS.best_params_


# In[ ]:





# In[21]:


grad_100=GradientBoostingClassifier(learning_rate=0.5,max_depth=4,min_impurity_decrease=0.0,n_estimators=100)
grad_100.fit(X_train,y_train)
predictions3 = grad_100.predict(X_test)


# In[22]:


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
print(confusion_matrix(y_test, predictions3))
print('\n')
print(classification_report(y_test, predictions3))
print('\n Cross Validation \n')
scores=cross_val_score(grad_100,X,y, cv=10, scoring='accuracy')
print(scores.mean(), scores.std())


# In[29]:


sns.heatmap(confusion_matrix(y_test, predictions3), annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.title('100 estimators')
plt.show()


# In[10]:





# In[ ]:




