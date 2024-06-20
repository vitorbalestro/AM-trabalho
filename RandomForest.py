#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


obesity = pd.read_csv('Obesity.csv')


# In[3]:


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


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
X = obesity.drop('NObeyesdad', axis = 1)
y = obesity['NObeyesdad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# In[6]:


parameters={'n_estimators':[50,100,200,500],'criterion':['gini','entropy'],'max_depth':[2,4,6,8,10,None],'min_impurity_decrease':[0.0,0.2,0.5]}

rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
GS = GridSearchCV(estimator = rfc, 
                   param_grid = parameters, 
                   scoring = 'accuracy', 
                   refit = 'accuracy',
                   cv = 5,
                   verbose = 4 ,
                  error_score='raise'
                 )
GS.fit(X_train, y_train)


# In[11]:





# In[26]:


df=pd.DataFrame(GS.cv_results_)
df[['param_criterion','param_max_depth','param_min_impurity_decrease','param_n_estimators','mean_test_score','rank_test_score']][df['param_n_estimators']==500]
GS.best_estimator_


# In[33]:


rf_50=RandomForestClassifier(criterion='entropy',max_depth=None, min_impurity_decrease=0.0, n_estimators=50)
rf_500=GS.best_estimator_
rf_50.fit(X_train,y_train)
predictions50=rf_50.predict(X_test)
predictions500=rf_500.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
print('\n 50 estimators \n')
print(confusion_matrix(y_test, predictions50))
print('\n')
print(classification_report(y_test, predictions50))
print('\n')
print('Cross Validation\n')
print(cross_val_score(rf_500,X,y, cv=10, scoring='accuracy').mean())
print('\n 500 estimators \n')
print(confusion_matrix(y_test, predictions500))
print('\n')
print(classification_report(y_test, predictions500))
print('\n')
print('Cross Validation\n')
print(cross_val_score(rf_500,X,y, cv=10, scoring='accuracy').mean())


# In[39]:


fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot do heatmap para 500 estimators
sns.heatmap(confusion_matrix(y_test, predictions500), annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_xlabel('Predicted class')
axes[0].set_ylabel('True class')
axes[0].set_title('500 estimators')

# Plot do heatmap para 50 estimators
sns.heatmap(confusion_matrix(y_test, predictions50), annot=True, fmt="d", cmap="Blues", ax=axes[1])
axes[1].set_xlabel('Predicted class')
axes[1].set_ylabel('True class')
axes[1].set_title('50 estimators')

# Ajusta o layout para evitar sobreposição
plt.tight_layout()

# Mostra a figura
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




