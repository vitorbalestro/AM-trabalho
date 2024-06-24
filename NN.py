#!/usr/bin/env python
# coding: utf-8

# #### Imports

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy                 import stats  as ss
from scipy.stats import chi2_contingency

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# #### functions

# In[2]:


def cramer_v_old( x, y ):
    cm = pd.crosstab( x, y ).as_matrix()
    n = cm.sum()
    r, k = cm.shape
    
    chi2 = ss.chi2_contingency( cm )[0]
    chi2corr = max( 0, chi2 - (k-1)*(r-1)/(n-1) )
    
    kcorr = k - (k-1)**2/(n-1)
    rcorr = r - (r-1)**2/(n-1)
    
    return np.sqrt( (chi2corr/n) / ( min( kcorr-1, rcorr-1 ) ) )


# ##### Load data

# In[3]:


df = pd.read_csv('raw_data.csv')


# #### data description

# In[4]:


df.columns


# #### change columns name

# Dataset Information
# 
# This dataset include data for the estimation of obesity levels in individuals from the countries of Mexico, Peru and Colombia, based on their eating habits and physical condition. The data contains 17 attributes and 2111 records, the records are labeled with the class variable NObesity (Obesity Level), that allows classification of the data using the values of Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III. 77% of the data was generated synthetically using the Weka tool and the SMOTE filter, 23% of the data was collected directly from users through a web platform.
# 
# Gender: Feature, Categorical, "Gender"
# Age : Feature, Continuous, "Age"
# Height: Feature, Continuous
# Weight: Feature Continuous
# family_history_with_overweight: Feature, Binary, " Has a family member suffered or suffers from overweight? "
# 
# FAVC : Feature, Binary, " Do you eat high caloric food frequently? "
# FCVC : Feature, Integer, " Do you usually eat vegetables in your meals? "
# NCP : Feature, Continuous, " How many main meals do you have daily? "
# CAEC : Feature, Categorical, " Do you eat any food between meals? "
# SMOKE : Feature, Binary, " Do you smoke? "
# CH2O: Feature, Continuous, " How much water do you drink daily? "
# SCC: Feature, Binary, " Do you monitor the calories you eat daily? "
# FAF: Feature, Continuous, " How often do you have physical activity? "
# TUE : Feature, Integer, " How much time do you use technological devices such as cell phone, videogames, television, computer and others? "
# 
# CALC : Feature, Categorical, " How often do you drink alcohol? "
# MTRANS : Feature, Categorical, " Which transportation do you usually use? "
# NObeyesdad : Target, Categorical, "Obesity level"

# In[5]:


df.columns = ['Age', 'Gender', 'Height', 'Weight', 'drink_alcohol', 'caloric_food', 'vegetables', 'how_many_meals',
       'how_many_calories', 'SMOKE', 'CH2O', 'family_history_with_overweight', 'how_often_physical_activity', 'how_often_technological_devices',
       'eat_between_meals', 'transportation', 'obesity_level']


# In[6]:


unique_values = df.apply(pd.Series.nunique)
print(unique_values)


# In[7]:


unique_values = df.apply(lambda x: x.unique())
print(unique_values)


# In[8]:


null_values = df.isnull().sum().sum()
print(null_values)


# In[9]:


data_types = df.dtypes
print("\nData types:")
print(data_types)


# In[ ]:





# # Data Vizualization

# #### data dimensions

# In[10]:


print( 'Number of Rows: {}'.format( df.shape[0] ) )
print( 'Number of Cols: {}'.format( df.shape[1] ) )


# In[11]:


num_attributes = df.select_dtypes( include=['int64', 'float64'] )
cat_attributes = df.select_dtypes( exclude=['int64', 'float64'] )


# In[12]:


correlation = num_attributes.corr(method='pearson')

plt.figure(figsize=(10, 8)) 
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="Blues")
plt.title('Correlation Heatmap')
plt.savefig('num_correlation_heatmap.png')  
plt.show()


# In[13]:


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

categorical_attributes = ['Gender', 'drink_alcohol', 'caloric_food', 'how_many_calories', 'SMOKE',
                          'family_history_with_overweight', 'eat_between_meals', 'transportation',
                          'obesity_level']
correlation_matrix = pd.DataFrame(index=categorical_attributes, columns=categorical_attributes)

for i in range(len(categorical_attributes)):
    for j in range( len(categorical_attributes)):
        attribute1 = categorical_attributes[i]
        attribute2 = categorical_attributes[j]
        correlation = cramers_v(df[attribute1], df[attribute2])
        correlation_matrix.loc[attribute1, attribute2] = correlation
        correlation_matrix.loc[attribute2, attribute1] = correlation  # Since it's symmetric
        print(attribute1,attribute2,correlation)

correlation_matrix = correlation_matrix.apply(pd.to_numeric, errors='coerce')

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="Blues")
plt.title('Categorical Attributes Correlation Matrix')
plt.savefig('categorical_correlation_heatmap.png') 
plt.show()


# #### numerical atributes vizualization

# In[14]:


plt.figure(figsize=(12, 10))
numeric_cols = num_attributes.columns
for i, col in enumerate(numeric_cols):
    plt.subplot(len(numeric_cols) // 2, 2, i + 1)  
    df[col].hist(bins=20) 
    plt.title(col) 
    plt.xlabel(col)  
    plt.ylabel('Frequency')  

plt.tight_layout()  
plt.savefig('num_hist.png')
plt.show()


# In[15]:


ct1 = pd.DataFrame( num_attributes.apply( np.mean ) ).T
ct2 = pd.DataFrame( num_attributes.apply( np.median ) ).T

d1 = pd.DataFrame( num_attributes.apply( np.std ) ).T 
d2 = pd.DataFrame( num_attributes.apply( min ) ).T 
d3 = pd.DataFrame( num_attributes.apply( max ) ).T 
d4 = pd.DataFrame( num_attributes.apply( lambda x: x.max() - x.min() ) ).T 
d5 = pd.DataFrame( num_attributes.apply( lambda x: x.skew() ) ).T 
d6 = pd.DataFrame( num_attributes.apply( lambda x: x.kurtosis() ) ).T 

m = pd.concat( [d2, d3, d4, ct1, ct2, d1, d5, d6] ).T.reset_index()
m.columns = ['attributes', 'min', 'max', 'range', 'mean', 'median', 'std', 'skew', 'kurtosis']
m


# #### categorical atributes vizualization

# In[16]:


cat_attributes.apply( lambda x: x.unique().shape[0] )


# In[17]:


categorical_columns = df.select_dtypes(include=['object']).columns

num_plots = len(categorical_columns)
num_cols = 2  
num_rows = -(-num_plots // num_cols)  
categorical_columns = df.select_dtypes(include=['object']).columns
num_plots = len(categorical_columns)
num_cols = 2  
num_rows = -(-num_plots // num_cols)  

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows*5))

axes = axes.flatten()

for i, column in enumerate(categorical_columns):
    ax = axes[i] 
    df[column].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f'Frequency of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    ax.tick_params(axis='x', rotation=45)  

for i in range(num_plots, num_rows * num_cols):
    fig.delaxes(axes[i])

plt.tight_layout()

plt.show()
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows*5))

axes = axes.flatten()

for i, column in enumerate(categorical_columns):
    ax = axes[i] 
    df[column].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f'Frequency of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    ax.tick_params(axis='x', rotation=45)  
for i in range(num_plots, num_rows * num_cols):
    fig.delaxes(axes[i])

plt.tight_layout()

plt.savefig('categorical_count.png')
plt.show()


# # Prediction

# In[18]:


correlation = num_attributes.corr(method='pearson')

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="Blues")

plt.title('Correlation Heatmap')
plt.savefig('num_correlation_heatmap.png') 
plt.show()


# In[19]:


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


categorical_attributes = ['Gender', 'drink_alcohol', 'caloric_food', 'how_many_calories', 'SMOKE',
                          'family_history_with_overweight', 'eat_between_meals', 'transportation',
                          'obesity_level']

correlation_matrix = pd.DataFrame(index=categorical_attributes, columns=categorical_attributes)

for i in range(len(categorical_attributes)):
    for j in range( len(categorical_attributes)):
        attribute1 = categorical_attributes[i]
        attribute2 = categorical_attributes[j]
        correlation = cramers_v(df[attribute1], df[attribute2])
        correlation_matrix.loc[attribute1, attribute2] = correlation
        correlation_matrix.loc[attribute2, attribute1] = correlation  # Since it's symmetric
        print(attribute1,attribute2,correlation)

correlation_matrix = correlation_matrix.apply(pd.to_numeric, errors='coerce')

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="Blues")
plt.title('Categorical Attributes Correlation Matrix')
plt.savefig('categorical_correlation_heatmap.png') 
plt.show()


# In[ ]:





# #### Encoding

# In[20]:


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

categorical_attributes = ['Gender', 'drink_alcohol', 'caloric_food', 'how_many_calories', 'SMOKE',
                          'family_history_with_overweight', 'eat_between_meals', 'transportation',
                          'obesity_level']

correlation_matrix = pd.DataFrame(index=categorical_attributes, columns=categorical_attributes)

for i in range(len(categorical_attributes)):
    for j in range(i + 1, len(categorical_attributes)):
        attribute1 = categorical_attributes[i]
        attribute2 = categorical_attributes[j]
        correlation = cramers_v(df[attribute1], df[attribute2])
        correlation_matrix.loc[attribute1, attribute2] = correlation
        correlation_matrix.loc[attribute2, attribute1] = correlation

correlation_matrix = correlation_matrix.apply(pd.to_numeric, errors='coerce')
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Categorical Attributes Correlation Matrix')
plt.savefig('categorical_correlation_heatmap.png') 
plt.show()


# In[21]:


encoder = OneHotEncoder()
y = cat_attributes["obesity_level"]
cat_attributes = cat_attributes.drop(columns=["obesity_level"])
X_cat_encoded = encoder.fit_transform(cat_attributes)
X = np.concatenate((num_attributes, X_cat_encoded.toarray()), axis=1)
X = pd.DataFrame(X)
print("Shape of num_attributes:", num_attributes.shape)
print("Shape of cat_attributes:", cat_attributes.shape)
print("Shape of concatenated DataFrame X_cat_encoded:", X_cat_encoded.shape)
print("Shape of concatenated DataFrame X:", X.shape)
print(cat_attributes.columns)


# In[22]:


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = MLPClassifier(random_state=1, verbose=10)


param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'alpha': [1e-3, 1e-4, 1e-5],
    'learning_rate_init': [0.1, 0.01, 0.001],
    'solver': [ 'adam'],
    'max_iter': [100, 200, 300],
}
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100), (200,)],
    'alpha': [1e-3, 1e-4, 1e-5],
    'learning_rate_init': [ 0.001],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'max_iter': [500],
    'activation': ['relu', 'tanh', 'logistic'],
}
param_grid = {'activation': ['tanh'], 'alpha': [1e-05], 'hidden_layer_sizes': [(100, 100)], 'learning_rate_init': [0.001], 'max_iter': [500], 'solver': ['adam']}

grid_search = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy', verbose=10, n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

best_clf = grid_search.best_estimator_

y_pred = best_clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=y.unique()))


# In[23]:


from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.savefig('nn_map.png')  
plt.show()

