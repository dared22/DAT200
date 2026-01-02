#!/usr/bin/env python
# coding: utf-8

# # CA3

# ### Imports

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# ### Reading data

# In[2]:


df = pd.read_csv('Assets/train.csv')
df


# ### Data exploration and visualisation

# In[3]:


df[df.isnull().any(axis=1)]


# In[4]:


df.isnull().sum()


# In[5]:


# Checking what the shape and what are the datatypes we are working
print("The shape of the data: ", df.shape)
print()
print("The datatypes in each column are: ", df.dtypes)


# In[6]:


df.describe()


# In[7]:


# Cheking edible vs non edible
plt.figure(figsize=(10, 6))
sns.countplot(x='Edible', data=df, palette="Set2", hue='Edible')
plt.title('Distrubution of Edible')
plt.xlabel('Ediblity')
plt.ylabel('count')
plt.show()


# In[8]:


# Creates plots of distribution
def plot_dist(df, column_names):
    plt.figure(figsize=(32,24))
    
    for i in range(1, len(column_names)-1):
        plt.subplot(4, 4, i)
        sns.histplot(df[column_names[i]], bins=20, kde=True)
        plt.title('Distrubution of {0}'.format(column_names[i]))
        
    # plt.tight_layout()
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.9)
    plt.show()


# In[9]:


# I want to see the distribution for all of the columns in our dataset
column_names = df.columns
plot_dist(df, column_names)


# ### Data cleaning

# Since there is some missing data, I will write some code in order to remove those follwing rows with the missing information.

# In[10]:


df = df.dropna()
df.isnull().sum()


# Removing outliers:

# In[11]:


z_scores = (df - np.mean(df, axis=0)) / np.std(df, axis=0)
outliers = np.abs(z_scores) > 3
df[~outliers]


# In[12]:


column_names = df.columns
plot_dist(df, column_names)


# ### Data preprocessing and visualisation

# In[13]:


# Separating our feautures from the targets 
X = df.drop(columns=['Edible'])
y = df['Edible']

y = y.to_numpy().ravel()


# ### Modelling

# In[14]:


n_estimators_vals = [100, 150, 250, 350, 450, 500]
max_depth_vals = [None, 10, 20]
min_samples_split_vals = [6, 12, 16]


best_params = {}
best_acc = 0

for n in n_estimators_vals:
    for max_d in max_depth_vals:
        for min_ss in min_samples_split_vals:
            
            accuracies = []
            for r in range(20):

                X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                                  test_size=0.25, 
                                                                  random_state=r)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                clf = RandomForestClassifier(n_estimators=n,
                                             random_state=r, 
                                             min_samples_split=min_ss,
                                             max_depth=max_d)
                clf.fit(X_train_scaled, y_train)

                y_pred = clf.predict(X_val_scaled)

                accuracy = accuracy_score(y_pred, y_val)
                accuracies.append(accuracy)

            accuracy = np.mean(accuracies)

            if accuracy > best_acc:
                best_acc = accuracy
                best_params = {'n_estimators': n,
                               'max_depth': max_d,
                               'min_samples_split': min_ss,
                               'random_state': r}


# In[15]:


best_params


# In[16]:


best_acc


# In[17]:


clf = RandomForestClassifier(**best_params)
X_train_scaled = scaler.fit_transform(X)
clf.fit(X_train_scaled, y)


# ### Final evaluation

# In[18]:


X_test = pd.read_csv('Assets/test.csv')

X_test_scaled = scaler.transform(X_test)
y_pred = clf.predict(X_test_scaled)


# ### Kaggle submission

# In[19]:


y_csv = np.savetxt('y_pred.csv', np.dstack((np.arange(0, y_pred.size),y_pred))[0],"%d,%d", header="index,Edible", comments='')

