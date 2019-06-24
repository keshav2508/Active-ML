# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# importing the dataset
dataset = pd.read_csv('tic-tac-toe.data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,9].values

# There is no missing data so we donot use imputer

'''
Encoding the categorical data 
'''
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
for i in range(0,9):
  labelencoder_x_i = LabelEncoder()
  X[:,i] = labelencoder_x_i.fit_transform(X[:,i])
# Label Encoded into 0,1,2

onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
# To remove the weights of the values 

#Label Encoding the Dependent Variable
labelencoder_y =  LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Removing the dummy variable trap
l=[]
for i in range(27):
  if (i+1)%3!=0:
    l.append(i)
X = X[:, l]
# Every third column will be removed

# To keep the mapping of X and y
y = y.reshape(957,1)
dataframe = np.concatenate((X,y),axis=1)

# In order to randomly select 100 rows every time we shuffle the data 
np.random.shuffle(dataframe)

# Splitting the dataset into Labeled and Unlabeled dataset
labeled_dataset = dataframe[0:400,:]
unlabeled_dataset = dataframe[labeled_dataset.shape[0]: , :]

# Splitting the dataset into training set and test set
X_train = labeled_dataset[:,:-1]
y_train = labeled_dataset[:,9]

# Every time randomly selecting the test set from unlabeled dataset
#np.random.shuffle(unlabeled_dataset)
test_size = 500
X_test = unlabeled_dataset[:test_size , :-1]
y_test = unlabeled_dataset[:test_size , 9]

# Unlabeled dataset excluding the test set
unlabeled_dataset = unlabeled_dataset[test_size:,:]

# No need of Feature Scaling in our dataset (already in 0s and 1s) 

# Train the Model using Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predicting the test set
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

print(classifier.score(X_test,y_test))







