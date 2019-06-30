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

# Label Encoded into 0,1,2
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
for i in range(0,9):
  labelencoder_x_i = LabelEncoder()
  X[:,i] = labelencoder_x_i.fit_transform(X[:,i])
'''
# To remove the weights of the values
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
 
#Label Encoding the Dependent Variable
labelencoder_y =  LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Removing the dummy variable trap
# Every third column will be removed
l=[]
for i in range(27):
  if (i+1)%3!=0:
    l.append(i)
X = X[:, l]


# To keep the mapping of X and y
y = y.reshape(-1,1)
dataframe = np.concatenate((X,y),axis=1)

''' In order to randomly select 100 rows every time we shuffle the data 
 and take equal no of samples of positive and negative class '''
np.random.shuffle(dataframe)
train_size = 100
positive = 0
negative = 0
for i in range(0,train_size):
  if (dataframe[i][18] == 0):
    positive = positive + 1
  else:
    negative = negative + 1
print(positive)
print(negative)
while (positive!=negative):
  positive = 0
  negative = 0
  np.random.shuffle(dataframe)
  for i in range(0,train_size):
    if (dataframe[i][18] == 0):
      positive = positive + 1
    else:
      negative = negative + 1

print("new values")    
print(positive)
print(negative)
test_size = 100
labeled_dataset = dataframe[0:train_size,:]
unlabeled_dataset = dataframe[labeled_dataset.shape[0]: , :]
X_test = unlabeled_dataset[:test_size , :-1]
y_test = unlabeled_dataset[:test_size ,18]

no_of_queries = 0 
while no_of_queries!=50:
  # Splitting the dataset into Labeled and Unlabeled dataset
  labeled_dataset = dataframe[0:train_size,:]
  unlabeled_dataset = dataframe[labeled_dataset.shape[0]: , :]
  #print("Size of labeled Dataset {}".format(labeled_dataset.shape))
  # Splitting the dataset into training set and test set
  X_train = labeled_dataset[:,:-1]
  y_train = labeled_dataset[:,18]

  # Every time randomly selecting the test set from unlabeled dataset
  #np.random.shuffle(unlabeled_dataset)
  
  # Unlabeled dataset excluding the test set
  unlabeled_dataset = unlabeled_dataset[test_size:,:]
  X_unlabel = unlabeled_dataset[:,:-1]
  y_unlabel = unlabeled_dataset[:,18]

  # No need of Feature Scaling in our dataset (already in 0s and 1s) 

  # Train the Model using Naive Bayes Classifier
  from sklearn.naive_bayes import GaussianNB
  classifier = GaussianNB()
  classifier.fit(X_train,y_train)

  # Predicting the accuracy of the model
  #accuracy = classifier.score(X_test,y_test)
  #print(accuracy)

  '''
  QUERY SYNTHESIS
  '''
  # Getting the class label for an unlabeled value
  index = np.random.randint(0,X_unlabel.shape[0])
  k = 20 # no_of_committee members
  y_train = y_train.reshape(-1,1)
  new_df = np.concatenate((X_train,y_train),axis=1)
  np.random.shuffle(new_df)
  X_comm = new_df[0:20,:-1]
  y_comm = new_df[0:20,18]
  classifier1 = GaussianNB()
  classifier1.fit(X_comm,y_comm)
  # In Loop
  query = X_unlabel[index,:][np.newaxis, :]
  #print(query)
  y_pred = classifier1.predict(query)
  #print(y_pred)
  actual_y_label = y_unlabel[index]
  #print(actual_y_label)
  '''if y_pred!=actual_y_label:
    print("Hey!! It's Different")
  else:
    print("It's a match")
  '''
  if (y_pred==actual_y_label):
    new_entry = unlabeled_dataset[index , :][np.newaxis , :]
    #print(new_entry)
    labeled_dataset = np.append(labeled_dataset,new_entry,axis=0)
    unlabeled_dataset = np.delete(unlabeled_dataset, index , axis=0)
    train_size = train_size + 1
    no_of_queries = no_of_queries + 1
  #print(labeled_dataset.shape)
  #print(unlabeled_dataset.shape)
  
print(classifier.score(X_test,y_test))
print(labeled_dataset.shape)












