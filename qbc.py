# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import copy

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
y = y.reshape(-1,1)
dataframe = np.concatenate((X,y),axis=1)
new_df = copy.deepcopy(dataframe)
#print(new_df[:,18])
# In order to randomly select 100 rows every time we shuffle the data 
np.random.shuffle(dataframe)
print(dataframe.shape)
# Splitting the dataset into Labeled and Unlabeled dataset
train_size = 100
labeled_dataset = dataframe[0:train_size,:]
unlabeled_dataset = dataframe[labeled_dataset.shape[0]: , :]

# Splitting the dataset into training set and test set
X_train = labeled_dataset[:,:-1]
y_train = labeled_dataset[:,18]

# Every time randomly selecting the test set from unlabeled dataset
#np.random.shuffle(unlabeled_dataset)
test_size = 100
X_test = unlabeled_dataset[:test_size , :-1]
y_test = unlabeled_dataset[:test_size ,18]

# Unlabeled dataset excluding the test set
unlabeled_dataset = unlabeled_dataset[test_size:,:]
X_unlabel = unlabeled_dataset[:,:-1]
y_unlabel = unlabeled_dataset[:,18]
#print(X_unlabel)
#print(y_unlabel)

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

# Predicting the accuracy of the model
accuracy = classifier.score(X_test,y_test)
print(accuracy)

# Taking Samples in the ratio of 7:3 
print(new_df.shape)
df1 = new_df[new_df[:,18] == 1]
df2 = new_df[new_df[:,18] == 0]
np.random.shuffle(df1) # All +ve values
np.random.shuffle(df2) # All -ve values

# Splitting the dataset into Labeled and Unlabeled dataset (ratio 1:1)
train_size1 = 100
labeled_dataset1 = np.concatenate((df1[0:49,:] , df2[0:49,:]), axis = 0)
unlabeled_dataset1 = np.concatenate((df1[50:,:] , df2[50:,:]) , axis = 0)

# Splitting the dataset into training set and test set
X_train1 = labeled_dataset1[:,:-1]
y_train1 = labeled_dataset1[:,18]

# Every time randomly selecting the test set from unlabeled dataset
#np.random.shuffle(unlabeled_dataset)
test_size1 = 100
X_test1 = unlabeled_dataset1[:test_size , :-1]
y_test1 = unlabeled_dataset1[:test_size ,18]

# Unlabeled dataset excluding the test set
unlabeled_dataset1 = unlabeled_dataset1[test_size1:,:]
X_unlabel1 = unlabeled_dataset1[:,:-1]
y_unlabel1 = unlabeled_dataset1[:,18]
#print(X_unlabel)
#print(y_unlabel)

# No need of Feature Scaling in our dataset (already in 0s and 1s) 

# Train the Model using Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
classifier1 = GaussianNB()
classifier1.fit(X_train1,y_train1)

# Predicting the test set
y_pred1 = classifier.predict(X_test1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test,y_pred)
print(cm1)

# Predicting the accuracy of the model
new_accuracy = classifier.score(X_test1,y_test1)
print(new_accuracy)

# Processing committee members using K-Means Clustering
'''Classification algorithm A = naive bayes
The number of committee members: K
Few labeled examples: L = 100
Unlabeled examples: UL = rest
The condition of stopping: ζ
The threshold of Vote Entropy: θ
The threshold of KL-d min α'''

from sklearn.cluster import KMeans
WCSS = []
for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++',max_iter = 300, n_init = 10 , random_state = 0)
  kmeans.fit(X_train1)
  WCSS.append(kmeans.inertia_)
''' 
plt.plot(range(1,11),WCSS)
plt.title("The Elbow Method")
plt.xlabel('No. of Clusters')
plt.ylabel('WCSS')
plt.show()
'''
kmeans = KMeans(n_clusters=5, init='k-means++',max_iter = 300, n_init = 10 , random_state = 0)
y_means = kmeans.fit_predict(X_train1)
print(y_means)




