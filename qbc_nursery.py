# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# Splitting the dataset into test set and training set
from sklearn.model_selection import train_test_split
X_train ,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x =  StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

# Training the Model using naive bayes classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predicting the test results
y_pred = classifier.predict(X_test)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(classifier.score(X_test,y_test))

# three different scatter series so the class labels in the legend are distinct
'''plt.scatter(X[y==0]['Non-Diabetic Patients'], label='Class 1', c='red')
plt.scatter(X[y==1]['Diabetic Patients'],label='Class 2', c='lightgreen')

# Prettify the graph
plt.legend()
plt.xlabel('')
plt.ylabel('')

# display
plt.show()
'''