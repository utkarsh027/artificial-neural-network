# Importing the libraries
#artificial neural network
#data preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#now lets make ann
#import keras library
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialise neural network
classifier=Sequential()

#adding the input layer and first hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
#adding the second layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
#adding the output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
#compiling the ann
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#if dependent variable has more thn two categories then use softmax instead of sigmoid
#fiting the ann to the training set
#optimizer is algorithm we want to use stochastic grdient descent algoritm and one is adam
classifier.fit(X_train,y_train,batch_size=10,epochs=100)

#making the prediction and evaluting the model


# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred=(y_pred>0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)