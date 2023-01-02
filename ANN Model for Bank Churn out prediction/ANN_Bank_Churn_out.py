# Artificial Neural Network for Predicting Bank Customer Churning Out 

# REQUIRED LIBRARIES
import pandas as pd
import numpy as np
import tensorflow as tf

# IMPORTING DATA
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# ENCODING THE VARIABLES
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
X[:,2] = LE.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# SPLITTING THE DATA SET
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2, random_state=0)

# FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# ANN MODELING
# Initializing ANN and adding input, hidden, output layers
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units = 6, activation='relu'))
model.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
model.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

# Training the Model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics= ['accuracy'])
model.fit(X_train, Y_train, batch_size=32, epochs= 30)

# Testing the model
Y_pred = model.predict(X_test)
Y_pred = Y_pred > 0.5

# EVALUATING THE MODEL
from sklearn.metrics import confusion_matrix, accuracy_score
CF = confusion_matrix(Y_test, Y_pred)
AS = accuracy_score(Y_test, Y_pred)
print("CONFUSION MATRIX")
print(CF)
print("..............")
print("RESULTS AFTER EVALUATION USING ACCURACY THROUGH CONFUSION MATRIX")
print("Accuracy: {:.2f} %".format(AS*100))