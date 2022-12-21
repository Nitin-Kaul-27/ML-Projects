# LOGISTIC REGRESION ON BREAST CANCER CLASSIFICATION

# REQUIRED LIBRARIES
import pandas as pd

# IMPORTING DATA
data = pd.read_csv("breast_cancer.csv")
X = data.iloc[:,1:-1].values
Y = data.iloc[:,-1].values

# SPLITING THE DATASET
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

# TRAINING THE MODEL
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)
model.fit(X_train,Y_train)

# TESTING THE MODEL
Y_pred = model.predict(X_test)

# EVALUATING THE MODEL WITH CONFUSION MATRIX
from sklearn.metrics import confusion_matrix, accuracy_score
CF = confusion_matrix(Y_test, Y_pred)
AS = accuracy_score(Y_test, Y_pred)
print("CONFUSION MATRIX")
print(CF)
print("..............")
print("RESULTS AFTER EVALUATION USING ACCURACY THROUGH CONFUSION MATRIX")
print("Accuracy: {:.2f} %".format(AS*100))

# EVALUATING THE MODEL USING K-FOLD CROSS VALIDATION
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = Y_train, cv=10)
print("..............")
print("RESULTS AFTER EVALUATION USING K-FOLD CROSS VALIDATION")
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))