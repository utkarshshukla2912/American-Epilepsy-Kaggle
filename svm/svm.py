import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = pd.read_csv('../../files/Dog_3_train.csv')
X = df.drop(['Class'], 1)
Y = df['Class']
print("1 In Dataset: ", list(Y).count(1))
print("0 In Dataset: ", list(Y).count(0))
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
print("1 in Y train: ",list(y_train).count(1),"1 in Y test: ",list(y_test).count(1))
print("0 in Y train: ",list(y_train).count(0),"0 in Y test: ",list(y_test).count(0))
clf = SVC(kernel = 'poly')
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
print("SVM: Prediction",list(set(prediction)))
print("SVM: ",clf.score(x_test, y_test))



