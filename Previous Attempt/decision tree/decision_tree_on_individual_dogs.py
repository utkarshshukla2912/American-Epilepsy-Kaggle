import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report
from sklearn.metrics import roc_auc_score


df = pd.read_csv('../../files/Dog1_train_sample_feature_ext.csv')
X = df.drop(['Class',], axis = 1)
Y = df['Class']

# Dataset Values
print("Dataset length:", len(df))
print("1 In Dataset: ", list(Y).count(1))
print("0 In Dataset: ", list(Y).count(0))

# Splitting of Dataset
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

# Dataset Values
print("1 in Y train: ",list(y_train).count(1),"1 in Y test: ",list(y_test).count(1))
print("0 in Y train: ",list(y_train).count(0),"0 in Y test: ",list(y_test).count(0))

# Classifier Declaration
clf = DecisionTreeClassifier(random_state=0)
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
y_test_list = list(y_test)
true_values = []
number_of_correct1 = 0
number_of_wrong1 = 0

# Finding Total Number of misclassifications 
if len(prediction) == len(y_test_list):
	for i in range(0,len(prediction)):
		if prediction[i] == y_test_list[i]:
			if prediction[i] == 1:
				number_of_correct1 += 1
			true_values.append(1)
		else:
			if prediction[i] == 1:
				number_of_wrong1 += 1
			true_values.append(0)

# Dataset Values After Prediction			
print('Number of correct predictions: ',true_values.count(1))
print('Number of wrong predictions: ',true_values.count(0))			
print("DT: Predictions ",list(set(prediction)))
print("DT: ",clf.score(x_test, y_test))


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,prediction)
np.set_printoptions(precision=2)

print "Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])

# Calculation of Precision for our Prediction
print "Precision: ", np.true_divide(cnf_matrix[1,1],(cnf_matrix[0,1]+cnf_matrix[1,1]))

# Calculation of Recall for our Prediction
recall=np.true_divide(cnf_matrix[1,1],(cnf_matrix[1,0]+cnf_matrix[1,1]))
print "Recall: ", recall

# Calculation of F1score for our Prediction
precision=np.true_divide(cnf_matrix[1,1],(cnf_matrix[0,1]+cnf_matrix[1,1]))
f1score=2*np.true_divide(precision*recall,(precision+recall))
print "F1 Score: ", f1score

print('ROC curve',roc_auc_score(y_test, prediction))

# printing important features
column_list = list(x_train)
arr = clf.feature_importances_
print(arr)



