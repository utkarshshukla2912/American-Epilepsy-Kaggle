import pandas as pd
import numpy as np
import sklearn
import itertools
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing 
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 
from sklearn.metrics import roc_auc_score


# Reading Training Data
df = pd.read_csv('../../Files/Dog2_testing_train_sample.csv')
x_train = df.drop(["Class"], 1)
X_headers=list(x_train)


# Normalizing the Values Column Wise
for val in X_headers:
	x_train[val] = sklearn.preprocessing.StandardScaler().fit_transform(x_train[val].reshape(-1, 1))
y_train = df['Class']


# Reading Testing Data
dftest = pd.read_csv('../../Files/Dog2_testing_test_sample.csv')
x_test = dftest.drop(["Class"], 1)
X_headers_test=list(x_test)


# Normalizing the Values Column Wise
for val in X_headers_test:
	x_test[val] = sklearn.preprocessing.StandardScaler().fit_transform(x_test[val].reshape(-1, 1))
y_test = dftest['Class']


# Dataset Values
print('Test size:', len(x_test))
print('Train size:', len(x_train))
print("1 in Y train: ",list(y_train).count(1),"1 in Y test: ",list(y_test).count(1))
print("0 in Y train: ",list(y_train).count(0),"0 in Y test: ",list(y_test).count(0))


# Defining The Classifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
correct_prediction = []
correct_one = 0
wrong_one = 0

# finding the number of correct and wrong predictions
for i in range(0,len(prediction)):
	if prediction[i] == list(y_test)[i]:
		correct_prediction.append(1)
		# Finding the total number of correct ones predicted
		if prediction[i] == 1:
			correct_one += 1

	else:
		correct_prediction.append(0)
		# Finding the total number of wrong ones predicted
		if prediction[i] == 1:
			wrong_one += 1


print('Number of correct prediction: ', correct_prediction.count(1))
print('Number of wrong prediction: ', correct_prediction.count(0))
print("DT: Predictions ",list(set(prediction)))
print("DT: ",clf.score(x_test, y_test))
unique, counts = np.unique(prediction, return_counts=True)
print (dict(zip(unique, counts)))


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

print 'ROC curve',roc_auc_score(y_test, prediction)
