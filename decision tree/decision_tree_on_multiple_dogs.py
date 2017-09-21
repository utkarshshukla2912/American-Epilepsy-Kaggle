import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report
import itertools


df_train = pd.read_csv('../../files/Train_sample.csv')
df_test = pd.read_csv('../../files/Test_sample.csv')

x_train = df_train.drop(['Class'], axis = 1)
y_train = df_train['Class']

x_test = df_test.drop(['Class'], axis = 1)
y_test = df_test['Class']


# Implementing Standard Scaler on Power Spectral Density
'''
#scaler = StandardScaler()
columns_to_scale = ['psd1','psd2','psd3','psd4','psd5']
for column in columns_to_scale:
	scaler.fit(np.array(x_train[column]))
	x_train[column] = scaler.transform(np.array(x_train[column]))
	scaler.fit(np.array(x_test[column]))
	x_test[column] = scaler.transform(x_test[column])
'''


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
print("Number of correct one prediction: ",number_of_correct1)
print("Number of wrong one prediction: ",number_of_wrong1)
print("DT: Predictions ",list(set(prediction)))
print("DT: ",clf.score(x_test, y_test))


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,prediction)
np.set_printoptions(precision=2)

print "Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')


# Calculation of Precision for our Prediction
print "Precision: ", np.true_divide(cnf_matrix[1,1],(cnf_matrix[0,1]+cnf_matrix[1,1]))

# Calculation of Recall for our Prediction
recall=np.true_divide(cnf_matrix[1,1],(cnf_matrix[1,0]+cnf_matrix[1,1]))
print "Recall: ", recall

# Calculation of F1score for our Prediction
precision=np.true_divide(cnf_matrix[1,1],(cnf_matrix[0,1]+cnf_matrix[1,1]))
f1score=2*np.true_divide(precision*recall,(precision+recall))
print "F1 Score: ", f1score

# printing important features
column_list = list(x_train)
arr = clf1.feature_importances_
print(arr)

