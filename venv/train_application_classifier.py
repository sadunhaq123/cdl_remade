import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

train_data_pickle = ['train_classifier/all_combined.pkl']
test_data_pickle = ['test_classifier/all_combined.pkl']

train_data = pd.read_pickle(train_data_pickle[0])
test_data = pd.read_pickle(test_data_pickle[0])

train_data_x = train_data.iloc[:, :-1]
train_data_y = train_data.iloc[:, -1:]
#print(train_data_y)

test_data_x = test_data.iloc[:, :-1]
test_data_y = test_data.iloc[:, -1:]
#print(test_data_y)

sc = StandardScaler()
scaled_train_data_x = sc.fit_transform(train_data_x)
scaled_test_data_x = sc.fit_transform(test_data_x)

label_encoder = preprocessing.LabelEncoder()
label_encoded_train_y = label_encoder.fit_transform(train_data_y)
label_encoded_test_y = label_encoder.fit_transform(test_data_y)

clf = RandomForestClassifier(n_estimators = 200)

clf.fit(scaled_train_data_x, label_encoded_train_y)
y_pred = clf.predict(scaled_test_data_x)
accuracy = metrics.accuracy_score(y_pred, label_encoded_test_y)

print(accuracy)



