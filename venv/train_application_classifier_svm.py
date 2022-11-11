import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV



def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')

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

#sc = StandardScaler()
sc = Normalizer()
scaled_train_data_x = sc.fit_transform(train_data_x)
scaled_test_data_x = sc.fit_transform(test_data_x)

#onehot_encoder = OneHotEncoder()
#onehot_encoded_train_y = onehot_encoder.fit_transform(train_data_y)
#onehot_encoded_test_y = onehot_encoder.fit_transform(test_data_y)

label_encoder = preprocessing.LabelEncoder()
label_encoded_train_y = label_encoder.fit_transform(train_data_y)
label_encoded_test_y = label_encoder.fit_transform(test_data_y)
print(np.unique(label_encoded_train_y))

svc = SVC(kernel='poly', degree=4)
parameters = {
    "n_estimators":[200]
}

#parameters = {
#    "n_estimators":[50,100,150,200,250],
#    "max_depth":[2,4,8,16,32,None]
#}

#cv = GridSearchCV(rfc,parameters,cv=5)
#cv.fit(scaled_train_data_x,label_encoded_train_y)

#display(cv)



svc.fit(scaled_train_data_x, label_encoded_train_y)
y_pred = svc.predict(scaled_test_data_x)
accuracy = metrics.accuracy_score(y_pred, label_encoded_test_y)

print(accuracy)
print(confusion_matrix(label_encoded_test_y,y_pred))
print(classification_report(label_encoded_test_y,y_pred))
print(accuracy_score(label_encoded_test_y, y_pred))



