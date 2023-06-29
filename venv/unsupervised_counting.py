import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import pickle

supervised_path = 'C:/Users/12103/PycharmProjects/cdl_remade/venv/train_test_supervised_with_timestamp/'
file1 = open('apps-sok.txt', 'r')
Lines = file1.readlines()
list_of_train = [1,2,4]
list_of_test = [3]
df_train_all = pd.DataFrame()
df_test_all  = pd.DataFrame()

def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')



Lines1 = ['train_autoencoder_with_timestamp/train_autoencoder_all.pkl']
Lines2 = ['test_autoencoder_with_timestamp/test_autoencoder_all.pkl']



#exit()


df_train_all = pd.read_pickle(Lines1[0])
train_data_x = df_train_all.iloc[:, :-1]
train_data_y = df_train_all.iloc[:, -1:]

df_test_all  = pd.read_pickle(Lines2[0])
test_data_x = df_test_all.iloc[:, :-1]
test_data_y = df_test_all.iloc[:, -1:]


print(df_train_all.shape)
print(df_test_all.shape)

count_anomalies = 0
count_normal = 0

list_train_data_y = train_data_y[train_data_y.columns[0]].values.tolist()
for  values in list_train_data_y:
    #print(values)
    if values == -1:
        #print(values)
        count_anomalies += 1
    else:
        count_normal +=1

list_test_data_y = test_data_y[test_data_y.columns[0]].values.tolist()
for  values in list_test_data_y:
    #print(values)
    if values == -1:
        #print(values)
        count_anomalies += 1
    else:
        count_normal +=1

print("count_anomalies", count_anomalies)
print("count_normal", count_normal)


#count_anomalies 18182
#count_normal 712199