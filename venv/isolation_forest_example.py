import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
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

#train_data_pickle = ['train_classifier/all_combined.pkl']
#test_data_pickle = ['test_classifier/all_combined.pkl']
#CVE-2012-1823

train_data_combined = pd.DataFrame()
test_data_combined  = pd.DataFrame()
for i in range(1, 5):
    train_data_pickle = ['train_autoencoder_with_timestamp/CVE-2012-1823-' + str(i) + '.pkl']
    test_data_pickle  = ['test_autoencoder_with_timestamp/CVE-2012-1823-' + str(i) +'.pkl']
    train_data = pd.read_pickle(train_data_pickle[0])
    test_data = pd.read_pickle(test_data_pickle[0])
    train_data_combined = pd.concat([train_data_combined, train_data], axis=0)
    test_data_combined  = pd.concat([test_data_combined, test_data], axis=0)


#print(test_data)
#exit()

train_data_x = train_data_combined.iloc[:, :-1] #All rows and columns without label
train_data_y = train_data_combined.iloc[:, -1:] # All rows and 1 column with label ONLY
#print(train_data_y)


test_data_x = test_data_combined.iloc[:, :-1]
test_data_y = test_data_combined.iloc[:, -1:]
print(test_data_y)
#print(type(test_data_y))

# for i in range(len(test_data_y[test_data_y.columns[0]])):
#     if test_data_y.iloc[i, 0] == -1:
#         print("YES", i)

#exit()
sc = StandardScaler()
#sc = Normalizer()
scaled_train_data_x = sc.fit_transform(train_data_x)
scaled_test_data_x = sc.fit_transform(test_data_x)
print(scaled_test_data_x)

#onehot_encoder = OneHotEncoder()
#onehot_encoded_train_y = onehot_encoder.fit_transform(train_data_y)
#onehot_encoded_test_y = onehot_encoder.fit_transform(test_data_y)

#exit()

#label_encoder = preprocessing.LabelEncoder()
#label_encoded_train_y = label_encoder.fit_transform(train_data_y)
#label_encoded_test_y = label_encoder.fit_transform(test_data_y)
#print(np.unique(onehot_encoded_train_y))
#print(np.unique(onehot_encoded_test_y))
#print(set(list(label_encoder.inverse_transform(label_encoded_train_y))))

#model = IsolationForest(contamination='auto', max_features=555)
model = IsolationForest(contamination=0.01, max_features=555)

df_train_x = pd.DataFrame(scaled_train_data_x)
df_test_x  = pd.DataFrame(scaled_test_data_x)

full_df = pd.concat([df_train_x, df_test_x], axis=0)
model.fit(full_df)

y_pred = model.predict(scaled_test_data_x)
print(y_pred)
print(len(y_pred))
print(y_pred.shape)

print(type(y_pred))

actual_numpy =  test_data_y.to_numpy()
actual = np.where(actual_numpy == -1)
print(actual)
predicted = np.where(y_pred == -1)
print(predicted)


#accuracy = metrics.accuracy_score(y_pred, label_encoded_test_y)


#print(accuracy)
#print(confusion_matrix(label_encoded_test_y,y_pred))
#print(classification_report(label_encoded_test_y,y_pred))
#print(accuracy_score(label_encoded_test_y, y_pred))
#print("Precision:", precision_score(label_encoded_test_y, y_pred))
#print("Recall   :", recall_score(label_encoded_test_y, y_pred))


