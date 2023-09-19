import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

#%matplotlib inline
matplotlib.rcParams.update({'font.size': 20})

supervised_path = 'train_test_supervised_with_timestamp/'


train_path = 'train_autoencoder_with_timestamp/'
test_path  = 'test_autoencoder_with_timestamp/'

file1 = open('apps-sok-second-part.txt', 'r')
Lines1 = file1.readlines()
file2 = open('apps-sok-third-part.txt', 'r')
Lines2 = file2.readlines()
list_of_train = [1,2,3,4]
list_of_test  = [1,2,3,4]
df_train_all = pd.DataFrame()
df_test_all  = pd.DataFrame()


df_train_all = pd.DataFrame()
df_test_all  = pd.DataFrame()

predicted_list = []
fprs, tprs, scores = [], [], []


#Lines = file1.readlines()
#Lines1 = ['train_autoencoder_with_timestamp/train_autoencoder_all.pkl']
#Lines2 = ['test_autoencoder_with_timestamp/test_autoencoder_all.pkl']
#sc = StandardScaler()
#norm = MinMaxScaler()


def compute_roc_auc_train(scaled_train_data_x, train_data_y):
    y_predict = rfc.predict(scaled_train_data_x)
    fpr, tpr, thresholds = roc_curve(train_data_y, y_predict)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score

def compute_roc_auc_test(scaled_test_data_x, test_data_y):
    y_predict = rfc.predict(scaled_test_data_x)
    fpr, tpr, thresholds = roc_curve(test_data_y, y_predict)
    print("FPR:", fpr)
    print("TPR:", tpr)
    print("THR:", thresholds)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score



train_count=0

for test_element in range (1, 5):
    print(test_element)

    for line1 in Lines1:

        content = line1.strip()
        # print(content)
        for k in list_of_train:
            if k == test_element:
                continue

            else:
                path = train_path + content + '-' + str(k) + '.pkl'
                # path = 'C:/Users/sadun/PycharmProjects/cdl_remade/venv/data-CDL/' + content + '/' + content + '-' + str(k) + '_freqvector_full.csv'
                print("TRAIN")
                print(path)
                picklefile_train = open(path, 'rb')
                df_individual_train = pickle.load(picklefile_train)
                picklefile_train.close()
                df_train_all = pd.concat([df_train_all, df_individual_train], axis=0)


            #break

        #break


        train_count += 1

    for line2 in Lines2:

        content = line2.strip()

        for ki in list_of_test:
            if ki == test_element:
                path = test_path + content + '-' + str(ki) + '.pkl'
                # path = 'C:/Users/sadun/PycharmProjects/cdl_remade/venv/data-CDL/' + content + '/' + content + '-' + str(k) + '_freqvector_full.csv'
                print("TEST")
                print(path)
                picklefile_test = open(path, 'rb')
                df_individual_test = pickle.load(picklefile_test)
                picklefile_test.close()
                df_test_all = pd.concat([df_test_all, df_individual_test], axis=0)

            else:
                continue

            #break

        #break

    print(df_train_all.shape)
    print(df_test_all.shape)

    # With time
    # train_data_x = df_train_all.iloc[:, :-1]
    # train_data_y = df_train_all.iloc[:, -1:]
    #
    # test_data_x = df_test_all.iloc[:, :-1]
    # test_data_y = df_test_all.iloc[:, -1:]

    train_data_x = df_train_all.iloc[:, :-1]
    train_data_y = df_train_all.iloc[:, -1:]

    test_data_x = df_test_all.iloc[:, :-1]
    test_data_y = df_test_all.iloc[:, -1:]

    # Example
    # train_data_x = df_train_all.iloc[:4155, :-1]
    # train_data_y = df_train_all.iloc[:4155, -1:]

    # test_data_x = df_test_all.iloc[:4155, :-1]
    # test_data_y = df_test_all.iloc[:4155, -1:]

    print(train_data_x.shape)
    print(test_data_x.shape)

    sc = StandardScaler()
    # sc = Normalizer()
    scaled_train_data_x = sc.fit_transform(train_data_x)
    scaled_test_data_x = sc.fit_transform(test_data_x)
    print(type(scaled_train_data_x))
    # exit()

    # train_data_y = sc.fit_transform(train_data_y)
    # test_data_y  = sc.fit_transform(test_data_y)

    # print(train_data_y.values.ravel())
    # rfc = RandomForestClassifier(n_estimators=200, max_depth=16, max_features=100)
    # rfc = RandomForestClassifier(n_estimators=200, class_weight='balanced')

    # dict_weights = {0:16.10, 1:0.51}

    #rfc = OneClassSVM(kernel='linear', gamma=0.001, verbose=True)
    rfc = OneClassSVM(kernel='rbf', gamma=0.001, verbose=True)




    #X = np.concatenate((scaled_train_data_x, scaled_test_data_x), axis=0)
    #y = np.concatenate((train_data_y.values.ravel(), test_data_y.values.ravel()), axis=0)

    #X = pd.DataFrame(X)
    #y = pd.DataFrame(y)

    #print(X.shape)
    #print(y.shape)

    rfc.fit(scaled_train_data_x)
    _, _, auc_score_train = compute_roc_auc_train(scaled_train_data_x, train_data_y)
    fpr, tpr, auc_score = compute_roc_auc_test(scaled_test_data_x, test_data_y)
    scores.append((auc_score_train, auc_score))
    fprs.append(fpr)
    tprs.append(tpr)
    #scaled_test_data_x = X.iloc[test]
    #test_data_y = y.iloc[test].values.ravel()
    y_pred = rfc.predict(scaled_test_data_x)
    y_pred = [0 if i == -1 else 1 for i in y_pred]
    accuracy = metrics.accuracy_score(y_pred, test_data_y)
    print(accuracy)
    print(confusion_matrix(test_data_y, y_pred))
    print(classification_report(test_data_y, y_pred))
    print(accuracy_score(test_data_y, y_pred))

    precision = precision_score(test_data_y, y_pred, average='macro')
    print('Precision: %.3f', precision)
    recall = recall_score(test_data_y, y_pred, average='macro')
    print('Recall: %.3f', recall)
    score = f1_score(test_data_y, y_pred, average='macro')
    print('F-Measure: %.3f', score)
    #break

#exit()

def plot_roc_curve_simple(fprs, tprs):
    plt.figure(figsize=(8,8))
    for i in range(len(fprs)):
        roc_auc = auc(fprs[i], tprs[i])
        ysmoothed = gaussian_filter1d(tprs[i], sigma=2)
        plt.plot(fprs[i], tprs[i], label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))
        i +=1
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for One-Class-SVM')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig('OCSVM_scenario_s2_d2d3.png',  bbox_inches='tight')
    plt.show()




plot_roc_curve_simple(fprs, tprs);
pd.DataFrame(scores, columns=['AUC Train', 'AUC Test'])



#exit()
#df_test_reversed = df_test_all[::-1]
#df_test_all.drop(df_test_all.index, inplace=True)
#df_test_all = df_test_reversed








# #Below The 4fold
# #rfc.fit(scaled_train_data_x, train_data_y.values.ravel())
# #rfc.fit(scaled_train_data_x, train_data_y)
# y_pred = rfc.predict(scaled_test_data_x)
# print(y_pred)
# print(y_pred.shape)
# accuracy = metrics.accuracy_score(y_pred, test_data_y)
#
# #apple
# #---------------Metrics------------
#
# print(accuracy)
# print(confusion_matrix(test_data_y,y_pred))
# print(classification_report(test_data_y,y_pred))
# print(accuracy_score(test_data_y, y_pred))
#
#
# precision = precision_score(test_data_y, y_pred)
# print('Precision: %.3f', precision)
# recall = recall_score(test_data_y, y_pred)
# print('Recall: %.3f', recall)
# score = f1_score(test_data_y, y_pred)
# print('F-Measure: %.3f', score)
