import keras
import pickle
from keras import layers
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Model, Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

file1 = open('apps-sok.txt', 'r')


#train_path = 'train_test_supervised_with_timestamp/'
#test_path  = 'train_test_supervised_with_timestamp/'

train_path = 'train_autoencoder_with_timestamp/'
test_path  = 'test_autoencoder_with_timestamp/'
file1 = open('apps-sok-third-part.txt', 'r')
Lines1= file1.readlines()

file2 = open('apps-sok-third-part.txt', 'r')
Lines2= file2.readlines()
list_of_train = [1,2,3,4]
list_of_test =  [1,2,3,4]



df_train_all = pd.DataFrame()
df_test_all  = pd.DataFrame()

predicted_list = []
fprs, tprs, scores = [], [], []


#Lines = file1.readlines()
#Lines1 = ['train_autoencoder_with_timestamp/train_autoencoder_all.pkl']
#Lines2 = ['test_autoencoder_with_timestamp/test_autoencoder_all.pkl']
sc = StandardScaler()
norm = MinMaxScaler()




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

    print(df_train_all.shape)
    print(df_test_all.shape)

    # df_train_all = pd.read_pickle(Lines1[0])
    train_data_x = df_train_all.iloc[:, :-1]
    train_data_y = df_train_all.iloc[:, -1:]

    # df_test_all  = pd.read_pickle(Lines2[0])
    test_data_x = df_test_all.iloc[:, :-1]
    test_data_y = df_test_all.iloc[:, -1:]
    # dataset_test = (test_x - min_val) / (max_val - min_val)
    # dataset_test = tf.cast(dataset_test, tf.float32)

    print(df_train_all.shape)
    print(df_test_all.shape)

    scaled_train_data_x = sc.fit_transform(train_data_x)
    scaled_test_data_x = sc.fit_transform(test_data_x)


    # dataset_labels = norm.fit_transform(test_labels)
    # dataset_test = dataset_test[:10000]
    # dataset_train = dataset_train[:10000]
    # print(dataset_test.shape)

    # dataset_train_flatten = train_data_x.flatten()
    # print(dataset_train_flatten.shape)

    # print(test_labels[5])

    class AutoEncoder(Model):
        def __init__(self):
            super().__init__()
            print("IN ENC")
            self.encoder = Sequential([
                keras.Input(shape=(556), name='enc'),
                # encoder_input = keras.Input(shape=(1500, 555), name ='enc')
                # encoder_flatten = keras.layers.Flatten()(encoder_input)
                # encoder_layer1 = keras.layers.Dense(36630, activation='sigmoid')(encoder_flatten)
                keras.layers.Dense(240, activation='sigmoid'),
                keras.layers.Dense(64, activation='sigmoid')
            ])
            self.decoder = Sequential([
                keras.layers.Dense(64, activation="sigmoid"),
                keras.layers.Dense(240, activation='sigmoid'),
                # decoder_layer2 = keras.layers.Dense(36630, activation='sigmoid')(decoder_layer1)
                keras.layers.Dense(556, activation='sigmoid')
            ])

        def call(self, inputs):
            encoded = self.encoder(inputs)
            decoded = self.decoder(encoded)
            return decoded


    model = AutoEncoder()
    # model.compile(loss='mse', metrics=['mse'], optimizer='RMSprop')
    model.compile(loss='mse', metrics=['mse'], optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01))

    history = model.fit(
        scaled_train_data_x,
        scaled_train_data_x,
        epochs=20,
        batch_size=512
        # batch_size=512
        # validation_data=(dataset_test, dataset_test)
    )


    # opt = tf.keras.RMSprop(0.001, decay=1e-6)
    # autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
    # autoencoder.summary()

    def find_threshold(model, scaled_train_data_x):
        reconstructions = model.predict(scaled_train_data_x)
        # provides losses of individual instances
        reconstruction_errors = tf.keras.losses.mse(reconstructions, scaled_train_data_x)
        # threshold for anomaly scores
        threshold = np.mean(reconstruction_errors.numpy()) \
                    + np.std(reconstruction_errors.numpy())
        # threshold = np.percentile(reconstruction_errors, 99)
        print(threshold)
        # threshold = 0.1
        # plt.hist(reconstruction_errors[None, :], bins=20)
        # plt.xlabel("Train loss")
        # plt.ylabel("No of examples")
        # plt.show()

        return threshold


    def get_predictions(model, scaled_test_data_x, threshold):
        predictions = model.predict(scaled_test_data_x)
        # provides losses of individual instances
        errors = tf.keras.losses.mse(predictions, scaled_test_data_x)

        # plt.hist(errors[None, :], bins=20)
        # plt.xlabel("Test loss")
        # plt.ylabel("No of examples")
        # plt.show()
        # 0 = anomaly, 1 = normal
        # anomaly_mask = pd.Series(errors) > threshold
        # preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
        print("ERRORS SHAPE:", errors.shape)
        for i in range(len(errors)):
            if errors[i] < threshold:
                predicted_list.append(True)
            else:
                predicted_list.append(False)
        return tf.math.less(errors, threshold)
        # return predictions


    threshold = find_threshold(model, scaled_train_data_x)
    print(f"Threshold: {threshold}")
    # Threshold: 0.01001314025746261
    predictions = get_predictions(model, scaled_test_data_x, threshold)
    print(predictions.shape)
    print(scaled_test_data_x.shape)
    print(predictions)
    print(scaled_test_data_x)
    # y_pred=predictions
    # y_test=dataset_test
    # y_pred=np.argmax(predictions, axis=1)
    # y_test=np.argmax(dataset_test, axis=1)
    # print(y_pred)
    # print(y_test)
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)

    fpr, tpr, thresholds = roc_curve(test_data_y, predictions)
    auc_score = auc(fpr, tpr)
    fprs.append(fpr)
    tprs.append(tpr)

    print(confusion_matrix(test_data_y, predictions))
    print(classification_report(test_data_y, predictions))

    print("Accuracy :", accuracy_score(test_data_y, predictions))
    # print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(test_data_y, predictions))
    print("Recall   :", recall_score(test_data_y, predictions))
    # Epoch 1, threshold 0.00287, accuracy 92.247
    score = f1_score(test_data_y, predictions)
    print('F-Measure: %.3f', score)

    print(tf.get_static_value(predictions[5]))
    print(type(tf.get_static_value(predictions[5])))
    if (tf.get_static_value(predictions[5]) == 'True'):
        print("HAPPY")

    total_number = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    number_of_labels = len(test_data_y)
    test_labels = test_data_y
    lines_of_fpr = []
    lines_of_fnr = []

    break


def plot_roc_curve_simple(fprs, tprs):
    plt.figure(figsize=(8,8))
    for i in range(len(fprs)):
        roc_auc = auc(fprs[i], tprs[i])
        ysmoothed = gaussian_filter1d(tprs[i], sigma=2)
        plt.plot(fprs[i], tprs[i], label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))
        i +=1
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AE')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig('AE_scenario_s1_d3d3.png',  bbox_inches='tight')
    plt.show()




plot_roc_curve_simple(fprs, tprs)

exit()


for i in range(len(test_labels)):
    if test_labels[i] == 1 and predicted_list[i] is True:
        true_positive +=1
        #print("TP")
    if test_labels[i] == 0 and predicted_list[i] is False:
        true_negative +=1
    if test_labels[i] == 0 and predicted_list[i] is True:
        false_negative +=1
        lines_of_fnr.append(i)
    if test_labels[i] == 1 and predicted_list[i] is False:
        false_positive +=1
        lines_of_fpr.append(i)

tpr = (true_positive/i)*100
tnr = (true_negative/i)*100
fpr = (false_positive/i)*100
fnr = (false_negative/i)*100

print("FALSE NEGATIVE:", false_negative)
print("FALSE POSITIVE:", false_positive)
print("TPR:", tpr)
print("TNR:", tnr)
print("FPR:", fpr)
print("FNR:", fnr)
#print(lines_of_fpr)
#print(lines_of_fnr)




