import keras
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
from sklearn.metrics import accuracy_score, precision_score, recall_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

file1 = open('train_autoencoder_pickle.txt', 'r')

#Lines = file1.readlines()
Lines = ['train_autoencoder/train_autoencoder_all.pkl']
Line2 = ['test_autoencoder/test_autoencoder_all.pkl']
scaler = StandardScaler()
norm = MinMaxScaler()
for line in Lines:
    content = line.strip()
    train_data = pd.read_pickle(content)
    #train_data= train_data.values
    print(type(train_data))
    print(content, train_data.shape)
    rows, columns = train_data.shape
    print(rows, columns)
    train_x = train_data.iloc[:, 0:-1]
    #print(train_x)
    train_labels = train_data.iloc[:, -1]
    print(train_x.shape)
    #min_val = tf.reduce_min(train_x)
    #max_val = tf.reduce_max(train_x)
    #dataset_train = (train_x - min_val) / (max_val - min_val)
    #dataset_train = tf.cast(dataset_train, tf.float32)
    dataset_train = norm.fit_transform(train_x)
    #dataset_labels = norm.fit_transform(train_labels)
    #print(dataset_train)
    print(dataset_train.shape)
    #break

test_data = pd.read_pickle(Line2[0])
#test_data = test_data.values
test_x = test_data.iloc[:, 0:-1]
test_labels = test_data.iloc[:, -1]
#dataset_test = (test_x - min_val) / (max_val - min_val)
#dataset_test = tf.cast(dataset_test, tf.float32)
dataset_test = norm.fit_transform(test_x)
#dataset_labels = norm.fit_transform(test_labels)
#dataset_test = dataset_test[:10000]
#dataset_train = dataset_train[:10000]
print(dataset_test.shape)

dataset_train_flatten = dataset_train.flatten()
print(dataset_train_flatten.shape)


class AutoEncoder(Model):
    def __init__(self):
        super().__init__()
        print("IN ENC")
        self.encoder = Sequential([
            keras.Input(shape=(555), name='enc'),
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
            keras.layers.Dense(555, activation='sigmoid')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded



model = AutoEncoder()
model.compile(loss='mse', metrics=['mse'], optimizer='RMSprop')

history = model.fit(
    dataset_train,
    dataset_train,
    epochs=10,
    batch_size=32,
    validation_data=(dataset_test, dataset_test)
)

#opt = tf.keras.RMSprop(0.001, decay=1e-6)
#autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
#autoencoder.summary()


def find_threshold(model, x_train_scaled):
  reconstructions = model.predict(x_train_scaled)
  # provides losses of individual instances
  reconstruction_errors = tf.keras.losses.mse(reconstructions, x_train_scaled)
  # threshold for anomaly scores
  threshold = np.mean(reconstruction_errors.numpy()) \
      + np.std(reconstruction_errors.numpy())
  #threshold = 0.1
  plt.hist(reconstruction_errors[None, :], bins=20)
  plt.xlabel("Train loss")
  plt.ylabel("No of examples")
  plt.show()

  return threshold

def get_predictions(model, x_test_scaled, threshold):
  predictions = model.predict(x_test_scaled)
  # provides losses of individual instances
  errors = tf.keras.losses.mse(predictions, x_test_scaled)

  plt.hist(errors[None, :], bins=20)
  plt.xlabel("Test loss")
  plt.ylabel("No of examples")
  plt.show()
  # 0 = anomaly, 1 = normal
  #anomaly_mask = pd.Series(errors) > threshold
  #preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
  return tf.math.less(errors, threshold)
  #return predictions

threshold = find_threshold(model, dataset_train)
print(f"Threshold: {threshold}")
# Threshold: 0.01001314025746261
predictions = get_predictions(model, dataset_test, threshold)
print(predictions.shape)
print(dataset_test.shape)
print(predictions)
print(dataset_test)
#y_pred=predictions
#y_test=dataset_test
#y_pred=np.argmax(predictions, axis=1)
#y_test=np.argmax(dataset_test, axis=1)
#print(y_pred)
#print(y_test)
#cm = confusion_matrix(y_test, y_pred)
#print(cm)
print("Accuracy :", accuracy_score(test_labels, predictions))
#print("Accuracy :", accuracy_score(y_test, y_pred))
#print("Precision:", precision_score(y_test, y_pred))
#print("Recall   :", recall_score(y_test, y_pred))
# 0.944