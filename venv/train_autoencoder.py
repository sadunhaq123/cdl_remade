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
from sklearn.metrics import accuracy_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

file1 = open('train_autoencoder_pickle.txt', 'r')

#Lines = file1.readlines()
Lines = ['train_autoencoder/all_combined_train_auto_encoder.pkl']
Line2 = ['test_models/all_combined_test_auto_encoder.pkl']
scaler = StandardScaler()
norm = MinMaxScaler()
for line in Lines:
    content = line.strip()
    train_data = pd.read_pickle(content)
    print(content, train_data.shape)
    rows, columns = train_data.shape
    print(rows, columns)
    dataset_train = norm.fit_transform(train_data)
    #print(dataset_train)
    print(dataset_train.shape)
    #break

test_data = pd.read_pickle(Line2[0])
dataset_test = norm.fit_transform(test_data)
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

def encoder():
    print("IN ENC")
    encoder_input = keras.Input(shape=(8791200), name ='enc')
    #encoder_input = keras.Input(shape=(1500, 555), name ='enc')
    #encoder_flatten = keras.layers.Flatten()(encoder_input)
    #encoder_layer1 = keras.layers.Dense(36630, activation='sigmoid')(encoder_flatten)
    encoder_layer2 = keras.layers.Dense(2400, activation='sigmoid')(encoder_input)
    encoder_output = keras.layers.Dense(64, activation='sigmoid')(encoder_layer2)

#encoder()
print("HAPPY")
#encoder = keras.Model(encoder_input, encoder_output, name='encoder')

def decoder():
    decoder_input = keras.layers.Dense(64, activation="sigmoid")(encoder_output)
    decoder_layer1 = keras.layers.Dense(2400, activation='sigmoid')(decoder_input)
    #decoder_layer2 = keras.layers.Dense(36630, activation='sigmoid')(decoder_layer1)
    decoder_layer3 = keras.layers.Dense(8791200, activation='sigmoid')(decoder_layer1)
    #decoder_output = keras.Layers.Reshape((158400, 555))(decoder_layer3)


model = AutoEncoder()
model.compile(loss='msle', metrics=['mse'], optimizer='RMSprop')

history = model.fit(
    dataset_train,
    dataset_train,
    epochs=20,
    batch_size=32,
    validation_data=(dataset_test, dataset_test)
)

#opt = tf.keras.RMSprop(0.001, decay=1e-6)
#autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
#autoencoder.summary()


def find_threshold(model, x_train_scaled):
  reconstructions = model.predict(x_train_scaled)
  # provides losses of individual instances
  reconstruction_errors = tf.keras.losses.msle(reconstructions, x_train_scaled)
  # threshold for anomaly scores
  threshold = np.mean(reconstruction_errors.numpy()) \
      + np.std(reconstruction_errors.numpy())
  return threshold

def get_predictions(model, x_test_scaled, threshold):
  predictions = model.predict(x_test_scaled)
  # provides losses of individual instances
  errors = tf.keras.losses.msle(predictions, x_test_scaled)
  # 0 = anomaly, 1 = normal
  #anomaly_mask = pd.Series(errors) > threshold
  #preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
  return predictions

threshold = find_threshold(model, dataset_train)
print(f"Threshold: {threshold}")
# Threshold: 0.01001314025746261
predictions = get_predictions(model, dataset_test, threshold)
print(predictions.shape)
print(dataset_test.shape)
print(predictions)
print(dataset_test)
y_pred=np.argmax(predictions, axis=1)
y_test=np.argmax(dataset_test, axis=1)
print(y_pred)
print(y_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
# 0.944
