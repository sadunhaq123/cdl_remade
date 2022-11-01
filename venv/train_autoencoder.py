import keras
from keras import layers
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
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
print(dataset_test.shape)


encoder_input = keras.Input(shape=(158440, 555), name ='enc')
#encoder_input = keras.Input(shape=(1500, 555), name ='enc')
encoder_flatten = keras.layers.Flatten()(encoder_input)
encoder_layer1 = keras.layers.Dense(36630, activation='sigmoid')(encoder_flatten)
encoder_layer2 = keras.layers.Dense(2400, activation='sigmoid')(encoder_layer1)
encoder_output = keras.layers.Dense(64, activation='sigmoid')(encoder_layer2)

print("HAPPY")
encoder = keras.Model(encoder_input, encoder_output, name='encoder')

decoder_input = keras.layers.Dense(64, activation="sigmoid")(encoder_output)
decoder_layer1 = keras.layers.Dense(2400, activation='sigmoid')(decoder_input)
decoder_layer2 = keras.layers.Dense(36630, activation='sigmoid')(decoder_layer1)
decoder_layer3 = keras.layers.Dense(8791200, activation='sigmoid')(decoder_layer2)
decoder_output = keras.Layers.Reshape((158400, 555))(decoder_layer3)



opt = tf.keras.RMSprop(0.001, decay=1e-6)
autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
autoencoder.summary()
