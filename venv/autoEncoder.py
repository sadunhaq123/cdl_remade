import keras
from keras import layers
import pandas as pd

file1 = open('train_autoencoder_pickle.txt', 'r')

#Lines = file1.readlines()
Lines = ['train_autoencoder/all_combined_train.pkl']

for line in Lines:
    content = line.strip()
    unpickled_df = pd.read_pickle(content)
    print(content, unpickled_df.shape)
    rows, columns = unpickled_df.shape
    print(rows, columns)
    #break