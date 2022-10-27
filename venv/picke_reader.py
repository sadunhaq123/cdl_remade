import pandas as pd

unpickled_df = pd.read_pickle("C:/Users/12103/PycharmProjects/cdl_remade/venv/train_classifier/CVE-2012-1823.pkl")
#unpickled_df = pd.read_pickle("C:/Users/12103/PycharmProjects/cdl_remade/venv/train_autoencoder/CVE-2012-1823.pkl")
print(unpickled_df)