import pandas as pd
from sklearn.preprocessing import StandardScaler

def remove_duplicates(df):
    return df.drop_duplicates()

# def encode_categorical(df):
#     return pd.get_dummies(df, drop_first=True, dtype=int)

def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X) 

