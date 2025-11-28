from sklearn.preprocessing import MinMaxScaler
import numpy as np

def minMax(df): 

    num_cols=df.select_dtypes(include=[np.number]).columns.drop("id")
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df
