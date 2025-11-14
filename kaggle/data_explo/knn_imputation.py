import pandas as pd 
import numpy as np
from sklearn.impute import KNNImputer

def knn_impute_numeric(df, n_neighbors=5):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    non_num_cols=df.select_dtypes(exclude=[np.number]).columns
    df_num = df[num_cols]

    imputer = KNNImputer(n_neighbors = n_neighbors, weights="distance")
    df_num_imputed = pd.DataFrame(imputer.fit_transform(df_num), columns=num_cols, index=df.index)
    df_imputed = pd.concat([df_num_imputed, df[non_num_cols]], axis=1)

    return df_imputed

