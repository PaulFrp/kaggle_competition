# pipelines/cleaning_pipeline.py

import pandas as pd
import numpy as np

from .knn_imputation import knn_impute_numeric  
from .cat_cleaning import clean_categorical
from .one_hot import cat_encoding
from .minmax import minMax
from .seasonality_features import add_seasonality_features


def full_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:

    df = clean_categorical(df)

    df = add_seasonality_features(df)

    df = cat_encoding(df)

    df = knn_impute_numeric(df, n_neighbors=5)

    df = minMax(df)

    return df
 