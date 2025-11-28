import numpy as np
import pandas as pd

from cleaning.knn_imputation import knn_impute_numeric  
from cleaning.cleaning import clean_categorical
from cleaning.one_hot import cat_encoding
from cleaning.minmax import minMax
from cleaning.seasonality_features import add_seasonality_features

#----------------------NEED TO DO IT ALSO FOR TEST ------------------------
df = pd.read_csv('../train_dataset_M1_with_id.csv')

#Clean the data (I only clean the 3 columns maybe need to expand to the rest)
df = clean_categorical(df)

#Verify categorical was well cleaned
for col in ["Time_of_Day", "Payment_Method", "Referral_Source"]:
    print(f"Unique values for {col}: {df[col].dropna().unique()}")

df = add_seasonality_features(df)

#Encode categorical variable using one hot encoding 
df = cat_encoding(df)

#Impute numerical, Need to add categorical once Encoded
df_imputed = knn_impute_numeric(df,n_neighbors=5)

num_cols = df.select_dtypes(include=[np.number]).columns.drop("id").tolist()

#Return the df after minmax normalisation
clean_df = minMax(df_imputed)
print(clean_df.head())

df_imputed.to_csv("df_imputed.csv")



