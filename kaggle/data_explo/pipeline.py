import numpy as np
import pandas as pd

from knn_imputation import knn_impute_numeric  
from cleaning import clean_categorical
from one_hot import cat_encoding
from minmax import minMax
from seasonality_features import add_seasonality_features


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

#Make sure numerical was well split

print("Purchase value counts:")
print(df_imputed['Purchase'].value_counts())
print("\nPurchase proportions:")
print(df_imputed['Purchase'].value_counts(normalize=True))

num_cols = df.select_dtypes(include=[np.number]).columns.drop("id").tolist()

print("\n📊 Summary statistics for numeric columns after KNN imputation:")
print(df_imputed[num_cols].describe().T[['mean', 'std', 'min', 'max']])

missing = df_imputed[num_cols].isna().sum()
print("\n  Missing values remaining per numeric column:")
print(missing[missing > 0] if missing.sum() > 0 else "✅ No missing values left!")

try:
    num_cols_orig = df.select_dtypes(include=[np.number]).columns
    comparison = pd.DataFrame({
        'mean_before': df[num_cols_orig].mean(),
        'mean_after': df_imputed[num_cols_orig].mean(),
        'std_before': df[num_cols_orig].std(),
        'std_after': df_imputed[num_cols_orig].std()
    })
    print("\n📈 Mean and Std before vs after imputation:")
    print(comparison)
except Exception as e:
    print("\n(⚠️ Skipping before/after comparison – original df not available or mismatched columns.)")


#Save the csv
df_imputed.to_csv("df_imputed.csv")

#Return the df after minmax normalisation
clean_df = minMax(df_imputed)
print(clean_df.head())

#Add dimensionality reduction


