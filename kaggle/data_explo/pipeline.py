import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(__file__))
from knn_imputation import knn_impute_numeric  
from cleaning import clean_categorical

df = pd.read_csv('../train_dataset_M1_with_id.csv')

#Clean the data
df = clean_categorical(df)

#Verify categorical was well cleaned
for col in ["Time_of_Day", "Payment_Method", "Referral_Source"]:
    print(f"Unique values for {col}: {df[col].dropna().unique()}")

#Impute numerical
df_imputed = knn_impute_numeric(df,n_neighbors=5)

#Make sure numerical was well split

print("Purchase value counts:")
print(df_imputed['Purchase'].value_counts())
print("\nPurchase proportions:")
print(df_imputed['Purchase'].value_counts(normalize=True))

num_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()

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

#NEED to Normalize the numerical data (Use log ?) 

#Save the csv
df_imputed.to_csv("df_imputed.csv")
