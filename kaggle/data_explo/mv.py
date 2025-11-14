import pandas as pd 

df = pd.read_csv('../train_dataset_M1_with_id.csv')
print(df.head())
print(df.info())
print(df.describe())
print("test")
print("Missing values per column:")
print(df.isnull().sum())

missing_percent = df.isnull().sum() / len(df) * 100
print("Percentage of missing values per column:")   
print(missing_percent)
