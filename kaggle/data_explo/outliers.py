import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
df = pd.read_csv('../train_dataset_M1_with_id.csv')

#Identify numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
print("Numeric columns:", list(numeric_cols))

#Find outliers
def find_outliers (series , threshold=5):
    median = np.median(series.dropna())
    mad = np.median(np.abs(series.dropna() - median))
    if mad == 0:
        return pd.Series([False] * len(series), index=series.index)
    return np.abs(series-median)>  threshold * mad

outlier_summary = {}

for col in numeric_cols:
    mask = find_outliers(df[col])
    outliers = df[mask]
    outlier_count = mask.sum()
    outlier_summary[col] = outlier_count

    plt.figure(figsize=(5,3))
    plt.boxplot(df[col].dropna(), vert = False)
    plt.title(f"{col} ({outlier_count} outlietrs)")
    plt.xlabel(col)
    plt.show()

print("========Summary=======")
for col, count in outlier_summary.items():
    print(f"{col}: {count} outliers")


plt.plot(df["Price"])
plt.ylabel(df["Time_of_Day"])
plt.show()
