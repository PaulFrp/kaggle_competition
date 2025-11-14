
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 

df = pd.read_csv('../train_dataset_M1_with_id.csv')


corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

