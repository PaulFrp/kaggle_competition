
from sklearn.linear_model import LinearRegression
import pandas as pd
df = pd.read_csv('../train_dataset_M1_with_id.csv')
X = df[['Price','Discount','Items_In_Cart']].fillna(0)
y = df['Purchase']

model = LinearRegression()
model.fit(X, y)
print(model.score(X,y))
print(dict(zip(X.columns, model.coef_)))


from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from tqdm import tqdm

X = pd.get_dummies(df.drop(columns=['Purchase'])).fillna(0)
y = df['Purchase']

# Create RF with warm_start
rf = RandomForestClassifier(n_estimators=1, warm_start=True, random_state=42)

n_trees = 100
importances_list = []

with tqdm(total=n_trees, desc="Training Random Forest") as pbar:
    for i in range(1, n_trees + 1):
        rf.n_estimators = i
        rf.fit(X, y)
        pbar.update(1)

# Feature importances after full fit
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances.head(10))
