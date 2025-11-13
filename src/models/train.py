import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def train_baseline_model(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Train a simple baseline model. This is just a placeholder so that
    the repo has a working end-to-end example.
    """
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    preds = clf.predict(X)
    f1 = f1_score(y, preds)

    return {
        "model": clf,
        "f1_train": f1,
    }