from pathlib import Path
import pandas as pd


def load_train_data(path: str | Path = "data/raw/train_dataset_M1_with_id.csv") -> pd.DataFrame:
    """
    Load the training dataset for the Kaggle challenge.

    Parameters
    ----------
    path : str or Path
        Path to the train CSV (local only, not in git).

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    return pd.read_csv(path)


def load_test_data(path: str | Path = "data/raw/test_dataset_M1_with_id.csv") -> pd.DataFrame:
    """Load the test dataset (same idea as train)."""
    path = Path(path)
    return pd.read_csv(path)
    