from typing import Tuple
import pandas as pd


def temporal_train_val_split(
    df: pd.DataFrame,
    day_column: str = "Day",
    val_start: int = 70,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple temporal hold-out split:
    - train: df[day < val_start]
    - val  : df[day >= val_start]

    Adjust val_start later if needed.
    """
    train = df[df[day_column] < val_start].copy()
    val = df[df[day_column] >= val_start].copy()
    return train, val