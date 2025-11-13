import pandas as pd


def make_features(df: pd.DataFrame, as_of_day: int | None = None, seed: int = 42) -> pd.DataFrame:
    """
    Shared feature engineering entry point.

    For now this just returns df unchanged. Each teammate will later
    add their own feature blocks here (or call separate functions).
    """
    # TODO: add real feature engineering
    return df.copy()