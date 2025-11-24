
# seasonality_features.py

import numpy as np
import pandas as pd

def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds seasonality and temporal features to the dataframe.
    
    Features added:
    - day_sin, day_cos: cyclic encoding of the 'Day' column (weekly cycle)
    - time_sin, time_cos: cyclic encoding of 'Time_of_Day' (morning/afternoon/evening)
    - campaign_effect: numeric encoding of 'Campaign_Period' (0/1)
    - seasonality_score: combination of day, time, and campaign effects

    Parameters:
    ----------
    df : pd.DataFrame
        Input dataframe with columns: 'Day', 'Time_of_Day', 'Campaign_Period'

    Returns:
    -------
    pd.DataFrame
        DataFrame with new seasonality features added
    """
    
    # --- 1. Encode weekly cyclicality for 'Day' ---
    if 'Day' in df.columns:
        df['day_sin'] = np.sin(2 * np.pi * df['Day'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['Day'] / 7)
    
    # --- 2. Encode 'Time_of_Day' cyclically ---
    if 'Time_of_Day' in df.columns:
        time_mapping = {'morning': 0, 'afternoon': 1, 'evening': 2}
        df['time_numeric'] = df['Time_of_Day'].map(time_mapping)
        df['time_sin'] = np.sin(2 * np.pi * df['time_numeric'] / 3)
        df['time_cos'] = np.cos(2 * np.pi * df['time_numeric'] / 3)
        df.drop(columns=['time_numeric'], inplace=True)
    
    # --- 3. Encode campaign period ---
    if 'Campaign_Period' in df.columns:
        # Convert strings "true"/"false" to 1/0
        df['campaign_effect'] = df['Campaign_Period'].map({'true': 1, 'false': 0})
    # --- 4. Combine into a single 'seasonality_score' ---
    df['seasonality_score'] = (
        df.get('day_sin', 0) * 0.4 +
        df.get('time_sin', 0) * 0.3 +
        df.get('campaign_effect', 0) * 0.3
    )
    
    return df
