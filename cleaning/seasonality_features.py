import numpy as np
import pandas as pd

def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    
    if 'Day' in df.columns:
        df['day_sin'] = np.sin(2 * np.pi * df['Day'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['Day'] / 7)
    
    if 'Time_of_Day' in df.columns:
        time_mapping = {'morning': 0, 'afternoon': 1, 'evening': 2}
        df['time_numeric'] = df['Time_of_Day'].map(time_mapping)
        df['time_sin'] = np.sin(2 * np.pi * df['time_numeric'] / 3)
        df['time_cos'] = np.cos(2 * np.pi * df['time_numeric'] / 3)
        df.drop(columns=['time_numeric'], inplace=True)
    
    if 'Campaign_Period' in df.columns:
        df['campaign_effect'] = df['Campaign_Period'].map({'true': 1, 'false': 0})

    # Composite seasonality score
    df['seasonality_score'] = (
        df.get('day_sin', 0) * 0.4 +
        df.get('time_sin', 0) * 0.3 +
        df.get('campaign_effect', 0) * 0.3
    )
    
    return df
