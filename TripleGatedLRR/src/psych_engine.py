# src/psych_engine.py
import pandas as pd
import numpy as np
from src.config import DISTORTIONS


def calculate_omega(df):
    """
    Calculates the Individual Rationality Score (ω) per tweet.

    Formula:
        If 'exclusivereasoning' is present:
            ω = ((1 - mean_distortion) + exclusive_reasoning) / 2
        Else:
            ω = (1 - mean_distortion), clipped to [0.10, 1.00]

    Interpretation:
        ω ≈ 1.0 → high rationality (evidence-based reasoning, low cognitive noise)
        ω ≈ 0.1 → low rationality (dominated by cognitive distortions)

    The 14 distortion columns are binary indicators derived from a fine-tuned
    NLP classifier applied to each tweet's text.
    """
    df = df.copy()
    available = [c for c in DISTORTIONS if c in df.columns]

    if not available:
        # No distortion features present — assign neutral rationality
        df['omega'] = 0.5
        return df

    distortion_mean = df[available].mean(axis=1)

    if 'exclusivereasoning' in df.columns:
        # Balanced composite: penalise distortions, reward reasoned inference
        df['omega'] = ((1.0 - distortion_mean) + df['exclusivereasoning']) / 2.0
    else:
        df['omega'] = 1.0 - distortion_mean

    df['omega'] = df['omega'].clip(lower=0.10, upper=1.00)
    return df


def calculate_rolling_omega(df, user_col='source_user', time_col='time', window=30):
    """
    Computes a temporally-decayed omega per user per day.

    Addresses the static-omega limitation: a user's rationality score
    should evolve over time, not be fixed to a single number across
    the entire dataset.

    Returns a DataFrame with ['source_user', 'time', 'omega_rolling'] columns.
    Used as an optional enhancement when temporal omega evolution is needed.
    """
    df = df.copy().sort_values(time_col)
    df = calculate_omega(df)

    rolling_omega = (
        df.groupby(user_col)['omega']
        .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    )
    df['omega_rolling'] = rolling_omega
    return df
