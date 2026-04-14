# src/regime_engine.py
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def detect_market_regimes(asset_df, n_states=2):
    """
    Hidden Markov Model (HMM) market regime classifier.

    Features: daily returns + 7-day rolling volatility.
    States:   0 = Calm (low volatility), 1 = Crisis (high volatility).

    The state labels are normalised so that State 1 always corresponds
    to the higher-volatility regime, regardless of HMM initialisation order.

    Note: if the HMM fails to converge (common on short sub-samples),
    the function returns regime=2 for all rows — a deliberate design choice
    that converts model uncertainty into a volatility signal.

    IMPORTANT: always returns a DataFrame with BOTH 'time' and 'regime'
    columns. asset_df must have a 'time' column (not index) — which is
    guaranteed by the loader in this pipeline.

    Returns:
        DataFrame with columns ['time', 'regime'].
    """
    data = asset_df.copy()

    # Ensure 'time' is a column, not just the index
    if 'time' not in data.columns:
        data = data.reset_index()
        if 'index' in data.columns:
            data = data.rename(columns={'index': 'time'})

    data['time']       = pd.to_datetime(data['time'], errors='coerce').dt.date
    data['returns']    = data['close'].pct_change()
    data['volatility'] = data['returns'].rolling(window=7).std()
    data_clean         = data.dropna(subset=['returns', 'volatility']).copy()

    if len(data_clean) < 30:
        # Not enough data — assign Calm (0) to everything
        fallback = data[['time']].copy()
        fallback['regime'] = 0
        return fallback.reset_index(drop=True)

    X = data_clean[['returns', 'volatility']].values

    try:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type='full',
            n_iter=1000,
            random_state=42
        )
        model.fit(X)
        data_clean['regime'] = model.predict(X)

        # Normalise: State 1 = high volatility
        vol_by_regime = data_clean.groupby('regime')['volatility'].mean()
        if vol_by_regime.get(1, 0) < vol_by_regime.get(0, 0):
            data_clean['regime'] = data_clean['regime'].map({0: 1, 1: 0})

        return data_clean[['time', 'regime']].reset_index(drop=True)

    except Exception as e:
        print(f"   ! HMM non-convergence: {e} — tagging as Regime 2 (volatility signal)")
        fallback = data_clean[['time']].copy()
        fallback['regime'] = 2
        return fallback.reset_index(drop=True)