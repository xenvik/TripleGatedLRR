# src/anchor_utils.py
# Standalone V-Anchor computation extracted from reputation_engine.py
# Import this in LOO and rolling reputation scripts

import numpy as np
import pandas as pd
from src.config import PHI


def compute_anchor_vector(tw_df, asset_df, asset_name=None, train_end_date=None):
    """
    Compute per-user V-Anchor: correlation between user's daily mean
    sentiment and 7-day forward returns, on training data only.
    
    Returns: dict {user_id: anchor_value} where anchor_value >= 0.01
    """
    tw = tw_df.copy()
    tw['time'] = pd.to_datetime(tw['time'], errors='coerce').dt.date
    
    asset = asset_df.copy()
    if 'time' in asset.columns:
        asset['time'] = pd.to_datetime(asset['time'], errors='coerce').dt.date
    elif hasattr(asset.index, 'date'):
        asset = asset.reset_index()
        asset['time'] = pd.to_datetime(asset['time'], errors='coerce').dt.date
    
    # Filter to training period if specified
    if train_end_date is not None:
        train_end = pd.to_datetime(train_end_date).date()
        tw = tw[tw['time'] <= train_end]
        asset = asset[asset['time'] <= train_end]
    
    if 'close' not in asset.columns:
        # No price data — return adaptive floor anchors
        users = tw['source_user'].dropna().unique()
        adaptive_floor = 1.0 / max(len(users), 1)
        return {str(u): adaptive_floor for u in users}
    
    # 7-day forward returns
    asset = asset.sort_values('time')
    asset['target_anchor'] = asset['close'].pct_change(7).shift(-7)
    
    # Daily mean sentiment per user
    daily_sen = (
        tw.groupby(['source_user', 'time'])['sen']
        .mean()
        .unstack(level=0)
    )
    
    # Align with asset prices
    asset_idx = asset.set_index('time')
    
    # Compute per-user correlation
    all_users = tw['source_user'].dropna().unique()
    adaptive_floor = 1.0 / max(len(all_users), 1)
    anchor_vector = {str(u): adaptive_floor for u in all_users}
    
    for u in all_users:
        u_str = str(u)
        if u in daily_sen.columns:
            corr = daily_sen[u].corr(asset_idx['target_anchor'])
            if pd.notnull(corr) and corr > 0:
                anchor_vector[u_str] = float(corr)
    
    n_active = sum(1 for v in anchor_vector.values() if v > adaptive_floor)
    
    return anchor_vector
