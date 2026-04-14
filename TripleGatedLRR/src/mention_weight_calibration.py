# src/mention_weight_calibration.py
# Phase 0.5: Empirical calibration of mention edge weight
# Must run BEFORE reputation computation (Phase 1)
#
# Methodology:
#   For retweets: compute correlation between retweeter's sentiment
#                 and retweeted user's recent mean sentiment
#   For mentions: compute same correlation for mentioner vs mentioned
#   Ratio = mention_corr / retweet_corr → empirical mention weight
#
# This provides a statistical defense for the mention weight parameter.

import numpy as np
import pandas as pd
import os


def log(msg, indent=0):
    print('  ' * indent + msg)


def calibrate_mention_weight(tw_df, results_path, lookback_days=7):
    """
    Empirically calibrate mention weight via sentiment correlation analysis.
    
    For each retweet edge (A retweets B):
        corr_rt = correlation between A's tweet sentiment and B's 
                  mean sentiment over prior `lookback_days` days
    
    For each mention edge (A mentions B):
        corr_mn = correlation between A's tweet sentiment and B's
                  mean sentiment over prior `lookback_days` days
    
    Mention weight = corr_mn / corr_rt (clipped to [0.1, 1.0])
    
    Returns:
        float: calibrated mention weight
    """
    log('>>> Phase 0.5: Mention Weight Calibration')
    
    tw = tw_df.copy()
    tw['time'] = pd.to_datetime(tw['time'], errors='coerce').dt.date
    
    # Pre-compute per-user daily mean sentiment
    user_daily_sen = (
        tw.groupby(['source_user', 'time'])['sen']
        .mean()
        .reset_index()
        .rename(columns={'sen': 'user_daily_sen'})
    )
    
    # Build user->date->sentiment lookup for fast access
    user_sen_dict = {}
    for _, row in user_daily_sen.iterrows():
        u = str(row['source_user'])
        d = row['time']
        if u not in user_sen_dict:
            user_sen_dict[u] = {}
        user_sen_dict[u][d] = row['user_daily_sen']
    
    def _get_user_recent_mean(user, date, lookback=lookback_days):
        """Get user's mean sentiment over prior lookback_days."""
        if user not in user_sen_dict:
            return np.nan
        user_dates = user_sen_dict[user]
        recent_vals = []
        for d_offset in range(1, lookback + 1):
            d = date - pd.Timedelta(days=d_offset)
            if hasattr(d, 'date'):
                d = d.date()
            # Handle both datetime.date and Timestamp
            try:
                import datetime
                if isinstance(date, datetime.date):
                    d = date - datetime.timedelta(days=d_offset)
                else:
                    d = (pd.Timestamp(date) - pd.Timedelta(days=d_offset)).date()
            except Exception:
                continue
            if d in user_dates:
                recent_vals.append(user_dates[d])
        return np.mean(recent_vals) if recent_vals else np.nan
    
    # Sample for efficiency (full dataset = 350K tweets, sample 50K)
    n_sample = min(50000, len(tw))
    tw_sample = tw.sample(n=n_sample, random_state=42)
    
    # --- Retweet correlations ---
    rt_mask = tw_sample['rt_target'].notna()
    rt_data = tw_sample[rt_mask].copy()
    
    rt_pairs = []
    for _, row in rt_data.iterrows():
        src_sen = row['sen']
        tgt = str(row['rt_target'])
        date = row['time']
        tgt_recent = _get_user_recent_mean(tgt, date)
        if not np.isnan(tgt_recent):
            rt_pairs.append((src_sen, tgt_recent))
    
    if len(rt_pairs) > 30:
        rt_src, rt_tgt = zip(*rt_pairs)
        rt_corr = np.corrcoef(rt_src, rt_tgt)[0, 1]
    else:
        rt_corr = np.nan
    
    # --- Mention correlations ---
    mn_pairs = []
    for _, row in tw_sample.iterrows():
        mentions = row.get('mentions', [])
        if not isinstance(mentions, list) or len(mentions) == 0:
            continue
        src_sen = row['sen']
        date = row['time']
        for m in mentions:
            m_str = str(m)
            tgt_recent = _get_user_recent_mean(m_str, date)
            if not np.isnan(tgt_recent):
                mn_pairs.append((src_sen, tgt_recent))
    
    if len(mn_pairs) > 30:
        mn_src, mn_tgt = zip(*mn_pairs)
        mn_corr = np.corrcoef(mn_src, mn_tgt)[0, 1]
    else:
        mn_corr = np.nan
    
    # --- Compute calibrated weight ---
    if np.isnan(rt_corr) or np.isnan(mn_corr) or abs(rt_corr) < 1e-6:
        calibrated_weight = 0.5  # fallback to default
        log(f'    Insufficient data for calibration — using default w_m=0.5')
    else:
        raw_ratio = mn_corr / rt_corr
        calibrated_weight = float(np.clip(raw_ratio, 0.1, 1.0))
    
    log(f'    Retweet sentiment correlation: r={rt_corr:.4f} (n={len(rt_pairs)})')
    log(f'    Mention sentiment correlation: r={mn_corr:.4f} (n={len(mn_pairs)})')
    log(f'    Raw ratio (mention/retweet):   {mn_corr/rt_corr:.4f}' if not np.isnan(rt_corr) and abs(rt_corr) > 1e-6 else '    Raw ratio: N/A')
    log(f'    Calibrated mention weight:     w_m={calibrated_weight:.3f}')
    
    # Save report
    report_path = os.path.join(results_path, 'mention_weight_calibration.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('=== Mention Weight Empirical Calibration ===\n\n')
        f.write('Methodology:\n')
        f.write(f'  For each retweet (A->B), compute corr(A_sentiment, B_recent_{lookback_days}d_mean)\n')
        f.write(f'  For each mention (A->B), compute corr(A_sentiment, B_recent_{lookback_days}d_mean)\n')
        f.write(f'  Mention weight = mention_corr / retweet_corr, clipped to [0.1, 1.0]\n\n')
        f.write(f'Results:\n')
        f.write(f'  Retweet pairs analysed: {len(rt_pairs)}\n')
        f.write(f'  Retweet sentiment correlation: r = {rt_corr:.4f}\n')
        f.write(f'  Mention pairs analysed: {len(mn_pairs)}\n')
        f.write(f'  Mention sentiment correlation: r = {mn_corr:.4f}\n')
        if not np.isnan(rt_corr) and abs(rt_corr) > 1e-6:
            f.write(f'  Raw ratio: {mn_corr/rt_corr:.4f}\n')
        f.write(f'  Calibrated mention weight: w_m = {calibrated_weight:.3f}\n\n')
        f.write(f'Interpretation:\n')
        if calibrated_weight < 0.4:
            f.write(f'  Mentions carry substantially less sentiment alignment than retweets.\n')
            f.write(f'  This supports differential weighting in the LRR propagation.\n')
        elif calibrated_weight > 0.7:
            f.write(f'  Mentions carry nearly as much sentiment alignment as retweets.\n')
            f.write(f'  The two edge types are relatively interchangeable.\n')
        else:
            f.write(f'  Mentions carry moderate sentiment alignment relative to retweets.\n')
            f.write(f'  The 0.5 default is a reasonable approximation.\n')
    
    log(f'    mention_weight_calibration.txt saved')
    
    return calibrated_weight
