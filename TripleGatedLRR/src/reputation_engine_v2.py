# src/reputation_engine_v2.py
# Vectorised LRR propagation with sparse matrices + sensitivity support
# v3.0 — MAJOR REVISION:
#   1. Con REMOVED from propagation (moved to aggregation-only)
#   2. Degree-normalised flow (quality over volume)
#   3. Adaptive floor (1/N instead of fixed 0.001
#
# Design rationale for Con removal from propagation:
#   Con = sqrt(pos * |neg|) is a TWEET-level property measuring whether
#   a single post contains mixed sentiment (informationally rich).
#   In propagation, Con=0 for 68.7% of tweets, zeroing out those edges
#   and destroying graph structure. Con belongs in AGGREGATION where it
#   filters which tweets contribute to the daily signal.

import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from src.psych_engine import calculate_omega
from src.config import PHI, ITERATIONS, CONVERGENCE_TOL


def compute_lrr_vectorised(tw_df, unique_users, use_omega=True, use_con=True,
                            anchor_vector=None, phi=PHI, n_iterations=ITERATIONS,
                            omega_floor=0.10, con_in_propagation=False,
                            mention_weight=None):
    """
    Vectorised Liquid Reputation Propagation (LRR).
    
    v3.0 changes:
        - con_in_propagation now defaults to False (Con gates aggregation only)
        - Degree-normalised incoming flow (sqrt(in_degree))
        - Adaptive floor = 1/N (scales with population size)
    
    Propagation equation (per iteration):
        flow_ij = R_j(t-1) * omega_j * penalty_j * edge_weight
        raw_i   = Σ_j flow_ij
        norm_i  = raw_i / sqrt(in_degree_i + 1)     [degree normalisation]
        R_i(t)  = (1-phi) * max_normalise(log_i) + phi * anchor_i
    
    Parameters:
        use_omega          : gate flows by individual rationality score
        use_con            : LEGACY flag, only affects con_in_propagation 
                             if con_in_propagation=True (not recommended)
        anchor_vector      : V-Anchor economic alignment vector
        phi                : V-Anchor blending weight (default 0.15)
        n_iterations       : propagation iterations (default 5)
        omega_floor        : minimum omega score (default 0.10)
        con_in_propagation : whether Con enters the flow equation
                             Default FALSE — Con should gate aggregation only
        mention_weight     : weight for mention edges (default from config)
    
    Returns:
        dict {user_id: reputation_score} (scores in (floor, 1])
    """
    from src.config import MENTION_WEIGHT as DEFAULT_MW
    if mention_weight is None:
        mention_weight = DEFAULT_MW
    
    # Map users to integer indices
    user_list = list(unique_users)
    user_to_idx = {str(u): i for i, u in enumerate(user_list)}
    n_users = len(user_list)
    
    # Adaptive floor: 1/N instead of fixed 0.001
    adaptive_floor = 1.0 / max(n_users, 1)
    
    # Compute per-user statistics
    activity_count = tw_df['source_user'].value_counts().to_dict()
    
    # Pre-compute per-tweet values
    sources = tw_df['source_user'].astype(str).values
    omegas = tw_df['omega'].values.astype(float) if use_omega and 'omega' in tw_df.columns else np.ones(len(tw_df))
    
    # Con in propagation: only if explicitly requested (legacy/ablation use)
    if use_con and con_in_propagation and 'con' in tw_df.columns:
        cons = tw_df['con'].values.astype(float)
    else:
        cons = np.ones(len(tw_df))
    
    # Apply omega floor
    omegas = np.clip(omegas, omega_floor, 1.0) if use_omega else omegas
    
    # Pre-compute penalties per user
    user_penalties = {}
    if use_omega:
        user_omega_means = tw_df.groupby('source_user')['omega'].mean().to_dict() if 'omega' in tw_df.columns else {}
        for u in unique_users:
            u_str = str(u)
            mean_w = user_omega_means.get(u, 0.5)
            if mean_w < 0.5:
                user_penalties[u_str] = 1.0 / (1.0 + np.log1p(activity_count.get(u_str, 0)))
            else:
                user_penalties[u_str] = 1.0
    
    # Build edge lists: (source_idx, target_idx, tweet_idx)
    rt_src, rt_tgt, rt_tidx = [], [], []
    mn_src, mn_tgt, mn_tidx = [], [], []
    
    rt_targets = tw_df['rt_target'].values
    mentions_list = tw_df['mentions'].values
    
    for t_idx in range(len(tw_df)):
        src_str = str(sources[t_idx])
        if src_str not in user_to_idx:
            continue
        src_idx = user_to_idx[src_str]
        
        # Retweet edges
        rt = rt_targets[t_idx]
        if pd.notnull(rt):
            tgt_str = str(rt)
            if tgt_str in user_to_idx:
                rt_src.append(src_idx)
                rt_tgt.append(user_to_idx[tgt_str])
                rt_tidx.append(t_idx)
        
        # Mention edges
        m_list = mentions_list[t_idx]
        if isinstance(m_list, list):
            for m in m_list:
                tgt_str = str(m)
                if tgt_str in user_to_idx:
                    mn_src.append(src_idx)
                    mn_tgt.append(user_to_idx[tgt_str])
                    mn_tidx.append(t_idx)
    
    rt_src = np.array(rt_src)
    rt_tgt = np.array(rt_tgt)
    rt_tidx = np.array(rt_tidx)
    mn_src = np.array(mn_src)
    mn_tgt = np.array(mn_tgt)
    mn_tidx = np.array(mn_tidx)
    
    # Pre-compute in-degree for degree-normalised flow
    in_degree = np.zeros(n_users, dtype=float)
    if len(rt_tgt) > 0:
        np.add.at(in_degree, rt_tgt, 1.0)
    if len(mn_tgt) > 0:
        np.add.at(in_degree, mn_tgt, 1.0)
    # sqrt(in_degree + 1) — the +1 prevents division by zero for isolates
    degree_norm = np.sqrt(in_degree + 1.0)
    
    # Initialise reputation
    reputation = np.ones(n_users, dtype=float)
    
    for iteration in range(n_iterations):
        prev_reputation = reputation.copy()
        
        # Compute per-tweet flows
        # For retweet edges
        if len(rt_src) > 0:
            rt_rep = reputation[rt_src]
            rt_omega = omegas[rt_tidx]
            rt_con = cons[rt_tidx]
            rt_penalty = np.array([user_penalties.get(str(user_list[s]), 1.0) for s in rt_src])
            rt_flows = rt_rep * rt_omega * rt_con * rt_penalty * 1.0
        
        # For mention edges
        if len(mn_src) > 0:
            mn_rep = reputation[mn_src]
            mn_omega = omegas[mn_tidx]
            mn_con = cons[mn_tidx]
            mn_penalty = np.array([user_penalties.get(str(user_list[s]), 1.0) for s in mn_src])
            mn_flows = mn_rep * mn_omega * mn_con * mn_penalty * mention_weight
        
        # Accumulate flows at target users
        temp_rep = np.zeros(n_users, dtype=float)
        if len(rt_src) > 0:
            np.add.at(temp_rep, rt_tgt, rt_flows)
        if len(mn_src) > 0:
            np.add.at(temp_rep, mn_tgt, mn_flows)
        
        # --- Degree-normalised flow ---
        # Divide by sqrt(in_degree + 1) to reward quality over volume
        temp_rep = temp_rep / degree_norm
        
        # --- Kolonin logarithmic compression ---
        # lP_i = sign(dP_i) * log10(1 + |dP_i|)
        # Spreads mid-range, prevents hub monopolisation
        temp_rep = np.sign(temp_rep) * np.log10(1.0 + np.abs(temp_rep))
        
        # Max-normalise (after log compression)
        max_val = temp_rep.max()
        if max_val > 0:
            temp_rep /= max_val
        
        # Blend with V-Anchor
        if anchor_vector is not None:
            anchor_vals = np.array([anchor_vector.get(str(u), adaptive_floor) for u in user_list])
            reputation = (1.0 - phi) * temp_rep + phi * anchor_vals
        else:
            reputation = temp_rep.copy()
        
        # Adaptive floor: 1/N
        reputation = np.maximum(reputation, adaptive_floor)
        
        # Convergence check
        max_change = np.max(np.abs(reputation - prev_reputation))
        if max_change < CONVERGENCE_TOL:
            break
    
    # Convert back to dict
    return {str(user_list[i]): float(reputation[i]) for i in range(n_users)}


def compute_anchor_vector(tw_df, asset_df, unique_users, train_end_date=None):
    """
    Compute V-Anchor vector from training data.
    
    Extracted from run_benchmarked_reputation so the sensitivity suite
    can access it independently.
    
    Returns:
        anchor_vector : dict {user_id: anchor_weight}
    """
    n_users = len(unique_users)
    adaptive_floor = 1.0 / max(n_users, 1)
    
    def _get_time_col(df):
        if 'time' in df.columns:
            return pd.to_datetime(df['time'], errors='coerce').dt.date
        return pd.to_datetime(df.index, errors='coerce').date

    if train_end_date is not None:
        train_end = pd.to_datetime(train_end_date).date()
        asset_time = _get_time_col(asset_df)
        mask = asset_time <= train_end
        anchor_asset = asset_df[mask.values].copy()

        tw_time = pd.to_datetime(tw_df['time'], errors='coerce').dt.date
        tw_mask = tw_time <= train_end
        anchor_tw = tw_df[tw_mask.values].copy()
    else:
        anchor_asset = asset_df.copy()
        anchor_tw = tw_df.copy()

    anchor_asset = anchor_asset.copy()
    if 'close' not in anchor_asset.columns:
        anchor_asset = anchor_asset.reset_index()

    if 'close' not in anchor_asset.columns:
        return {str(u): adaptive_floor for u in unique_users}

    anchor_asset['target_anchor'] = anchor_asset['close'].pct_change(7).shift(-7)

    if 'time' in anchor_tw.columns:
        anchor_tw = anchor_tw.copy()
        anchor_tw['time'] = pd.to_datetime(anchor_tw['time'], errors='coerce').dt.date

    anchor_asset['time'] = pd.to_datetime(
        anchor_asset['time'] if 'time' in anchor_asset.columns
        else anchor_asset.index, errors='coerce'
    )
    if hasattr(anchor_asset['time'], 'dt'):
        anchor_asset['time'] = anchor_asset['time'].dt.date

    # Pivot: rows = dates, cols = users, values = mean daily sentiment
    daily_sen = (
        anchor_tw.groupby(['source_user', 'time'])['sen']
        .mean()
        .unstack(level=0)
    )

    if 'time' in anchor_asset.columns:
        anchor_asset_idx = anchor_asset.set_index('time')
    else:
        anchor_asset_idx = anchor_asset

    anchor_vector = {str(u): adaptive_floor for u in unique_users}
    for u in unique_users:
        u_str = str(u)
        if u_str in daily_sen.columns:
            corr = daily_sen[u_str].corr(anchor_asset_idx['target_anchor'])
            if pd.notnull(corr) and corr > 0:
                anchor_vector[u_str] = corr

    return anchor_vector


def compute_lrr_expanding_window(tw_df, unique_users, asset_df,
                                  window_days=30, train_end_date=None,
                                  **kwargs):
    """
    Expanding-window reputation computation for causal validation.
    
    Recomputes reputation every `window_days` using only past data.
    Returns: dict {date: reputation_dict}
    """
    from datetime import timedelta
    
    tw_df = tw_df.copy()
    tw_df['date'] = pd.to_datetime(tw_df['time'], errors='coerce').dt.date
    
    all_dates = sorted(tw_df['date'].dropna().unique())
    if not all_dates:
        return {}
    
    recomp_dates = []
    current = all_dates[0]
    while current <= all_dates[-1]:
        recomp_dates.append(current)
        current = current + timedelta(days=window_days)
    if recomp_dates[-1] != all_dates[-1]:
        recomp_dates.append(all_dates[-1])
    
    anchor_vector = kwargs.get('anchor_vector', None)
    
    lrr_kwargs = {k: v for k, v in kwargs.items()
                  if k not in ('anchor_vector', 'train_end_date')}
    
    date_to_reputation = {}
    
    for i, recomp_date in enumerate(recomp_dates):
        mask = tw_df['date'] <= recomp_date
        tw_subset = tw_df[mask]
        
        if len(tw_subset) < 10:
            continue
        
        rep = compute_lrr_vectorised(
            tw_subset, unique_users,
            anchor_vector=anchor_vector,
            **lrr_kwargs
        )
        
        if i + 1 < len(recomp_dates):
            next_date = recomp_dates[i + 1]
        else:
            next_date = all_dates[-1] + timedelta(days=1)
        
        for d in all_dates:
            if recomp_date <= d < next_date:
                date_to_reputation[d] = rep
    
    return date_to_reputation


def compute_daily_signal_with_config(tw_df, reputation_dict, 
                                      con_in_aggregation=True,
                                      con_in_denominator=False):
    """
    Compute daily LRR signal with configurable Con placement.
    
    con_in_aggregation: if True, multiply numerator by Con
    con_in_denominator: if True, also include Con in denominator weights
    """
    tw_df = tw_df.copy()
    n_users = max(len(reputation_dict), 1)
    adaptive_floor = 1.0 / n_users
    tw_df['rep_weight'] = tw_df['source_user'].astype(str).map(reputation_dict).fillna(adaptive_floor)
    
    def _daily_agg(x):
        s = x['sen'].values
        w = x['rep_weight'].values
        con = x['con'].values if 'con' in x.columns else np.ones(len(x))
        
        if con_in_aggregation:
            numerator = np.sum(s * con * w)
        else:
            numerator = np.sum(s * w)
        
        if con_in_denominator:
            denominator = np.sum(con * w) + 1e-9
        else:
            denominator = np.sum(w) + 1e-9
        
        return numerator / denominator
    
    daily = tw_df.groupby('time').apply(_daily_agg)
    return daily


def compute_independence_daily_deviation(tw_df):
    """
    Alternative A: Deviation from daily consensus.
    IndA(t) = |s_t - mean(s_d)| / (std(s_d) + eps)
    """
    tw_df = tw_df.copy()
    daily_stats = tw_df.groupby('time')['sen'].agg(['mean', 'std']).reset_index()
    daily_stats.columns = ['time', 'daily_mean', 'daily_std']
    tw_df = tw_df.merge(daily_stats, on='time', how='left')
    tw_df['ind_daily_dev'] = np.abs(tw_df['sen'] - tw_df['daily_mean']) / (tw_df['daily_std'] + 1e-9)
    return tw_df


def compute_independence_temporal_novelty(tw_df, window=7):
    """
    Alternative B: Temporal novelty.
    IndB(t) = |s_t - mean(user's last N days)| / (std(user's last N days) + eps)
    """
    tw_df = tw_df.copy()
    tw_df = tw_df.sort_values(['source_user', 'time'])
    
    def user_novelty(group):
        group = group.sort_values('time')
        rolling_mean = group['sen'].rolling(window=window, min_periods=1).mean().shift(1)
        rolling_std = group['sen'].rolling(window=window, min_periods=1).std().shift(1).fillna(1.0)
        group['ind_temporal_nov'] = np.abs(group['sen'] - rolling_mean) / (rolling_std + 1e-9)
        return group
    
    tw_df = tw_df.groupby('source_user', group_keys=False).apply(user_novelty)
    tw_df['ind_temporal_nov'] = tw_df['ind_temporal_nov'].fillna(0.5)
    return tw_df


def detect_suspected_bots(tw_df, freq_percentile=0.95, omega_threshold=0.5,
                           min_tweets=5):
    """
    Heuristic bot detection using available features.
    
    Flags users as suspected bots if:
    1. Tweet frequency > freq_percentile
    2. Mean omega < omega_threshold
    3. At least min_tweets posted
    
    Returns: set of suspected bot user IDs, feature DataFrame for analysis
    """
    user_stats = tw_df.groupby('source_user').agg(
        tweet_count=('sen', 'count'),
        mean_sen=('sen', 'mean'),
        std_sen=('sen', 'std'),
        unique_dates=('time', 'nunique'),
    ).reset_index()
    
    if 'omega' in tw_df.columns:
        user_omega = tw_df.groupby('source_user')['omega'].mean().reset_index()
        user_omega.columns = ['source_user', 'mean_omega']
        user_stats = user_stats.merge(user_omega, on='source_user', how='left')
        user_stats['mean_omega'] = user_stats['mean_omega'].fillna(0.5)
    else:
        user_stats['mean_omega'] = 0.5
    
    user_stats['sen_diversity'] = user_stats['std_sen'].fillna(0)
    user_stats['tweets_per_day'] = user_stats['tweet_count'] / (user_stats['unique_dates'] + 1)
    
    freq_threshold = user_stats['tweet_count'].quantile(freq_percentile)
    
    bot_mask = (
        (user_stats['tweet_count'] >= freq_threshold) &
        (user_stats['mean_omega'] < omega_threshold) &
        (user_stats['tweet_count'] >= min_tweets)
    )
    
    suspected_bots = set(user_stats.loc[bot_mask, 'source_user'].values)
    
    return suspected_bots, user_stats


def run_hmm_robustness(asset_df, winsorize_pcts=(1, 99)):
    """
    HMM regime robustness: compare original Gaussian HMM labels
    with winsorised returns.
    
    Returns dict with concordance rates and alternative regime labels.
    """
    from src.regime_engine import detect_market_regimes
    
    results = {}
    
    try:
        orig_regimes = detect_market_regimes(asset_df.copy())
        orig_labels = orig_regimes.set_index('time')['regime']
        results['original'] = orig_labels
    except Exception:
        return results
    
    for lo_pct, hi_pct in [(1, 99), (5, 95), (10, 90)]:
        label = f'winsorised_{lo_pct}_{hi_pct}'
        try:
            asset_win = asset_df.copy()
            if 'price_change' not in asset_win.columns:
                if 'close' in asset_win.columns:
                    asset_win['price_change'] = asset_win['close'].pct_change()
                else:
                    continue
            
            pc = asset_win['price_change'].dropna()
            if len(pc) < 50:
                continue
            lo = np.percentile(pc, lo_pct)
            hi = np.percentile(pc, hi_pct)
            asset_win['price_change'] = asset_win['price_change'].clip(lo, hi)
            
            win_regimes = detect_market_regimes(asset_win)
            win_labels = win_regimes.set_index('time')['regime']
            
            common_dates = orig_labels.index.intersection(win_labels.index)
            if len(common_dates) > 0:
                concordance = (orig_labels[common_dates] == win_labels[common_dates]).mean()
                results[f'{label}_concordance'] = float(concordance)
                results[label] = win_labels
        except Exception as e:
            results[f'{label}_error'] = str(e)
    
    return results
