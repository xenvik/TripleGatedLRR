# src/sensitivity_suite.py
# Phase 20: Comprehensive Sensitivity & Robustness Suite
# v2.2 — FIXED: anchor passthrough, rolling reputation, channel robustness
#        REPLACED: bot detection → channel concentration + leave-one-out


import os
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from scipy.stats import wilcoxon

from src.reputation_engine_v2 import (
    compute_lrr_vectorised,
    compute_lrr_expanding_window,
    compute_daily_signal_with_config,
    compute_independence_daily_deviation,
    compute_independence_temporal_novelty,
    compute_anchor_vector,
    run_hmm_robustness,
)
from src.sensitivity_config import (
    ROLLING_WINDOW_DAYS, GATE_ABLATION_CONFIGS,
    CON_SENSITIVITY_CONFIGS, PHI_VALUES, ITERATION_VALUES,
    OMEGA_FLOOR_VALUES,
    TEMPORAL_NOVELTY_WINDOW,
)
from src.config import PHI, ITERATIONS, TRAIN_RATIO
from src.risk_metrics import compute_tail_dependence_extended


def log(msg, indent=0):
    print('  ' * indent + msg)


def _run_granger_quick(signal_series, omega_series, price_series, max_lag=7):
    """Quick Granger test: returns F-stat and p-value for signal->omega."""
    data = pd.DataFrame({
        'signal': signal_series,
        'omega': omega_series,
        'price': price_series,
    }).dropna()
    
    if len(data) < 50:
        return np.nan, np.nan
    
    try:
        model = VAR(data).fit(maxlags=max_lag, ic='bic')
        gc = model.test_causality('omega', 'signal', kind='f')
        return float(gc.test_statistic), float(gc.pvalue)
    except Exception:
        return np.nan, np.nan


def _gini(arr):
    """Gini coefficient for a non-negative array."""
    arr = np.array(arr, dtype=float)
    arr = arr[arr >= 0]
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1) / n)


def run_sensitivity_suite(final_data, tw_raw, assets, results_path):
    """
    Master sensitivity suite. Runs all Category B analyses:
    
    20a. Con=0 fraction per asset per regime
    20b. Gate-by-gate ablations (Granger F-stats + crash counts)
    20c. Con sensitivity (propagation/aggregation/denominator)
    20d. Hyperparameter sweep (phi, iterations, omega floor) — BTC only
    20e. Rolling/expanding reputation + re-test Granger
    20f. HMM robustness (multi-level winsorisation)
    20g. Channel concentration + leave-one-out robustness
    20h. Alternative independence measures
    20i. CDS validation sample extraction
    """
    log('\n>>> Phase 20: Sensitivity & Robustness Suite')
    sens_path = os.path.join(results_path, 'sensitivity')
    os.makedirs(sens_path, exist_ok=True)
    
    # We need tw_df with omega computed
    from src.psych_engine import calculate_omega
    tw = calculate_omega(tw_raw.copy())
    
    sources = tw['source_user'].dropna().unique().tolist()
    rt_targets = tw['rt_target'].dropna().unique().tolist()
    all_mentions = [m for m_list in tw['mentions']
                    for m in m_list if isinstance(m_list, list)]
    unique_users = list(set(sources + rt_targets + all_mentions))
    
    n_channels = len(sources)
    log(f'    Source channels: {n_channels}')
    log(f'    Total graph nodes (channels + RT targets + mentions): {len(unique_users):,}')
    
    # ---------------------------------------------------------------
    # PRE-COMPUTE: V-Anchor vectors per asset
    # ---------------------------------------------------------------
    log('    Pre-computing V-Anchor vectors per asset...')
    asset_anchors = {}
    for asset_name, asset_df in assets.items():
        asset_df_c = asset_df.copy()
        if 'time' not in asset_df_c.columns:
            asset_df_c = asset_df_c.reset_index().rename(columns={'index': 'time'})
        asset_df_c['time'] = pd.to_datetime(asset_df_c['time'], errors='coerce').dt.date
        asset_df_c = asset_df_c.dropna(subset=['time'])
        
        if 'price_change' not in asset_df_c.columns:
            if 'close' in asset_df_c.columns:
                asset_df_c['price_change'] = asset_df_c['close'].pct_change()
        
        valid_dates = sorted(asset_df_c.dropna(subset=['price_change'])['time'].tolist())
        if not valid_dates:
            asset_anchors[asset_name] = None
            continue
        split_idx = int(len(valid_dates) * TRAIN_RATIO)
        train_end_date = valid_dates[split_idx - 1]
        
        try:
            anchor = compute_anchor_vector(
                tw, asset_df_c, unique_users, train_end_date=train_end_date)
            asset_anchors[asset_name] = anchor
            n_active = sum(1 for v in anchor.values() if v > 0.01)
            log(f'      {asset_name}: {n_active} users with active anchor')
        except Exception as e:
            log(f'      {asset_name} anchor failed: {e}')
            asset_anchors[asset_name] = None
    
    default_anchor = asset_anchors.get('BTC', list(asset_anchors.values())[0])
    
    # ===================================================================
    # 20a. Con=0 fraction
    # ===================================================================
    log('  20a. Con=0 fraction per asset per regime')
    con_zero_rows = []
    
    if 'con' in tw.columns:
        tw_with_date = tw.copy()
        tw_with_date['date'] = pd.to_datetime(tw_with_date['time'], errors='coerce').dt.date
        
        total_tweets = len(tw_with_date)
        con_zero_total = (tw_with_date['con'] == 0).sum()
        con_near_zero = (tw_with_date['con'] < 0.01).sum()
        
        con_zero_rows.append({
            'scope': 'ALL',
            'total_tweets': total_tweets,
            'con_zero': int(con_zero_total),
            'con_near_zero': int(con_near_zero),
            'pct_zero': round(con_zero_total / total_tweets * 100, 1),
            'pct_near_zero': round(con_near_zero / total_tweets * 100, 1),
        })
        
        for asset_name, final in final_data.items():
            if 'regime' in final.columns:
                for regime_id, regime_name in [(0, 'bull'), (1, 'bear')]:
                    regime_dates = set(final[final['regime'] == regime_id]['time'].values)
                    regime_tw = tw_with_date[tw_with_date['date'].isin(regime_dates)]
                    if len(regime_tw) > 0:
                        n = len(regime_tw)
                        cz = (regime_tw['con'] == 0).sum()
                        cnz = (regime_tw['con'] < 0.01).sum()
                        con_zero_rows.append({
                            'scope': f'{asset_name}_{regime_name}',
                            'total_tweets': n,
                            'con_zero': int(cz),
                            'con_near_zero': int(cnz),
                            'pct_zero': round(cz / n * 100, 1),
                            'pct_near_zero': round(cnz / n * 100, 1),
                        })
                break
    
    pd.DataFrame(con_zero_rows).to_csv(
        os.path.join(sens_path, 'con_zero_fraction.csv'), index=False)
    log(f'    Con=0: {con_zero_rows[0]["pct_zero"]}% of tweets' if con_zero_rows else '    No con data')
    
    # ===================================================================
    # 20b. Gate-by-gate ablations (pass actual anchor_vector)
    # ===================================================================
    log('  20b. Gate-by-gate ablations')
    ablation_rows = []
    
    # Adaptive floor for fillna
    n_users_total = len(unique_users)
    adaptive_floor = 1.0 / max(n_users_total, 1)
    
    for asset_name, final in final_data.items():
        anchor = asset_anchors.get(asset_name)
        
        for config_name, use_omega, use_anchor, con_in_agg in GATE_ABLATION_CONFIGS:
            try:
                rep = compute_lrr_vectorised(
                    tw, unique_users,
                    use_omega=use_omega,
                    anchor_vector=anchor if use_anchor else None,
                )
                
                tw_temp = tw.copy()
                tw_temp['rep_w'] = tw_temp['source_user'].astype(str).map(rep).fillna(adaptive_floor)
                
                daily = tw_temp.groupby('time').apply(lambda x: pd.Series({
                    'signal': np.average(
                        x['sen'] * (x['con'] if con_in_agg and 'con' in x.columns else 1.0),
                        weights=x['rep_w'] + 1e-9),
                    'omega': x['omega'].mean() if 'omega' in x.columns else 0.5,
                }), include_groups=False).reset_index()
                
                daily['time'] = pd.to_datetime(daily['time'], errors='coerce').dt.date
                merged = pd.merge(daily, final[['time', 'price_change']], on='time', how='inner')
                
                if len(merged) < 50:
                    continue
                
                f_stat, p_val = _run_granger_quick(
                    merged['signal'], merged['omega'], merged['price_change'])
                
                _, joint_n, _ = compute_tail_dependence_extended(
                    merged['signal'], merged['price_change'])
                
                ablation_rows.append({
                    'asset': asset_name,
                    'config': config_name,
                    'granger_F': round(f_stat, 3) if not np.isnan(f_stat) else np.nan,
                    'granger_p': round(p_val, 4) if not np.isnan(p_val) else np.nan,
                    'joint_crashes': joint_n,
                })
            except Exception as e:
                log(f'      {asset_name}/{config_name} failed: {e}')
    
    abl_df = pd.DataFrame(ablation_rows)
    abl_df.to_csv(
        os.path.join(sens_path, 'gate_ablation_results.csv'), index=False)
    log(f'    {len(ablation_rows)} ablation results saved')
    
    # Print gate ablation summary (mean F-stat across assets per config)
    if len(abl_df) > 0:
        abl_summary = abl_df.groupby('config')['granger_F'].mean().sort_values(ascending=False)
        log('    --- Gate Ablation Summary (mean F across 6 assets) ---')
        for config_name, mean_f in abl_summary.items():
            sig_count = ((abl_df[abl_df['config'] == config_name]['granger_p'] < 0.05).sum())
            total = len(abl_df[abl_df['config'] == config_name])
            log(f'      {config_name:<15} F={mean_f:.3f}  ({sig_count}/{total} significant)')
    
    # ===================================================================
    # 20c. Con sensitivity
    # ===================================================================
    log('  20c. Con sensitivity analysis')
    con_sens_rows = []
    
    btc_final = final_data.get('BTC', list(final_data.values())[0] if final_data else None)
    if btc_final is not None:
        for config_name, con_prop, con_agg, con_denom in CON_SENSITIVITY_CONFIGS:
            try:
                rep = compute_lrr_vectorised(
                    tw, unique_users, use_omega=True,
                    use_con=con_prop, con_in_propagation=con_prop,
                    anchor_vector=default_anchor)
                
                daily = compute_daily_signal_with_config(
                    tw, rep, con_in_aggregation=con_agg,
                    con_in_denominator=con_denom)
                
                daily_df = daily.reset_index()
                daily_df.columns = ['time', 'signal']
                daily_df['time'] = pd.to_datetime(daily_df['time'], errors='coerce').dt.date
                
                omega_daily = tw.groupby('time')['omega'].mean().reset_index()
                omega_daily['time'] = pd.to_datetime(omega_daily['time'], errors='coerce').dt.date
                
                merged = pd.merge(daily_df, btc_final[['time', 'price_change']], on='time', how='inner')
                merged = pd.merge(merged, omega_daily, on='time', how='left')
                
                if len(merged) < 50:
                    continue
                
                f_stat, p_val = _run_granger_quick(
                    merged['signal'], merged['omega'], merged['price_change'])
                _, joint_n, _ = compute_tail_dependence_extended(
                    merged['signal'], merged['price_change'])
                
                con_sens_rows.append({
                    'config': config_name,
                    'granger_F': round(f_stat, 3) if not np.isnan(f_stat) else np.nan,
                    'granger_p': round(p_val, 4) if not np.isnan(p_val) else np.nan,
                    'joint_crashes': joint_n,
                })
            except Exception as e:
                log(f'      Con config {config_name} failed: {e}')
    
    pd.DataFrame(con_sens_rows).to_csv(
        os.path.join(sens_path, 'con_sensitivity_results.csv'), index=False)
    log(f'    {len(con_sens_rows)} Con sensitivity results saved')
    
    # ===================================================================
    # 20d. Hyperparameter sweep (BTC only) — pass anchor
    # ===================================================================
    log('  20d. Hyperparameter sweep (BTC)')
    hyper_rows = []
    
    if btc_final is not None:
        for phi_val in PHI_VALUES:
            for n_iter in ITERATION_VALUES:
                for omega_fl in OMEGA_FLOOR_VALUES:
                    try:
                        rep = compute_lrr_vectorised(
                            tw, unique_users, use_omega=True, use_con=True,
                            anchor_vector=default_anchor,
                            phi=phi_val, n_iterations=n_iter,
                            omega_floor=omega_fl)
                        
                        tw_temp = tw.copy()
                        tw_temp['rep_w'] = tw_temp['source_user'].astype(str).map(rep).fillna(adaptive_floor)
                        daily = tw_temp.groupby('time').apply(lambda x: pd.Series({
                            'signal': np.average(
                                x['sen'] * x.get('con', pd.Series(1.0, index=x.index)),
                                weights=x['rep_w'] + 1e-9),
                            'omega': x['omega'].mean() if 'omega' in x.columns else 0.5,
                        }), include_groups=False).reset_index()
                        daily['time'] = pd.to_datetime(daily['time'], errors='coerce').dt.date
                        
                        merged = pd.merge(daily, btc_final[['time', 'price_change']],
                                         on='time', how='inner')
                        if len(merged) < 50:
                            continue
                        
                        f_stat, p_val = _run_granger_quick(
                            merged['signal'], merged['omega'], merged['price_change'])
                        _, joint_n, _ = compute_tail_dependence_extended(
                            merged['signal'], merged['price_change'])
                        
                        hyper_rows.append({
                            'phi': phi_val,
                            'iterations': n_iter,
                            'omega_floor': omega_fl,
                            'granger_F': round(f_stat, 3) if not np.isnan(f_stat) else np.nan,
                            'granger_p': round(p_val, 4) if not np.isnan(p_val) else np.nan,
                            'joint_crashes': joint_n,
                        })
                    except Exception as e:
                        log(f'      phi={phi_val} iter={n_iter} floor={omega_fl} failed: {e}')
    
    pd.DataFrame(hyper_rows).to_csv(
        os.path.join(sens_path, 'hyperparameter_sweep.csv'), index=False)
    log(f'    {len(hyper_rows)} hyperparameter combinations tested')
    
    # ===================================================================
    # 20d2. Mention weight sensitivity sweep (BTC)
    # ===================================================================
    log('  20d2. Mention weight sensitivity sweep (BTC)')
    from src.config import MENTION_WEIGHT_SWEEP
    mw_rows = []
    
    if btc_final is not None:
        for mw_val in MENTION_WEIGHT_SWEEP:
            try:
                rep = compute_lrr_vectorised(
                    tw, unique_users, use_omega=True, use_con=True,
                    anchor_vector=default_anchor, mention_weight=mw_val)
                
                tw_temp = tw.copy()
                tw_temp['rep_w'] = tw_temp['source_user'].astype(str).map(rep).fillna(adaptive_floor)
                tw_temp['time_d'] = pd.to_datetime(tw_temp['time'], errors='coerce').dt.date
                
                daily = tw_temp.groupby('time_d').apply(lambda x: pd.Series({
                    'signal': np.average(x['sen'].values * x['con'].values,
                                         weights=x['rep_w'].values + 1e-9),
                    'omega': x['omega'].mean() if 'omega' in x.columns else 0.5,
                }), include_groups=False).reset_index()
                daily.columns = ['time', 'signal', 'omega']
                
                merged = pd.merge(daily, btc_final[['time', 'price_change']],
                                  on='time', how='inner')
                
                if len(merged) >= 50:
                    F_val, p_val = _run_granger_quick(
                        merged['signal'], merged['omega'], merged['price_change'])
                else:
                    F_val, p_val = np.nan, np.nan
                
                mw_rows.append({
                    'mention_weight': mw_val,
                    'granger_F': round(F_val, 3) if not np.isnan(F_val) else np.nan,
                    'granger_p': round(p_val, 4) if not np.isnan(p_val) else np.nan,
                    'significant': '*' if (not np.isnan(p_val) and p_val < 0.05) else '',
                })
                log(f'      w_m={mw_val:.2f}: F={F_val:.3f} p={p_val:.4f}'
                    f'{" *" if not np.isnan(p_val) and p_val < 0.05 else ""}')
            except Exception as e:
                log(f'      w_m={mw_val:.2f} failed: {e}')
    
    pd.DataFrame(mw_rows).to_csv(
        os.path.join(sens_path, 'mention_weight_sweep.csv'), index=False)
    log(f'    {len(mw_rows)} mention weight values tested')
    
    # ===================================================================
    # 20e. Rolling/expanding reputation (date alignment + anchor)
    # ===================================================================
    log('  20e. Rolling reputation (expanding window)')
    rolling_rows = []
    
    for asset_name, final in final_data.items():
        try:
            anchor = asset_anchors.get(asset_name)
            asset_df = assets.get(asset_name, pd.DataFrame())
            
            date_to_rep = compute_lrr_expanding_window(
                tw, unique_users, asset_df,
                window_days=ROLLING_WINDOW_DAYS,
                use_omega=True, use_con=True,
                anchor_vector=anchor)
            
            if not date_to_rep:
                log(f'    {asset_name}: no rolling reputations computed')
                continue
            
            tw_with_date = tw.copy()
            tw_with_date['date'] = pd.to_datetime(tw_with_date['time'], errors='coerce').dt.date
            
            daily_signals = []
            for d in sorted(date_to_rep.keys()):
                rep = date_to_rep[d]
                day_tw = tw_with_date[tw_with_date['date'] == d]
                if len(day_tw) == 0:
                    continue
                
                weights = day_tw['source_user'].astype(str).map(rep).fillna(adaptive_floor).values
                con_vals = day_tw['con'].values if 'con' in day_tw.columns else np.ones(len(day_tw))
                sen_vals = day_tw['sen'].values
                omega_val = day_tw['omega'].mean() if 'omega' in day_tw.columns else 0.5
                
                sig = np.sum(sen_vals * con_vals * weights) / (np.sum(weights) + 1e-9)
                daily_signals.append({'time': d, 'rolling_signal': sig, 'omega': omega_val})
            
            if not daily_signals:
                log(f'    {asset_name}: no daily signals from rolling')
                continue
            
            rolling_df = pd.DataFrame(daily_signals)
            
            final_c = final.copy()
            final_c['time'] = pd.to_datetime(final_c['time'], errors='coerce').dt.date
            
            merged = pd.merge(rolling_df, final_c[['time', 'price_change']],
                             on='time', how='inner')
            
            log(f'    {asset_name}: {len(merged)} days merged from rolling')
            
            if len(merged) < 50:
                rolling_rows.append({
                    'asset': asset_name,
                    'method': f'expanding_{ROLLING_WINDOW_DAYS}d',
                    'granger_F': np.nan,
                    'granger_p': np.nan,
                    'n_windows': len(date_to_rep),
                    'n_merged': len(merged),
                })
                continue
            
            f_stat, p_val = _run_granger_quick(
                merged['rolling_signal'], merged['omega'], merged['price_change'])
            
            rolling_rows.append({
                'asset': asset_name,
                'method': f'expanding_{ROLLING_WINDOW_DAYS}d',
                'granger_F': round(f_stat, 3) if not np.isnan(f_stat) else np.nan,
                'granger_p': round(p_val, 4) if not np.isnan(p_val) else np.nan,
                'n_windows': len(date_to_rep),
                'n_merged': len(merged),
            })
            log(f'    {asset_name}: F={f_stat:.3f} p={p_val:.4f}' if not np.isnan(f_stat) else f'    {asset_name}: Granger test failed')
        except Exception as e:
            log(f'    {asset_name} rolling failed: {e}')
            import traceback
            traceback.print_exc()
    
    pd.DataFrame(rolling_rows).to_csv(
        os.path.join(sens_path, 'rolling_reputation_results.csv'), index=False)
    
    # ===================================================================
    # 20f. HMM robustness (multi-level winsorisation)
    # ===================================================================
    log('  20f. HMM robustness (multi-level winsorisation)')
    hmm_rows = []
    
    for asset_name, asset_df in assets.items():
        try:
            hmm_results = run_hmm_robustness(asset_df, winsorize_pcts=(1, 99))
            for key, val in hmm_results.items():
                if 'concordance' in key:
                    level = key.replace('_concordance', '')
                    hmm_rows.append({
                        'asset': asset_name,
                        'winsorisation': level,
                        'concordance': round(val, 3),
                    })
                    log(f'    {asset_name} [{level}]: concordance = {val:.1%}')
        except Exception as e:
            log(f'    {asset_name} HMM robustness failed: {e}')
    
    pd.DataFrame(hmm_rows).to_csv(
        os.path.join(sens_path, 'hmm_robustness.csv'), index=False)
    
    # ===================================================================
    # 20g. Channel Concentration + Leave-One-Out Robustness
    #      (replaced bot detection — dataset is from 76 curated channels)
    # ===================================================================
    log('  20g. Channel concentration & leave-one-out robustness')
    
    # --- 20g-i: Channel concentration analysis ---
    channel_stats = tw.groupby('source_user').agg(
        tweet_count=('sen', 'count'),
        mean_sen=('sen', 'mean'),
        std_sen=('sen', 'std'),
        mean_omega=('omega', 'mean') if 'omega' in tw.columns else ('sen', lambda x: np.nan),
        unique_dates=('time', 'nunique'),
    ).reset_index()
    channel_stats = channel_stats.sort_values('tweet_count', ascending=False)
    
    # Add omega properly if column exists
    if 'omega' in tw.columns:
        ch_omega = tw.groupby('source_user')['omega'].mean().reset_index()
        ch_omega.columns = ['source_user', 'mean_omega']
        if 'mean_omega' in channel_stats.columns:
            channel_stats = channel_stats.drop(columns=['mean_omega'])
        channel_stats = channel_stats.merge(ch_omega, on='source_user', how='left')
    
    # Compute concentration metrics
    tweet_counts = channel_stats['tweet_count'].values
    tweet_gini = _gini(tweet_counts)
    total_tweets = tweet_counts.sum()
    
    # Top-N share
    sorted_counts = np.sort(tweet_counts)[::-1]
    top1_share = sorted_counts[0] / total_tweets if len(sorted_counts) > 0 else 0
    top3_share = sorted_counts[:3].sum() / total_tweets if len(sorted_counts) >= 3 else 0
    top5_share = sorted_counts[:5].sum() / total_tweets if len(sorted_counts) >= 5 else 0
    top10_share = sorted_counts[:10].sum() / total_tweets if len(sorted_counts) >= 10 else 0
    
    concentration_info = {
        'n_channels': n_channels,
        'total_tweets': int(total_tweets),
        'tweet_gini': round(tweet_gini, 4),
        'top1_channel_share': round(top1_share, 4),
        'top3_channel_share': round(top3_share, 4),
        'top5_channel_share': round(top5_share, 4),
        'top10_channel_share': round(top10_share, 4),
        'mean_tweets_per_channel': round(total_tweets / max(n_channels, 1), 1),
        'max_tweets_one_channel': int(sorted_counts[0]) if len(sorted_counts) > 0 else 0,
        'min_tweets_one_channel': int(sorted_counts[-1]) if len(sorted_counts) > 0 else 0,
    }
    
    channel_stats.to_csv(
        os.path.join(sens_path, 'channel_statistics.csv'), index=False)
    
    log(f'    Channels: {n_channels}, Tweet Gini: {tweet_gini:.3f}')
    log(f'    Top-1 channel: {top1_share:.1%}, Top-5: {top5_share:.1%}, Top-10: {top10_share:.1%}')
    
    # --- 20g-ii: Leave-one-out channel robustness ---
    log('    Running leave-one-out channel robustness (BTC)...')
    loo_rows = []
    
    # Only do this for BTC to save time
    if btc_final is not None:
        # Baseline Granger F-stat (full data)
        baseline_F, baseline_p = np.nan, np.nan
        for row in ablation_rows:
            if row['asset'] == 'BTC' and row['config'] == 'Full_Oracle':
                baseline_F = row['granger_F']
                baseline_p = row['granger_p']
                break
        
        for ch_idx, ch_id in enumerate(sources):
            try:
                # Drop this channel's tweets
                tw_loo = tw[tw['source_user'] != ch_id].copy()
                n_dropped = len(tw) - len(tw_loo)
                
                if len(tw_loo) < 100:
                    continue
                
                # Recompute users (some RT targets/mentions may become orphaned)
                loo_sources = tw_loo['source_user'].dropna().unique().tolist()
                loo_rt = tw_loo['rt_target'].dropna().unique().tolist()
                loo_mentions = [m for ml in tw_loo['mentions']
                               for m in ml if isinstance(ml, list)]
                loo_users = list(set(loo_sources + loo_rt + loo_mentions))
                
                rep = compute_lrr_vectorised(
                    tw_loo, loo_users, use_omega=True, use_con=True,
                    anchor_vector=default_anchor)
                
                tw_temp = tw_loo.copy()
                tw_temp['rep_w'] = tw_temp['source_user'].astype(str).map(rep).fillna(adaptive_floor)
                
                daily = tw_temp.groupby('time').apply(lambda x: pd.Series({
                    'signal': np.average(
                        x['sen'] * (x['con'] if 'con' in x.columns else 1.0),
                        weights=x['rep_w'] + 1e-9),
                    'omega': x['omega'].mean() if 'omega' in x.columns else 0.5,
                }), include_groups=False).reset_index()
                daily['time'] = pd.to_datetime(daily['time'], errors='coerce').dt.date
                
                merged = pd.merge(daily, btc_final[['time', 'price_change']],
                                 on='time', how='inner')
                
                if len(merged) < 50:
                    continue
                
                f_stat, p_val = _run_granger_quick(
                    merged['signal'], merged['omega'], merged['price_change'])
                
                loo_rows.append({
                    'dropped_channel': str(ch_id),
                    'tweets_dropped': n_dropped,
                    'pct_dropped': round(n_dropped / len(tw) * 100, 2),
                    'granger_F': round(f_stat, 3) if not np.isnan(f_stat) else np.nan,
                    'granger_p': round(p_val, 4) if not np.isnan(p_val) else np.nan,
                    'F_change_pct': round((f_stat - baseline_F) / baseline_F * 100, 1)
                                    if not np.isnan(f_stat) and not np.isnan(baseline_F)
                                       and baseline_F > 0 else np.nan,
                })
                
                if (ch_idx + 1) % 10 == 0:
                    log(f'      {ch_idx+1}/{len(sources)} channels processed')
                    
            except Exception as e:
                log(f'      Channel {ch_id} LOO failed: {e}')
    
    loo_df = pd.DataFrame(loo_rows)
    loo_df.to_csv(os.path.join(sens_path, 'leave_one_out_channel.csv'), index=False)
    
    # Summarise LOO results
    if len(loo_rows) > 0:
        f_vals = [r['granger_F'] for r in loo_rows if not np.isnan(r.get('granger_F', np.nan))]
        p_vals = [r['granger_p'] for r in loo_rows if not np.isnan(r.get('granger_p', np.nan))]
        n_sig = sum(1 for p in p_vals if p < 0.05)
        
        log(f'    LOO: {n_sig}/{len(p_vals)} still significant at p<0.05')
        log(f'    LOO F-stat range: {min(f_vals):.3f} - {max(f_vals):.3f} (baseline: {baseline_F})')
    
    # --- Write channel robustness report ---
    with open(os.path.join(sens_path, 'channel_robustness_report.txt'), 'w', encoding='utf-8') as f:
        f.write('=== Channel Concentration & Leave-One-Out Robustness ===\n\n')
        
        f.write('--- Data Provenance ---\n')
        f.write(f'The corpus is sourced from {n_channels} curated cryptocurrency-focused\n')
        f.write('Twitter channels, selected for sustained activity and domain relevance.\n')
        f.write('This pre-selection acts as a first-pass quality filter — the reputation\n')
        f.write('propagation then differentiates information quality within this\n')
        f.write('already-curated population. Traditional bot detection is not applicable\n')
        f.write('to this setting, as channels are established accounts rather than\n')
        f.write('anonymous users from an unfiltered keyword crawl.\n\n')
        
        f.write('--- Channel Concentration ---\n')
        for k, v in concentration_info.items():
            f.write(f'  {k}: {v}\n')
        f.write('\n')
        
        if tweet_gini > 0.5:
            f.write('NOTE: High tweet Gini indicates substantial concentration.\n')
            f.write('A few channels produce the majority of tweets.\n')
        else:
            f.write('NOTE: Moderate tweet distribution across channels.\n')
        f.write('\n')
        
        f.write('--- Leave-One-Out Robustness (BTC) ---\n')
        f.write(f'Baseline Granger F-stat (all channels): {baseline_F}\n')
        f.write(f'Baseline p-value: {baseline_p}\n\n')
        
        if len(loo_rows) > 0:
            f_vals = [r['granger_F'] for r in loo_rows if not np.isnan(r.get('granger_F', np.nan))]
            p_vals = [r['granger_p'] for r in loo_rows if not np.isnan(r.get('granger_p', np.nan))]
            n_sig = sum(1 for p in p_vals if p < 0.05)
            
            f.write(f'Channels tested: {len(loo_rows)}\n')
            f.write(f'Still significant (p<0.05) after dropping any single channel: {n_sig}/{len(p_vals)}\n')
            f.write(f'F-stat range across LOO: {min(f_vals):.3f} - {max(f_vals):.3f}\n')
            f.write(f'Mean F-stat: {np.mean(f_vals):.3f}\n')
            f.write(f'Std F-stat: {np.std(f_vals):.3f}\n\n')
            
            if n_sig == len(p_vals):
                f.write('CONCLUSION: Results are fully robust to single-channel removal.\n')
                f.write('No single channel is necessary for the reported findings.\n')
            elif n_sig >= len(p_vals) * 0.9:
                f.write('CONCLUSION: Results are largely robust to single-channel removal.\n')
                f.write(f'{len(p_vals) - n_sig} channel(s) are influential but not critical.\n')
            else:
                f.write(f'WARNING: {len(p_vals) - n_sig} channels substantially affect results.\n')
                f.write('Findings may be driven by a small number of channels.\n')
            
            # List most influential channels (largest F change)
            f.write('\n--- Most Influential Channels (by F-stat change) ---\n')
            sorted_loo = sorted(loo_rows,
                               key=lambda r: abs(r.get('F_change_pct', 0) or 0),
                               reverse=True)
            for row in sorted_loo[:10]:
                f.write(f'  {row["dropped_channel"]}: '
                        f'F={row["granger_F"]} '
                        f'(change: {row.get("F_change_pct", "?")}%) '
                        f'tweets: {row["tweets_dropped"]} ({row["pct_dropped"]}%)\n')
    
    log('    channel_robustness_report.txt saved')
    
    # ===================================================================
    # 20h. Alternative independence measures
    # ===================================================================
    log('  20h. Alternative independence measures')
    alt_indep_rows = []
    
    tw_alt = compute_independence_daily_deviation(tw)
    tw_alt = compute_independence_temporal_novelty(tw_alt, TEMPORAL_NOVELTY_WINDOW)
    
    for alt_name, alt_col in [('daily_deviation', 'ind_daily_dev'),
                               ('temporal_novelty', 'ind_temporal_nov')]:
        try:
            rep = compute_lrr_vectorised(tw, unique_users, use_omega=True, use_con=False,
                                          anchor_vector=default_anchor)
            
            tw_temp = tw_alt.copy()
            tw_temp['rep_w'] = tw_temp['source_user'].astype(str).map(rep).fillna(adaptive_floor)
            tw_temp['alt_weight'] = tw_temp[alt_col].fillna(0.5)
            
            daily = tw_temp.groupby('time').apply(lambda x: pd.Series({
                'signal': np.average(
                    x['sen'] * x['alt_weight'],
                    weights=x['rep_w'] + 1e-9),
                'omega': x['omega'].mean() if 'omega' in x.columns else 0.5,
            }), include_groups=False).reset_index()
            daily['time'] = pd.to_datetime(daily['time'], errors='coerce').dt.date
            
            for asset_name, final in final_data.items():
                merged = pd.merge(daily, final[['time', 'price_change']],
                                 on='time', how='inner')
                if len(merged) >= 50:
                    f_stat, p_val = _run_granger_quick(
                        merged['signal'], merged['omega'], merged['price_change'])
                    _, joint_n, _ = compute_tail_dependence_extended(
                        merged['signal'], merged['price_change'])
                    
                    alt_indep_rows.append({
                        'independence_measure': alt_name,
                        'asset': asset_name,
                        'granger_F': round(f_stat, 3) if not np.isnan(f_stat) else np.nan,
                        'granger_p': round(p_val, 4) if not np.isnan(p_val) else np.nan,
                        'joint_crashes': joint_n,
                    })
        except Exception as e:
            log(f'    {alt_name} failed: {e}')
    
    pd.DataFrame(alt_indep_rows).to_csv(
        os.path.join(sens_path, 'alternative_independence.csv'), index=False)
    log(f'    {len(alt_indep_rows)} alternative independence results saved')
    
    # ===================================================================
    # 20i. CDS validation sample extraction
    # ===================================================================
    log('  20i. Extracting CDS validation sample')
    if 'omega' in tw.columns:
        distorted = tw[tw['omega'] < 0.9].sample(n=min(100, len(tw[tw['omega'] < 0.9])),
                                                   random_state=42)
        clean = tw[tw['omega'] >= 0.95].sample(n=min(100, len(tw[tw['omega'] >= 0.95])),
                                                 random_state=42)
        
        validation_sample = pd.concat([
            distorted.assign(cds_label='distorted'),
            clean.assign(cds_label='clean')
        ])
        
        cols_to_keep = ['source_user', 'time', 'sen', 'omega', 'cds_label']
        for text_col in ['text', 'content', 'tweet_text', 'body']:
            if text_col in validation_sample.columns:
                cols_to_keep.append(text_col)
                break
        
        cols_to_keep = [c for c in cols_to_keep if c in validation_sample.columns]
        validation_sample[cols_to_keep].to_csv(
            os.path.join(sens_path, 'cds_validation_sample.csv'), index=False)
        log(f'    CDS validation sample: {len(distorted)} distorted + {len(clean)} clean tweets')
    
    log('>>> Phase 20 complete. Results saved to /results/sensitivity/')
    
    # ===================================================================
    # Summary report
    # ===================================================================
    with open(os.path.join(sens_path, 'SENSITIVITY_SUMMARY.txt'), 'w', encoding='utf-8') as f:
        f.write('=== Phase 20: Sensitivity & Robustness Suite Summary ===\n')
        f.write('=== v2.2 — anchor fixes + channel robustness ===\n\n')
        
        f.write(f'Data: {n_channels} curated channels, {len(unique_users):,} graph nodes, '
                f'{len(tw):,} tweets\n\n')
        
        f.write('--- 20a. Con=0 Fraction ---\n')
        for row in con_zero_rows:
            f.write(f"  {row['scope']}: {row['pct_zero']}% zero, {row['pct_near_zero']}% near-zero\n")
        
        f.write('\n--- 20b. Gate Ablation (Granger F-stats) ---\n')
        for row in ablation_rows:
            f.write(f"  {row['asset']}/{row['config']}: F={row['granger_F']} p={row['granger_p']} crashes={row['joint_crashes']}\n")
        
        f.write('\n--- 20c. Con Sensitivity ---\n')
        for row in con_sens_rows:
            f.write(f"  {row['config']}: F={row['granger_F']} p={row['granger_p']} crashes={row['joint_crashes']}\n")
        
        f.write('\n--- 20d. Hyperparameter Sweep ---\n')
        f.write(f"  {len(hyper_rows)} combinations tested\n")
        if hyper_rows:
            sig_count = sum(1 for r in hyper_rows if r.get('granger_p', 1) < 0.05)
            f.write(f"  Significant at p<0.05: {sig_count}/{len(hyper_rows)}\n")
            f_vals = [r['granger_F'] for r in hyper_rows if not np.isnan(r.get('granger_F', np.nan))]
            if f_vals:
                f.write(f"  F-stat range: {min(f_vals):.3f} - {max(f_vals):.3f}\n")
        
        f.write('\n--- 20e. Rolling Reputation ---\n')
        for row in rolling_rows:
            f.write(f"  {row['asset']}: F={row['granger_F']} p={row['granger_p']} (n_merged={row.get('n_merged', '?')})\n")
        
        f.write('\n--- 20f. HMM Robustness ---\n')
        for row in hmm_rows:
            f.write(f"  {row['asset']} [{row['winsorisation']}]: concordance={row['concordance']}\n")
        
        f.write(f'\n--- 20g. Channel Robustness ---\n')
        f.write(f"  Channels: {n_channels}\n")
        f.write(f"  Tweet Gini: {concentration_info['tweet_gini']}\n")
        f.write(f"  Top-5 channel share: {concentration_info['top5_channel_share']:.1%}\n")
        if len(loo_rows) > 0:
            f_vals_loo = [r['granger_F'] for r in loo_rows if not np.isnan(r.get('granger_F', np.nan))]
            p_vals_loo = [r['granger_p'] for r in loo_rows if not np.isnan(r.get('granger_p', np.nan))]
            n_sig_loo = sum(1 for p in p_vals_loo if p < 0.05)
            f.write(f"  LOO: {n_sig_loo}/{len(p_vals_loo)} still significant after dropping any channel\n")
            f.write(f"  LOO F-stat range: {min(f_vals_loo):.3f} - {max(f_vals_loo):.3f}\n")
        
        f.write('\n--- 20h. Alternative Independence ---\n')
        for row in alt_indep_rows:
            f.write(f"  {row['independence_measure']}/{row['asset']}: F={row['granger_F']} p={row['granger_p']} crashes={row.get('joint_crashes', '?')}\n")
    
    log('  SENSITIVITY_SUMMARY.txt saved')