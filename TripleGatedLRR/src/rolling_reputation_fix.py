# src/rolling_reputation_fix.py  (v2)
# Fixed: uses identical Granger test as main pipeline
# Fixed: passes V-Anchor per asset
# Fixed: no signal rescaling

import os
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


def log(msg, indent=0):
    print('  ' * indent + msg)


def _run_granger_standard(signal_series, omega_series, price_series, max_lag=7):
    """Standard Granger test - identical to main pipeline. No rescaling."""
    data = pd.DataFrame({
        'signal': signal_series,
        'omega': omega_series,
        'price': price_series,
    }).dropna()
    if len(data) < 50:
        return np.nan, np.nan, 0
    try:
        model = VAR(data).fit(maxlags=max_lag, ic='bic')
        gc = model.test_causality('omega', 'signal', kind='f')
        return float(gc.test_statistic), float(gc.pvalue), model.k_ar
    except Exception:
        return np.nan, np.nan, 0


def run_rolling_reputation_fix(tw_raw, final_data, results_path):
    """Rolling reputation validation - FIXED version."""
    log('\n>>> Rolling Reputation Fix v2 (causal validation)')
    
    from src.reputation_engine_v2 import compute_lrr_expanding_window
    from src.anchor_utils import compute_anchor_vector
    from src.psych_engine import calculate_omega
    
    sens_path = os.path.join(results_path, 'sensitivity')
    os.makedirs(sens_path, exist_ok=True)
    
    tw = calculate_omega(tw_raw.copy())
    tw['date'] = pd.to_datetime(tw['time'], errors='coerce').dt.date
    
    sources = tw['source_user'].dropna().unique().tolist()
    rt_targets = tw['rt_target'].dropna().unique().tolist()
    mentions_list = []
    for ml in tw['mentions']:
        if isinstance(ml, list):
            mentions_list.extend(ml)
    unique_users = list(set(sources + rt_targets + mentions_list))
    
    from src.config import TRAIN_RATIO
    
    results_rows = []
    
    for asset_name, final in final_data.items():
        log(f'  {asset_name}...')
        try:
            final_c = final.copy()
            final_c['time'] = pd.to_datetime(final_c['time'], errors='coerce').dt.date
            
            # Compute train_end_date (same logic as main.py)
            valid_dates = final_c.dropna(subset=['price_change'])['time'].tolist()
            valid_dates_sorted = sorted(valid_dates)
            split_idx = int(len(valid_dates_sorted) * TRAIN_RATIO)
            train_end_date = valid_dates_sorted[split_idx - 1] if split_idx > 0 else None
            
            # Asset-specific anchor (training data only)
            try:
                anchor = compute_anchor_vector(tw, final_c, asset_name,
                                               train_end_date=train_end_date)
                log(f'    Anchor computed for {asset_name}')
            except Exception:
                anchor = None
                log(f'    No anchor for {asset_name}')
            
            # Expanding-window reputation WITH anchor
            date_to_rep = compute_lrr_expanding_window(
                tw, unique_users, pd.DataFrame(),
                window_days=30,
                use_omega=True, use_con=True,
                anchor_vector=anchor)
            
            if not date_to_rep:
                log(f'    No rolling reputation computed')
                continue
            
            # Build daily signal from rolling reputation
            daily_signals = []
            for d in sorted(date_to_rep.keys()):
                rep = date_to_rep[d]
                day_tw = tw[tw['date'] == d]
                if len(day_tw) == 0:
                    continue
                weights = day_tw['source_user'].astype(str).map(rep).fillna(
                    1.0 / max(len(unique_users), 1)).values
                con_vals = day_tw['con'].values if 'con' in day_tw.columns else np.ones(len(day_tw))
                sen_vals = day_tw['sen'].values
                sig = np.sum(sen_vals * con_vals * weights) / (np.sum(weights) + 1e-9)
                omega_mean = day_tw['omega'].mean() if 'omega' in day_tw.columns else 0.5
                daily_signals.append({'time': d, 'rolling_lrr': sig, 'omega_raw': omega_mean})
            
            if len(daily_signals) < 50:
                continue
            
            rolling_df = pd.DataFrame(daily_signals)
            log(f'    {len(rolling_df)} rolling days computed')
            
            # Merge with final data
            merged = pd.merge(rolling_df, 
                            final_c[['time', 'price_change', 'omega', 'LRR_Oracle_Sen']],
                            on='time', how='inner')
            log(f'    {len(merged)} days after merge')
            
            if len(merged) < 50:
                continue
            
            corr = np.corrcoef(merged['rolling_lrr'].values, 
                              merged['LRR_Oracle_Sen'].values)[0, 1]
            log(f'    Static-Rolling r={corr:.4f}')
            
            roll_std = merged['rolling_lrr'].std()
            static_std = merged['LRR_Oracle_Sen'].std()
            log(f'    Std: rolling={roll_std:.6f} static={static_std:.6f}')
            
            # Granger on rolling signal (NO rescaling)
            roll_F, roll_p, roll_lag = _run_granger_standard(
                merged['rolling_lrr'], merged['omega'], merged['price_change'])
            if not np.isnan(roll_F):
                log(f'    Rolling Granger: F={roll_F:.3f} p={roll_p:.4f} lag={roll_lag}')
            else:
                log(f'    Rolling Granger: FAILED')
            
            # Static on SAME date range
            stat_F, stat_p, stat_lag = _run_granger_standard(
                merged['LRR_Oracle_Sen'], merged['omega'], merged['price_change'])
            if not np.isnan(stat_F):
                log(f'    Static Granger (same dates): F={stat_F:.3f} p={stat_p:.4f} lag={stat_lag}')
            else:
                log(f'    Static Granger: FAILED')
            
            # Also with raw omega
            roll_F2, roll_p2, _ = _run_granger_standard(
                merged['rolling_lrr'], merged['omega_raw'], merged['price_change'])
            if not np.isnan(roll_F2):
                log(f'    Rolling (raw omega): F={roll_F2:.3f} p={roll_p2:.4f}')
            
            results_rows.append({
                'asset': asset_name, 'n_days': len(merged),
                'static_rolling_corr': round(corr, 4),
                'rolling_std': round(roll_std, 6), 'static_std': round(static_std, 6),
                'rolling_F': round(roll_F, 3) if not np.isnan(roll_F) else np.nan,
                'rolling_p': round(roll_p, 4) if not np.isnan(roll_p) else np.nan,
                'rolling_lag': roll_lag,
                'static_F': round(stat_F, 3) if not np.isnan(stat_F) else np.nan,
                'static_p': round(stat_p, 4) if not np.isnan(stat_p) else np.nan,
                'static_lag': stat_lag,
                'rolling_F_raw_omega': round(roll_F2, 3) if not np.isnan(roll_F2) else np.nan,
                'rolling_p_raw_omega': round(roll_p2, 4) if not np.isnan(roll_p2) else np.nan,
            })
        except Exception as e:
            log(f'    {asset_name} failed: {e}')
            import traceback
            traceback.print_exc()
    
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(os.path.join(sens_path, 'rolling_reputation_fixed.csv'), index=False)
    
    with open(os.path.join(sens_path, 'rolling_reputation_report.txt'), 'w', encoding='utf-8') as f:
        f.write('=== Rolling Reputation Validation (v2) ===\n\n')
        f.write('Question: Does computing reputation causally (expanding window)\n')
        f.write('rather than from the full corpus change the Granger results?\n\n')
        f.write(f'{"Asset":<8} {"r(S,R)":<10} {"StaticF":<10} {"StaticP":<10} '
                f'{"RollF":<10} {"RollP":<10}\n')
        f.write('-' * 60 + '\n')
        for _, row in results_df.iterrows():
            f.write(f'{row["asset"]:<8} {row.get("static_rolling_corr",""):<10} '
                    f'{row.get("static_F",""):<10} {row.get("static_p",""):<10} '
                    f'{row.get("rolling_F",""):<10} {row.get("rolling_p",""):<10}\n')
        
        mean_corr = results_df['static_rolling_corr'].mean()
        f.write(f'\nMean static-rolling correlation: r={mean_corr:.4f}\n')
        f.write(f'NOTE: Main pipeline reports F~8.17 on 604 days.\n')
        f.write(f'This uses {results_df["n_days"].iloc[0] if len(results_df)>0 else "?"} days.\n')
    
    log('  rolling_reputation_report.txt saved')
    log('>>> Rolling Reputation Fix v2 complete.')
