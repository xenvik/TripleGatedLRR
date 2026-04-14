# src/lrr_enhanced_comparison.py
# Enhanced LRR vs HITS vs PageRank comparison suite
# Adds: Neut gate, signal stability, regime discrimination, flexible crash thresholds
#
# Call from main.py after Phase 6, or standalone after pipeline completes.
# Usage: run_enhanced_comparison(final_data, tw, results_path)

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def log(msg, indent=0):
    print('  ' * indent + msg)


def compute_neut(p_vals, n_vals):
    """
    Neutrality gate: Neut(t) = sqrt((1 - p(t)) * (1 - |n(t)|))
    HIGH when NEITHER sentiment is strong (calm/neutral posts).
    Opposite of Con which rewards BOTH sentiments being strong.
    """
    return np.sqrt((1.0 - p_vals) * (1.0 - np.abs(n_vals)))


def compute_ltd_flexible(signal, price, quantile=0.10, min_events=2):
    """
    Lower Tail Dependence with flexible minimum event threshold.
    
    Returns: (ltd_value, joint_count, tail_count)
    ltd_value = joint_count / tail_count if joint_count >= min_events, else NaN
    """
    signal = np.array(signal, dtype=float)
    price = np.array(price, dtype=float)
    
    # Remove NaN
    mask = ~(np.isnan(signal) | np.isnan(price))
    signal = signal[mask]
    price = price[mask]
    
    if len(signal) < 20:
        return np.nan, 0, 0
    
    sig_threshold = np.percentile(signal, quantile * 100)
    price_threshold = np.percentile(price, quantile * 100)
    
    sig_tail = signal <= sig_threshold
    price_tail = price <= price_threshold
    
    joint = np.sum(sig_tail & price_tail)
    tail_n = np.sum(sig_tail)
    
    if tail_n == 0 or joint < min_events:
        return np.nan, int(joint), int(tail_n)
    
    return float(joint) / float(tail_n), int(joint), int(tail_n)


def compute_signal_stability(signal_series, price_series, crash_quantile=0.10):
    """
    Signal stability during crashes:
    - std of signal during worst crash_quantile% of price days
    - std of signal during normal days
    - ratio (lower = more stable during crashes)
    """
    signal = np.array(signal_series, dtype=float)
    price = np.array(price_series, dtype=float)
    
    mask = ~(np.isnan(signal) | np.isnan(price))
    signal = signal[mask]
    price = price[mask]
    
    if len(signal) < 20:
        return {'crash_std': np.nan, 'normal_std': np.nan, 'stability_ratio': np.nan}
    
    price_threshold = np.percentile(price, crash_quantile * 100)
    crash_days = price <= price_threshold
    normal_days = ~crash_days
    
    crash_std = np.std(signal[crash_days]) if crash_days.sum() > 2 else np.nan
    normal_std = np.std(signal[normal_days]) if normal_days.sum() > 2 else np.nan
    
    ratio = crash_std / normal_std if (normal_std and normal_std > 0) else np.nan
    
    return {
        'crash_std': float(crash_std) if not np.isnan(crash_std) else np.nan,
        'normal_std': float(normal_std) if not np.isnan(normal_std) else np.nan,
        'stability_ratio': float(ratio) if not np.isnan(ratio) else np.nan,
        'crash_mean': float(np.mean(signal[crash_days])) if crash_days.sum() > 0 else np.nan,
        'normal_mean': float(np.mean(signal[normal_days])) if normal_days.sum() > 0 else np.nan,
    }


def compute_crash_correlation(signal_series, price_series, crash_quantile=0.10):
    """
    Signal-price correlation during crash vs normal periods.
    A good monitoring signal maintains LOWER correlation during crashes
    (it doesn't crash with the market).
    """
    signal = np.array(signal_series, dtype=float)
    price = np.array(price_series, dtype=float)
    
    mask = ~(np.isnan(signal) | np.isnan(price))
    signal = signal[mask]
    price = price[mask]
    
    if len(signal) < 30:
        return {'crash_corr': np.nan, 'normal_corr': np.nan, 'full_corr': np.nan}
    
    price_threshold = np.percentile(price, crash_quantile * 100)
    crash_days = price <= price_threshold
    normal_days = ~crash_days
    
    try:
        crash_corr, _ = pearsonr(signal[crash_days], price[crash_days]) if crash_days.sum() > 5 else (np.nan, np.nan)
    except Exception:
        crash_corr = np.nan
    try:
        normal_corr, _ = pearsonr(signal[normal_days], price[normal_days]) if normal_days.sum() > 5 else (np.nan, np.nan)
    except Exception:
        normal_corr = np.nan
    try:
        full_corr, _ = pearsonr(signal, price)
    except Exception:
        full_corr = np.nan
    
    return {
        'crash_corr': float(crash_corr) if not np.isnan(crash_corr) else np.nan,
        'normal_corr': float(normal_corr) if not np.isnan(normal_corr) else np.nan,
        'full_corr': float(full_corr) if not np.isnan(full_corr) else np.nan,
    }


def compute_regime_discrimination(signal_series, regime_series):
    """
    How well does the signal differentiate between regimes?
    Higher absolute difference = better regime discrimination.
    """
    signal = np.array(signal_series, dtype=float)
    regime = np.array(regime_series)
    
    mask = ~np.isnan(signal)
    signal = signal[mask]
    regime = regime[mask]
    
    bull = signal[regime == 0]
    bear = signal[regime == 1]
    
    if len(bull) < 5 or len(bear) < 5:
        return {'bull_mean': np.nan, 'bear_mean': np.nan, 'discrimination': np.nan}
    
    bull_mean = float(np.mean(bull))
    bear_mean = float(np.mean(bear))
    pooled_std = float(np.std(signal))
    
    # Cohen's d effect size
    discrimination = abs(bull_mean - bear_mean) / pooled_std if pooled_std > 0 else 0
    
    return {
        'bull_mean': bull_mean,
        'bear_mean': bear_mean,
        'bull_std': float(np.std(bull)),
        'bear_std': float(np.std(bear)),
        'discrimination_d': discrimination,
    }


def run_enhanced_comparison(final_data, tw_raw, results_path):
    """
    Run all enhanced comparison analyses.
    Call after the main pipeline has populated final_data.
    """
    log('\n>>> Enhanced LRR Comparison Suite')
    comp_path = os.path.join(results_path, 'comparison')
    os.makedirs(comp_path, exist_ok=True)
    
    # ===================================================================
    # Step 1: Compute Neut-weighted signal for each asset
    # ===================================================================
    log('  Computing Neut-weighted signals...')
    
    tw = tw_raw.copy()
    
    # Compute Neut values
    if 'pos' in tw.columns and 'neg' in tw.columns:
        p_vals = tw['pos'].fillna(0).values
        n_vals = tw['neg'].fillna(0).values
    elif 'p_sen' in tw.columns and 'n_sen' in tw.columns:
        p_vals = tw['p_sen'].fillna(0).values
        n_vals = tw['n_sen'].fillna(0).values
    elif 'sen' in tw.columns:
        # Approximate: p = max(sen, 0), n = min(sen, 0)
        log('    WARNING: Using approximated pos/neg from sen — less accurate')
        p_vals = np.maximum(tw['sen'].fillna(0).values, 0)
        n_vals = np.minimum(tw['sen'].fillna(0).values, 0)
    else:
        log('    ! Cannot compute Neut: no sentiment components found')
        p_vals = np.zeros(len(tw))
        n_vals = np.zeros(len(tw))
    
    tw['neut'] = compute_neut(p_vals, n_vals)
    
    # Also ensure Con exists
    if 'con' not in tw.columns:
        tw['con'] = np.sqrt(np.maximum(p_vals, 0) * np.abs(np.minimum(n_vals, 0)))
    
    log(f'    Neut stats: mean={tw["neut"].mean():.3f}, '
        f'std={tw["neut"].std():.3f}, '
        f'zero_frac={( tw["neut"] == 0).mean():.1%}')
    log(f'    Con stats:  mean={tw["con"].mean():.3f}, '
        f'std={tw["con"].std():.3f}, '
        f'zero_frac={(tw["con"] == 0).mean():.1%}')
    
    # ===================================================================
    # Step 2: For each asset, compute all comparison metrics
    # ===================================================================
    all_comparison_rows = []
    all_crash_rows = []
    all_stability_rows = []
    all_regime_rows = []
    
    signal_cols = {
        'Full_LRR': 'LRR_Oracle_Sen',
        'PageRank': 'PageRank_Sen',
        'HITS': 'HITS_Sen',
        'Simple': 'Simple_Sen',
        'LRR_Social': 'LRR_Social_Sen',
    }
    
    for asset_name, final in final_data.items():
        log(f'  Processing {asset_name}...')
        
        final_c = final.copy()
        final_c['time'] = pd.to_datetime(final_c['time'], errors='coerce').dt.date
        
        # --- Compute Neut-weighted and NoCon signals from raw tweets ---
        # We need reputation weights. Since all assets share the same graph,
        # we compute LRR weights once from the sensitivity suite's vectorised engine.
        tw_temp = tw.copy()
        tw_temp['time_date'] = pd.to_datetime(tw_temp['time'], errors='coerce').dt.date
        
        neut_signal_available = False
        nocon_signal_available = False
        
        try:
            from src.reputation_engine_v2 import compute_lrr_vectorised
            from src.psych_engine import calculate_omega
            
            if 'omega' not in tw_temp.columns:
                tw_temp = calculate_omega(tw_temp)
            
            sources = tw_temp['source_user'].dropna().unique().tolist()
            rt_targets = tw_temp['rt_target'].dropna().unique().tolist()
            all_mentions_list = [m for ml in tw_temp['mentions']
                               for m in ml if isinstance(ml, list)]
            unique_users = list(set(sources + rt_targets + all_mentions_list))
            
            # Compute LRR weights (same for all assets since graph is shared)
            if 'LRR_Oracle_W' not in tw_temp.columns:
                rep = compute_lrr_vectorised(
                    tw_temp, unique_users, use_omega=True, use_con=True)
                tw_temp['LRR_Oracle_W'] = tw_temp['source_user'].astype(str).map(rep).fillna(
                    1.0 / max(len(unique_users), 1))
            
            # Neut-weighted daily signal
            neut_daily = tw_temp.groupby('time_date').apply(lambda x: pd.Series({
                'Neut_LRR_Sen': np.average(
                    x['sen'].values * x['neut'].values,
                    weights=x['LRR_Oracle_W'].values + 1e-9),
            }), include_groups=False).reset_index()
            neut_daily.columns = ['time', 'Neut_LRR_Sen']
            final_c = pd.merge(final_c, neut_daily, on='time', how='left')
            neut_signal_available = 'Neut_LRR_Sen' in final_c.columns
            
            # NoCon daily signal (reputation-weighted but no Con filter)
            nocon_daily = tw_temp.groupby('time_date').apply(lambda x: pd.Series({
                'LRR_NoCon_Sen2': np.average(
                    x['sen'].values,
                    weights=x['LRR_Oracle_W'].values + 1e-9),
            }), include_groups=False).reset_index()
            nocon_daily.columns = ['time', 'LRR_NoCon_Sen2']
            final_c = pd.merge(final_c, nocon_daily, on='time', how='left')
            nocon_signal_available = 'LRR_NoCon_Sen2' in final_c.columns
            
            log(f'    Neut and NoCon signals computed successfully')
        except Exception as e:
            log(f'    ! Could not compute Neut/NoCon signals: {e}')
        
        # Build signal dictionary for this asset
        signal_cols_extended = {}
        for sig_name, sig_col in signal_cols.items():
            if sig_col in final_c.columns:
                signal_cols_extended[sig_name] = sig_col
        if neut_signal_available:
            signal_cols_extended['Neut_LRR'] = 'Neut_LRR_Sen'
        if nocon_signal_available:
            signal_cols_extended['LRR_NoCon'] = 'LRR_NoCon_Sen2'
        elif 'LRR_NoCon_Sen' in final_c.columns:
            signal_cols_extended['LRR_NoCon'] = 'LRR_NoCon_Sen'
        
        price = final_c['price_change'].values
        
        for sig_name, sig_col in signal_cols_extended.items():
            if sig_col not in final_c.columns:
                continue
            
            signal = final_c[sig_col].values
            
            # --- Joint crashes at multiple thresholds ---
            for min_ev in [2, 4, 5]:
                ltd_val, joint_n, tail_n = compute_ltd_flexible(
                    signal, price, quantile=0.10, min_events=min_ev)
                all_crash_rows.append({
                    'asset': asset_name,
                    'signal': sig_name,
                    'min_events': min_ev,
                    'ltd': round(ltd_val, 4) if not np.isnan(ltd_val) else np.nan,
                    'joint_crashes': joint_n,
                    'tail_days': tail_n,
                })
            
            # --- Signal stability during crashes ---
            stab = compute_signal_stability(signal, price)
            stab['asset'] = asset_name
            stab['signal'] = sig_name
            all_stability_rows.append(stab)
            
            # --- Crash vs normal correlation ---
            corr_metrics = compute_crash_correlation(signal, price)
            
            # --- Regime discrimination ---
            regime_disc = {'asset': asset_name, 'signal': sig_name}
            if 'regime' in final_c.columns:
                rd = compute_regime_discrimination(signal, final_c['regime'].values)
                regime_disc.update(rd)
            all_regime_rows.append(regime_disc)
            
            # --- Compile main comparison row ---
            all_comparison_rows.append({
                'asset': asset_name,
                'signal': sig_name,
                'joint_crashes_t2': all_crash_rows[-3]['joint_crashes'],  # min_events=2
                'joint_crashes_t4': all_crash_rows[-2]['joint_crashes'],  # min_events=4
                'joint_crashes_t5': all_crash_rows[-1]['joint_crashes'],  # min_events=5
                'ltd_t2': all_crash_rows[-3]['ltd'],
                'ltd_t4': all_crash_rows[-2]['ltd'],
                'crash_std': stab['crash_std'],
                'normal_std': stab['normal_std'],
                'stability_ratio': stab['stability_ratio'],
                'crash_corr': corr_metrics['crash_corr'],
                'normal_corr': corr_metrics['normal_corr'],
                'regime_d': regime_disc.get('discrimination_d', np.nan),
            })
    
    # ===================================================================
    # Step 3: Save results
    # ===================================================================
    comp_df = pd.DataFrame(all_comparison_rows)
    comp_df.to_csv(os.path.join(comp_path, 'enhanced_comparison.csv'),
                   index=False, float_format='%.4f')
    
    crash_df = pd.DataFrame(all_crash_rows)
    crash_df.to_csv(os.path.join(comp_path, 'crash_thresholds.csv'),
                    index=False, float_format='%.4f')
    
    stab_df = pd.DataFrame(all_stability_rows)
    stab_df.to_csv(os.path.join(comp_path, 'signal_stability.csv'),
                   index=False, float_format='%.4f')
    
    regime_df = pd.DataFrame(all_regime_rows)
    regime_df.to_csv(os.path.join(comp_path, 'regime_discrimination.csv'),
                     index=False, float_format='%.4f')
    
    # ===================================================================
    # Step 4: Generate comparison charts
    # ===================================================================
    log('  Generating comparison charts...')
    
    # --- Chart 1: Joint crashes across signals (all assets, threshold=2) ---
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        key_signals = ['Full_LRR', 'Neut_LRR', 'LRR_NoCon', 'PageRank', 'HITS', 'Simple']
        avail_signals = [s for s in key_signals if s in comp_df['signal'].unique()]
        
        assets = sorted(comp_df['asset'].unique())
        x = np.arange(len(assets))
        width = 0.8 / len(avail_signals)
        colors = ['#1D9E75', '#5DCAA5', '#85B7EB', '#D85A30', '#F0997B', '#B4B2A9']
        
        for i, sig in enumerate(avail_signals):
            vals = []
            for asset in assets:
                row = comp_df[(comp_df['asset'] == asset) & (comp_df['signal'] == sig)]
                vals.append(row['joint_crashes_t2'].values[0] if len(row) > 0 else 0)
            ax.bar(x + i * width, vals, width, label=sig, color=colors[i % len(colors)],
                   edgecolor='white', linewidth=0.5)
        
        ax.set_xlabel('Asset')
        ax.set_ylabel('Joint crash days (threshold ≥ 2)')
        ax.set_title('Crash decoupling: joint crashes with price (lower = better)')
        ax.set_xticks(x + width * len(avail_signals) / 2)
        ax.set_xticklabels(assets)
        ax.legend(fontsize=8, ncol=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(comp_path, 'crash_comparison_all_signals.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        log('    crash_comparison_all_signals.png saved')
    except Exception as e:
        log(f'    ! Crash comparison chart failed: {e}')
    
    # --- Chart 2: Signal stability ratio (crash_std / normal_std) ---
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, sig in enumerate(avail_signals):
            vals = []
            for asset in assets:
                row = comp_df[(comp_df['asset'] == asset) & (comp_df['signal'] == sig)]
                vals.append(row['stability_ratio'].values[0] if len(row) > 0 else np.nan)
            ax.bar(x + i * width, vals, width, label=sig, color=colors[i % len(colors)],
                   edgecolor='white', linewidth=0.5)
        
        ax.axhline(y=1.0, color='#888780', linewidth=0.8, linestyle='--', label='Equal stability')
        ax.set_xlabel('Asset')
        ax.set_ylabel('Stability ratio (crash std / normal std)')
        ax.set_title('Signal stability during crashes (lower = more stable)')
        ax.set_xticks(x + width * len(avail_signals) / 2)
        ax.set_xticklabels(assets)
        ax.legend(fontsize=8, ncol=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(comp_path, 'stability_comparison.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        log('    stability_comparison.png saved')
    except Exception as e:
        log(f'    ! Stability chart failed: {e}')
    
    # --- Chart 3: Regime discrimination (Cohen's d) ---
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, sig in enumerate(avail_signals):
            vals = []
            for asset in assets:
                row = comp_df[(comp_df['asset'] == asset) & (comp_df['signal'] == sig)]
                vals.append(row['regime_d'].values[0] if len(row) > 0 else 0)
            ax.bar(x + i * width, vals, width, label=sig, color=colors[i % len(colors)],
                   edgecolor='white', linewidth=0.5)
        
        ax.set_xlabel('Asset')
        ax.set_ylabel("Cohen's d (regime discrimination)")
        ax.set_title('Regime discrimination power (higher = better bull/bear separation)')
        ax.set_xticks(x + width * len(avail_signals) / 2)
        ax.set_xticklabels(assets)
        ax.legend(fontsize=8, ncol=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(comp_path, 'regime_discrimination.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        log('    regime_discrimination.png saved')
    except Exception as e:
        log(f'    ! Regime discrimination chart failed: {e}')
    
    # --- Chart 4: Crash correlation vs normal correlation ---
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for ax_idx, (period, col) in enumerate([('During crashes', 'crash_corr'),
                                                  ('Normal periods', 'normal_corr')]):
            ax = axes[ax_idx]
            for i, sig in enumerate(avail_signals):
                vals = []
                for asset in assets:
                    row = comp_df[(comp_df['asset'] == asset) & (comp_df['signal'] == sig)]
                    vals.append(row[col].values[0] if len(row) > 0 else 0)
                ax.bar(x + i * width, vals, width, label=sig, color=colors[i % len(colors)],
                       edgecolor='white', linewidth=0.5)
            
            ax.set_xlabel('Asset')
            ax.set_ylabel('Pearson r (signal vs price)')
            ax.set_title(f'Signal-price correlation: {period}')
            ax.set_xticks(x + width * len(avail_signals) / 2)
            ax.set_xticklabels(assets)
            if ax_idx == 0:
                ax.legend(fontsize=7, ncol=2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(comp_path, 'crash_vs_normal_correlation.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        log('    crash_vs_normal_correlation.png saved')
    except Exception as e:
        log(f'    ! Correlation chart failed: {e}')
    
    # --- Chart 5: Con vs Neut direct comparison ---
    try:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        
        con_data = comp_df[comp_df['signal'] == 'Full_LRR']
        neut_data = comp_df[comp_df['signal'] == 'Neut_LRR']
        
        if len(neut_data) > 0:
            # Panel 1: Joint crashes
            ax = axes[0]
            con_crashes = [con_data[con_data['asset'] == a]['joint_crashes_t2'].values[0] 
                          for a in assets if len(con_data[con_data['asset'] == a]) > 0]
            neut_crashes = [neut_data[neut_data['asset'] == a]['joint_crashes_t2'].values[0]
                           for a in assets if len(neut_data[neut_data['asset'] == a]) > 0]
            x2 = np.arange(len(assets))
            ax.bar(x2 - 0.2, con_crashes, 0.35, label='LRR + Con', color='#1D9E75')
            ax.bar(x2 + 0.2, neut_crashes, 0.35, label='LRR + Neut', color='#378ADD')
            ax.set_xticks(x2)
            ax.set_xticklabels(assets)
            ax.set_ylabel('Joint crashes')
            ax.set_title('Crash decoupling')
            ax.legend(fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Panel 2: Stability ratio
            ax = axes[1]
            con_stab = [con_data[con_data['asset'] == a]['stability_ratio'].values[0]
                       for a in assets if len(con_data[con_data['asset'] == a]) > 0]
            neut_stab = [neut_data[neut_data['asset'] == a]['stability_ratio'].values[0]
                        for a in assets if len(neut_data[neut_data['asset'] == a]) > 0]
            ax.bar(x2 - 0.2, con_stab, 0.35, label='LRR + Con', color='#1D9E75')
            ax.bar(x2 + 0.2, neut_stab, 0.35, label='LRR + Neut', color='#378ADD')
            ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)
            ax.set_xticks(x2)
            ax.set_xticklabels(assets)
            ax.set_ylabel('Stability ratio')
            ax.set_title('Crash stability (lower=better)')
            ax.legend(fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Panel 3: Regime discrimination
            ax = axes[2]
            con_rd = [con_data[con_data['asset'] == a]['regime_d'].values[0]
                     for a in assets if len(con_data[con_data['asset'] == a]) > 0]
            neut_rd = [neut_data[neut_data['asset'] == a]['regime_d'].values[0]
                      for a in assets if len(neut_data[neut_data['asset'] == a]) > 0]
            ax.bar(x2 - 0.2, con_rd, 0.35, label='LRR + Con', color='#1D9E75')
            ax.bar(x2 + 0.2, neut_rd, 0.35, label='LRR + Neut', color='#378ADD')
            ax.set_xticks(x2)
            ax.set_xticklabels(assets)
            ax.set_ylabel("Cohen's d")
            ax.set_title('Regime discrimination (higher=better)')
            ax.legend(fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(comp_path, 'con_vs_neut_comparison.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        log('    con_vs_neut_comparison.png saved')
    except Exception as e:
        log(f'    ! Con vs Neut chart failed: {e}')
    
    # ===================================================================
    # Step 5: Summary report
    # ===================================================================
    with open(os.path.join(comp_path, 'COMPARISON_SUMMARY.txt'), 'w', encoding='utf-8') as f:
        f.write('=== Enhanced LRR Comparison Suite ===\n\n')
        
        f.write('--- Neut Gate Statistics ---\n')
        f.write(f'Neut mean: {tw["neut"].mean():.4f}\n')
        f.write(f'Neut zero fraction: {(tw["neut"] == 0).mean():.1%}\n')
        f.write(f'Con mean: {tw["con"].mean():.4f}\n')
        f.write(f'Con zero fraction: {(tw["con"] == 0).mean():.1%}\n\n')
        
        f.write('--- Cross-Signal Comparison (mean across assets) ---\n')
        f.write(f'{"Signal":<15} {"Crashes_t2":>10} {"Crashes_t5":>10} '
                f'{"StabRatio":>10} {"CrashCorr":>10} {"RegimeD":>10}\n')
        f.write('-' * 70 + '\n')
        
        for sig in comp_df['signal'].unique():
            sub = comp_df[comp_df['signal'] == sig]
            f.write(f'{sig:<15} '
                    f'{sub["joint_crashes_t2"].mean():>10.1f} '
                    f'{sub["joint_crashes_t5"].mean():>10.1f} '
                    f'{sub["stability_ratio"].mean():>10.3f} '
                    f'{sub["crash_corr"].mean():>10.3f} '
                    f'{sub["regime_d"].mean():>10.3f}\n')
        
        f.write('\n--- Interpretation ---\n')
        f.write('Crashes_t2: Joint crash count with min_events=2 (lower=better)\n')
        f.write('Crashes_t5: Joint crash count with min_events=5 (lower=better)\n')
        f.write('StabRatio: crash_std/normal_std (lower=more stable during crashes)\n')
        f.write('CrashCorr: signal-price correlation during crashes (lower=more independent)\n')
        f.write('RegimeD: Cohen\'s d for bull vs bear signal means (higher=better discrimination)\n')
    
    log('  COMPARISON_SUMMARY.txt saved')
    log('>>> Enhanced comparison complete.')
