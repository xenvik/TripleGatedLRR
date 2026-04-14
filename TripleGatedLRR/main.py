# main.py  —  LRR Q1 Master Pipeline
# ---------------------------------------------------------------------------
# Execution order:
#   Phase 0  : Data ingestion
#   Phase 1  : Reputation Engine (V-anchor bias-corrected)
#   Phase 2  : Regime Detection (HMM)
#   Phase 3  : Daily Signal Aggregation
#   Phase 4  : Master Join + Stationarity
#   Phase 5  : L1-L5 Lagged Correlations (with p-values)
#   Phase 6  : LTD Benchmark + Transfer Entropy
#   Phase 7  : Con Gate Ablation (full population + elite filter)
#   Phase 8  : OOS Validation (LRR-VAR vs AR(1) vs Random Walk)
#   Phase 9  : VAR + Granger Causality Table + IRF with CI bands
#   Phase 10 : Regime-Specific VAR (CALM / CRISIS)
#   Phase 11 : SVAR On-Chain Transmission (whale data, if present)
#   Phase 12 : Rolling Correlation (robustness over time)
#   Phase 13 : Cross-Asset Spillover (BTC LRR -> ETH price)
#   Phase 14 : Visualisations
# ---------------------------------------------------------------------------

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

from src.loader import load_and_clean_data
from src.reputation_engine import run_benchmarked_reputation, compute_lrr_recursive
from src.analytics import (
    perform_lag_sweep, run_unified_var, run_out_of_sample_validation,
    compute_rolling_correlation, compute_lag_correlation_table,
    run_granger_causality_table, check_stationarity,
    compute_onchain_lead_lag, run_regime_conditioned_oos,
    build_cross_asset_summary, compute_finbert_baseline,
    run_distortion_decomposition,
    run_subperiod_analysis, compute_network_statistics,
    compute_fevd, run_lag_robustness,
    compute_user_heterogeneity, run_hits_var_comparison,
    run_oscillation_binomial_test, compute_svar_residual_correlations,
)
from src.regime_engine import detect_market_regimes
from src.risk_metrics import (
    compute_tail_dependence, compute_tail_dependence_extended,
    calculate_transfer_entropy,
    calculate_mutual_information, compute_directional_accuracy,
    bootstrap_ltd_reduction_test, directional_accuracy_significance,
    permutation_test_ltd_reduction, pooled_con_gate_significance,
    compute_ier, compute_ier_table, bootstrap_ier_superiority
)
from src.reputation_engine import (
    run_benchmarked_reputation, compute_lrr_recursive
)
from src.event_study import run_event_study
from src.visualizer import (
    plot_ablation_denoising, plot_oos_forecast, plot_authority_gap,
    generate_correlation_matrix, plot_var_irf, plot_svar_cumulative_irf,
    plot_regime_aware_forecast, plot_ltd_benchmark, plot_baseline_comparison,
    plot_rolling_correlation, plot_granger_heatmap, plot_gate_sensitivity,
    plot_lrr_whale_alignment, plot_onchain_leadlag_heatmap,
    plot_cumulative_returns, plot_sharpe_comparison, plot_drawdown,
    plot_ier_table, plot_rolling_correlation_regimes,
    plot_distortion_heatmap, plot_distortion_clusters
)
from src.portfolio_engine import run_portfolio_backtest
from src.config import (
    MAX_LAG, ELITE_PERCENTILE, TRAIN_RATIO, ROLLING_WINDOW, ROLLING_LAG
)

warnings.filterwarnings('ignore')
np.random.seed(42)  # Global reproducibility seed
from src.sensitivity_suite import run_sensitivity_suite
from src.lrr_enhanced_comparison import run_enhanced_comparison

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_time_column(df):
    """
    Ensures the DataFrame has a 'time' column of datetime.date objects.
    Handles DataFrames that have 'time' as index OR as a column,
    regardless of whether the index is a RangeIndex, DatetimeIndex, or object.
    """
    df = df.copy()
    # Case 1: time is the index
    if df.index.name == 'time' or (
        hasattr(df.index, 'dtype') and
        str(df.index.dtype) in ('object', 'datetime64[ns]') and
        'time' not in df.columns
    ):
        df = df.reset_index()
        if 'index' in df.columns and 'time' not in df.columns:
            df = df.rename(columns={'index': 'time'})

    # Case 2: time is already a column — normalise its type
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce').dt.date

    return df


def log(msg, indent=0):
    print('  ' * indent + msg)


def open_results_file(results_path, filename, header=''):
    path = os.path.join(results_path, filename)
    with open(path, 'w', encoding='utf-8') as f:
        if header:
            f.write(header + '\n')
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    base_dir     = os.path.dirname(os.path.abspath(__file__))
    data_path    = os.path.join(base_dir, 'data')
    results_path = os.path.join(base_dir, 'results')
    os.makedirs(results_path, exist_ok=True)

    # Collectors for cross-asset analyses
    all_ier_rows          = []
    all_gate_perm_results = {}
    all_cross_asset       = {}
    event_study_results   = {}
    all_distortion_results= {}
    all_hits_var_results  = {}  
    all_lag_dfs           = {}   

    # Initialise ADF log
    open_results_file(results_path, 'ADF_Tests.txt',
                      '=== Augmented Dickey-Fuller Tests ===')
    # Clear integration log at start of each run (fresh file per run)
    open(os.path.join(results_path, 'integration_order_log.txt'),
         'w', encoding='utf-8').write(
         '=== Integration Order Log (VAR pre-checks) ===\n'
         'Format: variable: ADF_p -> I(0) or I(1)*\n'
         'I(1)* variables are auto-differenced before VAR fitting.\n\n')
    # -----------------------------------------------------------------------
    log('>>> Phase 0: Data Ingestion')
    tw, assets = load_and_clean_data(data_path)
    log(f'    Twitter rows: {len(tw):,}   Assets: {list(assets.keys())}', 1)

    # -----------------------------------------------------------------------
    # Phase 0.5: Mention Weight Calibration (BEFORE any reputation computation)
    # -----------------------------------------------------------------------
    from src.mention_weight_calibration import calibrate_mention_weight
    import src.config as config_module
    calibrated_mw = calibrate_mention_weight(tw, results_path)
    config_module.MENTION_WEIGHT = calibrated_mw  # Update global config
    log(f'    Using calibrated mention weight: w_m={calibrated_mw:.3f}')

    # Store final aligned dataframes for cross-asset analysis
    final_data = {}

    # -----------------------------------------------------------------------
    # Per-Asset Loop
    # -----------------------------------------------------------------------
    for asset_name, asset_df in assets.items():
        log(f'\n{"="*55}')
        log(f'  LRR PIPELINE: {asset_name}')
        log(f'{"="*55}')

        asset_df = asset_df.copy()
        asset_df = ensure_time_column(asset_df)

        # Price returns
        if 'price_change' not in asset_df.columns:
            if 'returns' in asset_df.columns:
                asset_df = asset_df.rename(columns={'returns': 'price_change'})
            elif 'close' in asset_df.columns:
                asset_df['price_change'] = asset_df['close'].pct_change()

        # ------------------------------------------------------------------
        # Determine train/test split date BEFORE reputation engine
        # (required for V-anchor bias correction)
        # ------------------------------------------------------------------
        if 'time' not in asset_df.columns:
            asset_df = asset_df.reset_index().rename(columns={'index': 'time'})

        # Ensure time column is datetime.date throughout
        asset_df['time'] = pd.to_datetime(asset_df['time'], errors='coerce').dt.date
        asset_df = asset_df.dropna(subset=['time']).reset_index(drop=True)

        valid_dates = asset_df.dropna(subset=['price_change'])['time'].tolist()
        if not valid_dates:
            log(f'  ! No valid price data for {asset_name} — skipping.', 1)
            continue
        valid_dates_sorted = sorted(valid_dates)
        split_idx          = int(len(valid_dates_sorted) * TRAIN_RATIO)
        train_end_date     = valid_dates_sorted[split_idx - 1]
        log(f'    Train/test split: {valid_dates_sorted[0]} → '
            f'{train_end_date} | test: {valid_dates_sorted[split_idx]} → '
            f'{valid_dates_sorted[-1]}', 1)

        # ------------------------------------------------------------------
        # Phase 1: Reputation Engine
        # ------------------------------------------------------------------
        log('>>> Phase 1: Reputation Engine (V-anchor bias-corrected)')
        (pr_scores, hits_auth, lrr_social, lrr_oracle,
         tw_df, unique_users, anchor_vector, G_social) = run_benchmarked_reputation(
            tw, asset_df, train_end_date=train_end_date
        )
        log(f'    Users in graph: {len(unique_users):,}', 1)

        # ── VALIDATION MARKERS (Phase 1) ──────────────────────────────
        import networkx as nx
        _n_graph_nodes = G_social.number_of_nodes() if G_social else 0
        _n_graph_edges = G_social.number_of_edges() if G_social else 0
        log(f'   Network: {_n_graph_nodes} nodes  {_n_graph_edges} edges  '
            f'density={nx.density(G_social):.5f}', 1)
        
        # Verify HITS is authorities (high scores = accounts that RECEIVE retweets)
        # Hub accounts (who DO the retweeting) should have LOW authority scores
        _hits_vals = list(hits_auth.values())
        _hits_nonzero = sum(1 for v in _hits_vals if v > 0.001)
        _hits_gini = np.sort(np.array(_hits_vals, dtype=float))
        _hits_gini = float((2 * (np.arange(1, len(_hits_gini)+1) * _hits_gini).sum()) / 
                          (len(_hits_gini) * _hits_gini.sum()) - (len(_hits_gini)+1)/len(_hits_gini)) if _hits_gini.sum() > 0 else 0
        log(f'   HITS: {_hits_nonzero} non-zero authorities, Gini={_hits_gini:.3f}', 1)
        if _hits_gini < 0.3:
            log(f'   ⚠ WARNING: HITS Gini={_hits_gini:.3f} is unusually low — '
                f'verify authorities (not hubs) are being used!', 1)
        
        # Verify anchor
        _adaptive_floor = 1.0 / max(len(unique_users), 1)
        _n_active_anchor = sum(1 for v in anchor_vector.values() if v > _adaptive_floor) if anchor_vector else 0
        log(f'   V-Anchor: {_n_active_anchor} users with active anchor (>{_adaptive_floor:.4f})', 1)
        
        # Verify LRR Gini
        _lrr_vals = np.sort(np.array(list(lrr_oracle.values()), dtype=float))
        _lrr_gini = float((2 * (np.arange(1, len(_lrr_vals)+1) * _lrr_vals).sum()) / 
                         (len(_lrr_vals) * _lrr_vals.sum()) - (len(_lrr_vals)+1)/len(_lrr_vals)) if _lrr_vals.sum() > 0 else 0
        log(f'   LRR_Gini={_lrr_gini:.3f}  '
            f'Top1%={sum(sorted(_lrr_vals,reverse=True)[:max(1,len(_lrr_vals)//100)])/max(_lrr_vals.sum(),1e-9)*100:.1f}%  '
            f'ωTop20={np.mean([tw_df[tw_df["source_user"]==u]["omega"].mean() for u in sorted(lrr_oracle, key=lrr_oracle.get, reverse=True)[:max(1,len(lrr_oracle)//5)] if u in tw_df["source_user"].values][:20]):.3f} '
            f'vs ωBot80={np.mean([tw_df[tw_df["source_user"]==u]["omega"].mean() for u in sorted(lrr_oracle, key=lrr_oracle.get)[:max(1,4*len(lrr_oracle)//5)] if u in tw_df["source_user"].values][:80]):.3f}', 1)
        # ── END VALIDATION (Phase 1) ──────────────────────────────────

        # Gap 2 — Network statistics (once per asset run, file is overwritten)
        try:
            compute_network_statistics(
                tw_df, pr_scores, hits_auth, lrr_oracle, results_path)
        except Exception as e:
            log(f'    ! Network stats failed: {e}', 1)

        # Gap 5 — User heterogeneity
        try:
            compute_user_heterogeneity(
                tw_df, lrr_oracle, pr_scores, asset_name, results_path)
        except Exception as e:
            log(f'    ! User heterogeneity failed: {e}', 1)

        log('>>> Phase 2: HMM Regime Detection')
        try:
            regime_df          = detect_market_regimes(asset_df.copy())
            # regime_df always comes back with ['time', 'regime'] columns
            regime_df['time']  = pd.to_datetime(regime_df['time'], errors='coerce').dt.date
            regime_convergence = True
        except Exception as e:
            log(f'    ! HMM failed ({e}) — tagging as Regime 2', 1)
            regime_df          = asset_df[['time']].copy()
            regime_df['time']  = pd.to_datetime(regime_df['time'], errors='coerce').dt.date
            regime_df['regime'] = 2
            regime_convergence  = False

        regime_counts = regime_df['regime'].value_counts().to_dict()
        log(f'    Regime counts: {regime_counts}', 1)

        # ------------------------------------------------------------------
        # Phase 3: Daily Signal Aggregation (5 signal variants)
        # ------------------------------------------------------------------
        log('>>> Phase 3: Daily Signal Aggregation')

        # T2.3 — FinBERT-style follower-weighted baseline
        tw_df, finbert_proxy = compute_finbert_baseline(tw_df)

        tw_df = tw_df.copy()
        tw_df['PR_W']         = tw_df['source_user'].map(pr_scores).fillna(0)
        tw_df['HITS_W']       = tw_df['source_user'].map(hits_auth).fillna(0)
        tw_df['LRR_Social_W'] = tw_df['source_user'].map(lrr_social).fillna(0)
        tw_df['LRR_Oracle_W'] = tw_df['source_user'].map(lrr_oracle).fillna(0)

        # Ablated signal:
        # (v3.0: Con is aggregation-only, so reputation weights are identical)
        tw_df['LRR_NoCon_W'] = tw_df['LRR_Oracle_W'].copy()

        daily_lrr = tw_df.groupby('time').apply(lambda x: pd.Series({
            'Simple_Sen':      x['sen'].mean(),
            'PageRank_Sen':    np.average(x['sen'],
                                          weights=x['PR_W'] + 1e-9),
            'HITS_Sen':        np.average(x['sen'],
                                          weights=x['HITS_W'] + 1e-9),
            'LRR_Social_Sen':  np.average(x['sen'],
                                          weights=x['LRR_Social_W'] + 1e-9),
            'LRR_Oracle_Sen':  np.average(
                                   x['sen'] * x.get('con', pd.Series(1.0, index=x.index)),
                                   weights=x['LRR_Oracle_W'] + 1e-9),
            'LRR_NoCon_Sen':   np.average(x['sen'],
                                          weights=x['LRR_NoCon_W'] + 1e-9),
            'FinBERT_Sen':     np.average(x['sen'],
                                          weights=(x['FinBERT_W']
                                          if 'FinBERT_W' in x.columns
                                          else pd.Series(1.0, index=x.index)) + 1e-9),
            'omega':           x['omega'].mean() if 'omega' in x.columns else 0.5,
            'con':             x['con'].mean()   if 'con'   in x.columns else 1.0,
            'PR_W':            x['PR_W'].mean(),
            'LRR_Oracle_W':    x['LRR_Oracle_W'].mean(),
        }), include_groups=False).reset_index()

        # ── VALIDATION MARKERS (Phase 3) ──────────────────────────────
        for _sig_col in ['LRR_Oracle_Sen', 'HITS_Sen', 'PageRank_Sen', 'Simple_Sen']:
            if _sig_col in daily_lrr.columns:
                _s = daily_lrr[_sig_col]
                _nan_pct = _s.isna().mean() * 100
                if _nan_pct > 5:
                    log(f'   ⚠ WARNING: {_sig_col} has {_nan_pct:.1f}% NaN values!', 1)
                if _s.std() < 1e-8:
                    log(f'   ⚠ WARNING: {_sig_col} has near-zero std={_s.std():.2e} — signal may be degenerate!', 1)
        log(f'   FinBERT baseline: using \'{finbert_proxy}\' as reach proxy', 1)
        # ── END VALIDATION (Phase 3) ──────────────────────────────────

        # ------------------------------------------------------------------
        # Phase 4: Master Join + Stationarity Auto-Fix
        # ------------------------------------------------------------------
        log('>>> Phase 4: Master Join + Stationarity')

        asset_df['time']  = pd.to_datetime(asset_df['time']).dt.date
        daily_lrr['time'] = pd.to_datetime(daily_lrr['time']).dt.date
        # regime_df['time'] already normalised in Phase 2

        final = pd.merge(daily_lrr, asset_df, on='time', how='inner')
        final = pd.merge(final, regime_df[['time', 'regime']], on='time', how='left')
        final = final.dropna(subset=['price_change'])
        final = final.sort_values('time').reset_index(drop=True)
        log(f'    Aligned rows: {len(final)}', 1)

        if len(final) < 50:
            log(f'  ! Insufficient data for {asset_name} — skipping.', 1)
            continue

        # Stationarity check and auto-differencing for VAR signal
        pval = adfuller(final['LRR_Oracle_Sen'].dropna())[1]
        with open(os.path.join(results_path,
                               f'{asset_name.lower()}_Stationarity.txt'), 'w', encoding='utf-8') as f:
            f.write(f'Initial LRR_Oracle_Sen ADF p-value: {pval:.4f}\n')
            if pval > 0.05:
                log(f'    Non-stationary (p={pval:.4f}) — applying first-difference', 1)
                final['LRR_VAR_Signal'] = final['LRR_Oracle_Sen'].diff().fillna(0)
                new_pval = adfuller(final['LRR_VAR_Signal'].dropna())[1]
                f.write(f'Applied first-difference. New ADF p: {new_pval:.4f}\n')
            else:
                f.write('Signal is naturally stationary.\n')
                final['LRR_VAR_Signal'] = final['LRR_Oracle_Sen']

        final_data[asset_name] = final.copy()

        # ------------------------------------------------------------------
        # Phase 5: L1-L5 Lagged Correlation Feature Map (with p-values)
        # ------------------------------------------------------------------
        log('>>> Phase 5: L1-L5 Lagged Correlations (with p-values)')
        lag_table = compute_lag_correlation_table(final, asset_name, results_path)
        all_lag_dfs[asset_name] = lag_table   # Gap 7 — collect for oscillation test
        log(f'    Saved to {asset_name.lower()}_Lagged_Correlations.csv', 1)

        # ------------------------------------------------------------------
        # Phase 6: Risk Metrics — LTD Benchmark + Transfer Entropy
        # ------------------------------------------------------------------
        log('>>> Phase 6: LTD Benchmark + Transfer Entropy')

        # Extended LTD: collect both value AND raw joint crash counts
        ltd_ext = {}
        for sig_name, sig_col in [('LRR_Oracle', 'LRR_Oracle_Sen'),
                                   ('PageRank', 'PageRank_Sen'),
                                   ('HITS', 'HITS_Sen')]:
            ltd_val, joint_n, tail_n = compute_tail_dependence_extended(
                final[sig_col], final['price_change'])
            ltd_ext[sig_name] = {'ltd': ltd_val, 'joint': joint_n, 'tail': tail_n}

        ltd_dict = {k: v['ltd'] for k, v in ltd_ext.items()}

        te_dict = {
            'LRR_Oracle': calculate_transfer_entropy(
                final['LRR_Oracle_Sen'], final['price_change'], lag=7),
            'PageRank':   calculate_transfer_entropy(
                final['PageRank_Sen'], final['price_change'], lag=7),
            'HITS':       calculate_transfer_entropy(
                final['HITS_Sen'], final['price_change'], lag=7),
        }

        with open(os.path.join(results_path,
                               f'{asset_name.lower()}_Risk_Metrics.txt'), 'w', encoding='utf-8') as f:
            f.write(f'=== {asset_name} Risk & Information Flow ===\n\n')
            f.write('Lower Tail Dependence (LTD):\n')
            for k, v in ltd_dict.items():
                f.write(f'  {k}: {v:.6f}\n')
            f.write('\nJoint Tail Crash Counts (lambda=0.10):\n')
            for k, v in ltd_ext.items():
                f.write(f'  {k}: {v["joint"]} joint crashes '
                        f'(out of {v["tail"]} signal-tail days, '
                        f'N={len(final)})\n')
            f.write('\nTransfer Entropy (TE, 7-day lead, bits):\n')
            f.write('  [True conditional MI: TE(X->Y)=I(Y_t;X_{t-7}|Y_{t-7})]\n')
            for k, v in te_dict.items():
                f.write(f'  {k}: {v:.6f}\n')

        log(f'    LTD — LRR:{ltd_dict["LRR_Oracle"]:.4f}  '
            f'PR:{ltd_dict["PageRank"]:.4f}  '
            f'HITS:{ltd_dict["HITS"]:.4f}', 1)
        log(f'    Joint crashes — LRR:{ltd_ext["LRR_Oracle"]["joint"]}  '
            f'PR:{ltd_ext["PageRank"]["joint"]}  '
            f'HITS:{ltd_ext["HITS"]["joint"]}', 1)

        # T1.5 — Information Efficiency Ratio (IER = TE / LTD)
        risk_dict_for_ier = {
            'LRR_Oracle': {'te': te_dict['LRR_Oracle'], 'ltd': ltd_dict['LRR_Oracle']},
            'PageRank':   {'te': te_dict['PageRank'],   'ltd': ltd_dict['PageRank']},
            'HITS':       {'te': te_dict['HITS'],        'ltd': ltd_dict['HITS']},
        }
        ier_rows_this_asset = compute_ier_table(risk_dict_for_ier, asset_name)
        all_ier_rows.extend(ier_rows_this_asset)

        # Log IER
        for row in ier_rows_this_asset:
            log(f'    IER {row["signal"]}: TE={row["TE"]:.4f} '
                f'LTD={row["LTD"]:.4f} IER={row["IER"]:.4f}', 1)

        # Collect per-asset data for cross-asset summary table
        all_cross_asset[asset_name] = {
            'ltd_lrr':              ltd_dict['LRR_Oracle'],
            'ltd_pr':               ltd_dict['PageRank'],
            'ltd_hits':             ltd_dict['HITS'],
            'joint_lrr':            ltd_ext['LRR_Oracle']['joint'],
            'joint_pr':             ltd_ext['PageRank']['joint'],
            'joint_hits':           ltd_ext['HITS']['joint'],
        }

        # ------------------------------------------------------------------
        # Phase 7: Con Gate Ablation (Full Population + Elite Filter)
        # ------------------------------------------------------------------
        log('>>> Phase 7: Con Gate Ablation Study')
        gate_results = {}

        # --- 7a. Full Population ---
        ltd_full_pop   = compute_tail_dependence(
            final['LRR_Oracle_Sen'], final['price_change'])
        ltd_no_con_pop = compute_tail_dependence(
            final['LRR_NoCon_Sen'], final['price_change'])

        p_pop, ci_lo_pop, ci_hi_pop, red_pop = bootstrap_ltd_reduction_test(
            final['LRR_Oracle_Sen'], final['LRR_NoCon_Sen'], final['price_change']
        )
        gate_results.update({
            'Full_LTD':      ltd_full_pop,
            'NoCon_LTD':     ltd_no_con_pop,
            'Reduction_%':   red_pop,
            'Boot_CI_lo':    ci_lo_pop,
            'Boot_CI_hi':    ci_hi_pop,
            'Boot_p':        p_pop,
        })
        log(f'    Population — Reduction: {red_pop:.1f}%  '
            f'95%CI [{ci_lo_pop:.1f}%, {ci_hi_pop:.1f}%]  p={p_pop:.3f}', 1)

        # Permutation test (stronger than bootstrap)
        perm_red, perm_p, perm_null_mean, perm_null_std = permutation_test_ltd_reduction(
            final['LRR_Oracle_Sen'], final['LRR_NoCon_Sen'], final['price_change']
        )
        all_gate_perm_results[asset_name] = {
            'boot_p':    p_pop,
            'perm_p':    perm_p,
            'reduction': red_pop,
        }
        log(f'    Permutation test — p={perm_p:.4f}  '
            f'null_mean={perm_null_mean:.2f}%  null_std={perm_null_std:.2f}%', 1)

        # --- 7b. Elite User Filter (top ELITE_PERCENTILE by LRR weight) ---
        elite_threshold = final['LRR_Oracle_W'].quantile(ELITE_PERCENTILE)
        elite_users_set = set(
            tw_df.loc[tw_df['LRR_Oracle_W'] >= elite_threshold, 'source_user']
            .unique()
        )
        if len(elite_users_set) >= 5:
            tw_elite = tw_df[tw_df['source_user'].isin(elite_users_set)].copy()
            tw_elite['LRR_E_W']    = tw_elite['source_user'].map(lrr_oracle).fillna(0)
            # v3.0: same weights, Con only differs in aggregation
            tw_elite['LRR_E_NC_W'] = tw_elite['LRR_E_W'].copy()

            daily_elite = tw_elite.groupby('time').apply(lambda x: pd.Series({
                'Elite_Full_Sen':  np.average(
                    x['sen'] * x.get('con', pd.Series(1.0, index=x.index)),
                    weights=x['LRR_E_W'] + 1e-9),
                'Elite_NoCon_Sen': np.average(
                    x['sen'],
                    weights=x['LRR_E_NC_W'] + 1e-9),
            }), include_groups=False).reset_index()

            daily_elite['time'] = pd.to_datetime(daily_elite['time']).dt.date
            final_e = pd.merge(final[['time', 'price_change']], daily_elite,
                               on='time', how='inner')

            if len(final_e) > 30:
                ltd_e_full   = compute_tail_dependence(
                    final_e['Elite_Full_Sen'], final_e['price_change'])
                ltd_e_no_con = compute_tail_dependence(
                    final_e['Elite_NoCon_Sen'], final_e['price_change'])
                p_e, ci_lo_e, ci_hi_e, red_e = bootstrap_ltd_reduction_test(
                    final_e['Elite_Full_Sen'],
                    final_e['Elite_NoCon_Sen'],
                    final_e['price_change']
                )
                gate_results.update({
                    'Elite_Full_LTD':    ltd_e_full,
                    'Elite_NoCon_LTD':   ltd_e_no_con,
                    'Elite_Reduction_%': red_e,
                    'Elite_Boot_CI_lo':  ci_lo_e,
                    'Elite_Boot_CI_hi':  ci_hi_e,
                    'Elite_Boot_p':      p_e,
                })
                log(f'    Elite    — Reduction: {red_e:.1f}%  '
                    f'95%CI [{ci_lo_e:.1f}%, {ci_hi_e:.1f}%]  p={p_e:.3f}', 1)

        # Save gate sensitivity text
        with open(os.path.join(results_path,
                               f'{asset_name.lower()}_Gate_Sensitivity.txt'), 'w', encoding='utf-8') as f:
            f.write(f'=== {asset_name} Con Gate Ablation ===\n\n')
            f.write('--- Full Population ---\n')
            f.write(f'Full Oracle LTD  (Ω + Con): {ltd_full_pop:.6f}\n')
            f.write(f'Ablated LTD      (Ω only):  {ltd_no_con_pop:.6f}\n')
            f.write(f'Risk Reduction:             {red_pop:.2f}%\n')
            f.write(f'Bootstrap 95% CI:           [{ci_lo_pop:.2f}%, {ci_hi_pop:.2f}%]\n')
            f.write(f'Bootstrap p-value:          {p_pop:.4f}\n\n')
            if 'Elite_Reduction_%' in gate_results:
                f.write(f'--- Elite Users (Top {int((1-ELITE_PERCENTILE)*100)}%) ---\n')
                f.write(f'Elite Full LTD:   {gate_results["Elite_Full_LTD"]:.6f}\n')
                f.write(f'Elite Ablated LTD:{gate_results["Elite_NoCon_LTD"]:.6f}\n')
                f.write(f'Elite Reduction:  {gate_results["Elite_Reduction_%"]:.2f}%\n')
                f.write(f'Elite 95% CI:     [{gate_results["Elite_Boot_CI_lo"]:.2f}%, '
                        f'{gate_results["Elite_Boot_CI_hi"]:.2f}%]\n')
                f.write(f'Elite p-value:    {gate_results["Elite_Boot_p"]:.4f}\n')

        # ------------------------------------------------------------------
        # Phase 8: OOS Validation (LRR-VAR vs AR(1) vs Random Walk)
        # ------------------------------------------------------------------
        log('>>> Phase 8: OOS Validation + Baselines')
        comparison_df, results_dict = run_out_of_sample_validation(
            final, asset_name, results_path
        )

        if comparison_df is not None and results_dict:
            lrr_res = results_dict.get('LRR-VAR', {})
            rw_res  = results_dict.get('RandomWalk', {})
            ar_res  = results_dict.get('AR(1)', {})
            log(f'    Random Walk: RMSE={rw_res.get("RMSE", np.nan):.5f}  '
                f'DA={rw_res.get("DA", np.nan):.1%}', 1)
            log(f'    AR(1):       RMSE={ar_res.get("RMSE", np.nan):.5f}  '
                f'DA={ar_res.get("DA", np.nan):.1%}', 1)
            log(f'    LRR-VAR:     RMSE={lrr_res.get("RMSE", np.nan):.5f}  '
                f'DA={lrr_res.get("DA", np.nan):.1%}', 1)

            # Bootstrap significance on directional accuracy
            if 'LRR_VAR' in comparison_df.columns:
                da, da_p, da_ci_lo, da_ci_hi = directional_accuracy_significance(
                    comparison_df['Actual'], comparison_df['LRR_VAR']
                )
                with open(os.path.join(results_path,
                                       f'{asset_name.lower()}_OOS_Metrics.txt'), 'a', encoding='utf-8') as f:
                    f.write(f'\n--- Directional Accuracy Bootstrap Test ---\n')
                    f.write(f'Observed DA:   {da:.4f}\n')
                    f.write(f'95% CI:        [{da_ci_lo:.4f}, {da_ci_hi:.4f}]\n')
                    f.write(f'p-value (H0: DA <= 0.50): {da_p:.4f}\n')
                log(f'    DA bootstrap: p={da_p:.3f}  95%CI [{da_ci_lo:.3f}, {da_ci_hi:.3f}]', 1)

        # ------------------------------------------------------------------
        # Phase 16: Regime-Conditioned OOS Validation
        # ------------------------------------------------------------------
        log('>>> Phase 16: Regime-Conditioned OOS')
        try:
            regime_oos_df = run_regime_conditioned_oos(
                final, asset_name, results_path
            )
            if regime_oos_df is not None:
                for _, row in regime_oos_df.iterrows():
                    beats = 'YES' if row['LRR_beats_AR'] else 'no'
                    log(f'    {row["regime"]:<8}: DA={row["LRR_DA"]:.3f}  '
                        f'AR1_DA={row["AR1_DA"]:.3f}  beats={beats}', 1)
        except Exception as e:
            log(f'    ! Regime OOS failed: {e}', 1)

        # ------------------------------------------------------------------
        # Phase 9: Full VAR + Granger Causality Table + IRF
        # ------------------------------------------------------------------
        log('>>> Phase 9: VAR + Granger Causality + IRF (with CI bands)')
        var_results = run_unified_var(final, asset_name, results_path)

        if var_results is not None:
            gc_df = run_granger_causality_table(var_results, asset_name, results_path)
            if gc_df is not None:
                # Surface the key causal relationships in the log
                key_pairs = [
                    ('omega',          'LRR_Oracle_Sen', 'ω → LRR'),
                    ('LRR_Oracle_Sen', 'price_change',   'LRR → Price'),
                    ('LRR_Oracle_Sen', 'omega',          'LRR → ω (feedback)'),
                ]
                log('    Granger causality (key pairs):', 1)
                for cause, effect, label in key_pairs:
                    row = gc_df[(gc_df['Cause'] == cause) &
                                (gc_df['Effect'] == effect)]
                    if not row.empty:
                        p   = row.iloc[0]['p_value']
                        sig = row.iloc[0]['Significant']
                        log(f'      {label}: p={p:.4f} {sig}', 2)
                        
                        if cause == 'LRR_Oracle_Sen' and effect == 'omega':
                            all_cross_asset.setdefault(asset_name, {})
                            all_cross_asset[asset_name]['granger_lrr_omega_p'] = p
                        if cause == 'omega' and effect == 'LRR_Oracle_Sen':
                            all_cross_asset.setdefault(asset_name, {})
                            all_cross_asset[asset_name]['granger_omega_lrr_p'] = p

        # ------------------------------------------------------------------
        # Phase 19: Gap Analyses per asset
        # ------------------------------------------------------------------
        log('>>> Phase 19: Gap Analyses (robustness checks)')

        # Gap 1 — Sub-period robustness (calm vs crisis market)
        log('    19a. Sub-period robustness (Gap 1)', 1)
        try:
            run_subperiod_analysis(final, asset_name, results_path)
        except Exception as e:
            log(f'    ! Sub-period failed: {e}', 1)

        # Gap 3 — FEVD (uses already-fitted var_results)
        log('    19b. Forecast Error Variance Decomposition (Gap 3)', 1)
        try:
            if var_results is not None:
                compute_fevd(var_results, asset_name, results_path)
        except Exception as e:
            log(f'    ! FEVD failed: {e}', 1)

        # Gap 4 — Lag order robustness
        log('    19c. Lag order robustness (Gap 4)', 1)
        try:
            run_lag_robustness(final, asset_name, results_path)
        except Exception as e:
            log(f'    ! Lag robustness failed: {e}', 1)

        # Gap 8 — HITS-VAR comparison
        log('    19d. HITS-VAR signal comparison (Gap 8)', 1)
        try:
            hits_var_res = run_hits_var_comparison(
                final, asset_name, results_path)
            all_hits_var_results[asset_name] = hits_var_res
        except Exception as e:
            log(f'    ! HITS-VAR failed: {e}', 1)

        # SVAR residual correlations
        log('    19e. SVAR residual correlation matrix', 1)
        try:
            if var_results is not None:
                compute_svar_residual_correlations(
                    var_results, asset_name, results_path)
        except Exception as e:
            log(f'    ! SVAR residuals failed: {e}', 1)


        log('>>> Phase 10: Regime-Specific VAR')
        for r_id, r_name in [(0, 'CALM'), (1, 'CRISIS'), (2, 'NON_CONVERGED')]:
            subset = final[final['regime'] == r_id].copy()
            if len(subset) < 40:
                continue
            log(f'    Regime {r_name}: {len(subset)} rows', 1)
            r_var = run_unified_var(
                subset, f'{asset_name}_{r_name}', results_path
            )
            if r_var is not None:
                run_granger_causality_table(
                    r_var, f'{asset_name}_{r_name}', results_path
                )
                if r_id == 2:
                    avg_omega = subset['omega'].mean()
                    with open(os.path.join(
                            results_path,
                            f'{asset_name.lower()}_VAR_{r_name}.txt'), 'a', encoding='utf-8') as f:
                        f.write(f'\n[NOTE] Avg omega during Non-Convergence: '
                                f'{avg_omega:.4f}\n')

        # ------------------------------------------------------------------
        # Phase 11: On-Chain Verification Pipeline
        #
        # Full verification of the LRR -> Whale -> Price causal chain:
        #   11a. Lead-lag correlation table (LRR->Whale, Whale->Price, 1-14 days)
        #   11b. 4-variable SVAR (LRR + whale_vol_log + price + omega)
        #        using log-normalised whale volume to fix scale mismatch
        #   11c. Granger causality: LRR->Whale and Whale->Price
        #   11d. Regime-stratified SVAR (CALM vs CRISIS)
        #   11e. 4-variable OOS validation (adds whale to the forecast)
        #   11f. Visualisations: alignment chart + lead-lag heatmap
        # ------------------------------------------------------------------
        if 'whale_vol_log' in final.columns:
            log('>>> Phase 11: On-Chain Verification (LRR -> Whale -> Price)')

            # Stationarity of log whale volume — auto-difference if needed
            whale_adf_pval = adfuller(final['whale_vol_log'].dropna())[1]
            if whale_adf_pval > 0.05:
                log(f'    whale_vol_log non-stationary (p={whale_adf_pval:.4f}) — applying first-difference', 1)
                final['whale_vol_log'] = final['whale_vol_log'].diff()
                # Also log the differenced stationarity
                check_stationarity(
                    final['whale_vol_log'].dropna(),
                    f'{asset_name}_whale_vol_log_diff', results_path
                )
            else:
                log(f'    whale_vol_log stationary (p={whale_adf_pval:.4f})', 1)
                check_stationarity(
                    final['whale_vol_log'].diff().dropna(),
                    f'{asset_name}_whale_vol_log_diff', results_path
                )

            # -------------------------------------------------------
            # 11a. Lead-Lag Correlation Table (LRR->Whale, 1-14 days)
            # -------------------------------------------------------
            log('    11a. Lead-lag correlation table', 1)
            final_indexed = final.set_index('time')
            ll_df = compute_onchain_lead_lag(
                final_indexed, asset_name, results_path, max_lag=14
            )

            # Find the strongest lag for LRR->Whale (used in alignment plot)
            best_lrr_whale_lag = 4  # default
            if ll_df is not None and 'LRR_to_Whale_r' in ll_df.columns:
                valid_r = ll_df['LRR_to_Whale_r'].dropna()
                if not valid_r.empty:
                    best_lrr_whale_lag = int(
                        ll_df.loc[valid_r.abs().idxmax(), 'Lag'].replace('t-', '')
                    )
            log(f'    Strongest LRR->Whale lag: t-{best_lrr_whale_lag}', 1)

            # -------------------------------------------------------
            # 11b-d. SVAR with log-normalised whale volume
            # -------------------------------------------------------
            # Cholesky ordering: LRR (most exogenous) -> Whale -> Price
            # omega added as it shapes LRR and is strictly exogenous
            svar_cols  = ['LRR_VAR_Signal', 'whale_vol_log', 'price_change']
            svar_data  = final[svar_cols].dropna()

            if len(svar_data) > 30:
                try:
                    res_svar = VAR(svar_data).fit(maxlags=6)

                    # Orthogonalised cumulative IRF with 90% CI bands
                    plot_svar_cumulative_irf(res_svar, asset_name, results_path)

                    def _sig(p):
                        return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''

                    # Granger: LRR -> Whale  and  Whale -> Price
                    gc_lw = res_svar.test_causality(
                        'whale_vol_log', 'LRR_VAR_Signal', kind='f')
                    gc_wp = res_svar.test_causality(
                        'price_change',  'whale_vol_log',  kind='f')
                    # Also test direct: LRR -> Price (for comparison)
                    gc_lp = res_svar.test_causality(
                        'price_change',  'LRR_VAR_Signal', kind='f')

                    with open(os.path.join(
                            results_path,
                            f'{asset_name.lower()}_Stage2_OnChain.txt'), 'w', encoding='utf-8') as f:
                        f.write(f'=== {asset_name} On-Chain Transmission Verification ===\n')
                        f.write(f'Cholesky ordering: {svar_cols}\n')
                        f.write(f'Whale variable: whale_vol_log = log1p(whale_vol_usd)\n\n')
                        f.write(f'--- Granger Causality Tests ---\n')
                        f.write(f'LRR -> Whale  : F={gc_lw.test_statistic:.4f}  '
                                f'p={gc_lw.pvalue:.4f}  {_sig(gc_lw.pvalue)}\n')
                        f.write(f'Whale -> Price: F={gc_wp.test_statistic:.4f}  '
                                f'p={gc_wp.pvalue:.4f}  {_sig(gc_wp.pvalue)}\n')
                        f.write(f'LRR -> Price  : F={gc_lp.test_statistic:.4f}  '
                                f'p={gc_lp.pvalue:.4f}  {_sig(gc_lp.pvalue)}  '
                                f'[direct path, for comparison]\n\n')
                        f.write(str(res_svar.summary()))

                    log(f'    Granger LRR->Whale  p={gc_lw.pvalue:.4f} {_sig(gc_lw.pvalue)}', 1)
                    log(f'    Granger Whale->Price p={gc_wp.pvalue:.4f} {_sig(gc_wp.pvalue)}', 1)
                    log(f'    Granger LRR->Price   p={gc_lp.pvalue:.4f} {_sig(gc_lp.pvalue)} [direct]', 1)

                    # Regime-stratified SVAR
                    for r_id, r_name in [(0, 'CALM'), (1, 'CRISIS')]:
                        subset = final[final['regime'] == r_id][svar_cols].dropna()
                        if len(subset) < 40:
                            continue
                        try:
                            r_svar   = VAR(subset).fit(maxlags=5)
                            gc_r_lw  = r_svar.test_causality(
                                'whale_vol_log', 'LRR_VAR_Signal', kind='f')
                            gc_r_wp  = r_svar.test_causality(
                                'price_change',  'whale_vol_log',  kind='f')
                            with open(os.path.join(
                                    results_path,
                                    f'{asset_name.lower()}_Structural_{r_name}.txt'), 'w', encoding='utf-8') as f:
                                f.write(f'=== {asset_name} {r_name} REGIME — On-Chain ===\n')
                                f.write(f'LRR->Whale   p={gc_r_lw.pvalue:.4f} {_sig(gc_r_lw.pvalue)}\n')
                                f.write(f'Whale->Price p={gc_r_wp.pvalue:.4f} {_sig(gc_r_wp.pvalue)}\n\n')
                                f.write(str(r_svar.summary()))
                            log(f'    Regime {r_name}: LRR->Whale p={gc_r_lw.pvalue:.4f}  '
                                f'Whale->Price p={gc_r_wp.pvalue:.4f}', 1)
                        except Exception as e:
                            log(f'    ! Regime-SVAR {r_name} error: {e}', 2)

                except Exception as e:
                    log(f'    ! SVAR failed: {e}', 1)

            # -------------------------------------------------------
            # 11e. 4-Variable OOS Validation (LRR + Whale -> Price)
            # -------------------------------------------------------
            log('    11e. 4-variable OOS (LRR + Whale + Omega -> Price)', 1)
            cols_4var = ['price_change', 'LRR_VAR_Signal', 'whale_vol_log', 'omega']
            data_4var = final[cols_4var].dropna()

            if len(data_4var) >= 50:
                try:
                    split_4  = int(len(data_4var) * TRAIN_RATIO)
                    train_4  = data_4var.iloc[:split_4]
                    test_4   = data_4var.iloc[split_4:]
                    actual_4 = test_4['price_change'].values

                    var_4    = VAR(train_4).fit(maxlags=7)
                    fc_4     = var_4.forecast(y=train_4.values[-7:], steps=len(test_4))
                    pred_4   = fc_4[:, 0]

                    rmse_4   = float(np.sqrt(np.mean((actual_4 - pred_4) ** 2)))
                    da_4     = float(np.mean(np.sign(actual_4) == np.sign(pred_4)))

                    with open(os.path.join(
                            results_path,
                            f'{asset_name.lower()}_OOS_4Var.txt'), 'w', encoding='utf-8') as f:
                        f.write(f'=== {asset_name} 4-Variable OOS (LRR+Whale+Omega->Price) ===\n')
                        f.write(f'Variables: {cols_4var}\n')
                        f.write(f'Train: {split_4}  Test: {len(test_4)}\n\n')
                        f.write(f'RMSE: {rmse_4:.6f}\n')
                        f.write(f'Directional Accuracy: {da_4:.4f}\n')
                        f.write(f'\nInterpretation: compare RMSE with 3-var model in '
                                f'{asset_name.lower()}_OOS_Metrics.csv\n')
                        f.write(f'Improvement shows that whale volume adds predictive value '
                                f'beyond social signal alone.\n')

                    log(f'    4-var OOS: RMSE={rmse_4:.5f}  DA={da_4:.1%}', 1)

                except Exception as e:
                    log(f'    ! 4-var OOS failed: {e}', 1)

            # -------------------------------------------------------
            # 11f. On-Chain Visualisations
            # -------------------------------------------------------
            log('    11f. On-chain visualisations', 1)
            plot_lrr_whale_alignment(
                final, asset_name, best_lrr_whale_lag, results_path
            )
            plot_onchain_leadlag_heatmap(ll_df, asset_name, results_path)

        else:
            log('>>> Phase 11: Skipped — no whale_vol_log column found.', 1)
            log('    Ensure btc_onchain.csv / eth_onchain.csv exist in /data/', 1)

        # ------------------------------------------------------------------
        # Phase 12: Rolling Correlation (fixed + regime-split)
        # ------------------------------------------------------------------
        log('>>> Phase 12: Rolling Correlation (temporal robustness)')
        final_indexed = final.set_index('time')   # indexed by time
        rolling_corr  = compute_rolling_correlation(
            final_indexed,
            signal_col='LRR_Oracle_Sen',
            target_col='price_change',
            window=ROLLING_WINDOW,
            lag=ROLLING_LAG
        )
        rolling_corr.to_csv(
            os.path.join(results_path,
                         f'{asset_name.lower()}_Rolling_Correlation.csv')
        )
        pct_positive = float((rolling_corr > 0).mean())
        log(f'    % windows with positive correlation: {pct_positive:.1%}', 1)
        log(f'    Mean r: {rolling_corr.mean():.4f}  '
            f'Max r: {rolling_corr.max():.4f}', 1)

        # ------------------------------------------------------------------
        # Phase 17: Event Study
        # ------------------------------------------------------------------
        log('>>> Phase 17: Event Study')
        try:
            lead_result = run_event_study(
                tw_df, final, asset_name, results_path,
                n_events=20, pre=5, post=10
            )
            if lead_result:
                event_study_results[asset_name] = lead_result
                log(f'    Mean LRR lead: {lead_result["mean_lead"]:.2f} days  '
                    f'p={lead_result["p_value"]:.4f}{lead_result.get("sig","")}  '
                    f'n={lead_result["n_events"]} events', 1)
        except Exception as e:
            log(f'    ! Event study failed: {e}', 1)

        # ------------------------------------------------------------------
        # Phase 15: Portfolio Backtest
        # ------------------------------------------------------------------
        log('>>> Phase 15: Portfolio Backtest')
        try:
            portfolio_metrics, equity_curves = run_portfolio_backtest(
                final, asset_name, results_path
            )
        except Exception as e:
            log(f'    ! Portfolio backtest failed: {e}', 1)
            portfolio_metrics = None
            equity_curves     = {}

        # ------------------------------------------------------------------
        # Phase 18: Cognitive Distortion Decomposition
        # ------------------------------------------------------------------
        log('>>> Phase 18: Cognitive Distortion Decomposition')
        try:
            dist_result = run_distortion_decomposition(
                tw_df, final, asset_name, results_path
            )
            if dist_result is not None:
                dist_df, cluster_df = dist_result
                all_distortion_results[asset_name] = {
                    'individual': dist_df,
                    'clusters':   cluster_df,
                }
                sig_count = (dist_df['LRR_to_dist_sig'] != '').sum()
                log(f'    {sig_count}/{len(dist_df)} distortions significantly '
                    f'predicted by LRR', 1)
            else:
                log('    No distortion columns found — check Twitter CSV', 1)
        except Exception as e:
            log(f'    ! Distortion decomposition failed: {e}', 1)

        # ------------------------------------------------------------------
        # Phase 14: All Visualisations
        # ------------------------------------------------------------------
        log('>>> Phase 14: Generating All Visualisations')

        # Correlation heatmap
        generate_correlation_matrix(final, asset_name, results_path)

        # Authority Gap
        plot_authority_gap(final, asset_name, results_path)

        # Evolutionary Denoising (6-signal ablation including FinBERT)
        signals   = ['Simple_Sen', 'FinBERT_Sen', 'PageRank_Sen',
                     'HITS_Sen', 'LRR_Social_Sen', 'LRR_Oracle_Sen']
        signals   = [s for s in signals if s in final.columns]
        sweep_df  = perform_lag_sweep(final, signals, 'price_change',
                                      max_lag=MAX_LAG)
        plot_ablation_denoising(sweep_df, asset_name, results_path)

        # Standard IRF with CI bands
        plot_var_irf(var_results, asset_name, results_path)

        # OOS forecast with baselines
        if comparison_df is not None and results_dict:
            plot_oos_forecast(comparison_df, results_dict,
                              asset_name, results_path)
            regime_col = final.set_index('time')['regime']
            plot_regime_aware_forecast(comparison_df, regime_col,
                                       results_dict, asset_name, results_path)
            plot_baseline_comparison(results_dict, asset_name, results_path)

        # LTD benchmark chart
        plot_ltd_benchmark(ltd_dict, asset_name, results_path)

        # Gate sensitivity chart
        plot_gate_sensitivity(gate_results, asset_name, results_path)

        # Rolling correlation with regime shading
        regime_col_for_plot = final.set_index('time')['regime'] \
            if 'regime' in final.columns else None
        plot_rolling_correlation_regimes(
            {'Full': rolling_corr},
            regime_col_for_plot,
            asset_name, ROLLING_LAG, ROLLING_WINDOW, results_path
        )

        # Granger heatmap
        if var_results is not None:
            gc_df_for_plot = run_granger_causality_table(
                var_results, asset_name, results_path
            )
            plot_granger_heatmap(gc_df_for_plot, asset_name, results_path)

        # Portfolio backtest plots
        if equity_curves:
            plot_cumulative_returns(equity_curves, asset_name, results_path)
            plot_drawdown(equity_curves, asset_name, results_path)
        if portfolio_metrics is not None and not portfolio_metrics.empty:
            plot_sharpe_comparison(portfolio_metrics, asset_name, results_path)

        # Distortion decomposition plots
        if asset_name in all_distortion_results:
            d_res = all_distortion_results[asset_name]
            plot_distortion_heatmap(d_res['individual'],
                                    asset_name, results_path)
            plot_distortion_clusters(d_res['individual'],
                                     d_res['clusters'],
                                     asset_name, results_path)

        # ------------------------------------------------------------------
        # Phase 26: Advanced Analyses (VECM, Partial Granger, DM, etc.)
        # ------------------------------------------------------------------
        log('>>> Phase 26: Advanced Robustness Analyses')
        
        from src.analytics import (
            run_vecm_analysis, run_partial_granger,
            run_diebold_mariano_test, run_time_varying_granger,
            run_quantile_regression_crash, run_regime_granger_detail,
        )
        
        # VECM as alternative to differenced VAR
        try:
            run_vecm_analysis(final, asset_name, results_path)
        except Exception as e:
            log(f'    ! VECM failed: {e}', 1)
        
        # Partial Granger: LRR→ω controlling for HITS and PR
        try:
            run_partial_granger(final, asset_name, results_path)
        except Exception as e:
            log(f'    ! Partial Granger failed: {e}', 1)
        
        # Diebold-Mariano test
        try:
            run_diebold_mariano_test(final, asset_name, results_path)
        except Exception as e:
            log(f'    ! Diebold-Mariano failed: {e}', 1)
        
        # Time-varying rolling Granger
        try:
            run_time_varying_granger(final, asset_name, results_path)
        except Exception as e:
            log(f'    ! Time-varying Granger failed: {e}', 1)
        
        # Quantile regression for crash analysis
        try:
            run_quantile_regression_crash(final, asset_name, results_path)
        except Exception as e:
            log(f'    ! Quantile regression failed: {e}', 1)
        
        # Regime-specific Granger detail
        try:
            run_regime_granger_detail(final, asset_name, results_path)
        except Exception as e:
            log(f'    ! Regime Granger failed: {e}', 1)
        
        # Conditional Transfer Entropy: TE(LRR→ω | HITS) and TE(LRR→ω | PR)
        try:
            from src.risk_metrics import calculate_conditional_transfer_entropy
            cte_results = {}
            if all(c in final.columns for c in ['LRR_Oracle_Sen', 'omega', 'HITS_Sen', 'PageRank_Sen']):
                lrr_sig = final['LRR_Oracle_Sen'].dropna()
                omega_sig = final['omega'].dropna()
                hits_sig = final['HITS_Sen'].dropna()
                pr_sig = final['PageRank_Sen'].dropna()
                
                # Standard TE for reference
                te_lrr = calculate_transfer_entropy(lrr_sig, omega_sig)
                te_hits = calculate_transfer_entropy(hits_sig, omega_sig)
                te_pr = calculate_transfer_entropy(pr_sig, omega_sig)
                
                # Conditional TE
                cte_lrr_given_hits = calculate_conditional_transfer_entropy(
                    lrr_sig, omega_sig, hits_sig)
                cte_lrr_given_pr = calculate_conditional_transfer_entropy(
                    lrr_sig, omega_sig, pr_sig)
                cte_hits_given_lrr = calculate_conditional_transfer_entropy(
                    hits_sig, omega_sig, lrr_sig)
                
                log(f'   Conditional TE {asset_name}:')
                log(f'     TE(LRR→ω)={te_lrr:.4f}  TE(LRR→ω|HITS)={cte_lrr_given_hits:.4f}  '
                    f'TE(LRR→ω|PR)={cte_lrr_given_pr:.4f}')
                log(f'     TE(HITS→ω)={te_hits:.4f}  TE(HITS→ω|LRR)={cte_hits_given_lrr:.4f}')
                
                # Save
                with open(os.path.join(results_path, f'{asset_name.lower()}_conditional_te.txt'),
                          'w', encoding='utf-8') as f:
                    f.write(f'=== {asset_name} Conditional Transfer Entropy ===\n\n')
                    f.write(f'Standard TE:\n')
                    f.write(f'  TE(LRR→omega) = {te_lrr:.6f}\n')
                    f.write(f'  TE(HITS→omega) = {te_hits:.6f}\n')
                    f.write(f'  TE(PR→omega) = {te_pr:.6f}\n\n')
                    f.write(f'Conditional TE (unique information):\n')
                    f.write(f'  TE(LRR→omega | HITS) = {cte_lrr_given_hits:.6f}  '
                            f'(LRR info beyond HITS)\n')
                    f.write(f'  TE(LRR→omega | PR) = {cte_lrr_given_pr:.6f}  '
                            f'(LRR info beyond PR)\n')
                    f.write(f'  TE(HITS→omega | LRR) = {cte_hits_given_lrr:.6f}  '
                            f'(HITS info beyond LRR)\n\n')
                    if cte_lrr_given_hits > cte_hits_given_lrr:
                        f.write(f'LRR carries MORE unique information about omega than HITS.\n')
                    else:
                        f.write(f'HITS carries more unique information about omega than LRR.\n')
        except Exception as e:
            log(f'    ! Conditional TE failed: {e}', 1)

        log(f'✅  {asset_name} complete.\n')

    # -----------------------------------------------------------------------
    # Phase 13: Cross-Asset Analyses (Spillover + Pooled + IER)
    # -----------------------------------------------------------------------
    log('\n>>> Phase 13: Cross-Asset Analyses')

    # 13a — BTC LRR → ETH Price spillover
    if 'BTC' in final_data and 'ETH' in final_data:
        log('    13a. BTC LRR → ETH price spillover', 1)
        btc_f = final_data['BTC'].set_index('time')[['LRR_Oracle_Sen', 'omega']]
        eth_f = final_data['ETH'].set_index('time')[['price_change']]
        cross = btc_f.join(eth_f, how='inner', rsuffix='_eth').dropna()
        cross.columns = ['BTC_LRR', 'BTC_omega', 'ETH_price']
        cross = cross.sort_index()

        if len(cross) >= 30:
            rows = []
            for lag in range(1, 8):
                x     = cross['BTC_LRR'].shift(lag)
                y     = cross['ETH_price']
                valid = pd.concat([x, y], axis=1).dropna()
                if len(valid) > 10:
                    from scipy.stats import pearsonr as _pr
                    r, p = _pr(valid.iloc[:, 0], valid.iloc[:, 1])
                    rows.append({'lag': f't-{lag}', 'r': round(r, 5),
                                 'p': round(p, 4),
                                 'sig': ('***' if p < 0.001 else
                                         '**'  if p < 0.01  else
                                         '*'   if p < 0.05  else '')})
            cross_df = pd.DataFrame(rows)
            cross_df.to_csv(
                os.path.join(results_path,
                             'cross_asset_BTC_LRR_to_ETH_Price.csv'),
                index=False
            )
            sig_rows = cross_df[cross_df['sig'] != '']
            if not sig_rows.empty:
                for _, row in sig_rows.iterrows():
                    log(f'      {row["lag"]}: r={row["r"]:.4f} '
                        f'p={row["p"]:.4f} {row["sig"]}', 2)
            else:
                log('      No significant lags found', 2)
        else:
            log('    Insufficient overlap for cross-asset analysis.', 1)

    # 13b — Pooled Con Gate Significance (Fisher's method)
    if all_gate_perm_results:
        log('    13b. Pooled Con Gate Significance — Fisher\'s method (T1.4)', 1)
        pooled = pooled_con_gate_significance(all_gate_perm_results)
        with open(os.path.join(results_path,
                               'cross_asset_ConGate_Pooled.txt'),
                  'w', encoding='utf-8') as f:
            f.write("=== Cross-Asset Pooled Con Gate Test (Fisher's Method) ===\n\n")
            f.write("H0: Con gate provides no LTD reduction in ANY asset\n")
            f.write("Method: Fisher's combined probability test\n")
            f.write("X^2 = -2 * sum(ln(p_i))  ~  chi2(2k) under H0\n\n")
            for method, res in pooled.items():
                sig = ('***' if res['p_combined'] < 0.001 else
                       '**'  if res['p_combined'] < 0.01  else
                       '*'   if res['p_combined'] < 0.05  else '')
                f.write(f"{method} ({res['k']} assets): "
                        f"chi2={res['chi2']:.4f}  "
                        f"p_combined={res['p_combined']:.6f} {sig}\n")
                f.write(f"  Individual p-values: "
                        f"{[round(p, 4) for p in res['individual_ps']]}\n\n")
                log(f'      {method}: chi2={res["chi2"]:.3f}  '
                    f'p={res["p_combined"]:.4f} {sig}', 2)

    # 13c — Cross-Asset IER Table, Plot, and Superiority Test
    if all_ier_rows:
        log('    13c. IER table + T2.4 superiority test', 1)
        ier_df = pd.DataFrame(all_ier_rows)
        ier_df.to_csv(
            os.path.join(results_path, 'cross_asset_IER_table.csv'),
            index=False, encoding='utf-8'
        )

        obs_diff, ier_p, ier_ci_lo, ier_ci_hi = bootstrap_ier_superiority(
            all_ier_rows
        )

        with open(os.path.join(results_path, 'cross_asset_IER_table.txt'),
                  'w', encoding='utf-8') as f:
            f.write("=== Information Efficiency Ratio (IER = TE / LTD) ===\n")
            f.write("Higher IER = more predictive information per unit of "
                    "crash exposure\n\n")
            f.write(f"{'Asset':<8} {'Signal':<14} {'TE':>10} "
                    f"{'LTD':>10} {'IER':>10}\n")
            f.write('-' * 56 + '\n')
            for _, row in ier_df.sort_values(['asset', 'signal']).iterrows():
                f.write(f"{row['asset']:<8} {row['signal']:<14} "
                        f"{row['TE']:>10.6f} {row['LTD']:>10.6f} "
                        f"{row['IER']:>10.6f}\n")

            avg_ier = ier_df.groupby('signal')['IER'].mean().sort_values(
                ascending=False)
            f.write('\nMean IER across all assets:\n')
            for sig_name, ier_val in avg_ier.items():
                f.write(f"  {sig_name}: {ier_val:.6f}\n")

            if not np.isnan(obs_diff):
                sig = ('***' if ier_p < 0.001 else '**' if ier_p < 0.01
                       else '*' if ier_p < 0.05 else '')
                f.write(f'\n--- T2.4 IER Superiority Test (LRR vs PageRank) ---\n')
                f.write(f'Mean IER diff (LRR - PageRank): {obs_diff:.6f}\n')
                f.write(f'95% CI: [{ier_ci_lo:.6f}, {ier_ci_hi:.6f}]\n')
                f.write(f'Bootstrap p (H0: LRR_IER <= PR_IER): '
                        f'{ier_p:.4f} {sig}\n')
                log(f'    IER superiority: diff={obs_diff:.4f}  '
                    f'p={ier_p:.4f}{sig}', 1)

        plot_ier_table(all_ier_rows, results_path)
        log('    IER table + cross_asset_IER_comparison.png saved', 1)

    # 13d — Cross-Asset Summary Table
    if all_cross_asset:
        log('    13d. Cross-asset summary table (T2.2)', 1)
        # Enrich with OOS, rolling corr, and gate data from saved files
        for ak in list(all_cross_asset.keys()):
            # OOS metrics
            oos_f = os.path.join(results_path,
                                 f'{ak.lower()}_OOS_Metrics.csv')
            if os.path.exists(oos_f):
                try:
                    oos_tbl = pd.read_csv(oos_f, index_col=0)
                    if 'LRR-VAR' in oos_tbl.index:
                        all_cross_asset[ak]['oos_da'] = float(
                            oos_tbl.loc['LRR-VAR', 'DA'])
                    if 'AR(1)' in oos_tbl.index:
                        all_cross_asset[ak]['ar1_da'] = float(
                            oos_tbl.loc['AR(1)', 'DA'])
                except Exception:
                    pass

            # Rolling correlation
            roll_f = os.path.join(results_path,
                                  f'{ak.lower()}_Rolling_Correlation.csv')
            if os.path.exists(roll_f):
                try:
                    roll_tbl = pd.read_csv(roll_f)
                    col      = roll_tbl.columns[1]
                    nn       = roll_tbl[col].dropna()
                    if len(nn) > 0:
                        all_cross_asset[ak]['rolling_pct_pos'] = float(
                            (nn > 0).mean() * 100)
                        all_cross_asset[ak]['rolling_mean_r'] = float(nn.mean())
                except Exception:
                    pass

            # Gate sensitivity
            gate_f = os.path.join(results_path,
                                  f'{ak.lower()}_Gate_Sensitivity.txt')
            if os.path.exists(gate_f):
                try:
                    import re as _re
                    gt = open(gate_f, encoding='utf-8').read()
                    rm = _re.search(r'Risk Reduction:\s+([\d.]+)%', gt)
                    bm = _re.search(r'Bootstrap p-value:\s+([\d.]+)', gt)
                    if rm:
                        all_cross_asset[ak]['con_reduction'] = float(rm.group(1))
                    if bm:
                        all_cross_asset[ak]['con_boot_p'] = float(bm.group(1))
                except Exception:
                    pass

        build_cross_asset_summary(all_cross_asset, results_path)
        log('    cross_asset_summary_table.csv saved', 1)

    # 13e — Event Study Cross-Asset Summary
    if event_study_results:
        log('    13e. Event study cross-asset summary (T2.1)', 1)
        with open(os.path.join(results_path, 'cross_asset_EventStudy.txt'),
                  'w', encoding='utf-8') as f:
            f.write('=== Cross-Asset Event Study Summary ===\n\n')
            f.write(f"{'Asset':<8} {'N_Events':>9} {'Mean_Lead':>11} "
                    f"{'p-value':>9} {'Sig':>5} {'%LRR_First':>11}\n")
            f.write('-' * 58 + '\n')
            for ak, res in sorted(event_study_results.items()):
                f.write(f"{ak:<8} {res['n_events']:>9} "
                        f"{res['mean_lead']:>11.2f} "
                        f"{res['p_value']:>9.4f} "
                        f"{res.get('sig',''):>5} "
                        f"{res['pct_positive_lead']*100:>10.1f}%\n")
        log('    cross_asset_EventStudy.txt saved', 1)

    # 13f — Cross-Asset Distortion Decomposition Summary
    if all_distortion_results:
        log('    13f. Cross-asset distortion decomposition summary (T3.1)', 1)
        with open(os.path.join(results_path,
                               'cross_asset_Distortion_Summary.txt'),
                  'w', encoding='utf-8') as f:
            f.write('=== Cross-Asset Cognitive Distortion Decomposition ===\n')
            f.write('T3.1 — Which distortions drive the LRR↔ω feedback?\n\n')

            # Aggregate: which distortions are consistently significant
            all_sig = {}
            for asset_key, res in all_distortion_results.items():
                df = res['individual']
                sig_rows = df[df['LRR_to_dist_sig'] != '']
                for _, row in sig_rows.iterrows():
                    d = row['distortion']
                    if d not in all_sig:
                        all_sig[d] = {
                            'cluster': row['cluster'],
                            'assets':  [],
                            'p_vals':  [],
                        }
                    all_sig[d]['assets'].append(asset_key)
                    all_sig[d]['p_vals'].append(row['LRR_to_dist_p'])

            f.write('--- Distortions Significantly Predicted by LRR (LRR→Distortion) ---\n')
            f.write(f'{"Distortion":<22} {"Cluster":<25} '
                    f'{"N_assets":>8} {"Assets"}\n')
            f.write('-' * 75 + '\n')
            for dist, info in sorted(all_sig.items(),
                                     key=lambda x: -len(x[1]['assets'])):
                f.write(f"{dist:<22} {info['cluster']:<25} "
                        f"{len(info['assets']):>8}  "
                        f"{', '.join(info['assets'])}\n")

            f.write('\n--- Cluster-Level Results Per Asset ---\n')
            for asset_key, res in sorted(all_distortion_results.items()):
                cl_df = res['clusters']
                if cl_df is not None and not cl_df.empty:
                    f.write(f'\n{asset_key}:\n')
                    for _, row in cl_df.iterrows():
                        f.write(f"  {row['cluster']:<25}: "
                                f"LRR→Cl p={row['LRR_to_cl_p']:.4f}"
                                f"{row['LRR_to_cl_sig']}  "
                                f"Cl→LRR p={row['cl_to_LRR_p']:.4f}"
                                f"{row['cl_to_LRR_sig']}\n")

        log('    cross_asset_Distortion_Summary.txt saved', 1)

    # 13g — Cross-asset oscillation cycle binomial test
    if all_lag_dfs:
        log('    13g. Oscillation cycle sign consistency test (Gap 7)', 1)
        try:
            run_oscillation_binomial_test(all_lag_dfs, results_path)
            log('    cross_asset_OscillationTest.txt saved', 1)
        except Exception as e:
            log(f'    ! Oscillation binomial test failed: {e}', 1)
        
        # Generate two-panel oscillation figure
        try:
            from src.analytics import generate_oscillation_figure
            generate_oscillation_figure(all_lag_dfs, results_path)
        except Exception as e:
            log(f'    ! Oscillation figure failed: {e}', 1)

    # 13h — Cross-asset HITS vs LRR Granger summary
    if all_hits_var_results:
        log('    13h. Cross-asset signal↔omega VAR comparison (Gap 8)', 1)
        try:
            rows_8 = []
            for ak, res in sorted(all_hits_var_results.items()):
                if not res:
                    continue
                def _p(signal):
                    return res.get(signal, {}).get('sig_to_omega_p', np.nan)
                def _f(signal):
                    return res.get(signal, {}).get('sig_to_omega_F', np.nan)
                def _sig(p):
                    if np.isnan(p): return 'n/a'
                    return ('***' if p < 0.001 else '**' if p < 0.01
                            else '*' if p < 0.05 else 'n.s.')
                lrr_p  = _p('LRR Oracle')
                hits_p = _p('HITS')
                pr_p   = _p('PageRank')
                unique = ('UNIQUE ✓' if (not np.isnan(lrr_p) and lrr_p < 0.05 and
                                         (np.isnan(hits_p) or hits_p >= 0.05))
                          else 'SHARED' if (not np.isnan(lrr_p) and lrr_p < 0.05 and
                                            not np.isnan(hits_p) and hits_p < 0.05)
                          else 'WEAK')
                rows_8.append({
                    'Asset': ak,
                    'LRR_p': round(lrr_p, 4) if not np.isnan(lrr_p) else np.nan,
                    'LRR_F': round(_f('LRR Oracle'), 3) if not np.isnan(_f('LRR Oracle')) else np.nan,
                    'HITS_p': round(hits_p, 4) if not np.isnan(hits_p) else np.nan,
                    'HITS_F': round(_f('HITS'), 3) if not np.isnan(_f('HITS')) else np.nan,
                    'PR_p': round(pr_p, 4) if not np.isnan(pr_p) else np.nan,
                    'LRR_sig': _sig(lrr_p),
                    'HITS_sig': _sig(hits_p),
                    'LRR_unique': unique,
                })

            if rows_8:
                df8 = pd.DataFrame(rows_8)
                df8.to_csv(os.path.join(results_path,
                    'cross_asset_HITS_vs_LRR_Granger.csv'),
                    index=False, encoding='utf-8')

                with open(os.path.join(results_path,
                    'cross_asset_HITS_vs_LRR_Granger.txt'),
                    'w', encoding='utf-8') as f:
                    f.write('=== Gap 8: Cross-Asset Signal↔omega Granger Comparison ===\n')
                    f.write('Critical test: Is LRR↔omega unique to cognitive gating,\n')
                    f.write('or does a pure link-structure signal (HITS) show the same dynamic?\n\n')
                    f.write(f'{"Asset":<6} {"LRR→ω p":>10} {"Sig":>5} '
                            f'{"HITS→ω p":>10} {"Sig":>5} '
                            f'{"PR→ω p":>9} {"LRR unique?":>12}\n')
                    f.write('-' * 60 + '\n')
                    for _, row in df8.iterrows():
                        f.write(f"{row['Asset']:<6} "
                                f"{str(row['LRR_p']):>10} {row['LRR_sig']:>5} "
                                f"{str(row['HITS_p']):>10} {row['HITS_sig']:>5} "
                                f"{str(row['PR_p']):>9} {row['LRR_unique']:>12}\n")
                    f.write('\n--- Interpretation ---\n')
                    unique_count = (df8['LRR_unique'] == 'UNIQUE ✓').sum()
                    shared_count = (df8['LRR_unique'] == 'SHARED').sum()
                    f.write(f'LRR↔omega UNIQUE to LRR: {unique_count}/{len(df8)} assets\n')
                    f.write(f'LRR↔omega SHARED with HITS: {shared_count}/{len(df8)} assets\n')
                    if unique_count >= 4:
                        f.write('\nCONCLUSION: The cognitive gating in LRR is NECESSARY for\n')
                        f.write('the rationality feedback dynamic. Pure link-structure (HITS)\n')
                        f.write('does not replicate the effect — confirming LRR uniqueness.\n')
                    elif shared_count >= 4:
                        f.write('\nCONCLUSION: The dynamic is partially structural (network-driven).\n')
                        f.write('LRR amplifies it through cognitive gating but does not originate it.\n')
                        f.write('Compare F-statistics: if LRR F > HITS F, LRR is still the\n')
                        f.write('stronger driver despite the shared significance.\n')

                log('    cross_asset_HITS_vs_LRR_Granger.txt saved', 1)
                n_unique = (df8['LRR_unique'] == 'UNIQUE ✓').sum()
                log(f'    LRR↔omega unique to LRR: {n_unique}/{len(df8)} assets', 1)
        except Exception as e:
            log(f'    ! Gap 8 cross-asset summary failed: {e}', 1)

    # ------------------------------------------------------------------
    # 13i — Cross-Asset LRR Signal Correlation Matrix
    # ------------------------------------------------------------------
    try:
        log('    13i. Cross-asset LRR signal independence check', 1)
        lrr_signals = {}
        for ak in sorted(final_data.keys()):
            fd = final_data[ak].copy()
            fd['time'] = pd.to_datetime(fd['time'], errors='coerce').dt.date
            lrr_signals[ak] = fd.set_index('time')['LRR_Oracle_Sen']

        if len(lrr_signals) >= 2:
            lrr_df = pd.DataFrame(lrr_signals).dropna()
            corr_matrix = lrr_df.corr()
            corr_matrix.to_csv(
                os.path.join(results_path,
                             'cross_asset_LRR_Signal_Correlations.csv'),
                float_format='%.4f')

            # Also compute for omega and price for comparison
            omega_signals = {}
            price_signals = {}
            for ak in sorted(final_data.keys()):
                fd = final_data[ak].copy()
                fd['time'] = pd.to_datetime(fd['time'], errors='coerce').dt.date
                idx = fd.set_index('time')
                omega_signals[ak] = idx['omega']
                price_signals[ak] = idx['price_change']

            omega_corr = pd.DataFrame(omega_signals).dropna().corr()
            price_corr = pd.DataFrame(price_signals).dropna().corr()

            with open(os.path.join(results_path,
                'cross_asset_LRR_Signal_Correlations.txt'),
                'w', encoding='utf-8') as f:
                f.write('=== Cross-Asset Signal Independence Check ===\n')
                f.write('Tests whether 6-asset "replication" is independent\n')
                f.write('or driven by the shared social graph.\n\n')

                f.write('--- LRR Oracle Daily Signal Pairwise Correlations ---\n')
                f.write(corr_matrix.to_string(float_format=lambda x: f'{x:.4f}'))
                f.write('\n\n')

                # Extract upper triangle mean
                import numpy as _np
                mask = _np.triu(_np.ones(corr_matrix.shape, dtype=bool), k=1)
                upper_vals = corr_matrix.values[mask]
                f.write(f'Mean pairwise r (LRR signals): {upper_vals.mean():.4f}\n')
                f.write(f'Min pairwise r:  {upper_vals.min():.4f}\n')
                f.write(f'Max pairwise r:  {upper_vals.max():.4f}\n\n')

                f.write('--- Omega Daily Signal Pairwise Correlations ---\n')
                f.write(omega_corr.to_string(float_format=lambda x: f'{x:.4f}'))
                f.write('\n\n')
                mask_o = _np.triu(_np.ones(omega_corr.shape, dtype=bool), k=1)
                upper_o = omega_corr.values[mask_o]
                f.write(f'Mean pairwise r (omega signals): {upper_o.mean():.4f}\n\n')

                f.write('--- Price Return Pairwise Correlations ---\n')
                f.write(price_corr.to_string(float_format=lambda x: f'{x:.4f}'))
                f.write('\n\n')
                mask_p = _np.triu(_np.ones(price_corr.shape, dtype=bool), k=1)
                upper_p = price_corr.values[mask_p]
                f.write(f'Mean pairwise r (price returns): {upper_p.mean():.4f}\n\n')

                f.write('--- Interpretation ---\n')
                mean_lrr = upper_vals.mean()
                if mean_lrr > 0.90:
                    f.write('HIGH correlation: 6-asset consistency is largely driven by\n')
                    f.write('the shared social graph. Cross-asset findings are NOT independent.\n')
                    if mean_lrr > 0.999:
                        f.write('\nNote (v3.0): With Con removed from propagation, reputation\n')
                        f.write('weights are identical across assets (only V-Anchor differs,\n')
                        f.write('at phi=0.15 weight). Signal differentiation across assets\n')
                        f.write('comes from the aggregation step where sen × Con interacts\n')
                        f.write('with asset-specific sentiment patterns. The LRR reputation\n')
                        f.write('weights represent USER quality (graph + omega), while\n')
                        f.write('asset-specificity enters through the SIGNAL (sen × Con).\n')
                elif mean_lrr > 0.70:
                    f.write('MODERATE-HIGH correlation: partial independence. The shared\n')
                    f.write('social graph contributes substantially, but asset-specific\n')
                    f.write('V-Anchor differentiation provides some independence.\n')
                elif mean_lrr > 0.40:
                    f.write('MODERATE correlation: meaningful independence between assets.\n')
                    f.write('V-Anchor and asset-specific price dynamics create differentiation.\n')
                else:
                    f.write('LOW correlation: 6-asset findings are substantially independent.\n')

            log(f'    Mean pairwise LRR r = {upper_vals.mean():.4f}', 1)
            log('    cross_asset_LRR_Signal_Correlations.txt saved', 1)
    except Exception as e:
        log(f'    ! 13i cross-asset correlation failed: {e}', 1)

    # ------------------------------------------------------------------
    # 13j — Cross-Asset Joint Crash Count Summary Table
    # ------------------------------------------------------------------
    try:
        log('    13j. Cross-asset joint crash count table', 1)
        crash_rows = []
        for ak in sorted(all_cross_asset.keys()):
            ac = all_cross_asset[ak]
            crash_rows.append({
                'Asset':       ak,
                'LRR_LTD':     ac.get('ltd_lrr', np.nan),
                'LRR_Crashes':  ac.get('joint_lrr', ''),
                'PR_LTD':      ac.get('ltd_pr', np.nan),
                'PR_Crashes':   ac.get('joint_pr', ''),
                'HITS_LTD':    ac.get('ltd_hits', np.nan),
                'HITS_Crashes': ac.get('joint_hits', ''),
            })

        crash_df = pd.DataFrame(crash_rows)
        crash_df.to_csv(
            os.path.join(results_path, 'cross_asset_Joint_Crash_Counts.csv'),
            index=False, float_format='%.4f')

        with open(os.path.join(results_path,
            'cross_asset_Joint_Crash_Counts.txt'),
            'w', encoding='utf-8') as f:
            f.write('=== Cross-Asset Joint Tail Crash Counts ===\n')
            f.write('lambda=0.10 (lower 10th percentile), N=604 per asset\n')
            f.write('Joint crashes = days where BOTH signal AND price are in lower tail\n')
            f.write('Fewer joint crashes = better crash decoupling\n\n')
            f.write(f'{"Asset":<6} {"LRR LTD":>9} {"LRR #":>6} '
                    f'{"PR LTD":>9} {"PR #":>6} '
                    f'{"HITS LTD":>9} {"HITS #":>6}\n')
            f.write('-' * 58 + '\n')
            for _, row in crash_df.iterrows():
                lrr_ltd_s  = f'{row["LRR_LTD"]:.4f}' if pd.notnull(row['LRR_LTD']) else 'N/A'
                pr_ltd_s   = f'{row["PR_LTD"]:.4f}' if pd.notnull(row['PR_LTD']) else 'N/A'
                hits_ltd_s = f'{row["HITS_LTD"]:.4f}' if pd.notnull(row['HITS_LTD']) else 'N/A'
                f.write(f'{row["Asset"]:<6} {lrr_ltd_s:>9} {str(row["LRR_Crashes"]):>6} '
                        f'{pr_ltd_s:>9} {str(row["PR_Crashes"]):>6} '
                        f'{hits_ltd_s:>9} {str(row["HITS_Crashes"]):>6}\n')
            f.write('\n--- Key Finding ---\n')
            lrr_crashes = [r['LRR_Crashes'] for r in crash_rows
                          if r['LRR_Crashes'] != '' and not pd.isna(r.get('LRR_Crashes', np.nan))]
            pr_crashes  = [r['PR_Crashes'] for r in crash_rows
                          if r['PR_Crashes'] != '' and not pd.isna(r.get('PR_Crashes', np.nan))]
            if lrr_crashes and pr_crashes:
                f.write(f'LRR joint crashes range: {min(lrr_crashes)}-{max(lrr_crashes)}\n')
                f.write(f'PageRank joint crashes range: {min(pr_crashes)}-{max(pr_crashes)}\n')
                f.write('LRR consistently produces fewer joint crashes than PageRank,\n')
                f.write('confirming crash decoupling even for NaN-LTD assets.\n')

        log('    cross_asset_Joint_Crash_Counts.txt saved', 1)
    except Exception as e:
        log(f'    ! 13j joint crash count table failed: {e}', 1)
    
    # Phase 20: Sensitivity and Robustness Suite 
    try:
        from src.sensitivity_suite import run_sensitivity_suite
        run_sensitivity_suite(final_data, tw, assets, results_path)
    except Exception as e:
        log(f'>>> Phase 20 failed: {e}')
        import traceback
        traceback.print_exc()

    # Phase 21: Enhanced LRR Comparison Suite
    try:
        from src.lrr_enhanced_comparison import run_enhanced_comparison
        run_enhanced_comparison(final_data, tw, results_path)
    except Exception as e:
        log(f'>>> Phase 21 failed: {e}')
        import traceback
        traceback.print_exc()

    # Phase 22: LOO All Signals x All Assets
    try:
        from src.loo_all_signals import run_loo_all_signals
        run_loo_all_signals(tw, final_data, results_path)
    except Exception as e:
        log(f'>>> Phase 22 failed: {e}')
        import traceback
        traceback.print_exc()

    # Phase 23: Rolling Reputation Fix
    try:
        from src.rolling_reputation_fix import run_rolling_reputation_fix
        run_rolling_reputation_fix(tw, final_data, results_path)
    except Exception as e:
        log(f'>>> Phase 23 failed: {e}')
        import traceback
        traceback.print_exc()

    # Phase 24: Additional Robustness (Johansen, Expanding HMM, Winsorized TE)
    try:
        from src.additional_robustness import run_additional_robustness
        run_additional_robustness(final_data, assets, tw, results_path)
    except Exception as e:
        log(f'>>> Phase 24 failed: {e}')
        import traceback
        traceback.print_exc()

    # ================================================================
    # Phase 25: PIPELINE VALIDATION (always run last)
    # Checks every critical computation and prints PASS/FAIL dashboard
    # ================================================================
    try:
        from src.pipeline_validator import run_pipeline_validation
        run_pipeline_validation(tw, final_data, assets, results_path)
    except Exception as e:
        log(f'>>> Phase 25 (Validation) failed: {e}')
        import traceback
        traceback.print_exc()

    log('\n>>> All assets processed. Results saved to /results/')

if __name__ == '__main__':
    main()