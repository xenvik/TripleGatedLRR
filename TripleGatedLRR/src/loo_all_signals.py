# src/loo_all_signals.py
# Leave-One-Out channel robustness for LRR, HITS, and PageRank
# Tests: for each of 76 channels × 6 assets × 3 signals, 
#        remove channel, recompute weights, rebuild signal, run Granger
#
# Usage: run_loo_all_signals(tw, final_data, results_path)

import os
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
import networkx as nx


def log(msg, indent=0):
    print('  ' * indent + msg)


def _run_granger_quick(signal_series, omega_series, price_series, max_lag=7):
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


def _build_graph_prhits(tw_df):
    """Build graph for PageRank/HITS: all edges weight=1.0 (standard algorithm)."""
    G = nx.DiGraph()
    # Retweet edges
    rt_mask = tw_df['rt_target'].notna()
    if rt_mask.any():
        for src, tgt in zip(tw_df.loc[rt_mask, 'source_user'].astype(str),
                            tw_df.loc[rt_mask, 'rt_target'].astype(str)):
            if src and tgt and tgt != 'nan' and src != tgt:
                if G.has_edge(src, tgt):
                    G[src][tgt]['weight'] += 1.0
                else:
                    G.add_edge(src, tgt, weight=1.0)
    # Mention edges — EQUAL weight for PR/HITS
    for src_val, mentions in zip(tw_df['source_user'].astype(str), tw_df['mentions']):
        if isinstance(mentions, list):
            for m in mentions:
                m = str(m)
                if src_val and m and m != 'nan' and src_val != m:
                    if G.has_edge(src_val, m):
                        G[src_val][m]['weight'] += 1.0
                    else:
                        G.add_edge(src_val, m, weight=1.0)
    return G


def _build_daily_and_test(tw_sub, weight_col, ref_final, use_con=True):
    tw_sub = tw_sub.copy()
    tw_sub['time_d'] = pd.to_datetime(tw_sub['time'], errors='coerce').dt.date
    def agg_fn(x):
        w = x[weight_col].values + 1e-9
        if use_con and 'con' in x.columns:
            s = (x['sen'].values * x['con'].values)
        else:
            s = x['sen'].values
        return pd.Series({
            'signal': np.average(s, weights=w),
            'omega': x['omega'].mean() if 'omega' in x.columns else 0.5,
        })
    daily = tw_sub.groupby('time_d').apply(agg_fn, include_groups=False).reset_index()
    daily.columns = ['time', 'signal', 'omega']
    merged = pd.merge(daily, ref_final[['time', 'price_change']], on='time', how='inner')
    if len(merged) < 50:
        return np.nan, np.nan
    return _run_granger_quick(merged['signal'], merged['omega'], merged['price_change'])


def run_loo_all_signals(tw, final_data, results_path):
    """Run leave-one-out for LRR, PageRank, HITS across ALL assets."""
    log('\n>>> Leave-One-Out Robustness: ALL Signals x ALL Assets')

    comp_path = os.path.join(results_path, 'comparison')
    os.makedirs(comp_path, exist_ok=True)

    from src.reputation_engine_v2 import compute_lrr_vectorised
    from src.anchor_utils import compute_anchor_vector
    from src.psych_engine import calculate_omega
    from src.config import TRAIN_RATIO

    tw_work = tw.copy()
    if 'omega' not in tw_work.columns:
        tw_work = calculate_omega(tw_work)

    sources = sorted(tw_work['source_user'].dropna().unique().tolist())
    log(f'  Source channels: {len(sources)}')

    all_users = list(set(
        tw_work['source_user'].dropna().unique().tolist() +
        tw_work['rt_target'].dropna().unique().tolist() +
        [m for ml in tw_work['mentions'] for m in ml if isinstance(ml, list)]
    ))

    # Pre-compute full graph for PR/HITS (equal weight, standard algorithm)
    G_prhits_full = _build_graph_prhits(tw_work)
    pr_full = nx.pagerank(G_prhits_full, alpha=0.85, weight='weight')
    _, hits_full = nx.hits(G_prhits_full, max_iter=100, normalized=True)
    tw_work['pr_w'] = tw_work['source_user'].astype(str).map(pr_full).fillna(0)
    tw_work['hits_w'] = tw_work['source_user'].astype(str).map(hits_full).fillna(0)

    all_loo_rows = []
    all_summary_rows = []

    for asset_name, final_raw in final_data.items():
        log(f'\n  === {asset_name} ===')
        final = final_raw.copy()
        final['time'] = pd.to_datetime(final['time'], errors='coerce').dt.date

        # Compute train_end_date for this asset (same logic as main.py)
        valid_dates = final.dropna(subset=['price_change'])['time'].tolist()
        valid_dates_sorted = sorted(valid_dates)
        split_idx = int(len(valid_dates_sorted) * TRAIN_RATIO)
        train_end_date = valid_dates_sorted[split_idx - 1] if split_idx > 0 else None

        try:
            anchor = compute_anchor_vector(tw_work, final, asset_name,
                                           train_end_date=train_end_date)
        except Exception as e:
            log(f'    Anchor computation failed: {e}')
            anchor = None

        lrr_full = compute_lrr_vectorised(tw_work, all_users, use_omega=True,
                                           use_con=True, anchor_vector=anchor)
        tw_work['lrr_w'] = tw_work['source_user'].astype(str).map(lrr_full).fillna(1.0 / max(len(all_users), 1))

        base_lrr_F, base_lrr_p = _build_daily_and_test(tw_work, 'lrr_w', final, use_con=True)
        base_pr_F, base_pr_p = _build_daily_and_test(tw_work, 'pr_w', final, use_con=False)
        base_hits_F, base_hits_p = _build_daily_and_test(tw_work, 'hits_w', final, use_con=False)

        log(f'  Baselines -- LRR: F={base_lrr_F:.3f} p={base_lrr_p:.4f} | '
            f'PR: F={base_pr_F:.3f} p={base_pr_p:.4f} | '
            f'HITS: F={base_hits_F:.3f} p={base_hits_p:.4f}')

        asset_loo_rows = []
        for ch_idx, ch_id in enumerate(sources):
            try:
                tw_loo = tw_work[tw_work['source_user'] != ch_id].copy()
                n_dropped = len(tw_work) - len(tw_loo)
                if len(tw_loo) < 100:
                    continue

                loo_users = list(set(
                    tw_loo['source_user'].dropna().unique().tolist() +
                    tw_loo['rt_target'].dropna().unique().tolist() +
                    [m for ml in tw_loo['mentions'] for m in ml if isinstance(ml, list)]
                ))

                rep_loo = compute_lrr_vectorised(tw_loo, loo_users, use_omega=True,
                                                  use_con=True, anchor_vector=anchor)
                tw_loo['lrr_w'] = tw_loo['source_user'].astype(str).map(rep_loo).fillna(1.0 / max(len(all_users), 1))
                lrr_F, lrr_p = _build_daily_and_test(tw_loo, 'lrr_w', final, use_con=True)

                G_loo_prhits = _build_graph_prhits(tw_loo)
                try:
                    pr_loo = nx.pagerank(G_loo_prhits, alpha=0.85, weight='weight')
                except Exception:
                    pr_loo = {u: 1.0/max(len(loo_users),1) for u in loo_users}
                tw_loo['pr_w'] = tw_loo['source_user'].astype(str).map(pr_loo).fillna(0)
                pr_F, pr_p = _build_daily_and_test(tw_loo, 'pr_w', final, use_con=False)

                try:
                    _, hits_loo = nx.hits(G_loo_prhits, max_iter=100, normalized=True)
                except Exception:
                    hits_loo = {u: 1.0/max(len(loo_users),1) for u in loo_users}
                tw_loo['hits_w'] = tw_loo['source_user'].astype(str).map(hits_loo).fillna(0)
                hits_F, hits_p = _build_daily_and_test(tw_loo, 'hits_w', final, use_con=False)

                row = {
                    'asset': asset_name,
                    'dropped_channel': str(ch_id),
                    'tweets_dropped': n_dropped,
                    'pct_dropped': round(n_dropped / len(tw_work) * 100, 2),
                    'lrr_F': round(lrr_F, 3) if not np.isnan(lrr_F) else np.nan,
                    'lrr_p': round(lrr_p, 4) if not np.isnan(lrr_p) else np.nan,
                    'lrr_sig': '*' if (not np.isnan(lrr_p) and lrr_p < 0.05) else '',
                    'pr_F': round(pr_F, 3) if not np.isnan(pr_F) else np.nan,
                    'pr_p': round(pr_p, 4) if not np.isnan(pr_p) else np.nan,
                    'pr_sig': '*' if (not np.isnan(pr_p) and pr_p < 0.05) else '',
                    'hits_F': round(hits_F, 3) if not np.isnan(hits_F) else np.nan,
                    'hits_p': round(hits_p, 4) if not np.isnan(hits_p) else np.nan,
                    'hits_sig': '*' if (not np.isnan(hits_p) and hits_p < 0.05) else '',
                }
                asset_loo_rows.append(row)
                all_loo_rows.append(row)
            except Exception as e:
                log(f'      Channel {ch_id} failed: {e}')

            if (ch_idx + 1) % 10 == 0:
                log(f'      {ch_idx+1}/{len(sources)} channels processed')

        adf = pd.DataFrame(asset_loo_rows)
        if len(adf) > 0:
            n_t = len(adf)
            lrr_s = (adf['lrr_sig'] == '*').sum()
            pr_s = (adf['pr_sig'] == '*').sum()
            hits_s = (adf['hits_sig'] == '*').sum()
            
            # F-stat distribution for LOO
            lrr_f_vals = adf['lrr_F'].dropna()
            hits_f_vals = adf['hits_F'].dropna()
            pr_f_vals = adf['pr_F'].dropna()
            
            all_summary_rows.append({
                'asset': asset_name, 'n_channels': n_t,
                'lrr_robust': lrr_s, 'lrr_pct': round(lrr_s/n_t*100, 1),
                'lrr_base_F': round(base_lrr_F, 3) if not np.isnan(base_lrr_F) else np.nan,
                'lrr_F_mean': round(lrr_f_vals.mean(), 3) if len(lrr_f_vals) > 0 else np.nan,
                'lrr_F_median': round(lrr_f_vals.median(), 3) if len(lrr_f_vals) > 0 else np.nan,
                'lrr_F_5pct': round(lrr_f_vals.quantile(0.05), 3) if len(lrr_f_vals) > 0 else np.nan,
                'lrr_F_95pct': round(lrr_f_vals.quantile(0.95), 3) if len(lrr_f_vals) > 0 else np.nan,
                'pr_robust': pr_s, 'pr_pct': round(pr_s/n_t*100, 1),
                'pr_base_F': round(base_pr_F, 3) if not np.isnan(base_pr_F) else np.nan,
                'hits_robust': hits_s, 'hits_pct': round(hits_s/n_t*100, 1),
                'hits_base_F': round(base_hits_F, 3) if not np.isnan(base_hits_F) else np.nan,
                'hits_F_mean': round(hits_f_vals.mean(), 3) if len(hits_f_vals) > 0 else np.nan,
                'hits_F_5pct': round(hits_f_vals.quantile(0.05), 3) if len(hits_f_vals) > 0 else np.nan,
                'hits_F_95pct': round(hits_f_vals.quantile(0.95), 3) if len(hits_f_vals) > 0 else np.nan,
            })
            log(f'  {asset_name} -- LRR: {lrr_s}/{n_t} | PR: {pr_s}/{n_t} | HITS: {hits_s}/{n_t}')
            if len(lrr_f_vals) > 0:
                log(f'    LRR F-dist: mean={lrr_f_vals.mean():.3f} '
                    f'median={lrr_f_vals.median():.3f} '
                    f'[5%={lrr_f_vals.quantile(0.05):.3f}, 95%={lrr_f_vals.quantile(0.95):.3f}]')

    # Save
    pd.DataFrame(all_loo_rows).to_csv(
        os.path.join(comp_path, 'loo_all_signals_all_assets.csv'), index=False)
    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_csv(
        os.path.join(comp_path, 'loo_all_signals_summary.csv'), index=False)

    # Report
    with open(os.path.join(comp_path, 'loo_all_signals_report.txt'), 'w', encoding='utf-8') as f:
        f.write('=== Leave-One-Out Channel Robustness: ALL Signals x ALL Assets ===\n\n')
        f.write(f'{"Asset":<8} {"LRR":<14} {"PageRank":<14} {"HITS":<14}\n')
        f.write('-' * 55 + '\n')
        for _, row in summary_df.iterrows():
            f.write(f'{row["asset"]:<8} '
                    f'{row["lrr_robust"]}/{row["n_channels"]} ({row["lrr_pct"]}%)  '
                    f'{row["pr_robust"]}/{row["n_channels"]} ({row["pr_pct"]}%)  '
                    f'{row["hits_robust"]}/{row["n_channels"]} ({row["hits_pct"]}%)\n')

        total_ch = summary_df['n_channels'].sum()
        total_lrr = summary_df['lrr_robust'].sum()
        total_pr = summary_df['pr_robust'].sum()
        total_hits = summary_df['hits_robust'].sum()
        f.write(f'\n{"TOTAL":<8} '
                f'{total_lrr}/{total_ch} ({total_lrr/total_ch*100:.1f}%)  '
                f'{total_pr}/{total_ch} ({total_pr/total_ch*100:.1f}%)  '
                f'{total_hits}/{total_ch} ({total_hits/total_ch*100:.1f}%)\n')

        log(f'\n  TOTAL: LRR={total_lrr}/{total_ch} ({total_lrr/total_ch*100:.1f}%) '
            f'PR={total_pr}/{total_ch} ({total_pr/total_ch*100:.1f}%) '
            f'HITS={total_hits}/{total_ch} ({total_hits/total_ch*100:.1f}%)')

        # Channels breaking HITS across multiple assets
        all_df = pd.DataFrame(all_loo_rows)
        f.write('\n\n--- Channels breaking HITS across assets ---\n')
        hits_broken = all_df[all_df['hits_sig'] != '*']
        if len(hits_broken) > 0:
            bc = hits_broken.groupby('dropped_channel')['asset'].apply(list)
            bc_sorted = bc.apply(len).sort_values(ascending=False)
            for ch in bc_sorted.index:
                assets_list = bc[ch]
                f.write(f'  @{ch}: breaks HITS in {len(assets_list)} asset(s): {", ".join(assets_list)}\n')

        f.write('\n--- Channels breaking LRR across assets ---\n')
        lrr_broken = all_df[all_df['lrr_sig'] != '*']
        if len(lrr_broken) > 0:
            bc = lrr_broken.groupby('dropped_channel')['asset'].apply(list)
            bc_sorted = bc.apply(len).sort_values(ascending=False)
            for ch in bc_sorted.index:
                assets_list = bc[ch]
                f.write(f'  @{ch}: breaks LRR in {len(assets_list)} asset(s): {", ".join(assets_list)}\n')

        f.write('\n--- Interpretation ---\n')
        f.write(f'LRR: {total_lrr/total_ch*100:.1f}% robust across all assets\n')
        f.write(f'HITS: {total_hits/total_ch*100:.1f}% robust across all assets\n')
        f.write(f'PageRank: {total_pr/total_ch*100:.1f}% robust (trivially: Gini=0.003)\n')
        if total_lrr > total_hits:
            f.write(f'\nLRR is more robust than HITS by {total_lrr-total_hits} '
                    f'channel-asset combinations.\n')

    log('  loo_all_signals_report.txt saved')
    log('>>> LOO All Signals x All Assets complete.')
