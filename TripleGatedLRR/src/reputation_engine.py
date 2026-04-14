# src/reputation_engine.py
import networkx as nx
import pandas as pd
import numpy as np
from src.psych_engine import calculate_omega
from src.config import PHI, ITERATIONS, CONVERGENCE_TOL


def compute_lrr_recursive(tw_df, unique_users, use_omega=True, use_con=True,
                           anchor_vector=None, mention_weight=None):
    """
    Liquid Reputation Propagation (LRR) — recursive reference implementation.

    v3.0 — Con REMOVED from propagation (gates aggregation only).
    Propagation gates:
        1. Omega (ω)   — individual rationality filter
        2. V-Anchor    — economic alignment personalization

    Penalty: high-volume low-rationality accounts receive a log-dampening
    penalty that reduces their effective reach without hard exclusion.

    Note: use_con parameter is retained for API compatibility but no longer
    affects propagation. Con is applied in signal aggregation (Phase 3).

    Returns:
        dict {user_id: reputation_score}  (scores in (floor, 1])
    """
    from src.config import MENTION_WEIGHT as DEFAULT_MW
    if mention_weight is None:
        mention_weight = DEFAULT_MW

    n_users = len(unique_users)
    adaptive_floor = 1.0 / max(n_users, 1)

    reputation     = {str(u): 1.0  for u in unique_users}
    activity_count = tw_df['source_user'].value_counts().to_dict()

    # Pre-compute in-degree for degree normalisation
    in_degree = {str(u): 0.0 for u in unique_users}
    for _, row in tw_df.iterrows():
        if pd.notnull(row.get('rt_target')):
            tgt = str(row['rt_target'])
            if tgt in in_degree:
                in_degree[tgt] += 1.0
        for m in row.get('mentions', []):
            tgt = str(m)
            if tgt in in_degree:
                in_degree[tgt] += 1.0

    for iteration in range(ITERATIONS):
        temp_rep = {k: 0.0 for k in unique_users}

        for _, row in tw_df.iterrows():
            src = str(row['source_user'])
            if src not in reputation:
                continue

            omega = float(row.get('omega', 0.5)) if use_omega else 1.0
            # Con removed from propagation (v3.0)

            # Volume-rationality penalty: spammers with low omega are dampened
            if use_omega and omega < 0.5:
                penalty = 1.0 / (1.0 + np.log1p(activity_count.get(src, 0)))
            else:
                penalty = 1.0

            effective_flow = reputation[src] * omega * penalty

            # Retweet: full reputation flow to original author
            if pd.notnull(row.get('rt_target')):
                target = str(row['rt_target'])
                if target in temp_rep:
                    temp_rep[target] += effective_flow * 1.0

            # Mentions: mention_weight reputation flow (empirically calibrated)
            for m in row.get('mentions', []):
                target = str(m)
                if target in temp_rep:
                    temp_rep[target] += effective_flow * mention_weight

        # Degree-normalised flow
        for k in unique_users:
            temp_rep[k] = temp_rep[k] / np.sqrt(in_degree.get(str(k), 0.0) + 1.0)

        # Kolonin logarithmic compression
        for k in unique_users:
            v = temp_rep[k]
            temp_rep[k] = np.sign(v) * np.log10(1.0 + abs(v))

        max_val = max(temp_rep.values()) if temp_rep.values() and max(temp_rep.values()) > 0 else 1.0

        prev_reputation = reputation.copy()
        for k in unique_users:
            base_score = temp_rep[k] / max_val
            if anchor_vector and k in anchor_vector:
                reputation[k] = (1.0 - PHI) * base_score + PHI * anchor_vector[k]
            else:
                reputation[k] = base_score

        # Adaptive floor: 1/N
        reputation = {k: max(v, adaptive_floor) for k, v in reputation.items()}

        # Convergence check
        max_change = max(abs(reputation[k] - prev_reputation[k]) for k in unique_users)
        if max_change < CONVERGENCE_TOL:
            break

    return reputation


def run_benchmarked_reputation(tw_df, asset_df, train_end_date=None,
                                mention_weight=None):
    """
    Computes all reputation benchmarks for a given asset.

    Look-ahead bias prevention:
        The V-Anchor (economic alignment vector) is computed by correlating
        each user's daily sentiment with FUTURE 7-day price returns.
        If computed over the full dataset this introduces look-ahead bias
        into OOS evaluation.

        Fix: when `train_end_date` is provided, anchor correlations are
        computed ONLY on the training period (dates <= train_end_date).
        The resulting reputation weights are then applied to the full
        dataset — reflecting how the model would perform at deployment.

    Graph design:
        - PR/HITS: full social graph (retweets + mentions), ALL edges weight=1.0
          (standard unweighted link analysis per Brin & Page 1998, Kleinberg 1999)
        - LRR: retweets (weight=1.0) + mentions (weight=mention_weight, empirically
          calibrated), plus triple cognitive gating (omega, con, V-anchor)

    Returns:
        pr_scores    : PageRank authority dict
        hits_auth    : HITS authority dict
        lrr_social   : LRR without gates (social-only baseline)
        lrr_oracle   : LRR with full triple-gating (Ω + Con + V-Anchor)
        tw_df        : tweet DataFrame with omega column added
        unique_users : list of all user IDs in the graph
    """
    from src.config import MENTION_WEIGHT
    if mention_weight is None:
        mention_weight = MENTION_WEIGHT

    tw_df = calculate_omega(tw_df)

    sources      = tw_df['source_user'].dropna().unique().tolist()
    rt_targets   = tw_df['rt_target'].dropna().unique().tolist()
    all_mentions = [m for m_list in tw_df['mentions']
                    for m in m_list if isinstance(m_list, list)]
    unique_users = list(set(sources + rt_targets + all_mentions))

    # -----------------------------------------------------------------------
    # 1. Social Graph for PR/HITS: ALL edges weight = 1.0 (standard algorithm)
    #    Retweets AND mentions are treated as equal links — this is how
    #    PageRank and HITS are designed (Brin & Page 1998; Kleinberg 1999).
    # -----------------------------------------------------------------------
    G_prhits = nx.DiGraph()
    # Retweet edges
    rt_mask = tw_df['rt_target'].notna()
    if rt_mask.any():
        for src, tgt in zip(tw_df.loc[rt_mask, 'source_user'].astype(str),
                            tw_df.loc[rt_mask, 'rt_target'].astype(str)):
            if src and tgt and tgt != 'nan' and src != tgt:
                if G_prhits.has_edge(src, tgt):
                    G_prhits[src][tgt]['weight'] += 1.0
                else:
                    G_prhits.add_edge(src, tgt, weight=1.0)
    # Mention edges — EQUAL weight for PR/HITS
    for src_val, mentions in zip(tw_df['source_user'].astype(str), tw_df['mentions']):
        if isinstance(mentions, list):
            for m in mentions:
                m_str = str(m)
                if src_val and m_str and m_str != 'nan' and src_val != m_str:
                    if G_prhits.has_edge(src_val, m_str):
                        G_prhits[src_val][m_str]['weight'] += 1.0
                    else:
                        G_prhits.add_edge(src_val, m_str, weight=1.0)

    pr_scores = nx.pagerank(G_prhits, alpha=0.85, weight='weight') if len(G_prhits) > 0 else {}
    try:
        _, hits_auth = nx.hits(G_prhits, max_iter=100, normalized=True) if len(G_prhits) > 0 else ({}, {})
    except Exception:
        hits_auth = {u: 0.001 for u in unique_users}

    # -----------------------------------------------------------------------
    # 2. Economic V-Anchor (LOOK-AHEAD BIAS FIX)
    #    Anchor correlations computed using ONLY training-period data.
    #    Uses the 'time' column (not index) since asset_df may have a
    #    RangeIndex after ensure_time_column() resets it in main.py.
    # -----------------------------------------------------------------------
    def _get_time_col(df):
        """Return the time column as a comparable Series of datetime.date."""
        if 'time' in df.columns:
            return pd.to_datetime(df['time'], errors='coerce').dt.date
        # Fallback: try the index
        return pd.to_datetime(df.index, errors='coerce').date

    if train_end_date is not None:
        train_end = pd.to_datetime(train_end_date).date()
        asset_time = _get_time_col(asset_df)
        mask        = asset_time <= train_end
        anchor_asset = asset_df[mask.values].copy()

        tw_time  = pd.to_datetime(tw_df['time'], errors='coerce').dt.date
        tw_mask  = tw_time <= train_end
        anchor_tw = tw_df[tw_mask.values].copy()
    else:
        anchor_asset = asset_df.copy()
        anchor_tw    = tw_df.copy()

    anchor_asset = anchor_asset.copy()

    # Ensure 'close' is accessible as a column
    if 'close' not in anchor_asset.columns:
        # May be in the index as the DataFrame was set_index('time') in older code
        anchor_asset = anchor_asset.reset_index()

    if 'close' not in anchor_asset.columns:
        # No price data available for anchor — use adaptive floor
        adaptive_floor = 1.0 / max(len(unique_users), 1)
        anchor_vector = {u: adaptive_floor for u in unique_users}
    else:
        anchor_asset['target_anchor'] = anchor_asset['close'].pct_change(7).shift(-7)

        # Align anchor_tw dates with anchor_asset dates for correlation
        if 'time' in anchor_tw.columns:
            anchor_tw = anchor_tw.copy()
            anchor_tw['time'] = pd.to_datetime(anchor_tw['time'], errors='coerce').dt.date
        anchor_asset['time'] = pd.to_datetime(
            anchor_asset['time'] if 'time' in anchor_asset.columns
            else anchor_asset.index, errors='coerce'
        ).dt.date if hasattr(anchor_asset.get('time', anchor_asset.index), 'dt') \
            else anchor_asset.get('time', pd.Series(dtype=object))

        # Pivot: rows = dates, cols = users, values = mean daily sentiment
        daily_sen = (
            anchor_tw.groupby(['source_user', 'time'])['sen']
            .mean()
            .unstack(level=0)
        )

        # Re-index anchor_asset by time for correlation
        if 'time' in anchor_asset.columns:
            anchor_asset_idx = anchor_asset.set_index('time')
        else:
            anchor_asset_idx = anchor_asset

        adaptive_floor = 1.0 / max(len(unique_users), 1)
        anchor_vector = {u: adaptive_floor for u in unique_users}
        for u in unique_users:
            if u in daily_sen.columns:
                corr = daily_sen[u].corr(anchor_asset_idx['target_anchor'])
                if pd.notnull(corr) and corr > 0:
                    anchor_vector[u] = corr

    # -----------------------------------------------------------------------
    # 3. LRR Variants
    # -----------------------------------------------------------------------
    # Baseline: social graph only (no cognitive gates, no anchor)
    lrr_social = compute_lrr_recursive(
        tw_df, unique_users, use_omega=False, use_con=False
    )

    # Full Oracle: Ω-gated propagation + V-Anchor (Con gates aggregation in Phase 3)
    lrr_oracle = compute_lrr_recursive(
        tw_df, unique_users, use_omega=True, use_con=True,
        anchor_vector=anchor_vector
    )

    return pr_scores, hits_auth, lrr_social, lrr_oracle, tw_df, unique_users, anchor_vector, G_prhits


# ===========================================================================
# Gap 7 — Network Statistics
# Gap 8 — User Heterogeneity
# ===========================================================================

def compute_network_statistics(G, pr_scores, hits_auth, lrr_oracle,
                                results_dir):
    """
    Gap 7: Compute social graph structural statistics.
    Reports degree distribution, density, clustering, and
    Gini coefficients of reputation score distributions.

    Gap 8: User heterogeneity — how concentrated is the LRR signal?
    Reports top 1%/5%/10% contribution share and Lorenz curve stats.
    """
    import numpy as np

    def gini(arr):
        """Gini coefficient for an array of non-negative values."""
        arr = np.array(arr, dtype=float)
        arr = arr[arr >= 0]
        if len(arr) == 0 or arr.sum() == 0:
            return 0.0
        arr = np.sort(arr)
        n    = len(arr)
        idx  = np.arange(1, n + 1)
        return float((2 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1) / n)

    def top_share(scores_dict, pct):
        vals = sorted(scores_dict.values(), reverse=True)
        n    = len(vals)
        k    = max(1, int(n * pct))
        return sum(vals[:k]) / sum(vals) if sum(vals) > 0 else 0.0

    stats = {}

    # Graph statistics
    if G is not None and len(G) > 0:
        stats['n_nodes']       = G.number_of_nodes()
        stats['n_edges']       = G.number_of_edges()
        stats['density']       = nx.density(G)
        in_degrees             = [d for _, d in G.in_degree()]
        out_degrees            = [d for _, d in G.out_degree()]
        stats['mean_in_deg']   = float(np.mean(in_degrees))  if in_degrees  else 0.0
        stats['max_in_deg']    = float(np.max(in_degrees))   if in_degrees  else 0.0
        stats['mean_out_deg']  = float(np.mean(out_degrees)) if out_degrees else 0.0
        # Clustering on undirected version
        try:
            G_und = G.to_undirected()
            stats['avg_clustering'] = nx.average_clustering(G_und)
        except Exception:
            stats['avg_clustering'] = np.nan
        # In-degree Gini (power-law check)
        stats['indegree_gini'] = gini(in_degrees)
    else:
        stats.update({'n_nodes': 0, 'n_edges': 0, 'density': 0,
                      'mean_in_deg': 0, 'max_in_deg': 0,
                      'mean_out_deg': 0, 'avg_clustering': 0,
                      'indegree_gini': 0})

    # Score distribution statistics (Gap 5)
    for score_name, score_dict in [
        ('PageRank', pr_scores),
        ('HITS',     hits_auth),
        ('LRR',      lrr_oracle),
    ]:
        vals = list(score_dict.values()) if score_dict else [0]
        stats[f'{score_name}_gini']     = round(gini(vals), 4)
        stats[f'{score_name}_top1pct']  = round(top_share(score_dict, 0.01), 4) if score_dict else 0
        stats[f'{score_name}_top5pct']  = round(top_share(score_dict, 0.05), 4) if score_dict else 0
        stats[f'{score_name}_top10pct'] = round(top_share(score_dict, 0.10), 4) if score_dict else 0
        stats[f'{score_name}_mean']     = round(float(np.mean(vals)), 6)
        stats[f'{score_name}_std']      = round(float(np.std(vals)), 6)

    # Save to file (appended — called once per run, covers all assets)
    out_path = os.path.join(results_dir, 'network_statistics.txt')
    mode = 'a' if os.path.exists(out_path) else 'w'
    with open(out_path, mode, encoding='utf-8') as f:
        if mode == 'w':
            f.write('=== Social Network Statistics (Gap 2 + Gap 5) ===\n\n')
        f.write('--- Graph Structure ---\n')
        f.write(f'  Nodes: {stats["n_nodes"]:,}  |  Edges: {stats["n_edges"]:,}\n')
        f.write(f'  Density: {stats["density"]:.6f}  |  '
                f'Mean in-degree: {stats["mean_in_deg"]:.2f}  |  '
                f'Max in-degree: {stats["max_in_deg"]:.0f}\n')
        f.write(f'  Average clustering: {stats["avg_clustering"]:.4f}\n')
        f.write(f'  In-degree Gini: {stats["indegree_gini"]:.4f}  '
                f'(1.0=perfectly concentrated, 0=uniform)\n\n')
        f.write('--- Reputation Score Concentration (Gap 5) ---\n')
        f.write(f'{"Signal":<12} {"Gini":>8} {"Top1%":>8} {"Top5%":>8} {"Top10%":>8}\n')
        f.write('-' * 48 + '\n')
        for s in ['PageRank', 'HITS', 'LRR']:
            f.write(f'{s:<12} '
                    f'{stats[f"{s}_gini"]:>8.4f} '
                    f'{stats[f"{s}_top1pct"]:>7.1%} '
                    f'{stats[f"{s}_top5pct"]:>7.1%} '
                    f'{stats[f"{s}_top10pct"]:>7.1%}\n')
        f.write('\n')

    return stats