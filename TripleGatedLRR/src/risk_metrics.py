# src/risk_metrics.py
import numpy as np
import pandas as pd
from src.config import BOOTSTRAP_N, BOOTSTRAP_SEED


# ---------------------------------------------------------------------------
# Lower Tail Dependence
# ---------------------------------------------------------------------------

def compute_tail_dependence(signal, returns, quantile=0.10, verbose=True):
    """
    Empirical Lower Tail Dependence (LTD).

    Measures the conditional probability that the market crashes given
    the signal is also in its lower tail.  A LOW LTD for LRR-Oracle
    means the signal decouples cleanly from price crashes — i.e., it
    retains predictive accuracy even during extreme market stress.

    LTD(λ) = P(Return ≤ Q_λ(Return) | Signal ≤ Q_λ(Signal))

    Dynamic quantile thresholding:
    For small samples (N < 200), the tail quantile is dynamically
    adjusted upward so the empirical tail contains enough observations
    to form a reliable conditional probability. If fewer than 5 joint
    tail events remain after adjustment, returns NaN with a warning.
    """
    df = pd.DataFrame({'s': signal, 'r': returns}).dropna()
    n  = len(df)

    if n < 30:
        return np.nan

    # Dynamic quantile: ensure at least 10 obs in the signal tail
    MIN_TAIL_OBS = 10
    MIN_JOINT    = 2   # Lowered from 5; values with <5 events flagged as low-reliability
    q = quantile

    if n < 200:
        # Scale up quantile to guarantee at least MIN_TAIL_OBS in tail
        q = max(quantile, MIN_TAIL_OBS / n)
        q = min(q, 0.25)   # cap at 25th percentile

    q_s = df['s'].quantile(q)
    q_r = df['r'].quantile(q)

    in_tail_s = df['s'] <= q_s
    in_tail_r = df['r'] <= q_r

    denom      = in_tail_s.sum()
    joint      = (in_tail_s & in_tail_r).sum()

    if denom == 0:
        return np.nan

    if joint < MIN_JOINT:
        if verbose:
            print(f'   ! LTD warning: only {joint} joint tail events '
                  f'(N={n}, q={q:.2f}) — returning NaN')
        return np.nan

    return float(joint / denom)


def compute_tail_dependence_extended(signal, returns, quantile=0.10, verbose=True):
    """
    Extended LTD: returns (ltd_value, joint_count, tail_count).

    Same logic as compute_tail_dependence but also reports the raw
    number of joint tail events and total signal-tail events — critical
    for interpreting NaN cases as crash decoupling evidence.
    """
    df = pd.DataFrame({'s': signal, 'r': returns}).dropna()
    n  = len(df)

    if n < 30:
        return np.nan, 0, 0

    MIN_TAIL_OBS = 10
    MIN_JOINT    = 2   # Lowered from 5; values with <5 events flagged as low-reliability
    q = quantile

    if n < 200:
        q = max(quantile, MIN_TAIL_OBS / n)
        q = min(q, 0.25)

    q_s = df['s'].quantile(q)
    q_r = df['r'].quantile(q)

    in_tail_s = df['s'] <= q_s
    in_tail_r = df['r'] <= q_r

    denom = int(in_tail_s.sum())
    joint = int((in_tail_s & in_tail_r).sum())

    if denom == 0:
        return np.nan, 0, denom

    if joint < MIN_JOINT:
        if verbose:
            print(f'   ! LTD warning: only {joint} joint tail events '
                  f'(N={n}, q={q:.2f}) — returning NaN')
        return np.nan, joint, denom

    return float(joint / denom), joint, denom


def bootstrap_ltd_reduction_test(signal_full, signal_no_con, returns,
                                  quantile=0.10,
                                  n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """
    Bootstrap paired confidence interval and p-value for the LTD reduction
    attributable to the Con gate.

    Test statistic: risk_reduction = (LTD_no_con - LTD_full) / LTD_no_con * 100
    H0: risk_reduction <= 0  (Con gate provides no tail-risk benefit)

    Approach: paired bootstrap (resample daily observations with replacement).
    Both signals are drawn from the same bootstrap sample, preserving the
    time-alignment between them and the returns.

    Returns:
        p_value   : P(bootstrap_reduction <= 0) under H0
        ci_lower  : 2.5th percentile of bootstrap distribution
        ci_upper  : 97.5th percentile of bootstrap distribution
        observed  : observed risk reduction (%)
    """
    df = pd.DataFrame({
        'full':   signal_full,
        'no_con': signal_no_con,
        'ret':    returns
    }).dropna()

    n   = len(df)
    rng = np.random.default_rng(seed)

    observed_ltd_full   = compute_tail_dependence(df['full'],   df['ret'], quantile)
    observed_ltd_no_con = compute_tail_dependence(df['no_con'], df['ret'], quantile)
    safe_nc  = max(observed_ltd_no_con, 0.001)
    observed = (safe_nc - max(observed_ltd_full, 0.001)) / safe_nc * 100.0

    boot_reductions = []
    for _ in range(n_boot):
        idx  = rng.integers(0, n, size=n)
        b    = df.iloc[idx].reset_index(drop=True)
        ltd_f  = compute_tail_dependence(b['full'],   b['ret'], quantile, verbose=False)
        ltd_nc = compute_tail_dependence(b['no_con'], b['ret'], quantile, verbose=False)
        if np.isnan(ltd_f) or np.isnan(ltd_nc):
            continue
        s_nc   = max(ltd_nc, 0.001)
        s_f    = max(ltd_f,  0.001)
        boot_reductions.append((s_nc - s_f) / s_nc * 100.0)

    if len(boot_reductions) < 10:
        return np.nan, np.nan, np.nan, observed

    boot_arr  = np.array(boot_reductions)
    p_value   = float(np.mean(boot_arr <= 0.0))
    ci_lower  = float(np.percentile(boot_arr, 2.5))
    ci_upper  = float(np.percentile(boot_arr, 97.5))

    return p_value, ci_lower, ci_upper, observed


# ---------------------------------------------------------------------------
# Transfer Entropy (True Conditional Mutual Information)
# ---------------------------------------------------------------------------

def calculate_transfer_entropy(source_signal, target_returns, lag=7, bins=8):
    """
    Transfer Entropy: TE(X → Y, lag) = I(Y_t ; X_{t-lag} | Y_{t-lag})

    Measures the UNIQUE predictive information flow from X to Y beyond
    what Y's own past already provides.  This is the correct information-
    theoretic measure of directed predictability, not simple MI(X, Y).

    Formula:
        TE = H(Y_t | Y_{t-lag}) - H(Y_t | Y_{t-lag}, X_{t-lag})
           = [H(Y_t, Y_{t-lag}) - H(Y_{t-lag})]
             - [H(Y_t, Y_{t-lag}, X_{t-lag}) - H(Y_{t-lag}, X_{t-lag})]

    Implementation: discrete histogram estimator with Laplace smoothing.
    bins=8 balances resolution vs. sample-size requirements for daily data.

    Returns non-negative float (bits).
    """
    df = pd.DataFrame({
        'y_t':    target_returns,
        'y_past': target_returns.shift(lag),
        'x_past': source_signal.shift(lag)
    }).dropna()

    min_obs = max(50, bins ** 2)
    if len(df) < min_obs:
        return 0.0

    def _discretize(series, n):
        lo = series.min() - 1e-10
        hi = series.max() + 1e-10
        edges = np.linspace(lo, hi, n + 1)
        return np.clip(np.digitize(series, edges[:-1]) - 1, 0, n - 1)

    yt  = _discretize(df['y_t'].values,    bins)
    yp  = _discretize(df['y_past'].values, bins)
    xp  = _discretize(df['x_past'].values, bins)

    eps = 1e-10  # Laplace smoothing

    def _entropy(counts):
        counts = counts + eps
        p = counts / counts.sum()
        return float(-np.sum(p * np.log(p)))

    # H(Y_t, Y_past)
    h_yy  = _entropy(np.histogram2d(yt, yp, bins=bins)[0])
    # H(Y_past)
    h_yp  = _entropy(np.bincount(yp, minlength=bins).astype(float))
    # H(Y_t, Y_past, X_past)
    h_yyx = _entropy(np.histogramdd(
                        np.column_stack([yt, yp, xp]), bins=bins)[0])
    # H(Y_past, X_past)
    h_yx  = _entropy(np.histogram2d(yp, xp, bins=bins)[0])

    # TE = H(Y|Y_past) - H(Y|Y_past, X_past)
    te = (h_yy - h_yp) - (h_yyx - h_yx)
    return max(float(te), 0.0)


def calculate_conditional_transfer_entropy(source, target, condition, lag=7, bins=6):
    """
    Conditional Transfer Entropy: TE(X -> Y | Z)
    
    Measures the information flow from X to Y that remains AFTER
    conditioning on Z. This is the information-theoretic analog of
    partial Granger causality.
    
    TE(X->Y|Z) = H(Y_t | Y_past, Z_past) - H(Y_t | Y_past, X_past, Z_past)
    
    If TE(LRR->omega|HITS) > 0 and significant, LRR carries unique
    information about omega that HITS doesn't capture.
    
    Uses bins=6 (lower than standard TE) to handle the 4D histogram
    without curse-of-dimensionality issues on 604 observations.
    """
    df = pd.DataFrame({
        'y_t':    target,
        'y_past': target.shift(lag),
        'x_past': source.shift(lag),
        'z_past': condition.shift(lag),
    }).dropna()
    
    min_obs = max(100, bins ** 3)
    if len(df) < min_obs:
        return 0.0
    
    def _discretize(series, n):
        lo = series.min() - 1e-10
        hi = series.max() + 1e-10
        edges = np.linspace(lo, hi, n + 1)
        return np.clip(np.digitize(series, edges[:-1]) - 1, 0, n - 1)
    
    yt = _discretize(df['y_t'].values, bins)
    yp = _discretize(df['y_past'].values, bins)
    xp = _discretize(df['x_past'].values, bins)
    zp = _discretize(df['z_past'].values, bins)
    
    eps = 1e-10
    
    def _entropy(counts):
        counts = counts + eps
        p = counts / counts.sum()
        return float(-np.sum(p * np.log(p)))
    
    # H(Y_t, Y_past, Z_past) — 3D
    h_yyz = _entropy(np.histogramdd(np.column_stack([yt, yp, zp]), bins=bins)[0])
    # H(Y_past, Z_past) — 2D
    h_yz = _entropy(np.histogram2d(yp, zp, bins=bins)[0])
    # H(Y_t, Y_past, X_past, Z_past) — 4D
    h_yyxz = _entropy(np.histogramdd(np.column_stack([yt, yp, xp, zp]), bins=bins)[0])
    # H(Y_past, X_past, Z_past) — 3D
    h_yxz = _entropy(np.histogramdd(np.column_stack([yp, xp, zp]), bins=bins)[0])
    
    # CTE = H(Y|Y_past,Z_past) - H(Y|Y_past,X_past,Z_past)
    cte = (h_yyz - h_yz) - (h_yyxz - h_yxz)
    return max(float(cte), 0.0)


def calculate_mutual_information(x, y, bins=10):
    """
    Symmetric Mutual Information I(X; Y).
    Used for the ablation denoising lag-sweep chart (not for TE claims).
    Clearly labelled as MI, not TE, in all outputs.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return 0.0
    c_xy = np.histogram2d(x, y, bins=bins)[0] + 1e-10
    p_xy = c_xy / c_xy.sum()
    p_x  = p_xy.sum(axis=1, keepdims=True)
    p_y  = p_xy.sum(axis=0, keepdims=True)
    mi   = np.sum(p_xy * np.log(p_xy / (p_x * p_y + 1e-10)))
    return max(float(mi), 0.0)


# ---------------------------------------------------------------------------
# Directional Accuracy
# ---------------------------------------------------------------------------

def compute_directional_accuracy(actual, predicted):
    """
    Percentage of correctly predicted price-movement direction (sign accuracy).

    This is the primary practitioner-relevant metric: regardless of the
    magnitude of the forecast error, does the model correctly identify
    whether the price will go up or down?

    Returns float in [0, 1].  0.5 = random baseline.
    """
    df = pd.DataFrame({'a': actual, 'p': predicted}).dropna()
    if len(df) == 0:
        return np.nan
    correct = (np.sign(df['a']) == np.sign(df['p'])).sum()
    return float(correct / len(df))


def directional_accuracy_significance(actual, predicted, n_boot=BOOTSTRAP_N,
                                       seed=BOOTSTRAP_SEED):
    """
    Bootstrap p-value for directional accuracy.
    H0: model accuracy <= 0.50 (no better than a coin flip).

    Returns:
        da       : observed directional accuracy
        p_value  : P(bootstrap_da >= observed | H0: da_true = 0.5)
        ci_lower : 2.5th percentile bootstrap CI
        ci_upper : 97.5th percentile bootstrap CI
    """
    df  = pd.DataFrame({'a': actual, 'p': predicted}).dropna().reset_index(drop=True)
    n   = len(df)
    rng = np.random.default_rng(seed)

    da_obs = compute_directional_accuracy(df['a'], df['p'])

    boot_das = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        b   = df.iloc[idx].reset_index(drop=True)
        boot_das.append(compute_directional_accuracy(b['a'], b['p']))

    boot_arr  = np.array(boot_das)
    p_value   = float(np.mean(boot_arr <= 0.50))
    ci_lower  = float(np.percentile(boot_arr, 2.5))
    ci_upper  = float(np.percentile(boot_arr, 97.5))

    return da_obs, p_value, ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Con Gate Permutation Test
# ---------------------------------------------------------------------------

def permutation_test_ltd_reduction(signal_full, signal_no_con, returns,
                                    quantile=0.10, n_perm=1000,
                                    seed=BOOTSTRAP_SEED):
    """
    Permutation test for Con gate LTD reduction.

    H0: observed LTD reduction = 0 (Con gate has no effect).
    Approach: randomly permute the Con scores across tweets,
    recompute LTD reduction, build null distribution.

    This is stronger than bootstrap because it directly tests
    whether the Con gate assignment is non-random.

    Returns:
        observed_reduction : float (%)
        p_value            : P(perm_reduction >= observed | H0)
        null_mean          : mean of permutation null distribution
        null_std           : std of permutation null distribution
    """
    df = pd.DataFrame({
        'full':   signal_full,
        'no_con': signal_no_con,
        'ret':    returns
    }).dropna()

    if len(df) < 30:
        return np.nan, np.nan, np.nan, np.nan

    # Observed
    ltd_f  = compute_tail_dependence(df['full'],   df['ret'], quantile)
    ltd_nc = compute_tail_dependence(df['no_con'], df['ret'], quantile)
    safe_nc  = max(ltd_nc, 0.001)
    safe_f   = max(ltd_f,  0.001)
    observed = (safe_nc - safe_f) / safe_nc * 100.0

    # Null distribution: permute the gap between full and no_con
    rng      = np.random.default_rng(seed)
    gap      = (df['full'] - df['no_con']).values
    perm_reds = []

    for _ in range(n_perm):
        perm_gap     = rng.permutation(gap)
        perm_full    = df['no_con'].values + perm_gap
        perm_ltd_f   = compute_tail_dependence(
            pd.Series(perm_full), df['ret'], quantile, verbose=False)
        perm_ltd_nc  = compute_tail_dependence(df['no_con'], df['ret'], quantile, verbose=False)
        if np.isnan(perm_ltd_f) or np.isnan(perm_ltd_nc):
            continue
        s_nc = max(perm_ltd_nc, 0.001)
        s_f  = max(perm_ltd_f,  0.001)
        perm_reds.append((s_nc - s_f) / s_nc * 100.0)

    if len(perm_reds) < 10:
        return observed, np.nan, np.nan, np.nan

    null_arr  = np.array(perm_reds)
    p_value   = float(np.mean(null_arr >= observed))
    null_mean = float(null_arr.mean())
    null_std  = float(null_arr.std())

    return observed, p_value, null_mean, null_std


def pooled_con_gate_significance(asset_results, alpha=0.05):
    """
    Fisher's method for pooling p-values across assets.

    asset_results : dict {asset_name: {'boot_p': float, 'perm_p': float}}

    Returns combined chi-squared statistic and p-value under H0
    that ALL assets have no Con gate effect simultaneously.
    Uses Fisher's combined probability test:
        X^2 = -2 * sum(ln(p_i))  ~ chi2(2k) under H0
    """
    from scipy.stats import chi2

    boot_ps = [v['boot_p'] for v in asset_results.values()
               if not np.isnan(v.get('boot_p', np.nan))]
    perm_ps = [v['perm_p'] for v in asset_results.values()
               if not np.isnan(v.get('perm_p', np.nan))]

    results = {}
    for label, ps in [('Bootstrap', boot_ps), ('Permutation', perm_ps)]:
        if len(ps) < 2:
            results[label] = {'chi2': np.nan, 'p_combined': np.nan, 'k': len(ps)}
            continue
        ps_clipped = np.clip(ps, 1e-10, 1.0)
        stat = -2.0 * np.sum(np.log(ps_clipped))
        df_  = 2 * len(ps_clipped)
        p_combined = float(1.0 - chi2.cdf(stat, df_))
        results[label] = {
            'chi2': round(stat, 4),
            'p_combined': round(p_combined, 6),
            'k': len(ps_clipped),
            'individual_ps': ps
        }

    return results


# ---------------------------------------------------------------------------
# Information Efficiency Ratio (IER)
# ---------------------------------------------------------------------------

def compute_ier(te_value, ltd_value):
    """
    Information Efficiency Ratio: IER = TE / LTD

    Higher IER = more predictive information per unit of crash exposure.
    A signal that maximises TE while minimising LTD is the ideal.

    LRR's design deliberately suppresses crash-correlated noise (reducing TE)
    but also dramatically reduces LTD. The IER quantifies whether this
    trade-off is favourable compared to naive benchmarks.

    Returns nan if ltd_value <= 0.
    """
    if ltd_value <= 0 or np.isnan(ltd_value) or np.isnan(te_value):
        return np.nan
    return round(float(te_value / ltd_value), 6)


def compute_ier_table(risk_dict, asset_name):
    """
    Compute IER for all three signals for a given asset.

    risk_dict: {'LRR_Oracle': {'te': float, 'ltd': float},
                'PageRank':   {'te': float, 'ltd': float},
                'HITS':       {'te': float, 'ltd': float}}

    Returns a list of dicts for table construction.
    """
    rows = []
    for signal, vals in risk_dict.items():
        ier = compute_ier(vals['te'], vals['ltd'])
        rows.append({
            'asset':  asset_name,
            'signal': signal,
            'TE':     round(vals['te'],  6),
            'LTD':    round(vals['ltd'], 6),
            'IER':    ier,
        })
    return rows


# ---------------------------------------------------------------------------
# IER Significance Test (LRR IER vs PageRank IER)
# ---------------------------------------------------------------------------

def bootstrap_ier_superiority(ier_data, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """
    Paired bootstrap test: is LRR IER > PageRank IER across assets?

    ier_data: list of dicts with keys 'asset', 'signal', 'IER'
    (output of compute_ier_table accumulated across all assets)

    H0: mean(IER_LRR - IER_PageRank) <= 0

    Returns:
        observed_diff : mean difference across assets
        p_value       : P(boot_diff <= 0 | H0)
        ci_lo, ci_hi  : 95% bootstrap CI for the difference
    """
    df = pd.DataFrame(ier_data)
    if df.empty:
        return np.nan, np.nan, np.nan, np.nan

    assets = df['asset'].unique().tolist()
    pairs  = []
    for asset in assets:
        lrr_row = df[(df['asset'] == asset) & (df['signal'] == 'LRR_Oracle')]
        pr_row  = df[(df['asset'] == asset) & (df['signal'] == 'PageRank')]
        if not lrr_row.empty and not pr_row.empty:
            pairs.append({
                'asset':   asset,
                'lrr_ier': float(lrr_row['IER'].values[0]),
                'pr_ier':  float(pr_row['IER'].values[0]),
            })

    if len(pairs) < 2:
        return np.nan, np.nan, np.nan, np.nan

    pairs_df   = pd.DataFrame(pairs)
    diffs      = (pairs_df['lrr_ier'] - pairs_df['pr_ier']).values
    observed   = float(diffs.mean())

    rng        = np.random.default_rng(seed)
    boot_diffs = []
    n          = len(diffs)
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_diffs.append(sample.mean())

    boot_arr   = np.array(boot_diffs)
    p_value    = float(np.mean(boot_arr <= 0.0))
    ci_lo      = float(np.percentile(boot_arr, 2.5))
    ci_hi      = float(np.percentile(boot_arr, 97.5))

    return observed, p_value, ci_lo, ci_hi