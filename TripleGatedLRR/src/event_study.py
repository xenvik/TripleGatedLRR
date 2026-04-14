# src/event_study.py
"""
Automated Event Study
==============================
Identifies high-impact social events programmatically from the Twitter
dataset and tests whether the LRR signal response precedes price response.

Event identification strategy:
    1. Compute daily tweet volume and mean sentiment shift (|delta_sen|)
    2. Score each day as: event_score = z(volume) * z(|delta_sen|)
    3. Select top-N days where event_score > threshold (default: z > 2.0)
    4. Enforce minimum spacing of 10 days between events (de-clustering)

Event window: [-5, +10] trading days around each identified event.

Normalisation: all signals divided by their mean over [-5, -1] pre-event
baseline, giving a "relative response" where 1.0 = baseline level.

Aggregation: mean across all events (± SEM for error bands).

Key test: does LRR peak BEFORE price? Does omega shift BEFORE LRR?
This directly addresses the correlation-vs-causation critique.
"""

import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp


# ---------------------------------------------------------------------------
# Event identification
# ---------------------------------------------------------------------------

def identify_social_events(tw_df, n_events=20, min_spacing=10,
                            zscore_threshold=1.5):
    """
    Identifies peak social activity days as quasi-natural experiments.

    Scoring: z(daily_volume) * z(|daily_sentiment_change|)
    High score = both unusually active AND unusually opinionated day.

    Parameters
    ----------
    tw_df            : tweet DataFrame with 'time', 'sen', 'omega' columns
    n_events         : maximum number of events to return
    min_spacing      : minimum days between events (de-clustering)
    zscore_threshold : minimum combined z-score to qualify as an event

    Returns
    -------
    event_dates : sorted list of datetime.date objects
    event_df    : DataFrame with date, volume, sentiment_shift, score
    """
    tw = tw_df.copy()
    tw['time'] = pd.to_datetime(tw['time'], errors='coerce').dt.date
    tw = tw.dropna(subset=['time', 'sen'])

    daily = tw.groupby('time').agg(
        volume=('sen', 'count'),
        mean_sen=('sen', 'mean'),
        mean_omega=('omega', 'mean') if 'omega' in tw.columns else ('sen', 'count')
    ).reset_index().sort_values('time')

    daily['sen_change']  = daily['mean_sen'].diff().abs().fillna(0)
    daily['z_volume']    = (daily['volume']     - daily['volume'].mean())     / (daily['volume'].std()     + 1e-10)
    daily['z_sen_shift'] = (daily['sen_change'] - daily['sen_change'].mean()) / (daily['sen_change'].std() + 1e-10)

    # Score = sum of absolute z-scores (not product).
    # The product z_v * z_s collapses to ~0 when either component is near
    # its mean — which is most days in a stable dataset. Using the sum of
    # absolute values ensures we capture days that are anomalous on EITHER
    # volume OR sentiment shift (logical OR, not AND).
    # We then rank and take the top-N days without a hard threshold.
    daily['event_score'] = daily['z_volume'].abs() + daily['z_sen_shift'].abs()

    # Take top candidates by rank (no threshold gate — avoids empty results)
    candidates = daily.sort_values('event_score', ascending=False
    ).head(n_events * 3).reset_index(drop=True)

    # De-cluster: enforce minimum spacing
    selected    = []
    used_dates  = []
    for _, row in candidates.iterrows():
        d = row['time']
        if all(abs((pd.Timestamp(d) - pd.Timestamp(ud)).days) >= min_spacing
               for ud in used_dates):
            selected.append(row)
            used_dates.append(d)
        if len(selected) >= n_events:
            break

    event_df    = pd.DataFrame(selected).reset_index(drop=True)

    # Diagnostic: how many events passed the threshold?
    n_cands = len(candidates)
    print(f"   Event study: {n_cands} candidate days (rank-based), {len(selected)} after de-clustering")

    if event_df.empty or 'time' not in event_df.columns:
        return [], event_df

    event_dates = sorted(event_df['time'].tolist())
    return event_dates, event_df


# ---------------------------------------------------------------------------
# Event window extraction
# ---------------------------------------------------------------------------

def extract_event_windows(final_df, event_dates, pre=5, post=10):
    """
    Extracts normalised signal windows around each event date.

    Normalisation: each signal divided by its mean over the [-pre, -1]
    pre-event window. This removes level differences and focuses on
    the shape of the response.

    Returns:
        windows : dict {signal: 2D array of shape (n_events, pre+post+1)}
        lags    : array of lag values [-pre ... 0 ... post]
    """
    df = final_df.copy()
    # Defensive: ensure 'time' is a plain column (already normalised at entry)
    if 'time' not in df.columns:
        df = df.reset_index()
        if 'index' in df.columns:
            df = df.rename(columns={'index': 'time'})
    df['time'] = pd.to_datetime(df['time'], errors='coerce').dt.date
    df = df.sort_values('time').reset_index(drop=True)

    signals = {
        'LRR_Oracle_Sen': [],
        'omega':          [],
        'price_change':   [],
        'PageRank_Sen':   [],
    }
    # Only include signals that exist
    signals = {k: v for k, v in signals.items() if k in df.columns}

    lags        = list(range(-pre, post + 1))
    valid_events = 0

    # Normalise 'time' to string for robust cross-type comparison
    df['_date_str'] = pd.to_datetime(df['time'], errors='coerce').dt.strftime('%Y-%m-%d')

    for event_date in event_dates:
        ed_str = pd.Timestamp(event_date).strftime('%Y-%m-%d')

        # Find the index of the event date (string comparison avoids type mismatches)
        idx_matches = df[df['_date_str'] == ed_str].index
        if len(idx_matches) == 0:
            # Find nearest date within 2 days
            df['_ts_tmp'] = pd.to_datetime(df['_date_str'])
            ed_ts         = pd.Timestamp(ed_str)
            diff          = (df['_ts_tmp'] - ed_ts).abs()
            nearest       = diff.idxmin()
            if diff[nearest].days > 2:
                continue
            event_idx = nearest
        else:
            event_idx = idx_matches[0]

        start_idx = event_idx - pre
        end_idx   = event_idx + post

        if start_idx < 0 or end_idx >= len(df):
            continue

        window_df = df.iloc[start_idx:end_idx + 1]

        # Compute baseline mean (pre-event period)
        pre_window = df.iloc[start_idx:event_idx]

        for sig in signals:
            baseline = pre_window[sig].mean()
            if abs(baseline) < 1e-10:
                baseline = 1e-10

            norm_series = (window_df[sig].values / abs(baseline))
            if len(norm_series) == len(lags):
                signals[sig].append(norm_series)

        valid_events += 1

    # Convert to arrays
    windows = {}
    for sig, event_list in signals.items():
        if event_list:
            windows[sig] = np.array(event_list)

    return windows, np.array(lags), valid_events


# ---------------------------------------------------------------------------
# Statistical test: does LRR lead price?
# ---------------------------------------------------------------------------

def test_lrr_price_lead(windows, lags, pre=5):
    """
    Tests whether LRR peaks (relative to baseline) BEFORE price peaks.

    For each event:
        - Find the lag at which LRR_Oracle_Sen is maximum (post event)
        - Find the lag at which price_change is maximum (post event)
        - Compute lead = lag_price_peak - lag_LRR_peak
        - Positive lead = LRR peaks first

    t-test: H0: mean_lead <= 0 (LRR does not precede price)

    Returns dict with test results.
    """
    if 'LRR_Oracle_Sen' not in windows or 'price_change' not in windows:
        return {}

    post_mask  = lags >= 0
    post_lags  = lags[post_mask]

    lrr_data   = windows['LRR_Oracle_Sen'][:, post_mask]
    price_data = windows['price_change'][:, post_mask]

    leads = []
    for i in range(len(lrr_data)):
        lrr_peak_lag   = post_lags[np.argmax(np.abs(lrr_data[i]))]
        price_peak_lag = post_lags[np.argmax(np.abs(price_data[i]))]
        leads.append(int(price_peak_lag) - int(lrr_peak_lag))

    leads    = np.array(leads)
    mean_l   = float(leads.mean())
    std_l    = float(leads.std(ddof=1))
    n        = len(leads)

    if n > 1:
        t_stat, p_val = ttest_1samp(leads, 0.0, alternative='greater')
    else:
        t_stat, p_val = 0.0, 1.0

    sig = ('***' if p_val < 0.001 else '**' if p_val < 0.01
           else '*' if p_val < 0.05 else '')

    return {
        'n_events':   n,
        'mean_lead':  round(mean_l, 3),
        'std_lead':   round(std_l,  3),
        't_stat':     round(t_stat, 4),
        'p_value':    round(p_val,  4),
        'sig':        sig,
        'pct_positive_lead': round(float((leads > 0).mean()), 3),
        'leads':      leads.tolist(),
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _ensure_time_column(df, name='df'):
    """
    Guarantees the DataFrame has 'time' as a plain column of datetime.date.
    Handles: time as index, time as DatetimeIndex, time as string column,
    time already as datetime.date.
    """
    df = df.copy()

    # If 'time' is the index, bring it back as a column
    if 'time' not in df.columns:
        if df.index.name == 'time' or (
            hasattr(df.index, 'dtype') and
            str(df.index.dtype) in ('object', 'datetime64[ns]',
                                    'datetime64[us]')
        ):
            df = df.reset_index()
            if 'index' in df.columns and 'time' not in df.columns:
                df = df.rename(columns={'index': 'time'})

    if 'time' not in df.columns:
        raise KeyError(f"Cannot find 'time' column in {name}. "
                       f"Available columns: {list(df.columns)}, "
                       f"index name: {df.index.name}")

    # Normalise to datetime.date
    df['time'] = pd.to_datetime(df['time'], errors='coerce').dt.date
    return df


def run_event_study(tw_df, final_df, asset_name, results_dir,
                    n_events=20, pre=5, post=10):
    """
    Full event study pipeline for one asset.

    1. Identify events from Twitter data
    2. Extract normalised windows
    3. Test LRR-price lead
    4. Save results and plots
    """
    # Normalise inputs before any sub-function call
    # This is the single point that ensures 'time' is always a plain column
    try:
        tw_df    = _ensure_time_column(tw_df,    'tw_df')
        final_df = _ensure_time_column(final_df, 'final_df')
    except Exception as e:
        print(f"   ! {asset_name}: input normalisation failed — {e}")
        return None

    # Step 1: Identify events
    event_dates, event_df = identify_social_events(tw_df, n_events=n_events)

    if len(event_dates) < 5:
        print(f"   ! {asset_name}: only {len(event_dates)} events found "
              f"— skipping event study")
        return None

    # Step 2: Extract windows
    windows, lags, valid_n = extract_event_windows(
        final_df, event_dates, pre=pre, post=post
    )

    if valid_n < 5:
        print(f"   ! {asset_name}: only {valid_n} valid event windows "
              f"— skipping")
        return None

    # Step 3: Test LRR lead
    lead_test = test_lrr_price_lead(windows, lags)

    # Step 4: Save results
    _save_event_results(windows, lags, lead_test, event_df,
                        asset_name, results_dir, pre, post, valid_n)

    return lead_test


def _save_event_results(windows, lags, lead_test, event_df,
                        asset_name, results_dir, pre, post, n_valid):
    """Saves text summary, event CSV, and event study plot."""

    # Text summary
    with open(os.path.join(results_dir,
                           f'{asset_name.lower()}_EventStudy.txt'),
              'w', encoding='utf-8') as f:
        f.write(f'=== {asset_name} Event Study ===\n')
        f.write(f'Events identified: {len(event_df)}  '
                f'Valid windows: {n_valid}\n')
        f.write(f'Window: [-{pre}, +{post}] days around peak social activity\n\n')

        if lead_test:
            sig = lead_test.get('sig', '')
            f.write(f'--- LRR Lead-Price Test ---\n')
            f.write(f'Mean LRR lead over price: {lead_test["mean_lead"]:.2f} days '
                    f'(std={lead_test["std_lead"]:.2f})\n')
            f.write(f't-statistic: {lead_test["t_stat"]:.4f}\n')
            f.write(f'p-value (H0: lead<=0): {lead_test["p_value"]:.4f} {sig}\n')
            f.write(f'% events where LRR peaks before price: '
                    f'{lead_test["pct_positive_lead"]*100:.1f}%\n\n')
            f.write(f'Interpretation: '
                    f'{"LRR systematically leads price around social events" if lead_test["p_value"] < 0.05 else "No significant systematic LRR lead detected"}\n\n')

        f.write('--- Event Dates Identified ---\n')
        for _, row in event_df.iterrows():
            f.write(f"  {row['time']}: score={row['event_score']:.3f}  "
                    f"vol={row['volume']}  "
                    f"sen_shift={row['sen_change']:.4f}\n")

    # Event dates CSV
    event_df.to_csv(
        os.path.join(results_dir, f'{asset_name.lower()}_EventDates.csv'),
        index=False, encoding='utf-8'
    )

    # Plot
    _plot_event_study(windows, lags, lead_test, asset_name, results_dir,
                      n_valid, pre)


def _plot_event_study(windows, lags, lead_test, asset_name,
                      results_dir, n_valid, pre):
    """
    Event study plot:
    - Top panel: mean normalised LRR and omega response (with SEM bands)
    - Bottom panel: mean normalised price response
    - Vertical line at event day (lag=0)
    """
    sig_map = {
        'LRR_Oracle_Sen': ('LRR Oracle', '#e74c3c', 2.0),
        'omega':          ('Omega (ω)', '#f39c12', 1.5),
        'PageRank_Sen':   ('PageRank', '#3498db', 1.2),
        'price_change':   ('Price Return', '#2c3e50', 2.0),
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: social signals
    for sig in ['LRR_Oracle_Sen', 'omega', 'PageRank_Sen']:
        if sig not in windows:
            continue
        label, color, lw = sig_map[sig]
        data  = windows[sig]
        mean  = data.mean(axis=0)
        sem   = data.std(axis=0) / np.sqrt(data.shape[0])
        ax1.plot(lags, mean, label=label, color=color, linewidth=lw)
        ax1.fill_between(lags, mean - sem, mean + sem,
                         alpha=0.15, color=color)

    ax1.axvline(0, color='black', linewidth=1.0, linestyle='--',
                alpha=0.7, label='Event day')
    ax1.axhline(1.0, color='grey', linewidth=0.5, linestyle=':')
    ax1.set_ylabel('Signal (normalised to pre-event baseline)', fontsize=10)
    ax1.legend(fontsize=9, frameon=True)
    ax1.grid(True, alpha=0.15)

    # Bottom: price
    if 'price_change' in windows:
        label, color, lw = sig_map['price_change']
        data  = windows['price_change']
        mean  = data.mean(axis=0)
        sem   = data.std(axis=0) / np.sqrt(data.shape[0])
        ax2.plot(lags, mean, label=label, color=color, linewidth=lw)
        ax2.fill_between(lags, mean - sem, mean + sem,
                         alpha=0.15, color=color)
        ax2.axvline(0, color='black', linewidth=1.0, linestyle='--', alpha=0.7)
        ax2.axhline(1.0, color='grey', linewidth=0.5, linestyle=':')
        ax2.set_ylabel('Price return (normalised)', fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.15)

    ax2.set_xlabel('Days relative to peak social event', fontsize=10)

    # Title with lead test result
    lead_str = ''
    if lead_test:
        lead_str = (f'\nMean LRR lead: {lead_test["mean_lead"]:.1f} days  '
                    f'p={lead_test["p_value"]:.3f}{lead_test.get("sig","")}  '
                    f'n={n_valid} events')

    fig.suptitle(
        f'{asset_name} — Event Study: LRR vs Price Response Around Peak Social Activity{lead_str}',
        fontsize=11, y=1.01
    )
    fig.tight_layout()

    plt.savefig(
        os.path.join(results_dir, f'{asset_name.lower()}_event_study.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close('all')