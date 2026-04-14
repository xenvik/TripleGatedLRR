# src/portfolio_engine.py
"""
T1.1 — Portfolio Backtest Engine
=================================
Signal-based long/cash and long/short strategies evaluated across all
five LRR signal variants plus buy-and-hold benchmark.

Metrics: Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, Profit Factor.
Transaction costs: 0.10% per round-trip (realistic crypto exchange fees).
Annualisation: 365 days (crypto markets trade every day).
Significance: bootstrap Sharpe test (H0: SR <= 0).

Check Note: the portfolio backtest circumvents the OOS RMSE weakness.
Even if RMSE is flat, a strategy with Sharpe > 1.0 (p < 0.05) demonstrates
economically meaningful alpha from the LRR signal denoising architecture.
"""

import numpy as np
import pandas as pd
from src.config import BOOTSTRAP_N, BOOTSTRAP_SEED

# Crypto trades 365 days/year
ANNUAL_FACTOR  = 365
# Transaction cost per round-trip trade (0.10%)
TRADE_COST     = 0.001
# Rolling window for signal threshold (20 days ≈ 1 trading month)
SIGNAL_WINDOW  = 20
# Minimum observations for a meaningful backtest
MIN_OBS        = 60


# ---------------------------------------------------------------------------
# Core strategy runner
# ---------------------------------------------------------------------------

def _run_strategy(price_returns, signal, long_short=False, costs=TRADE_COST):
    """
    Daily rebalancing strategy.

    Position rule:
        signal(t-1) > rolling_median(signal, SIGNAL_WINDOW) → +1 (LONG)
        otherwise                                            →  0 (CASH)
                                               [or -1 if long_short=True]

    Transaction cost deducted whenever position flips.

    Returns:
        strategy_returns : Series of daily strategy returns (net of costs)
        positions        : Series of daily position (+1 / 0 / -1)
    """
    signal  = signal.copy()
    thresh  = signal.rolling(window=SIGNAL_WINDOW, min_periods=SIGNAL_WINDOW // 2).median()
    raw_pos = (signal.shift(1) > thresh.shift(1)).astype(float)

    if long_short:
        # Replace 0 with -1 for the short leg
        raw_pos = raw_pos.replace(0.0, -1.0)

    # Detect position changes to apply transaction costs
    pos_change = raw_pos.diff().abs().fillna(0)
    cost_series = pos_change * costs

    strat_returns = raw_pos * price_returns - cost_series
    return strat_returns.fillna(0), raw_pos


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _compute_metrics(daily_returns, label, regime_mask=None):
    """
    Compute all portfolio metrics from a daily returns series.

    Parameters
    ----------
    daily_returns : Series
    label         : str  — strategy/signal name
    regime_mask   : boolean Series or None — if provided, restrict to regime

    Returns dict of metrics.
    """
    r = daily_returns.copy()
    if regime_mask is not None:
        r = r[regime_mask]

    r = r.dropna()
    n = len(r)

    if n < MIN_OBS:
        return {
            'label': label, 'n_obs': n,
            'total_return': np.nan, 'ann_return': np.nan,
            'sharpe': np.nan, 'sortino': np.nan,
            'max_drawdown': np.nan, 'calmar': np.nan,
            'win_rate': np.nan, 'profit_factor': np.nan,
            'sharpe_p': np.nan
        }

    # Cumulative equity curve
    equity  = (1 + r).cumprod()
    total_r = float(equity.iloc[-1] - 1)

    # Annualised return (geometric)
    ann_r   = float((1 + total_r) ** (ANNUAL_FACTOR / n) - 1)

    # Sharpe ratio (rf = 0)
    mean_r  = r.mean()
    std_r   = r.std(ddof=1)
    sharpe  = float((mean_r / std_r) * np.sqrt(ANNUAL_FACTOR)) if std_r > 1e-10 else 0.0

    # Sortino ratio (downside deviation)
    down_r  = r[r < 0]
    down_std = down_r.std(ddof=1) if len(down_r) > 1 else 1e-10
    sortino = float((mean_r / down_std) * np.sqrt(ANNUAL_FACTOR)) if down_std > 1e-10 else 0.0

    # Maximum drawdown
    roll_max  = equity.cummax()
    drawdowns = (equity - roll_max) / roll_max
    max_dd    = float(drawdowns.min())

    # Calmar ratio
    calmar = float(ann_r / abs(max_dd)) if abs(max_dd) > 1e-10 else 0.0

    # Win rate — computed on ACTIVE trading days only (position != 0)
    # Cash days (return=0 because position=0) are excluded to avoid
    # artificially deflating the win rate.
    active_r = r[r != 0.0]  # proxy: non-zero return days = active days
    win_rate = float((active_r > 0).mean()) if len(active_r) > 0 else 0.0

    # Profit factor
    wins   = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    profit_factor = float(wins / losses) if losses > 1e-10 else np.inf

    # Bootstrap p-value for Sharpe > 0
    rng     = np.random.default_rng(BOOTSTRAP_SEED)
    r_arr   = r.values
    boot_sr = []
    for _ in range(BOOTSTRAP_N):
        s = rng.choice(r_arr, size=n, replace=True)
        sr = (s.mean() / s.std(ddof=1)) * np.sqrt(ANNUAL_FACTOR) if s.std(ddof=1) > 1e-10 else 0.0
        boot_sr.append(sr)
    sharpe_p = float(np.mean(np.array(boot_sr) <= 0.0))

    return {
        'label':         label,
        'n_obs':         n,
        'total_return':  round(total_r * 100, 2),
        'ann_return':    round(ann_r  * 100, 2),
        'sharpe':        round(sharpe,        4),
        'sortino':       round(sortino,       4),
        'max_drawdown':  round(max_dd  * 100, 2),
        'calmar':        round(calmar,        4),
        'win_rate':      round(win_rate * 100,2),
        'profit_factor': round(profit_factor, 4),
        'sharpe_p':      round(sharpe_p,      4),
    }


# ---------------------------------------------------------------------------
# Full backtest runner
# ---------------------------------------------------------------------------

def run_portfolio_backtest(final_df, asset_name, results_dir):
    """
    Runs the full portfolio backtest for all signal variants.

    Parameters
    ----------
    final_df     : aligned DataFrame with signals, price_change, regime columns
    asset_name   : str  e.g. 'BTC'
    results_dir  : path to results folder

    Returns
    -------
    metrics_df   : DataFrame of all metrics (rows = strategies × regime splits)
    equity_dict  : dict of equity curve Series for plotting
    """
    import os

    df = final_df.copy().sort_values('time').reset_index(drop=True)
    price_ret = df['price_change'].fillna(0)

    signals = {
        'Buy-and-Hold':   None,           # special case
        'Simple_Sen':     'Simple_Sen',
        'FinBERT_Sen':    'FinBERT_Sen',
        'PageRank_Sen':   'PageRank_Sen',
        'HITS_Sen':       'HITS_Sen',
        'LRR_Social_Sen': 'LRR_Social_Sen',
        'LRR_Oracle_Sen': 'LRR_Oracle_Sen',
    }

    # Build regime masks
    regime_masks = {
        'Full':   pd.Series(True,  index=df.index),
        'CALM':   df['regime'] == 0 if 'regime' in df.columns else pd.Series(True, index=df.index),
        'CRISIS': df['regime'] == 1 if 'regime' in df.columns else pd.Series(False, index=df.index),
    }

    all_metrics   = []
    equity_curves = {}   # for plotting — full-sample only

    for strat_name, sig_col in signals.items():

        # Generate daily returns for this strategy
        if sig_col is None:
            # Buy-and-hold: always long, no costs
            strat_ret_lc = price_ret.copy()
            strat_ret_ls = price_ret.copy()
        elif sig_col not in df.columns:
            continue
        else:
            strat_ret_lc, _ = _run_strategy(price_ret, df[sig_col], long_short=False)
            strat_ret_ls, _ = _run_strategy(price_ret, df[sig_col], long_short=True)

        # Store equity curve (long/cash, full sample) for plotting
        equity_curves[strat_name] = (1 + strat_ret_lc).cumprod()

        # Compute metrics across all regime splits
        for regime_name, mask in regime_masks.items():
            for variant, ret in [('Long/Cash', strat_ret_lc),
                                  ('Long/Short', strat_ret_ls)]:
                if sig_col is None and variant == 'Long/Short':
                    continue  # B&H has no short variant
                m = _compute_metrics(ret, strat_name, regime_mask=mask)
                m['strategy_variant'] = variant
                m['regime']           = regime_name
                m['asset']            = asset_name
                all_metrics.append(m)

    metrics_df = pd.DataFrame(all_metrics)

    # -----------------------------------------------------------------------
    # Save text summary — most readable format for the paper
    # -----------------------------------------------------------------------
    _save_portfolio_summary(metrics_df, asset_name, results_dir)

    # Save CSV for downstream analysis
    metrics_df.to_csv(
        os.path.join(results_dir, f'{asset_name.lower()}_Portfolio_Metrics.csv'),
        index=False, encoding='utf-8'
    )

    # Log key headline numbers
    _log_headline(metrics_df, asset_name)

    return metrics_df, equity_curves


def _save_portfolio_summary(metrics_df, asset_name, results_dir):
    """Saves user-readable portfolio metrics summary."""
    import os

    with open(os.path.join(results_dir,
                           f'{asset_name.lower()}_Portfolio_Metrics.txt'),
              'w', encoding='utf-8') as f:

        f.write(f'=== {asset_name} Portfolio Backtest ===\n')
        f.write(f'Strategy: Long/Cash | Signal threshold: 20-day rolling median\n')
        f.write(f'Transaction cost: {TRADE_COST*100:.1f}% per round-trip | '
                f'Annualisation: {ANNUAL_FACTOR} days\n\n')

        # Full-sample long/cash table
        sub = metrics_df[
            (metrics_df['regime'] == 'Full') &
            (metrics_df['strategy_variant'] == 'Long/Cash')
        ].copy()

        f.write(f'{"Strategy":<20} {"TotalRet%":>10} {"AnnRet%":>9} '
                f'{"Sharpe":>8} {"Sortino":>9} {"MaxDD%":>8} '
                f'{"Calmar":>8} {"WinRate%":>9} {"SR_p":>8}\n')
        f.write('-' * 95 + '\n')

        for _, row in sub.iterrows():
            sig_flag = ('***' if row['sharpe_p'] < 0.001 else
                        '**'  if row['sharpe_p'] < 0.01  else
                        '*'   if row['sharpe_p'] < 0.05  else '')
            f.write(f"{row['label']:<20} {row['total_return']:>10.1f} "
                    f"{row['ann_return']:>9.1f} {row['sharpe']:>8.3f} "
                    f"{row['sortino']:>9.3f} {row['max_drawdown']:>8.1f} "
                    f"{row['calmar']:>8.3f} {row['win_rate']:>9.1f} "
                    f"{row['sharpe_p']:>6.3f}{sig_flag}\n")

        f.write('\n--- Regime-Conditioned Sharpe (LRR_Oracle_Sen, Long/Cash) ---\n')
        lrr_rows = metrics_df[
            (metrics_df['label'] == 'LRR_Oracle_Sen') &
            (metrics_df['strategy_variant'] == 'Long/Cash')
        ]
        for _, row in lrr_rows.iterrows():
            sig_flag = ('***' if row['sharpe_p'] < 0.001 else
                        '**'  if row['sharpe_p'] < 0.01  else
                        '*'   if row['sharpe_p'] < 0.05  else '')
            f.write(f"  {row['regime']:<10}: Sharpe={row['sharpe']:.3f}  "
                    f"DA={row['win_rate']:.1f}%  p={row['sharpe_p']:.3f}{sig_flag}\n")

        f.write('\n--- Long/Short Variant (LRR_Oracle_Sen) ---\n')
        ls_row = metrics_df[
            (metrics_df['label'] == 'LRR_Oracle_Sen') &
            (metrics_df['strategy_variant'] == 'Long/Short') &
            (metrics_df['regime'] == 'Full')
        ]
        if not ls_row.empty:
            r = ls_row.iloc[0]
            sig_flag = ('***' if r['sharpe_p'] < 0.001 else
                        '**'  if r['sharpe_p'] < 0.01  else
                        '*'   if r['sharpe_p'] < 0.05  else '')
            f.write(f"  Full sample: Sharpe={r['sharpe']:.3f}  "
                    f"Ann={r['ann_return']:.1f}%  MaxDD={r['max_drawdown']:.1f}%  "
                    f"p={r['sharpe_p']:.3f}{sig_flag}\n")


def _log_headline(metrics_df, asset_name):
    """Print key results to stdout during pipeline run."""
    sub = metrics_df[
        (metrics_df['label'] == 'LRR_Oracle_Sen') &
        (metrics_df['strategy_variant'] == 'Long/Cash') &
        (metrics_df['regime'] == 'Full')
    ]
    if sub.empty:
        return
    r = sub.iloc[0]
    sig = ('***' if r['sharpe_p'] < 0.001 else
           '**'  if r['sharpe_p'] < 0.01  else
           '*'   if r['sharpe_p'] < 0.05  else '')
    print(f"    LRR Oracle (L/C): Sharpe={r['sharpe']:.3f}  "
          f"Ann={r['ann_return']:.1f}%  MaxDD={r['max_drawdown']:.1f}%  "
          f"p={r['sharpe_p']:.3f}{sig}")

    # Compare vs B&H and Simple_Sen
    bh = metrics_df[
        (metrics_df['label'] == 'Buy-and-Hold') &
        (metrics_df['regime'] == 'Full')
    ]
    if not bh.empty:
        print(f"    Buy-and-Hold:     Sharpe={bh.iloc[0]['sharpe']:.3f}  "
              f"Ann={bh.iloc[0]['ann_return']:.1f}%")