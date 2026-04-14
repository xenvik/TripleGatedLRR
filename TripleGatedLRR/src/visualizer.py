# src/visualizer.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
import os

# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------
_FIG_DPI   = 300
_PALETTE_5 = ['#bdc3c7', '#3498db', '#9b59b6', '#e74c3c', '#2ecc71']
_PALETTE_3 = ['#e74c3c', '#3498db', '#2ecc71']   # LRR, PageRank, HITS


def _save(fig_or_plt, path, dpi=_FIG_DPI):
    """Unified save-and-close helper."""
    if hasattr(fig_or_plt, 'savefig'):
        fig_or_plt.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close('all')
    else:
        fig_or_plt.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close('all')


# ---------------------------------------------------------------------------
# 1. Authority Gap
# ---------------------------------------------------------------------------

def plot_authority_gap(df, asset_name, results_dir):
    """
    Scatter: PageRank (structural popularity) vs LRR Oracle (gated rationality).
    Points coloured by omega — reveals that high-omega accounts diverge most
    from their PageRank scores, validating the purpose of the ω gate.
    """
    if 'PR_W' not in df.columns or 'LRR_Oracle_W' not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(df['PR_W'], df['LRR_Oracle_W'],
                    c=df['omega'], cmap='viridis', alpha=0.65, s=18)
    max_x = df['PR_W'].max()
    ax.plot([0, max_x], [0, max_x], '--', color='#e74c3c',
            alpha=0.5, linewidth=1.2, label='Parity line')
    plt.colorbar(sc, ax=ax, label='Omega (ω) rationality score')
    ax.set_title(f'{asset_name} — Authority Gap: Structural vs. Rational Influence',
                 fontsize=13)
    ax.set_xlabel('PageRank score (structural centrality)')
    ax.set_ylabel('LRR Oracle score (triple-gated rationality)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, f'{asset_name.lower()}_authority_gap.png'))


# ---------------------------------------------------------------------------
# 2. Evolutionary Denoising (Ablation Sweep)
# ---------------------------------------------------------------------------

def plot_ablation_denoising(sweep_df, asset_name, results_dir):
    """
    MI(signal, price_future) vs lag — shows how successive gating layers
    push the peak lead time further forward.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    styles = [':', '-.', '--', '-', '-', '-']
    widths = [1.3, 1.3, 1.5, 1.5, 2.0, 2.5]
    colors = ['#bdc3c7', '#27ae60', '#3498db', '#9b59b6', '#e74c3c', '#2ecc71']

    for i, col in enumerate(sweep_df.columns):
        c = colors[i % len(colors)]
        ax.plot(sweep_df.index, sweep_df[col],
                label=col, color=c,
                linestyle=styles[i % len(styles)],
                linewidth=widths[i % len(widths)])
        peak_lag = sweep_df[col].idxmax()
        peak_val = sweep_df[col].max()
        ax.scatter(peak_lag, peak_val, color=c, s=55, zorder=6)
        ax.annotate(f't-{peak_lag}', xy=(peak_lag, peak_val),
                    xytext=(2, 3), textcoords='offset points',
                    fontsize=8, color=c)

    ax.set_title(f'{asset_name} — Evolutionary Denoising (MI Peak Shift)', fontsize=13)
    ax.set_xlabel('Lead time (days)', fontsize=11)
    ax.set_ylabel('Mutual Information (bits)', fontsize=11)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, f'{asset_name.lower()}_ablation_denoising.png'))


# ---------------------------------------------------------------------------
# 3. Correlation Heatmap
# ---------------------------------------------------------------------------

def generate_correlation_matrix(df, asset_name, results_dir):
    """Feature correlation heatmap — confirms signal independence."""
    cols       = ['price_change', 'LRR_Oracle_Sen', 'LRR_Social_Sen',
                  'PageRank_Sen', 'HITS_Sen', 'omega', 'con']
    exist_cols = [c for c in cols if c in df.columns]

    corr = df[exist_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title(f'{asset_name} — Feature Correlation Matrix', fontsize=13)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, f'{asset_name.lower()}_heatmap.png'))


# ---------------------------------------------------------------------------
# 4. VAR Impulse Response with Confidence Bands
# ---------------------------------------------------------------------------

def plot_var_irf(var_results, asset_name, results_dir):
    """
    Standard IRF: LRR Oracle shock → price_change.
    signif=0.10 adds 90% bootstrap confidence bands.
    Tries 'LRR_Oracle_Sen' first, falls back to 'LRR_VAR_Signal'.
    """
    if var_results is None:
        return
    try:
        # The VAR may have been fit on LRR_Oracle_Sen (Phase 9) or
        # LRR_VAR_Signal (Phase 11 SVAR) — pick whichever is present
        impulse_col = 'LRR_Oracle_Sen' if 'LRR_Oracle_Sen' in var_results.names \
                      else 'LRR_VAR_Signal'
        irf = var_results.irf(10)
        fig = irf.plot(impulse=impulse_col, response='price_change',
                       signif=0.10)
        fig.suptitle(f'{asset_name} — IRF: LRR shock → price (90% CI)',
                     y=1.02, fontsize=12)
        _save(fig, os.path.join(results_dir, f'{asset_name.lower()}_irf.png'))
    except Exception as e:
        print(f'   ! IRF plot skipped ({asset_name}): {e}')


def plot_svar_cumulative_irf(var_results, asset_name, results_dir):
    """
    Orthogonalised cumulative IRF (Cholesky ordering: LRR → Whale → Price).
    Provides structural identification of the LRR → Price transmission.
    Tries 'LRR_VAR_Signal' first (Phase 11 SVAR), falls back to 'LRR_Oracle_Sen'.
    """
    if var_results is None:
        return
    try:
        impulse_col = 'LRR_VAR_Signal' if 'LRR_VAR_Signal' in var_results.names \
                      else 'LRR_Oracle_Sen'
        irf = var_results.irf(10)
        fig = irf.plot_cum_effects(impulse=impulse_col,
                                   response='price_change', orth=True,
                                   signif=0.10)
        plt.suptitle(
            f'{asset_name} — Structural Cumulative IRF (Orthogonalised, 90% CI)',
            y=1.02, fontsize=12)
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        _save(fig, os.path.join(results_dir,
                                f'{asset_name.lower()}_svar_cumulative.png'))
    except Exception as e:
        print(f'   ! SVAR cumulative IRF skipped ({asset_name}): {e}')


# ---------------------------------------------------------------------------
# 5. OOS Forecast — Model vs Baselines
# ---------------------------------------------------------------------------

def plot_oos_forecast(comparison_df, results_dict, asset_name, results_dir):
    """
    OOS forecast plot showing LRR-VAR, AR(1), and actual returns.
    RMSE annotations distinguish model from baselines.
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(comparison_df.index, comparison_df['Actual'],
            label='Actual', color='#2c3e50', alpha=0.5, linewidth=1.2)

    if 'LRR_VAR' in comparison_df.columns:
        lrr_rmse = results_dict.get('LRR-VAR', {}).get('RMSE', np.nan)
        lrr_da   = results_dict.get('LRR-VAR', {}).get('DA',   np.nan)
        ax.plot(comparison_df.index, comparison_df['LRR_VAR'],
                label=f'LRR-VAR  RMSE={lrr_rmse:.5f}  DA={lrr_da:.1%}',
                color='#e74c3c', linestyle='--', linewidth=2.0)

    if 'AR1' in comparison_df.columns:
        ar_rmse = results_dict.get('AR(1)', {}).get('RMSE', np.nan)
        ar_da   = results_dict.get('AR(1)', {}).get('DA',   np.nan)
        ax.plot(comparison_df.index, comparison_df['AR1'],
                label=f'AR(1)    RMSE={ar_rmse:.5f}  DA={ar_da:.1%}',
                color='#3498db', linestyle=':', linewidth=1.5)

    ax.axhline(0, color='gray', linewidth=0.6, linestyle='--', alpha=0.5)
    ax.set_title(f'{asset_name} — Out-of-Sample Validation', fontsize=13)
    ax.legend(fontsize=9, frameon=True)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, f'{asset_name.lower()}_oos_forecast.png'))


# ---------------------------------------------------------------------------
# 6. Regime-Aware Forecast
# ---------------------------------------------------------------------------

def plot_regime_aware_forecast(comparison_df, regime_series, results_dict,
                                asset_name, results_dir):
    """Forecast over HMM-detected regime background. Shading = Crisis."""
    if regime_series is None or comparison_df.empty:
        return

    lrr_rmse = results_dict.get('LRR-VAR', {}).get('RMSE', np.nan)
    fig, ax  = plt.subplots(figsize=(14, 5))

    ax.plot(comparison_df.index, comparison_df['Actual'],
            label='Actual returns', color='#2c3e50', alpha=0.45, linewidth=1.0)
    ax.plot(comparison_df.index, comparison_df['LRR_VAR'],
            label='LRR-VAR forecast', color='#e74c3c', linewidth=2.0)

    # Shade crisis regimes
    r = regime_series.reindex(comparison_df.index)
    for i in range(len(r) - 1):
        if r.iloc[i] == 1:
            ax.axvspan(r.index[i], r.index[i + 1],
                       color='grey', alpha=0.18, linewidth=0)

    ax.set_title(
        f'{asset_name} — Regime-Aware OOS Forecast  '
        f'(grey = Crisis | RMSE: {lrr_rmse:.5f})',
        fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.12)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir,
                            f'{asset_name.lower()}_regime_forecast.png'))


# ---------------------------------------------------------------------------
# 7. LTD Benchmark Comparison Bar Chart
# ---------------------------------------------------------------------------

def plot_ltd_benchmark(ltd_dict, asset_name, results_dir):
    """
    Bar chart comparing LTD across LRR Oracle, PageRank, and HITS.

    Lower LTD = better crash protection.
    This is the primary risk metric chart for the paper.
    """
    labels = list(ltd_dict.keys())
    values = list(ltd_dict.values())
    colors = [_PALETTE_3[i % len(_PALETTE_3)] for i in range(len(labels))]

    # Replace NaN with 0 for plotting
    plot_values = [v if (v is not None and not np.isnan(v)) else 0.0 for v in values]
    nan_mask    = [(v is None or np.isnan(v)) for v in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, plot_values, color=colors, edgecolor='white',
                  linewidth=0.8, width=0.5)

    # Value labels on bars — show N/A for missing
    for bar, val, is_nan in zip(bars, values, nan_mask):
        label = 'N/A' if is_nan else f'{val:.4f}'
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                label, ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Lower Tail Dependence (LTD)', fontsize=11)
    ax.set_title(
        f'{asset_name} — Crash Coupling Benchmark  '
        f'(lower = safer signal)',
        fontsize=12)
    valid_vals = [v for v in plot_values if v > 0]
    ylim_top = max(valid_vals) * 1.25 if valid_vals else 0.5
    ax.set_ylim(0, ylim_top)
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir,
                            f'{asset_name.lower()}_ltd_benchmark.png'))


# ---------------------------------------------------------------------------
# 8. OOS Baseline Comparison Bar Chart
# ---------------------------------------------------------------------------

def plot_baseline_comparison(results_dict, asset_name, results_dir):
    """
    Bar chart: RMSE and Directional Accuracy for all three models.
    Side-by-side panels.  Key result: LRR-VAR vs AR(1) vs Random Walk.
    """
    models  = list(results_dict.keys())
    rmses   = [results_dict[m].get('RMSE', np.nan) for m in models]
    das     = [results_dict[m].get('DA',   np.nan) for m in models]
    colors  = ['#e74c3c', '#3498db', '#95a5a6'][:len(models)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # RMSE panel
    ax = axes[0]
    bars = ax.bar(models, rmses, color=colors, edgecolor='white',
                  linewidth=0.8, width=0.45)
    for bar, v in zip(bars, rmses):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0002,
                    f'{v:.5f}', ha='center', va='bottom', fontsize=8)
    ax.set_title(f'{asset_name} RMSE (lower = better)', fontsize=11)
    ax.set_ylabel('RMSE')
    ax.grid(axis='y', alpha=0.2)

    # Directional Accuracy panel
    ax = axes[1]
    bars = ax.bar(models, das, color=colors, edgecolor='white',
                  linewidth=0.8, width=0.45)
    ax.axhline(0.50, color='black', linewidth=1.0, linestyle='--',
               label='Random baseline (0.50)')
    for bar, v in zip(bars, das):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{v:.1%}', ha='center', va='bottom', fontsize=8)
    ax.set_title(f'{asset_name} Directional Accuracy (higher = better)', fontsize=11)
    ax.set_ylabel('Directional Accuracy')
    ax.set_ylim(0, 0.75)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.2)

    fig.suptitle(f'{asset_name} — Forecast Benchmark Comparison', fontsize=13)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir,
                            f'{asset_name.lower()}_baseline_comparison.png'))


# ---------------------------------------------------------------------------
# 9. Rolling Correlation
# ---------------------------------------------------------------------------

def plot_rolling_correlation(rolling_corr, asset_name, signal_label,
                              lag, window, results_dir):
    """
    Rolling Pearson r between lagged LRR signal and price returns.
    Demonstrates temporal stability of the predictive relationship.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rolling_corr.index, rolling_corr,
            color='#2ecc71', linewidth=1.6, label=f'{signal_label} (t-{lag})')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.fill_between(rolling_corr.index, 0, rolling_corr,
                    where=rolling_corr > 0, alpha=0.20, color='#2ecc71')
    ax.fill_between(rolling_corr.index, 0, rolling_corr,
                    where=rolling_corr < 0, alpha=0.20, color='#e74c3c')
    ax.set_title(
        f'{asset_name} — {window}-Day Rolling Correlation  '
        f'({signal_label}, lag={lag})',
        fontsize=12)
    ax.set_ylabel(f'Pearson r  ({window}-day window)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir,
                            f'{asset_name.lower()}_rolling_correlation.png'))


# ---------------------------------------------------------------------------
# 10. Granger Causality Heatmap
# ---------------------------------------------------------------------------

def plot_granger_heatmap(gc_df, asset_name, results_dir):
    """
    Heatmap of Granger causality p-values (lower = stronger causality).
    Coloured by -log10(p) so significant results are visually prominent.
    """
    if gc_df is None or gc_df.empty:
        return

    variables = sorted(set(gc_df['Cause'].tolist() + gc_df['Effect'].tolist()))
    n = len(variables)

    matrix = pd.DataFrame(np.nan, index=variables, columns=variables)
    for _, row in gc_df.iterrows():
        matrix.loc[row['Cause'], row['Effect']] = row['p_value']

    neg_log_p = -np.log10(matrix.fillna(1.0).clip(lower=1e-10))

    fig, ax = plt.subplots(figsize=(max(6, n + 1), max(5, n)))
    sns.heatmap(neg_log_p, annot=matrix.round(3), fmt='.3f',
                cmap='YlOrRd', ax=ax, linewidths=0.5,
                cbar_kws={'label': '-log₁₀(p-value)'})
    ax.set_title(f'{asset_name} — Granger Causality  '
                 f'(cell = p-value, colour = -log₁₀(p))',
                 fontsize=12)
    ax.set_xlabel('Effect variable')
    ax.set_ylabel('Cause variable')
    fig.tight_layout()
    _save(fig, os.path.join(results_dir,
                            f'{asset_name.lower()}_granger_heatmap.png'))


# ---------------------------------------------------------------------------
# 11. Gate Sensitivity Summary (Con Gate Ablation)
# ---------------------------------------------------------------------------

def plot_gate_sensitivity(gate_results, asset_name, results_dir):
    """
    Bar chart showing LTD for Full Oracle and No-Con variants,
    annotated with risk reduction % and bootstrap 95% CI.

    gate_results dict keys:
        'Full_LTD', 'NoCon_LTD', 'Reduction_%',
        'Boot_CI_lo', 'Boot_CI_hi', 'Boot_p',
        'Elite_Full_LTD', 'Elite_NoCon_LTD',
        'Elite_Reduction_%', 'Elite_Boot_p'
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

    for ax, (prefix, title_suffix) in zip(
        axes,
        [('', 'Full Population'), ('Elite_', 'Top-20% Elite Users')]
    ):
        full_key  = f'{prefix}Full_LTD'
        nocon_key = f'{prefix}NoCon_LTD'
        red_key   = f'{prefix}Reduction_%'
        ci_lo_key = f'{prefix}Boot_CI_lo'
        ci_hi_key = f'{prefix}Boot_CI_hi'
        p_key     = f'{prefix}Boot_p'

        if full_key not in gate_results:
            ax.set_visible(False)
            continue

        labels = ['Full Oracle\n(Ω + Con)', 'Ablated\n(Ω only, no Con)']
        vals   = [gate_results[full_key], gate_results[nocon_key]]
        colors = ['#2ecc71', '#e74c3c']
        bars   = ax.bar(labels, vals, color=colors,
                        edgecolor='white', linewidth=0.8, width=0.4)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=9)

        red  = gate_results.get(red_key, np.nan)
        p    = gate_results.get(p_key,   np.nan)
        ci_l = gate_results.get(ci_lo_key, np.nan)
        ci_h = gate_results.get(ci_hi_key, np.nan)

        ann = f'Reduction: {red:.1f}%\n95% CI [{ci_l:.1f}%, {ci_h:.1f}%]\np={p:.3f}'
        ax.text(0.97, 0.97, ann, transform=ax.transAxes,
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='grey', alpha=0.8))

        ax.set_title(f'{asset_name} Gate Sensitivity — {title_suffix}',
                     fontsize=10)
        ax.set_ylabel('Lower Tail Dependence (LTD)')
        valid_v = [v for v in vals if v is not None and not np.isnan(v) and v > 0]
        ax.set_ylim(0, max(valid_v) * 1.35 if valid_v else 0.5)
        ax.grid(axis='y', alpha=0.2)

    fig.suptitle(
        f'{asset_name} — Con Gate Ablation Study  (lower LTD = better crash protection)',
        fontsize=12)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir,
                            f'{asset_name.lower()}_gate_sensitivity.png'))


# ---------------------------------------------------------------------------
# 12. LRR vs Whale Alignment Time-Series
# ---------------------------------------------------------------------------

def plot_lrr_whale_alignment(df, asset_name, best_lag, results_dir):
    """
    Dual-axis time series: LRR Oracle signal (left axis, lead-shifted by
    best_lag days) vs log whale volume (right axis).

    This is the most direct visual proof of the LRR -> Whale link.
    The LRR signal is plotted at t-best_lag so its peaks visually align
    with subsequent whale movements — letting reviewers see the lead-lag
    relationship without statistical abstraction.
    """
    if 'whale_vol_log' not in df.columns:
        return

    plot_df = df.set_index('time') if 'time' in df.columns else df.copy()
    plot_df  = plot_df.sort_index()

    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Left axis: LRR signal shifted forward to show its LEAD
    lrr_shifted = plot_df['LRR_Oracle_Sen'].shift(-best_lag)
    ax1.plot(plot_df.index, lrr_shifted,
             color='#e74c3c', linewidth=1.8, alpha=0.85,
             label=f'LRR Oracle (lead-shifted {best_lag}d)')
    ax1.set_ylabel('LRR Oracle Sentiment Score', color='#e74c3c', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='#e74c3c')

    # Right axis: whale volume (log)
    ax2 = ax1.twinx()
    ax2.fill_between(plot_df.index, plot_df['whale_vol_log'],
                     alpha=0.25, color='#3498db')
    ax2.plot(plot_df.index, plot_df['whale_vol_log'],
             color='#3498db', linewidth=1.2, alpha=0.7,
             label='Log Whale Volume')
    ax2.set_ylabel('Log Whale Volume (log1p USD)', color='#3498db', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='#3498db')

    ax1.set_title(
        f'{asset_name} — LRR Signal (t-{best_lag} lead) vs Whale Volume\n'
        f'Visual verification of the LRR → On-Chain transmission',
        fontsize=12)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper left', fontsize=9, frameon=True)

    ax1.grid(True, alpha=0.12)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir,
                            f'{asset_name.lower()}_lrr_whale_alignment.png'))


# ---------------------------------------------------------------------------
# 13. On-Chain Lead-Lag Heatmap
# ---------------------------------------------------------------------------

def plot_onchain_leadlag_heatmap(leadlag_df, asset_name, results_dir):
    """
    Heatmap of Pearson r values across lags (rows) and the three
    relationship types: LRR->Whale, Whale->Price, LRR->Price (direct).

    Colour: red = positive correlation, blue = negative.
    Cells with p < 0.05 are annotated with r value + significance star.
    Cells with p >= 0.05 show r value in grey to distinguish.

    This gives reviewers a complete picture of the causal chain strength
    at every time horizon in a single figure.
    """
    if leadlag_df is None or leadlag_df.empty:
        return

    r_cols = {
        'LRR_to_Whale_r':  'LRR → Whale',
        'Whale_to_Price_r': 'Whale → Price',
        'LRR_to_Price_r':  'LRR → Price (direct)',
    }
    p_cols = {
        'LRR_to_Whale_r':  'LRR_to_Whale_p',
        'Whale_to_Price_r': 'Whale_to_Price_p',
        'LRR_to_Price_r':  'LRR_to_Price_p',
    }

    # Build r matrix and p matrix
    r_data = leadlag_df.set_index('Lag')[list(r_cols.keys())].rename(columns=r_cols)
    p_data = leadlag_df.set_index('Lag')[list(p_cols.values())]
    p_data.columns = list(r_cols.values())

    fig, ax = plt.subplots(figsize=(9, max(5, len(r_data) * 0.45 + 1)))

    # Draw heatmap
    sns.heatmap(r_data.astype(float), cmap='RdBu_r', center=0,
                vmin=-0.4, vmax=0.4,
                annot=False, linewidths=0.4,
                cbar_kws={'label': 'Pearson r'}, ax=ax)

    # Custom annotations: show r value, bold if significant
    for i, lag in enumerate(r_data.index):
        for j, col in enumerate(r_data.columns):
            r_val = r_data.loc[lag, col]
            p_val = p_data.loc[lag, col]
            if pd.isna(r_val):
                continue
            sig   = p_val < 0.05 if not pd.isna(p_val) else False
            star  = ('*' if p_val < 0.05 else '')
            color = 'black' if sig else '#888888'
            weight = 'bold' if sig else 'normal'
            ax.text(j + 0.5, i + 0.5,
                    f'{r_val:.3f}{star}',
                    ha='center', va='center',
                    fontsize=8, color=color, fontweight=weight)

    ax.set_title(
        f'{asset_name} — On-Chain Lead-Lag Heatmap\n'
        f'(bold + * = p < 0.05;  rows = lag of cause variable)',
        fontsize=11)
    ax.set_xlabel('Relationship type')
    ax.set_ylabel('Lag of predictor variable')
    fig.tight_layout()
    _save(fig, os.path.join(results_dir,
                            f'{asset_name.lower()}_onchain_leadlag_heatmap.png'))


# ---------------------------------------------------------------------------
#  Portfolio Backtest Plots
# ---------------------------------------------------------------------------

def plot_cumulative_returns(equity_dict, asset_name, results_dir):
    """
    Equity curves: all signal strategies vs buy-and-hold.
    LRR Oracle drawn thicker in red; others in muted palette.
    X-axis = observation index (daily); Y-axis = cumulative return multiple.
    """
    if not equity_dict:
        return

    fig, ax = plt.subplots(figsize=(13, 5))

    palette = {
        'Buy-and-Hold':   ('#2c3e50', 1.5, '-'),
        'Simple_Sen':     ('#bdc3c7', 1.2, '--'),
        'PageRank_Sen':   ('#3498db', 1.2, '-.'),
        'HITS_Sen':       ('#9b59b6', 1.2, ':'),
        'LRR_Social_Sen': ('#f39c12', 1.5, '--'),
        'LRR_Oracle_Sen': ('#e74c3c', 2.5, '-'),
    }

    for label, equity in equity_dict.items():
        color, lw, ls = palette.get(label, ('#888888', 1.0, '-'))
        ax.plot(range(len(equity)), equity.values,
                label=label, color=color, linewidth=lw, linestyle=ls, alpha=0.85)

    ax.axhline(1.0, color='black', linewidth=0.6, linestyle='--', alpha=0.4)
    ax.set_title(f'{asset_name} — Cumulative Returns: Signal Strategy vs Buy-and-Hold\n'
                 f'(Long/Cash, 0.1% transaction costs, 20-day rolling median threshold)',
                 fontsize=11)
    ax.set_xlabel('Trading days')
    ax.set_ylabel('Portfolio value (initial = 1.0)')
    ax.legend(fontsize=8, frameon=True, ncol=2)
    ax.grid(True, alpha=0.12)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir,
                            f'{asset_name.lower()}_cumulative_returns.png'))


def plot_sharpe_comparison(metrics_df, asset_name, results_dir):
    """
    Side-by-side Sharpe ratio bar charts for FULL / CALM / CRISIS regimes.
    Long/Cash only. Stars mark bootstrap significance.
    This is the primary evidence chart for the portfolio contribution.
    """
    if metrics_df is None or metrics_df.empty:
        return

    sub = metrics_df[
        (metrics_df['strategy_variant'] == 'Long/Cash') &
        (metrics_df['label'] != 'Buy-and-Hold')
    ].copy()

    regimes   = ['Full', 'CALM', 'CRISIS']
    labels    = ['Simple_Sen', 'PageRank_Sen', 'HITS_Sen',
                 'LRR_Social_Sen', 'LRR_Oracle_Sen']
    colors    = ['#bdc3c7', '#3498db', '#9b59b6', '#f39c12', '#e74c3c']

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for ax, regime in zip(axes, regimes):
        r_sub = sub[sub['regime'] == regime]
        bars_data = []
        for lbl in labels:
            row = r_sub[r_sub['label'] == lbl]
            bars_data.append(row.iloc[0]['sharpe'] if not row.empty else 0.0)

        bars = ax.bar(range(len(labels)), bars_data,
                      color=colors, edgecolor='white', linewidth=0.6, width=0.6)

        # Significance stars above bars
        for i, lbl in enumerate(labels):
            row = r_sub[r_sub['label'] == lbl]
            if not row.empty:
                p = row.iloc[0]['sharpe_p']
                star = ('***' if p < 0.001 else '**' if p < 0.01
                        else '*' if p < 0.05 else '')
                if star:
                    ax.text(i, bars_data[i] + 0.02,
                            star, ha='center', va='bottom',
                            fontsize=9, fontweight='bold', color='#2c3e50')

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(
            ['Simple', 'PageRank', 'HITS', 'LRR\nSocial', 'LRR\nOracle'],
            fontsize=8)
        ax.set_title(f'{regime} Regime', fontsize=11)
        ax.grid(axis='y', alpha=0.2)
        if ax == axes[0]:
            ax.set_ylabel('Annualised Sharpe Ratio (rf=0)', fontsize=10)

    fig.suptitle(
        f'{asset_name} — Sharpe Ratio by Signal & Regime\n'
        f'(Long/Cash | * p<0.05  ** p<0.01  *** p<0.001)',
        fontsize=12)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir,
                            f'{asset_name.lower()}_sharpe_comparison.png'))


def plot_drawdown(equity_dict, asset_name, results_dir):
    """
    Drawdown profile: LRR Oracle vs Buy-and-Hold.
    Shows that LRR strategy reduces drawdown depth and duration.
    """
    if not equity_dict:
        return

    fig, ax = plt.subplots(figsize=(13, 4))

    for label, equity in equity_dict.items():
        if label not in ('LRR_Oracle_Sen', 'Buy-and-Hold'):
            continue
        roll_max  = equity.cummax()
        drawdown  = (equity - roll_max) / roll_max * 100
        color     = '#e74c3c' if label == 'LRR_Oracle_Sen' else '#2c3e50'
        lw        = 2.0       if label == 'LRR_Oracle_Sen' else 1.2
        alpha     = 0.9       if label == 'LRR_Oracle_Sen' else 0.5
        name      = 'LRR Oracle (Long/Cash)' if label == 'LRR_Oracle_Sen' else 'Buy-and-Hold'
        ax.fill_between(range(len(drawdown)), drawdown.values, 0,
                        alpha=0.25, color=color)
        ax.plot(range(len(drawdown)), drawdown.values,
                label=name, color=color, linewidth=lw, alpha=alpha)

    ax.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.4)
    ax.set_title(f'{asset_name} — Drawdown Profile: LRR Oracle vs Buy-and-Hold',
                 fontsize=11)
    ax.set_xlabel('Trading days')
    ax.set_ylabel('Drawdown (%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.12)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir,
                            f'{asset_name.lower()}_drawdown.png'))


# ---------------------------------------------------------------------------
#  Information Efficiency Ratio (IER) Table Plot
# ---------------------------------------------------------------------------

def plot_ier_table(ier_rows, results_dir):
    """
    Grouped bar chart of IER = TE/LTD across all assets and signals.

    LRR Oracle bar in red, PageRank in blue, HITS in purple.
    Higher IER = more informative per unit of crash risk.

    ier_rows : list of dicts from compute_ier_table()
               keys: asset, signal, TE, LTD, IER
    """
    if not ier_rows:
        return

    import pandas as pd
    df = pd.DataFrame(ier_rows).dropna(subset=['IER'])
    if df.empty:
        return

    assets  = df['asset'].unique().tolist()
    signals = ['LRR_Oracle', 'PageRank', 'HITS']
    colors  = {'LRR_Oracle': '#e74c3c', 'PageRank': '#3498db', 'HITS': '#9b59b6'}

    x      = np.arange(len(assets))
    width  = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, sig in enumerate(signals):
        vals = []
        for asset in assets:
            row = df[(df['asset'] == asset) & (df['signal'] == sig)]
            vals.append(row['IER'].values[0] if not row.empty else 0.0)
        bars = ax.bar(x + (i - 1) * width, vals,
                      width=width, label=sig,
                      color=colors[sig], edgecolor='white', linewidth=0.6)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.0005,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(assets, fontsize=10)
    ax.set_ylabel('IER = Transfer Entropy / Lower Tail Dependence', fontsize=10)
    ax.set_title(
        'Information Efficiency Ratio (IER) — All Assets\n'
        'Higher IER = more predictive information per unit of crash exposure',
        fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir, 'cross_asset_IER_comparison.png'))


# ---------------------------------------------------------------------------
#  Enhanced Rolling Correlation with Regime Shading
# ---------------------------------------------------------------------------

def plot_rolling_correlation_regimes(rolling_dict, regime_series,
                                      asset_name, lag, window, results_dir):
    """
    Rolling correlation plot with HMM regime shading.
    rolling_dict: {'Full': Series, 'CALM': Series, 'CRISIS': Series}
    Shows that LRR predictive relationship strengthens in specific regimes.
    """
    if not rolling_dict or 'Full' not in rolling_dict:
        return

    fig, ax = plt.subplots(figsize=(13, 4))

    full_corr = rolling_dict['Full'].dropna()
    ax.plot(range(len(full_corr)), full_corr.values,
            color='#2ecc71', linewidth=1.6, label=f'Full (lag={lag})', alpha=0.9)
    ax.fill_between(range(len(full_corr)), 0, full_corr.values,
                    where=full_corr.values > 0, alpha=0.15, color='#2ecc71')
    ax.fill_between(range(len(full_corr)), 0, full_corr.values,
                    where=full_corr.values < 0, alpha=0.15, color='#e74c3c')

    # Shade crisis regimes if provided
    if regime_series is not None:
        r = regime_series.reindex(full_corr.index).fillna(0)
        for i in range(len(r) - 1):
            if r.iloc[i] == 1:
                ax.axvspan(i, i + 1, color='#e74c3c', alpha=0.08, linewidth=0)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(0.10, color='grey', linewidth=0.5, linestyle=':', alpha=0.6)
    ax.axhline(-0.10, color='grey', linewidth=0.5, linestyle=':', alpha=0.6)

    # Annotate % positive
    pct_pos = (full_corr > 0).mean()
    ax.text(0.02, 0.95, f'{pct_pos:.1%} windows positive',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_title(
        f'{asset_name} — {window}-Day Rolling Correlation '
        f'(LRR_Oracle_Sen → price, lag={lag})\n'
        f'Red shading = HMM Crisis regime',
        fontsize=11)
    ax.set_ylabel(f'Pearson r ({window}-day window)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.12)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir,
                            f'{asset_name.lower()}_rolling_correlation.png'))


# ---------------------------------------------------------------------------
#  Cognitive Distortion Decomposition Plots
# ---------------------------------------------------------------------------

def plot_distortion_heatmap(result_df, asset_name, results_dir):
    """
    Heatmap showing Granger p-values for LRR↔each distortion.

    Rows = distortions (grouped by cluster with dividers)
    Columns = LRR→Distortion and Distortion→LRR
    Colour = -log10(p) so significant results are visually prominent
    Cells annotated with actual p-value + significance stars

    This is the key figure — it shows at a glance which cognitive
    distortions are most tightly coupled with the LRR social signal.
    """
    if result_df is None or result_df.empty:
        return

    # Sort by cluster then by LRR_to_dist_p
    df = result_df.copy().sort_values(
        ['cluster', 'LRR_to_dist_p']).reset_index(drop=True)

    dist_labels = df['distortion'].tolist()
    n = len(dist_labels)

    # Build p-value matrix
    p_matrix = pd.DataFrame({
        'LRR → Distortion': df['LRR_to_dist_p'].values,
        'Distortion → LRR': df['dist_to_LRR_p'].values,
    }, index=dist_labels)

    neg_log = -np.log10(p_matrix.clip(lower=1e-10))

    fig, ax = plt.subplots(figsize=(7, max(5, n * 0.45 + 1.5)))
    im = ax.imshow(neg_log.values, cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=4)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['LRR → Distortion', 'Distortion → LRR'],
                       fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(dist_labels, fontsize=8)

    # Annotate with p-value + stars
    sig_cols = ['LRR_to_dist_sig', 'dist_to_LRR_sig']
    p_cols   = ['LRR_to_dist_p',   'dist_to_LRR_p']
    for i, (dist, row) in enumerate(df.iterrows()):
        for j, (pc, sc) in enumerate(zip(p_cols, sig_cols)):
            p_val  = result_df.iloc[i][pc]
            sig    = result_df.iloc[i][sc]
            if pd.isna(p_val):
                continue
            color  = 'white' if neg_log.values[i, j] > 2.0 else 'black'
            weight = 'bold'  if sig else 'normal'
            ax.text(j, i, f'{p_val:.3f}{sig}',
                    ha='center', va='center', fontsize=7,
                    color=color, fontweight=weight)

    # Cluster divider lines
    current_cluster = None
    for i, (_, row) in enumerate(df.iterrows()):
        cl = result_df.iloc[i]['cluster']
        if cl != current_cluster and i > 0:
            ax.axhline(i - 0.5, color='white', linewidth=1.5)
        current_cluster = cl

    plt.colorbar(im, ax=ax, label='-log₁₀(p-value)', shrink=0.6)
    ax.set_title(
        f'{asset_name} — Cognitive Distortion Decomposition\n'
        f'(p-value annotations; grouped by psychological cluster)',
        fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir,
                            f'{asset_name.lower()}_distortion_heatmap.png'))


def plot_distortion_clusters(result_df, cluster_df, asset_name, results_dir):
    """
    Two-panel chart:
    Left:  Bar chart of top significant individual distortions (LRR→distortion)
    Right: Cluster-level Granger p-values showing which psychological cluster
           is most strongly coupled with LRR

    This is the summary figure for the paper — one chart that tells the
    whole story.
    """
    if result_df is None or result_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: individual distortions ranked by LRR→dist significance
    ax = axes[0]
    plot_df = result_df.dropna(subset=['LRR_to_dist_p']).sort_values(
        'LRR_to_dist_p').head(10)

    colors = ['#e74c3c' if p < 0.05 else '#95a5a6'
              for p in plot_df['LRR_to_dist_p']]
    bars = ax.barh(range(len(plot_df)),
                   -np.log10(plot_df['LRR_to_dist_p'].clip(lower=1e-10)),
                   color=colors, edgecolor='white', height=0.6)

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(
        [f"{r['distortion']}\n({r['cluster'].split('_')[0]})"
         for _, r in plot_df.iterrows()],
        fontsize=8)
    ax.axvline(-np.log10(0.05), color='black', linewidth=1.0,
               linestyle='--', alpha=0.6, label='p=0.05')
    ax.set_xlabel('-log₁₀(p-value)  [LRR → Distortion]', fontsize=9)
    ax.set_title(f'{asset_name}\nIndividual Distortions (top 10)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.2)

    # Right panel: cluster-level results
    ax = axes[1]
    if cluster_df is not None and not cluster_df.empty:
        cluster_names = [c.replace('_', '\n') for c in cluster_df['cluster']]
        x = np.arange(len(cluster_names))
        w = 0.35

        lrr_bars = ax.bar(x - w/2,
                          -np.log10(cluster_df['LRR_to_cl_p'].clip(lower=1e-10)),
                          w, label='LRR → Cluster',
                          color='#e74c3c', edgecolor='white')
        cl_bars  = ax.bar(x + w/2,
                          -np.log10(cluster_df['cl_to_LRR_p'].clip(lower=1e-10)),
                          w, label='Cluster → LRR',
                          color='#3498db', edgecolor='white')

        ax.axhline(-np.log10(0.05), color='black', linewidth=1.0,
                   linestyle='--', alpha=0.6, label='p=0.05')
        ax.set_xticks(x)
        ax.set_xticklabels(cluster_names, fontsize=8)
        ax.set_ylabel('-log₁₀(p-value)', fontsize=9)
        ax.set_title(f'{asset_name}\nCluster-Level Coupling with LRR', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.2)
    else:
        axes[1].set_visible(False)

    fig.suptitle(
        f'{asset_name} — Cognitive Distortion Decomposition (T3.1)\n'
        f'Red = significant (p<0.05)',
        fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(results_dir,
                            f'{asset_name.lower()}_distortion_clusters.png'))