# src/additional_robustness.py
# Three additional robustness tests:
#   1. Johansen cointegration test
#   2. Expanding-window HMM concordance
#   3. Winsorized Transfer Entropy comparison
#
# Usage: run_additional_robustness(final_data, assets, tw, results_path)

import os
import numpy as np
import pandas as pd


def log(msg, indent=0):
    print('  ' * indent + msg)


# ================================================================
# 1. Johansen Cointegration Test
# ================================================================
def _johansen_test(final, asset_name):
    """
    Test whether LRR, omega, and price_change are cointegrated.
    If cointegrated: should use VECM instead of differenced VAR.
    If not: differenced VAR is the correct specification.
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    
    cols = ['LRR_Oracle_Sen', 'omega', 'price_change']
    data = final[cols].dropna()
    
    if len(data) < 50:
        return {'asset': asset_name, 'result': 'insufficient_data'}
    
    try:
        # det_order: -1 = no constant, 0 = constant, 1 = trend
        # k_ar_diff: number of lagged differences (5 is standard)
        result = coint_johansen(data, det_order=0, k_ar_diff=5)
        
        # Trace statistic test
        # result.lr1 = trace statistics
        # result.cvt = critical values (90%, 95%, 99%)
        trace_stats = result.lr1
        crit_95 = result.cvt[:, 1]  # 95% critical values
        
        # Number of cointegrating relationships at 95%
        n_coint = sum(1 for i in range(len(trace_stats)) 
                     if trace_stats[i] > crit_95[i])
        
        return {
            'asset': asset_name,
            'n_coint_relations': n_coint,
            'trace_stat_r0': round(float(trace_stats[0]), 3),
            'crit_95_r0': round(float(crit_95[0]), 3),
            'trace_stat_r1': round(float(trace_stats[1]), 3) if len(trace_stats) > 1 else np.nan,
            'crit_95_r1': round(float(crit_95[1]), 3) if len(crit_95) > 1 else np.nan,
            'cointegrated': n_coint > 0,
            'conclusion': 'COINTEGRATED — consider VECM' if n_coint > 0 
                         else 'NOT cointegrated — differenced VAR is valid',
        }
    except Exception as e:
        return {'asset': asset_name, 'result': f'failed: {e}'}


# ================================================================
# 2. Expanding-Window HMM Concordance
# ================================================================
def _expanding_hmm_concordance(asset_df, full_regime_df):
    """
    For each point in time, fit HMM on data up to that point,
    predict regime, compare with full-sample regime label.
    
    High concordance (>95%) means forward-looking HMM labels
    are not meaningfully different from real-time labels.
    """
    from hmmlearn.hmm import GaussianHMM
    
    data = asset_df.copy()
    data['time'] = pd.to_datetime(data['time'], errors='coerce').dt.date
    data['returns'] = data['close'].pct_change()
    data['volatility'] = data['returns'].rolling(window=7).std()
    data = data.dropna(subset=['returns', 'volatility']).reset_index(drop=True)
    
    full_regime = full_regime_df.copy()
    full_regime['time'] = pd.to_datetime(full_regime['time'], errors='coerce').dt.date
    
    # Merge to get full-sample labels
    merged = pd.merge(data, full_regime[['time', 'regime']], on='time', how='inner')
    
    if len(merged) < 100:
        return {'concordance': np.nan, 'n_points': len(merged)}
    
    # Start expanding window from day 60 (need enough data for HMM)
    min_window = 60
    concordant = 0
    total = 0
    
    for i in range(min_window, len(merged)):
        X_train = merged[['returns', 'volatility']].iloc[:i+1].values
        
        try:
            model = GaussianHMM(n_components=2, covariance_type='full',
                               n_iter=100, random_state=42)
            model.fit(X_train)
            predicted = model.predict(X_train)
            
            # Normalise: state with higher volatility = crisis (1)
            vol_by_state = {}
            for s in [0, 1]:
                mask = predicted == s
                if mask.sum() > 0:
                    vol_by_state[s] = merged['volatility'].iloc[:i+1][mask].mean()
            
            if vol_by_state.get(1, 0) < vol_by_state.get(0, 0):
                predicted = 1 - predicted
            
            # Compare last prediction with full-sample label
            expanding_label = predicted[-1]
            full_label = merged['regime'].iloc[i]
            
            if expanding_label == full_label:
                concordant += 1
            total += 1
            
        except Exception:
            continue
    
    concordance = concordant / total if total > 0 else np.nan
    return {
        'concordance': round(concordance, 4),
        'n_points': total,
        'concordant': concordant,
    }


# ================================================================
# 3. Winsorized Transfer Entropy
# ================================================================
def _winsorized_te(final, asset_name, pct=5):
    """
    Compute TE after winsorizing signals at pct/100-pct percentiles.
    Tests whether TE differences are driven by tail observations.
    """
    from src.risk_metrics import calculate_transfer_entropy
    
    signals = {
        'Full_LRR': 'LRR_Oracle_Sen',
        'PageRank': 'PageRank_Sen',
        'HITS': 'HITS_Sen',
        'Simple': 'Simple_Sen',
    }
    
    results = []
    for sig_name, sig_col in signals.items():
        if sig_col not in final.columns:
            continue
        
        sig = final[sig_col].dropna()
        price = final['price_change'].dropna()
        
        # Align
        common = sig.index.intersection(price.index)
        sig = sig.loc[common]
        price = price.loc[common]
        
        if len(sig) < 50:
            continue
        
        # Raw TE
        te_raw = calculate_transfer_entropy(sig, price, lag=7)
        
        # Winsorized TE
        lo = sig.quantile(pct / 100)
        hi = sig.quantile(1 - pct / 100)
        sig_wins = sig.clip(lo, hi)
        
        lo_p = price.quantile(pct / 100)
        hi_p = price.quantile(1 - pct / 100)
        price_wins = price.clip(lo_p, hi_p)
        
        te_wins = calculate_transfer_entropy(sig_wins, price_wins, lag=7)
        
        results.append({
            'asset': asset_name,
            'signal': sig_name,
            'te_raw': round(te_raw, 6),
            'te_winsorized': round(te_wins, 6),
            'te_change_pct': round((te_wins - te_raw) / max(te_raw, 1e-10) * 100, 1),
            'winsorize_pct': pct,
        })
    
    return results


# ================================================================
# Main runner
# ================================================================
def run_additional_robustness(final_data, assets, tw, results_path):
    """Run all three additional robustness tests."""
    log('\n>>> Additional Robustness Tests (Johansen, Expanding HMM, Winsorized TE)')
    
    from src.regime_engine import detect_market_regimes
    
    sens_path = os.path.join(results_path, 'sensitivity')
    os.makedirs(sens_path, exist_ok=True)
    
    # ---------------------------------------------------------------
    # 1. Johansen Cointegration
    # ---------------------------------------------------------------
    log('  1. Johansen Cointegration Tests')
    johansen_rows = []
    for asset_name, final in final_data.items():
        result = _johansen_test(final, asset_name)
        johansen_rows.append(result)
        coint = result.get('cointegrated', 'N/A')
        n_rel = result.get('n_coint_relations', 'N/A')
        log(f'    {asset_name}: {n_rel} cointegrating relations — '
            f'{"COINTEGRATED" if coint else "Not cointegrated"}')
    
    pd.DataFrame(johansen_rows).to_csv(
        os.path.join(sens_path, 'johansen_cointegration.csv'), index=False)
    
    # ---------------------------------------------------------------
    # 2. Expanding-Window HMM
    # ---------------------------------------------------------------
    log('  2. Expanding-Window HMM Concordance')
    hmm_rows = []
    for asset_name in final_data.keys():
        if asset_name not in assets:
            continue
        asset_df = assets[asset_name]
        
        # Get full-sample regime labels
        try:
            full_regime = detect_market_regimes(asset_df.copy())
        except Exception:
            continue
        
        result = _expanding_hmm_concordance(asset_df, full_regime)
        result['asset'] = asset_name
        hmm_rows.append(result)
        log(f'    {asset_name}: concordance={result["concordance"]:.1%} '
            f'({result.get("concordant", "?")}/{result.get("n_points", "?")})')
    
    pd.DataFrame(hmm_rows).to_csv(
        os.path.join(sens_path, 'expanding_hmm_concordance.csv'), index=False)
    
    # ---------------------------------------------------------------
    # 3. Winsorized TE
    # ---------------------------------------------------------------
    log('  3. Winsorized Transfer Entropy (5% trim)')
    te_rows = []
    for asset_name, final in final_data.items():
        results = _winsorized_te(final, asset_name, pct=5)
        te_rows.extend(results)
        for r in results:
            log(f'    {asset_name}/{r["signal"]}: raw={r["te_raw"]:.4f} '
                f'wins={r["te_winsorized"]:.4f} ({r["te_change_pct"]:+.1f}%)')
    
    pd.DataFrame(te_rows).to_csv(
        os.path.join(sens_path, 'winsorized_te.csv'), index=False)
    
    # ---------------------------------------------------------------
    # Summary report
    # ---------------------------------------------------------------
    with open(os.path.join(sens_path, 'additional_robustness_report.txt'), 'w', encoding='utf-8') as f:
        f.write('=== Additional Robustness Tests ===\n\n')
        
        f.write('--- 1. Johansen Cointegration ---\n')
        n_coint = sum(1 for r in johansen_rows if r.get('cointegrated', False))
        f.write(f'{n_coint}/{len(johansen_rows)} assets show cointegration\n')
        if n_coint == 0:
            f.write('No cointegration found — differenced VAR is the correct specification.\n')
        for r in johansen_rows:
            f.write(f'  {r.get("asset","?")}: {r.get("conclusion", "N/A")}\n')
        
        f.write('\n--- 2. Expanding-Window HMM Concordance ---\n')
        for r in hmm_rows:
            f.write(f'  {r.get("asset","?")}: {r.get("concordance",0):.1%} concordance '
                    f'({r.get("concordant","?")}/{r.get("n_points","?")} points)\n')
        mean_conc = np.mean([r['concordance'] for r in hmm_rows 
                           if not np.isnan(r.get('concordance', np.nan))])
        f.write(f'Mean concordance: {mean_conc:.1%}\n')
        if mean_conc > 0.95:
            f.write('High concordance confirms forward-looking HMM labels\n')
            f.write('are not meaningfully different from real-time predictions.\n')
        
        f.write('\n--- 3. Winsorized Transfer Entropy (5% trim) ---\n')
        te_df = pd.DataFrame(te_rows)
        if len(te_df) > 0:
            for sig in te_df['signal'].unique():
                sub = te_df[te_df['signal'] == sig]
                f.write(f'  {sig}: mean raw={sub["te_raw"].mean():.4f} '
                        f'mean wins={sub["te_winsorized"].mean():.4f} '
                        f'({sub["te_change_pct"].mean():+.1f}%)\n')
            f.write('If rankings preserved after winsorization: TE differences are structural.\n')
    
    log('  additional_robustness_report.txt saved')
    log('>>> Additional Robustness Tests complete.')
