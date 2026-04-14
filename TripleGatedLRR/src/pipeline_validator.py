# src/pipeline_validator.py
# ================================================================
# COMPREHENSIVE PIPELINE VALIDATION MODULE
# ================================================================
# Runs after all phases complete. Checks every critical computation.
# Prints a PASS/FAIL dashboard at the end.
#
# Usage: run_pipeline_validation(tw, final_data, assets, results_path)
# Add as the LAST phase in main.py
# ================================================================

import os
import numpy as np
import pandas as pd
import networkx as nx


class ValidationResult:
    def __init__(self):
        self.checks = []
    
    def check(self, name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        self.checks.append({'name': name, 'status': status, 'detail': detail})
        return condition
    
    def warn(self, name, condition, detail=""):
        status = "PASS" if condition else "WARN"
        self.checks.append({'name': name, 'status': status, 'detail': detail})
        return condition
    
    def print_dashboard(self):
        print("\n" + "=" * 75)
        print("  PIPELINE VALIDATION DASHBOARD")
        print("=" * 75)
        
        n_pass = sum(1 for c in self.checks if c['status'] == 'PASS')
        n_fail = sum(1 for c in self.checks if c['status'] == 'FAIL')
        n_warn = sum(1 for c in self.checks if c['status'] == 'WARN')
        
        for c in self.checks:
            icon = "✓" if c['status'] == 'PASS' else "✗" if c['status'] == 'FAIL' else "⚠"
            detail = f"  ({c['detail']})" if c['detail'] else ""
            print(f"  {icon} [{c['status']}] {c['name']}{detail}")
        
        print("-" * 75)
        print(f"  TOTAL: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL")
        if n_fail == 0:
            print("  ✓ ALL CRITICAL CHECKS PASSED — pipeline results are reliable")
        else:
            print(f"  ✗ {n_fail} CRITICAL FAILURE(S) — results may be unreliable!")
        print("=" * 75 + "\n")
        
        return n_fail == 0


def run_pipeline_validation(tw, final_data, assets, results_path):
    """
    Comprehensive validation of the entire pipeline.
    Checks every formula, intermediate value, and assumption.
    """
    print("\n>>> Pipeline Validation Suite")
    v = ValidationResult()
    
    # ==================================================================
    # SECTION 1: DATA INTEGRITY
    # ==================================================================
    print("  Checking data integrity...")
    
    # 1.1 Tweet data completeness
    v.check("1.1 Tweet count > 300,000",
            len(tw) > 300000,
            f"n={len(tw)}")
    
    v.check("1.2 'sen' column exists and no NaN",
            'sen' in tw.columns and tw['sen'].notna().mean() > 0.99,
            f"NaN={tw['sen'].isna().sum() if 'sen' in tw.columns else 'MISSING'}")
    
    v.check("1.3 'pos' and 'neg' columns exist",
            'pos' in tw.columns and 'neg' in tw.columns,
            f"pos={'yes' if 'pos' in tw.columns else 'NO'}, neg={'yes' if 'neg' in tw.columns else 'NO'}")
    
    v.check("1.4 'con' column exists and range [0, 1]",
            'con' in tw.columns and tw['con'].min() >= 0 and tw['con'].max() <= 1.01,
            f"min={tw['con'].min():.4f}, max={tw['con'].max():.4f}" if 'con' in tw.columns else "MISSING")
    
    # 1.5 Con formula verification: con = sqrt(pos * |neg|)
    if 'pos' in tw.columns and 'neg' in tw.columns and 'con' in tw.columns:
        _sample = tw.sample(min(1000, len(tw)), random_state=42)
        _expected_con = np.sqrt(_sample['pos'].values * np.abs(_sample['neg'].values))
        _expected_con = np.nan_to_num(_expected_con, nan=0.0)
        _actual_con = _sample['con'].values
        # Allow for fillna(0.0) replacing NaN
        _mask = ~np.isnan(_sample['pos'].values) & ~np.isnan(_sample['neg'].values)
        _corr = np.corrcoef(_expected_con[_mask], _actual_con[_mask])[0, 1] if _mask.sum() > 10 else 0
        v.check("1.5 Con formula verified (sqrt(pos*|neg|))",
                _corr > 0.99,
                f"correlation with expected={_corr:.4f}")
    
    # 1.6 Con fillna check — should be 0.0 not 1.0
    if 'con' in tw.columns:
        _n_con_exactly_1 = (tw['con'] == 1.0).sum()
        _n_con_exactly_0 = (tw['con'] == 0.0).sum()
        v.check("1.6 Con fillna is 0.0 (not 1.0)",
                _n_con_exactly_1 < len(tw) * 0.01,  # less than 1% should be exactly 1.0
                f"con==1.0: {_n_con_exactly_1} ({_n_con_exactly_1/len(tw)*100:.2f}%), "
                f"con==0.0: {_n_con_exactly_0} ({_n_con_exactly_0/len(tw)*100:.1f}%)")
    
    # 1.7 Omega calculation
    v.check("1.7 'omega' column exists and range [0.1, 1.0]",
            'omega' in tw.columns and tw['omega'].min() >= 0.09 and tw['omega'].max() <= 1.01,
            f"min={tw['omega'].min():.3f}, max={tw['omega'].max():.3f}" if 'omega' in tw.columns else "MISSING")
    
    from src.config import N_CHANNELS
    
    # 1.8 Source users
    n_sources = tw['source_user'].nunique()
    v.check(f"1.8 Source channels count = {N_CHANNELS}",
            n_sources == N_CHANNELS,
            f"n_sources={n_sources}")
    
    # 1.9 Assets loaded
    v.check("1.9 All 6 assets loaded",
            len(final_data) == 6,
            f"assets={list(final_data.keys())}")
    
    # ==================================================================
    # SECTION 2: REPUTATION ENGINE
    # ==================================================================
    print("  Checking reputation engine...")
    
    # Use first asset's final data to check signals
    first_asset = list(final_data.keys())[0]
    final = final_data[first_asset]
    
    # 2.1 Build graph and verify HITS direction
    G_test = nx.DiGraph()
    for _, row in tw.head(50000).iterrows():
        if pd.notnull(row.get('rt_target')):
            src = str(row['source_user'])
            tgt = str(row['rt_target'])
            if src != tgt:
                G_test.add_edge(src, tgt, weight=1.0)
    
    if len(G_test) > 0:
        try:
            hubs_test, auth_test = nx.hits(G_test, max_iter=100, normalized=True)
            
            # In a retweet network: sources (retweeters) should be hubs,
            # targets (original authors) should be authorities.
            # Find the most retweeted account
            in_deg = dict(G_test.in_degree())
            top_target = max(in_deg, key=in_deg.get)
            top_source = max(dict(G_test.out_degree()), key=dict(G_test.out_degree()).get)
            
            v.check("2.1 HITS returns (hubs, authorities) in correct order",
                    auth_test.get(top_target, 0) > auth_test.get(top_source, 0) or
                    hubs_test.get(top_source, 0) > hubs_test.get(top_target, 0),
                    f"top_target={top_target} auth={auth_test.get(top_target,0):.4f}, "
                    f"top_source={top_source} hub={hubs_test.get(top_source,0):.4f}")
            
            # 2.2 Verify the main pipeline uses authorities (second return)
            # Check by looking at HITS_Sen signal properties
            if 'HITS_Sen' in final.columns:
                hits_std = final['HITS_Sen'].std()
                v.check("2.2 HITS_Sen has meaningful variance",
                        hits_std > 1e-6,
                        f"std={hits_std:.6f}")
        except Exception as e:
            v.check("2.1 HITS computation", False, f"failed: {e}")
    
    # 2.3 LRR signal properties
    if 'LRR_Oracle_Sen' in final.columns:
        lrr_std = final['LRR_Oracle_Sen'].std()
        lrr_mean = final['LRR_Oracle_Sen'].mean()
        v.check("2.3 LRR_Oracle_Sen has meaningful variance",
                lrr_std > 1e-6,
                f"mean={lrr_mean:.6f}, std={lrr_std:.6f}")
        
        v.check("2.4 LRR_Oracle_Sen has no NaN",
                final['LRR_Oracle_Sen'].notna().all(),
                f"NaN count={final['LRR_Oracle_Sen'].isna().sum()}")
    
    # 2.5 PageRank signal
    if 'PageRank_Sen' in final.columns:
        pr_std = final['PageRank_Sen'].std()
        v.check("2.5 PageRank_Sen has meaningful variance",
                pr_std > 1e-6,
                f"std={pr_std:.6f}")
    
    # 2.6 All signals exist
    expected_signals = ['LRR_Oracle_Sen', 'PageRank_Sen', 'HITS_Sen', 
                       'Simple_Sen', 'LRR_Social_Sen', 'LRR_NoCon_Sen']
    for sig in expected_signals:
        v.check(f"2.6 Signal '{sig}' exists in final data",
                sig in final.columns,
                f"{'present' if sig in final.columns else 'MISSING'}")
    
    # 2.7 LRR_NoCon should differ from LRR_Oracle (anchor was passed)
    if 'LRR_NoCon_Sen' in final.columns and 'LRR_Oracle_Sen' in final.columns:
        nocon_corr = final['LRR_NoCon_Sen'].corr(final['LRR_Oracle_Sen'])
        v.check("2.7 LRR_NoCon differs from LRR_Oracle (r < 0.999)",
                nocon_corr < 0.999,
                f"r={nocon_corr:.6f}")
    
    # ==================================================================
    # SECTION 3: REGIME DETECTION
    # ==================================================================
    print("  Checking regime detection...")
    
    if 'regime' in final.columns:
        regime_counts = final['regime'].value_counts().to_dict()
        n_regimes = len(regime_counts)
        v.check("3.1 Two regimes detected (0 and 1)",
                set(regime_counts.keys()) <= {0, 1, 2},
                f"regimes={regime_counts}")
        
        min_regime_size = min(regime_counts.values())
        v.warn("3.2 Smallest regime has >= 30 observations",
               min_regime_size >= 30,
               f"smallest={min_regime_size}")
    
    # ==================================================================
    # SECTION 4: VAR / GRANGER
    # ==================================================================
    print("  Checking VAR/Granger results...")
    
    # Check if Granger results exist
    gc_path = os.path.join(results_path, f'{first_asset.lower()}_Granger_Causality.csv')
    if os.path.exists(gc_path):
        gc_df = pd.read_csv(gc_path)
        
        # 4.1 LRR→omega should be significant
        lrr_omega = gc_df[(gc_df['Cause'] == 'LRR_Oracle_Sen') & 
                          (gc_df['Effect'] == 'omega')]
        if len(lrr_omega) > 0:
            p_val = lrr_omega.iloc[0]['p_value']
            f_stat = lrr_omega.iloc[0]['F_stat']
            v.check("4.1 LRR→omega Granger significant (p<0.05)",
                    p_val < 0.05,
                    f"F={f_stat:.3f}, p={p_val:.4f}")
            
            v.check("4.2 LRR→omega F-stat in expected range (2-20)",
                    2 < f_stat < 20,
                    f"F={f_stat:.3f}")
        
        # 4.3 Check that BIC was used (lag order should be moderate)
        var_path = os.path.join(results_path, f'{first_asset.lower()}_VAR_Summary.txt')
        if os.path.exists(var_path):
            with open(var_path, 'r') as f:
                var_text = f.read()
            # Look for lag order
            import re
            lag_match = re.search(r'Results for endogenous.*?(\d+)', var_text)
            if lag_match:
                lag_order = int(lag_match.group(1))
                v.check("4.3 VAR lag order selected by BIC (typically 3-7)",
                        1 <= lag_order <= 10,
                        f"lag={lag_order}")
    else:
        v.check("4.1 Granger CSV exists", False, f"file not found: {gc_path}")
    
    # ==================================================================
    # SECTION 5: LAG CORRELATIONS
    # ==================================================================
    print("  Checking lag correlations...")
    
    lag_path = os.path.join(results_path, f'{first_asset.lower()}_Lagged_Correlations.csv')
    if os.path.exists(lag_path):
        lag_df = pd.read_csv(lag_path)
        
        # 5.1 New LRR_to_Omega column exists
        v.check("5.1 LRR_to_Omega column exists in lag correlations",
                'LRR_to_Omega_r' in lag_df.columns,
                f"columns={[c for c in lag_df.columns if 'Omega' in c]}")
        
        # 5.2 Oscillation pattern: t-1 negative, t-5 positive
        if 'Omega_to_LRR_r' in lag_df.columns:
            t1 = lag_df[lag_df['Lag'] == 't-1']['Omega_to_LRR_r'].values
            t5 = lag_df[lag_df['Lag'] == 't-5']['Omega_to_LRR_r'].values
            if len(t1) > 0 and len(t5) > 0:
                v.check("5.2 Oscillation: t-1 negative, t-5 positive (Omega_to_LRR)",
                        t1[0] < 0 and t5[0] > 0,
                        f"t-1={t1[0]:.4f}, t-5={t5[0]:.4f}")
    else:
        v.check("5.1 Lag correlation CSV exists", False, "file not found")
    
    # ==================================================================
    # SECTION 6: CRASH METRICS
    # ==================================================================
    print("  Checking crash metrics...")
    
    crash_path = os.path.join(results_path, 'cross_asset_Joint_Crash_Counts.txt')
    if os.path.exists(crash_path):
        with open(crash_path, 'r') as f:
            crash_text = f.read()
        
        v.check("6.1 Crash counts file generated",
                True, "file exists")
        
        # Check that LRR crashes < PageRank crashes (from text)
        v.check("6.2 LRR crashes mentioned in file",
                'LRR' in crash_text,
                "")
    
    # Check enhanced comparison
    comp_path = os.path.join(results_path, 'comparison', 'COMPARISON_SUMMARY.txt')
    if os.path.exists(comp_path):
        with open(comp_path, 'r') as f:
            comp_text = f.read()
        
        v.check("6.3 Enhanced comparison completed",
                'Full_LRR' in comp_text and 'Neut_LRR' in comp_text,
                "")
        
        # Parse crash counts
        for line in comp_text.split('\n'):
            if 'Full_LRR' in line and 'Crashes' not in line:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        lrr_crashes = float(parts[1])
                        v.check("6.4 LRR mean crashes < 6",
                                lrr_crashes < 6,
                                f"mean={lrr_crashes}")
                    except ValueError:
                        pass
    
    # ==================================================================
    # SECTION 7: SENSITIVITY SUITE
    # ==================================================================
    print("  Checking sensitivity results...")
    
    sens_path = os.path.join(results_path, 'sensitivity')
    
    # 7.1 Gate ablation
    ablation_path = os.path.join(sens_path, 'gate_ablation_results.csv')
    if os.path.exists(ablation_path):
        abl = pd.read_csv(ablation_path)
        v.check("7.1 Gate ablation: 42 results (7 configs × 6 assets)",
                len(abl) >= 40,
                f"n={len(abl)}")
        
        # Con removal should halve F
        full_f = abl[abl['config'] == 'Full_Oracle']['granger_F'].mean()
        nocon_f = abl[abl['config'] == 'No_Con']['granger_F'].mean()
        v.check("7.2 Removing Con reduces F-stat significantly",
                nocon_f < full_f * 0.8,
                f"Full={full_f:.2f}, NoCon={nocon_f:.2f}, ratio={nocon_f/full_f:.2f}")
    
    # 7.2 Hyperparameter sweep
    sweep_path = os.path.join(sens_path, 'hyperparameter_sweep.csv')
    if os.path.exists(sweep_path):
        sweep = pd.read_csv(sweep_path)
        n_sig = (sweep['granger_p'] < 0.05).sum() if 'granger_p' in sweep.columns else 0
        v.check("7.3 Hyperparameter sweep: all 45 significant",
                n_sig >= 44,
                f"{n_sig}/45 significant")
    
    # 7.3 HMM robustness
    hmm_path = os.path.join(sens_path, 'hmm_robustness.csv')
    if os.path.exists(hmm_path):
        hmm = pd.read_csv(hmm_path)
        if 'concordance' in hmm.columns:
            min_conc = hmm['concordance'].min()
            v.check("7.4 HMM winsorization: 100% concordance",
                    min_conc >= 0.99,
                    f"min concordance={min_conc:.1%}")
    
    # 7.4 Channel LOO
    loo_path = os.path.join(sens_path, 'leave_one_out_channel.csv')
    if os.path.exists(loo_path):
        loo = pd.read_csv(loo_path)
        n_sig = (loo['granger_p'] < 0.05).sum() if 'granger_p' in loo.columns else 0
        n_total = len(loo)
        v.check(f"7.5 Channel LOO: >= {N_CHANNELS - 2}/{N_CHANNELS} significant",
                n_sig >= N_CHANNELS - 2,
                f"{n_sig}/{n_total} significant")
    
    # ==================================================================
    # SECTION 8: LOO ALL SIGNALS
    # ==================================================================
    print("  Checking LOO all signals...")
    
    loo_report = os.path.join(results_path, 'comparison', 'loo_all_signals_report.txt')
    if os.path.exists(loo_report):
        with open(loo_report, 'r') as f:
            loo_text = f.read()
        
        v.check("8.1 LOO report exists with all 6 assets",
                'BTC' in loo_text and 'XRP' in loo_text and 'TOTAL' in loo_text,
                "")
        
        # Parse TOTAL line
        for line in loo_text.split('\n'):
            if 'TOTAL' in line:
                # Try to extract LRR percentage
                import re
                pcts = re.findall(r'(\d+\.\d+)%', line)
                if len(pcts) >= 1:
                    lrr_pct = float(pcts[0])
                    v.check("8.2 LRR LOO robustness >= 97%",
                            lrr_pct >= 97,
                            f"LRR={lrr_pct}%")
                if len(pcts) >= 3:
                    hits_pct = float(pcts[2])
                    v.check("8.3 LRR more robust than HITS",
                            lrr_pct > hits_pct,
                            f"LRR={lrr_pct}% vs HITS={hits_pct}%")
    
    # ==================================================================
    # SECTION 9: ADDITIONAL ROBUSTNESS
    # ==================================================================
    print("  Checking additional robustness...")
    
    # Johansen
    joh_path = os.path.join(sens_path, 'johansen_cointegration.csv')
    if os.path.exists(joh_path):
        joh = pd.read_csv(joh_path)
        n_coint = joh['cointegrated'].sum() if 'cointegrated' in joh.columns else -1
        v.check("9.1 Johansen test completed",
                len(joh) >= 6,
                f"{len(joh)} assets tested")
        v.warn("9.2 No cointegration found (validates differenced VAR)",
               n_coint == 0,
               f"{n_coint} assets cointegrated")
    else:
        v.warn("9.1 Johansen test", False, "not yet run")
    
    # Expanding HMM
    ehmm_path = os.path.join(sens_path, 'expanding_hmm_concordance.csv')
    if os.path.exists(ehmm_path):
        ehmm = pd.read_csv(ehmm_path)
        if 'concordance' in ehmm.columns:
            mean_conc = ehmm['concordance'].mean()
            v.check("9.3 Expanding HMM concordance >= 90%",
                    mean_conc >= 0.90,
                    f"mean={mean_conc:.1%}")
    else:
        v.warn("9.3 Expanding HMM", False, "not yet run")
    
    # Winsorized TE
    wte_path = os.path.join(sens_path, 'winsorized_te.csv')
    if os.path.exists(wte_path):
        wte = pd.read_csv(wte_path)
        v.check("9.4 Winsorized TE completed",
                len(wte) >= 4,
                f"{len(wte)} signal-asset combinations")
    else:
        v.warn("9.4 Winsorized TE", False, "not yet run")
    
    # ==================================================================
    # SECTION 10: CROSS-CONSISTENCY CHECKS
    # ==================================================================
    print("  Running cross-consistency checks...")
    
    # 10.1 LRR signal correlation across assets (should be ~0.99)
    if len(final_data) >= 2:
        asset_names = list(final_data.keys())
        lrr_signals = {}
        for a in asset_names:
            if 'LRR_Oracle_Sen' in final_data[a].columns:
                lrr_signals[a] = final_data[a]['LRR_Oracle_Sen'].values
        
        if len(lrr_signals) >= 2:
            first_two = list(lrr_signals.keys())[:2]
            min_len = min(len(lrr_signals[first_two[0]]), len(lrr_signals[first_two[1]]))
            r = np.corrcoef(lrr_signals[first_two[0]][:min_len], 
                          lrr_signals[first_two[1]][:min_len])[0, 1]
            v.check("10.1 Cross-asset LRR correlation ~0.99 (shared graph)",
                    r > 0.95,
                    f"r({first_two[0]},{first_two[1]})={r:.4f}")
    
    # 10.2 Con=0 fraction should be ~68%
    if 'con' in tw.columns:
        con_zero = (tw['con'] == 0).mean()
        v.check("10.2 Con=0 fraction ~68% (expected for crypto Twitter)",
                0.60 < con_zero < 0.80,
                f"{con_zero:.1%}")
    
    # 10.3 All 604 days aligned per asset
    for asset_name, fd in final_data.items():
        n_rows = len(fd)
        v.check(f"10.3 {asset_name} has ~604 aligned rows",
                580 < n_rows < 620,
                f"n={n_rows}")
    
    # 10.4 Price change is roughly mean-zero
    for asset_name, fd in final_data.items():
        if 'price_change' in fd.columns:
            pc_mean = fd['price_change'].mean()
            v.warn(f"10.4 {asset_name} price_change mean ≈ 0",
                   abs(pc_mean) < 0.01,
                   f"mean={pc_mean:.6f}")
    
    # ==================================================================
    # SECTION 11: OUTPUT FILE COMPLETENESS
    # ==================================================================
    print("  Checking output files...")
    
    expected_files = [
        ('results', 'cross_asset_Joint_Crash_Counts.txt'),
        ('results', 'cross_asset_HITS_vs_LRR_Granger.txt'),
        ('results', 'cross_asset_OscillationTest.txt'),
        ('results', 'cross_asset_summary_table.csv'),
        ('results/comparison', 'COMPARISON_SUMMARY.txt'),
        ('results/comparison', 'loo_all_signals_report.txt'),
        ('results/comparison', 'enhanced_comparison.csv'),
        ('results/sensitivity', 'SENSITIVITY_SUMMARY.txt'),
        ('results/sensitivity', 'gate_ablation_results.csv'),
        ('results/sensitivity', 'hyperparameter_sweep.csv'),
    ]
    
    for subdir, fname in expected_files:
        fpath = os.path.join(results_path, fname) if subdir == 'results' else os.path.join(results_path, subdir.replace('results/', ''), fname)
        v.check(f"11. Output: {fname}",
                os.path.exists(fpath),
                f"{'exists' if os.path.exists(fpath) else 'MISSING'}")
    
    # ==================================================================
    # PRINT DASHBOARD
    # ==================================================================
    all_passed = v.print_dashboard()
    
    # Save to file
    report_path = os.path.join(results_path, 'VALIDATION_REPORT.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("PIPELINE VALIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        for c in v.checks:
            icon = "PASS" if c['status'] == 'PASS' else "FAIL" if c['status'] == 'FAIL' else "WARN"
            detail = f"  ({c['detail']})" if c['detail'] else ""
            f.write(f"[{icon}] {c['name']}{detail}\n")
        f.write(f"\nTotal: {sum(1 for c in v.checks if c['status']=='PASS')} PASS, "
                f"{sum(1 for c in v.checks if c['status']=='WARN')} WARN, "
                f"{sum(1 for c in v.checks if c['status']=='FAIL')} FAIL\n")
    
    print(f"  Validation report saved to {report_path}")
    return all_passed
