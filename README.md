# Liquid Reputation and Rationality (LRR) Framework

**A Triple-Gated Social Information Framework for Regime-Contingent Cognitive Dynamics and Risk Characterisation in Cryptocurrency Markets**

Authors: Abhishek Saxena, Anton Kolonin  
Affiliation: Novosibirsk State University 
Paper: Submitted to SN Computer Science (Springer), Special Issue: AI for Intelligent, Secure and Trustworthy Systems

---

## Overview

LRR extends graph-based reputation propagation with three quality filters to produce crash-robust, reputation-weighted sentiment signals from cryptocurrency Twitter data:

1. **Omega (ω) Gate** — down-weights tweets exhibiting cognitive distortions (Beck's taxonomy via CDS n-gram detection)
2. **Non-Conformity (Con) Gate** — rewards balanced, independent posts over one-sided herd sentiment
3. **V-Anchor** — personalises user reputation based on directional alignment between past commentary and market outcomes

The framework is applied to six cryptocurrency assets (BTC, ETH, SOL, LTC, XRP, DOGE) over 604 trading days (June 2021 – January 2023), demonstrating bidirectional Granger causality between reputation-weighted sentiment and aggregate market rationality.

---

## Repository Structure

```
LRR/
├── src/
│   ├── main.py                    # Master pipeline (Phases 0–14)
│   ├── config.py                  # Hyperparameters and distortion taxonomy
│   ├── loader.py                  # Data ingestion and preprocessing
│   ├── reputation_engine.py       # LRR propagation + PageRank/HITS benchmarks
│   ├── reputation_engine_v2.py    # Alternative reputation variants
│   ├── psych_engine.py            # Omega (ω) rationality score computation
│   ├── regime_engine.py           # HMM regime detection (calm/crisis)
│   ├── analytics.py               # VAR, Granger causality, FEVD, SVAR, IRF
│   ├── risk_metrics.py            # LTD, Transfer Entropy, IER, MI
│   ├── sensitivity_suite.py       # Hyperparameter sweep + robustness tests
│   ├── sensitivity_config.py      # Sensitivity analysis configuration
│   ├── lrr_enhanced_comparison.py # Cross-signal comparison (Table S15)
│   ├── loo_all_signals.py         # Leave-one-out channel robustness
│   ├── event_study.py             # Event study analysis
│   ├── portfolio_engine.py        # Portfolio backtest (Long/Cash strategy)
│   ├── additional_robustness.py   # VECM, partial Granger, winsorisation
│   ├── rolling_reputation_fix.py  # Expanding-window reputation variant
│   ├── mention_weight_calibration.py # Mention weight (w_m) calibration
│   ├── anchor_utils.py            # V-Anchor computation utilities
│   ├── pipeline_validator.py      # Automated validation checks
│   ├── visualizer.py              # All figure generation
│   └── data/                      # Input data directory
│       ├── twitter.csv            # Tweet-level data (see format below)
│       ├── btc.csv                # BTC daily OHLCV prices
│       ├── eth.csv                # ETH daily OHLCV prices
│       ├── crypto_research_data.csv # SOL, LTC, XRP, DOGE prices (long format)
│       ├── btc_onchain.csv        # BTC whale transaction volume (optional)
│       └── eth_onchain.csv        # ETH whale transaction volume (optional)
├── results/                       # Output directory (auto-created)
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/[your-username]/LRR.git
cd LRR

# Create virtual environment (recommended)
python -m venv lrr_env
source lrr_env/bin/activate    # Linux/Mac
# lrr_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Data Format

### twitter.csv 

The primary input file containing tweet-level data with sentiment and cognitive distortion scores pre-computed by the Aigents platform.

| Column | Type | Description |
|--------|------|-------------|
| `permalink` | string | Tweet URL (e.g., `https://twitter.com/user/status/123`) — used to extract `source_user (main, retweet and mentions)` |
| `time` | datetime | Tweet timestamp (e.g., `2021-06-01 03:49:58+00:00`) |
| `text` | string | Tweet text (used to extract retweet targets and mentions via regex) |
| `sen` | float | Overall sentiment polarity, range [-1, +1] |
| `pos` | float | Positive sentiment component, range [0, 1] |
| `neg` | float | Negative sentiment component, range [-1, 0] |
| `con` | float | Non-conformity score (pre-computed as √(pos × \|neg\|)); recomputed by pipeline |
| `wordcnt` | int | Word count of tweet |
| `itemcnt` | int | Item count |
| `catastrophizing` | float | Cognitive distortion: catastrophizing, range [0 to 1] |
| `dichotoreasoning` | float | Cognitive distortion: dichotomous reasoning, range [0 to 1] |
| `disqualpositive` | float | Cognitive distortion: disqualifying positives, range [0 to 1] |
| `emotionreasoning` | float | Cognitive distortion: emotional reasoning, range [0 to 1] |
| `fortunetelling` | float | Cognitive distortion: fortune telling, range [0 to 1] |
| `labeling` | float | Cognitive distortion: labelling, range [0 to 1] |
| `magnification` | float | Cognitive distortion: magnification, range [0 to 1] |
| `mentalfiltering` | float | Cognitive distortion: mental filtering, range [0 to 1] |
| `mindreading` | float | Cognitive distortion: mind reading, range [0 to 1] |
| `overgeneralizing` | float | Cognitive distortion: overgeneralising, range [0 to 1] |
| `personalizing` | float | Cognitive distortion: personalising, range [0 to 1] |
| `shouldment` | float | Cognitive distortion: should-statements, range [0 to 1] |
| `exclusivereasoning` | float | Cognitive distortion: exclusive reasoning, range [0 to 1] |
| `negativereasoning` | float | Cognitive distortion: negative reasoning, range [0 to 1] |
| `mentalfilteringplus` | float | Cognitive distortion: mental filtering (variant), range [0 to 1] |

**Privacy note:** The original Twitter data is not publicly distributed. A sample file (`twitter.csv`) is provided with sample data to demonstrate the expected format and allow the pipeline to run. Contact the authors for access to the full dataset under appropriate data use agreements.

### Price CSVs (required)

**btc.csv / eth.csv** — Daily OHLCV data:

| Column | Type | Description |
|--------|------|-------------|
| `time` | date | Trading date |
| `open` | float | Opening price (USD) |
| `high` | float | Highest price (USD) |
| `low` | float | Lowest price (USD) |
| `close` | float | Closing price (USD) |

**crypto_research_data.csv** — Long-format price data for additional assets (SOL, LTC, XRP, DOGE):

| Column | Type | Description |
|--------|------|-------------|
| `time` | date | Trading date |
| `open` | float | Opening price (USD) |
| `high` | float | Highest price (USD) |
| `low` | float | Lowest price (USD) |
| `close` | float | Closing price (USD) |
| `volume` | int | Total traded volome (USD), not used in analysis |
| `symbol` | string | Asset ticker (e.g., "SOL", "LTC") |

### On-chain data (optional)

**btc_onchain.csv / eth_onchain.csv** — Whale transaction volume:

| Column | Type | Description |
|--------|------|-------------|
| `time` | date | Date |
| `btc/eth_price` | float | Btc/Eth price (USD) |
| `daily_active_addresses` | int | Daily active whale addresses |
| `whale_vol_btc/eth` | float | Daily whale transaction volume (BTC/ETH) |
| `whale_vol_usd` | float | Daily whale transaction volume (USD) |
| `whale_tx_count` | int | Daily whale transaction count) |

If on-chain files are not present, Phase 11 (SVAR whale transmission) is automatically skipped.

---

## Running the Pipeline

### Full pipeline execution

```bash
cd src
python main.py
```

This runs all 15 phases sequentially:

| Phase | Description | Key outputs |
|-------|-------------|-------------|
| 0 | Data ingestion & preprocessing | Cleaned tweet DataFrame |
| 0.5 | Mention weight calibration | `mention_weight_calibration.txt` |
| 1 | Reputation engine (LRR + benchmarks) | User reputation scores |
| 2 | HMM regime detection | Calm/crisis labels |
| 3 | Daily signal aggregation | Daily LRR, PageRank, HITS, Simple signals |
| 4 | Master join + stationarity tests | `ADF_Tests.txt` |
| 5 | Lag-specific correlations | `*_Lagged_Correlations.csv` |
| 6 | LTD + Transfer Entropy | `*_Joint_Crash_Counts.csv` |
| 7 | Con gate ablation | `*_ConGate_Ablation.csv` |
| 8 | Out-of-sample validation | `*_OOS_Validation.csv` |
| 9 | VAR + Granger causality + IRF | `*_Granger_Causality.csv`, `*_irf.png` |
| 10 | Regime-specific VAR | `*_regime_granger.txt` |
| 11 | SVAR on-chain transmission | (skipped if no on-chain data) |
| 12 | Rolling correlation | `*_rolling_correlation.png` |
| 13 | Cross-asset spillover | `cross_asset_*.csv` |
| 14 | Visualisations | All `.png` figures |

Additional analyses (run after main pipeline):

```bash
# Sensitivity suite (hyperparameter sweep, LOO, HMM robustness)
python -c "from src.sensitivity_suite import run_sensitivity_suite; run_sensitivity_suite('src/data', 'src/results')"

# Enhanced signal comparison (Table S15)
python -c "from src.lrr_enhanced_comparison import run_enhanced_comparison; run_enhanced_comparison('src/data', 'src/results')"
```

### Output

All results are written to the `src/results/` directory:
- **CSV files**: numerical results for all tables
- **TXT files**: human-readable reports and summaries
- **PNG files**: all figures (Granger heatmaps, oscillation cycle, IRFs, ablation denoising, etc.)

---

## Key Hyperparameters

Configurable in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PHI` | 0.15 | V-Anchor blending weight (φ) |
| `ITERATIONS` | 5 | Reputation propagation iterations (ℓ_max) |
| `MENTION_WEIGHT` | 0.5 | Default mention edge weight (calibrated at runtime to 0.833) |
| `LEAD_WINDOW` | 7 | V-Anchor forward-looking window (days) |
| `TRAIN_RATIO` | 0.80 | Chronological train/test split ratio |
| `ROLLING_WINDOW` | 60 | Rolling correlation window (days) |
| `BOOTSTRAP_N` | 1000 | Bootstrap resamples for LTD significance |
| `SIG_LEVEL` | 0.05 | Statistical significance threshold |
| `ELITE_PERCENTILE` | 0.80 | Top-percentile threshold for elite-user analysis |

The sensitivity suite (`sensitivity_suite.py`) sweeps over φ ∈ {0.05, 0.10, 0.15, 0.20, 0.30}, ℓ_max ∈ {3, 5, 7}, and ω_floor ∈ {0.00, 0.05, 0.10} (45 configurations total).

---

## Reproducing Paper Results

To reproduce the results reported in the paper:

1. Place the full dataset in `src/data/` (twitter.csv + price CSVs)
2. Run the full pipeline: `python main.py`
3. Run the sensitivity suite for robustness tables (S18, S19)
4. Results will be written to `src/results/`

The pipeline produces all figures and tables reported in the main paper and supplementary material. Expected runtime: ~120–180 minutes on a standard machine (depends on dataset size, full dataset estimation).

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{saxena2025lrr,
  author  = {Saxena, Abhishek and Kolonin, Anton},
  title   = {Liquid Reputation and Rationality: A Triple-Gated Social 
             Information Framework for Regime-Contingent Cognitive 
             Dynamics and Risk Characterisation in Cryptocurrency Markets},
  journal = {SN Computer Science},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

This project is open-source. See LICENSE file for details.

---

## Acknowledgements

We thank the Novosibirsk State University for providing the sentiment and cognitive distortion detection infrastructure.
