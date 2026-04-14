# src/sensitivity_config.py
# Central configuration for Phase 20: Sensitivity & Robustness Suite
# v2.5 — added multi-level HMM winsorization

# Rolling reputation: expanding window recomputed every N days
ROLLING_WINDOW_DAYS = 30  # monthly recomputation

# Gate ablation configurations
# v3.0: Con removed from propagation, now aggregation-only gate
# Each config: (name, use_omega, use_anchor, con_in_aggregation)
# Propagation uses: omega gate + V-Anchor + volume-rationality penalty
# Aggregation uses: signal = weighted_mean(sen mean con, weights=rep_w)
GATE_ABLATION_CONFIGS = [
    ('Full_Oracle',    True,  True,  True),    # baseline: omega + anchor + con_agg
    ('No_Omega',       False, True,  True),    # remove rationality from propagation
    ('No_Con',         True,  True,  False),   # remove con from aggregation
    ('No_VAnchor',     True,  False, True),    # remove economic anchor
    ('Omega_Only',     True,  False, False),   # only rationality in propagation
    ('Con_Only',       False, False, True),    # only con in aggregation (no propagation gates)
    ('Social_Only',    False, False, False),   # no gates at all
]

# Con sensitivity configurations
# v3.0: Con no longer enters propagation by default
# These configs test different Con placements in aggregation
# Each config: (name, con_in_propagation, con_in_aggregation, con_in_denominator)
CON_SENSITIVITY_CONFIGS = [
    ('Con_Agg_Only',   False, True,  False),  # current design (v3.0 default)
    ('Con_Both',       True,  True,  False),  # legacy: Con in both prop + agg
    ('Con_Prop_Only',  True,  False, False),  # only in propagation (legacy)
    ('Con_With_Denom', False, True,  True),   # agg + denominator
    ('No_Con',         False, False, False),  # baseline without Con
]

# Hyperparameter sweep (BTC only to save time)
PHI_VALUES = [0.05, 0.10, 0.15, 0.20, 0.30]
ITERATION_VALUES = [3, 5, 7]
OMEGA_FLOOR_VALUES = [0.0, 0.05, 0.10]

# HMM robustness: winsorisation percentiles (test multiple levels)
WINSORIZE_LEVELS = [(1, 99), (5, 95), (10, 90)]

# Bot detection: feature-based clustering
BOT_FREQ_PERCENTILE = 0.95     # flag users above this tweet frequency
BOT_OMEGA_THRESHOLD = 0.5      # and below this mean omega
BOT_MIN_TWEETS = 5             # minimum tweets to evaluate

# Alternative independence measures
INDEPENDENCE_ALTERNATIVES = ['daily_deviation', 'temporal_novelty']
TEMPORAL_NOVELTY_WINDOW = 7    # days lookback for user's own history