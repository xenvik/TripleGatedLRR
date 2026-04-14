# src/config.py

# ---------------------------------------------------------------------------
# Cognitive Distortion Dictionary for Omega (ω) Gating
# Source: Beck's Cognitive Distortion Taxonomy (14 categories)
# ---------------------------------------------------------------------------
DISTORTIONS = [
    'catastrophizing', 'dichotoreasoning', 'emotionreasoning', 'fortunetelling',
    'labeling', 'magnification', 'mentalfiltering', 'mindreading',
    'overgeneralizing', 'personalizing', 'shouldment', 'negativereasoning',
    'mentalfilteringplus', 'disqualpositive'
]

# ---------------------------------------------------------------------------
# LRR Core Hyperparameters
# ---------------------------------------------------------------------------
ITERATIONS  = 5     # Recursive convergence steps for reputation propagation
CONVERGENCE_TOL = 1e-6  # Early stop if max reputation change < tolerance
N_CHANNELS = 76    # Expected number of source channels (dataset-specific)
MAX_LAG     = 30    # Maximum lag (days) for CLMI Lead-Lag Discovery sweep
PHI         = 0.15  # Personalization weight: jump prob. to economic V-vector
LEAD_WINDOW = 7     # Forward-looking window (days) for V-Anchor correlation
MENTION_WEIGHT = 0.5  # Default mention edge weight (calibrated in Phase 0.5)

# Mention weight sensitivity sweep values
MENTION_WEIGHT_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]

# ---------------------------------------------------------------------------
# Statistical Thresholds
# ---------------------------------------------------------------------------
SIG_LEVEL = 0.05    # Two-sided p-value threshold (ADF, VAR, Granger)

# ---------------------------------------------------------------------------
# Gate Sensitivity — Elite User Analysis
# ---------------------------------------------------------------------------
# Top percentile by LRR score for the elite-user Con gate ablation.
# Rationale: the Con gate effect is masked in BTC by mega-influencers whose
# structural authority drowns out the Con differentiation signal.
# Restricting to top 20% surfaces within-elite contrarian differentiation.
ELITE_PERCENTILE = 0.80  # Top-20% threshold

# ---------------------------------------------------------------------------
# Bootstrap / Permutation Testing
# ---------------------------------------------------------------------------
BOOTSTRAP_N = 1000   # Resamples for LTD reduction significance test
BOOTSTRAP_SEED = 42  # Reproducibility

# ---------------------------------------------------------------------------
# Out-of-Sample Validation
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.80   # 80/20 chronological train-test split

# ---------------------------------------------------------------------------
# Rolling Robustness Window
# ---------------------------------------------------------------------------
ROLLING_WINDOW = 60  # Days for rolling Pearson correlation
ROLLING_LAG    = 5   # Signal lead (days) used in rolling window
