# -----------------------------------------------------------------------------
# MODULE: config.py -- central research configuration
# PURPOSE: One source of truth for the dataset, chronological folds and models.
# RESEARCH RULE: Periods 35-49 were already exposed; never call them a new test.
# -----------------------------------------------------------------------------
"""Fixed scientific protocol shared across experiments, plots and audits."""

# Public Elliptic++ repository. Real CSV files require Git LFS.
REPO = "https://github.com/git-disl/EllipticPlusPlus.git"
FNS = (
    "txs_features.csv",   # Transaction features and time step.
    "txs_classes.csv",    # Known illicit, known licit, or unknown.
    "txs_edgelist.csv",   # Directed transaction network edges.
)

# Development folds: (fit_start, fit_end, stop_start, stop_end, eval_start, eval_end).
# Stop windows choose tree budgets/neural epochs; eval windows rank candidates.
FOLDS = [
    (1, 10, 11, 12, 13, 15),
    (1, 15, 16, 17, 18, 20),
    (1, 20, 21, 22, 23, 25),
]

# Fixed retrospective protocol. Calibration and decision thresholds do not use 35-49.
FINAL = {
    "fit": (1, 26),
    "stop": (27, 29),
    "calibration": (30, 31),
    "policy": (32, 34),
    "diagnostic_test": (35, 49),  # Already exposed in earlier analysis!
}

# Model registry; the CLI --models flag may select a computational subset.
MODELS = [
    "LOGIT", "XGB", "LGB", "CAT", "MLP",
    "SAGE_BATCH", "SAGE_NO_GRAPH", "SAGE_REWIRED",
    "BIDIR_SAGE", "BIDIR_NO_GRAPH", "BIDIR_REWIRED",
]

# Fair architecture-controlled comparisons of real edges vs zero messages.
GRAPH_PAIRS = [
    ("SAGE_BATCH", "SAGE_NO_GRAPH"),
    ("BIDIR_SAGE", "BIDIR_NO_GRAPH"),
]
