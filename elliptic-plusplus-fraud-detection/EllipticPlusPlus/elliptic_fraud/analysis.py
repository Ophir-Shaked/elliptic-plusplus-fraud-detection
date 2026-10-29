# MODULE: analysis.py
# PURPOSE: Distribution-shift and feature-importance analysis
# RESEARCH RULE: Large PSI shows changed distributions, not causal fraud drift or information leakage.

"""Feature drift, feature importance, and label-coverage plots."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from .preprocessing import preprocess
from .evaluation import ap


# ---------------------------------------------------------
# Psi
# WHY: Compare probability mass in reference bins; constant features need special care.
# ---------------------------------------------------------
# FUNCTION: psi — Measure binned distribution shift (PSI is descriptive, not causal evidence).
# WHY: Drift and importances are descriptive and can be distorted by selective labeling.
def psi(a, b):
    # RESEARCH RATIONALE:
    # PSI compares binned feature distributions; it is neither causal nor a leakage test.
    # Compute reference quantiles from the earlier slice, then score both distributions.
    a = np.asarray(a)
    b = np.asarray(b)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    # IF (min(len(a), len(b)) < 20): Return undefined PSI when either sample is too small to estimate bins.
    if min(len(a), len(b)) < 20:
        return np.nan
    bin_edges = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
    # IF (len(bin_edges) < 3): Handle nearly constant reference features without duplicate-bin errors.
    if len(bin_edges) < 3:
        value = float(np.median(a))
        edges = np.array([-np.inf, np.nextafter(value, -np.inf), np.nextafter(value, np.inf), np.inf])
        bin_edges = edges
    else:
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
    pa = np.clip(np.histogram(a, bins=bin_edges)[0] / len(a), 1e-05, None)
    pb = np.clip(np.histogram(b, bins=bin_edges)[0] / len(b), 1e-05, None)
    return float(np.sum((pa - pb) * np.log(pa / pb)))


# ---------------------------------------------------------
# Diagnostics
# WHY: Investigate drift after training without choosing new models from exposed data.
# ---------------------------------------------------------
# FUNCTION: diagnostics — Inspect feature distribution changes in already exposed late periods.
# WHY: Drift and importances are descriptive and can be distorted by selective labeling.
def diagnostics(d, out):
    # RESEARCH RATIONALE:
    # Periods 35-49 were already examined: plots are EXPOSED descriptive diagnostics.
    # The unknown class remains separate from known licit and known illicit.
    a = d.ix(1, 26, False)
    b = d.ix(35, 42, False)
    c = d.ix(43, 49, False)
    rows = []
    strat = []
    # LOOP (range(d.p)): Process every item using the same fixed input/output contract.
    for j in range(d.p):
        name = f'local_{j + 1:03}'
        rows.append({'feature': name, 'PSI_fit_vs_35_42': psi(d.X[a, j], d.X[b, j]), 'PSI_fit_vs_43_49': psi(d.X[a, j], d.X[c, j]), 'PSI_35_42_vs_43_49': psi(d.X[b, j], d.X[c, j]), 'missing_fit': float(np.mean(~np.isfinite(d.X[a, j]))), 'missing_late': float(np.mean(~np.isfinite(d.X[c, j])))})
        # LOOP ((0, 1, -1)): Process every item using the same fixed input/output contract.
        for label in (0, 1, -1):
            eb = b[d.y[b] == label]
            ec = c[d.y[c] == label]
            strat.append({'feature': name, 'known_label': label, 'median_35_42': float(np.nanmedian(d.X[eb, j])) if len(eb) else np.nan, 'median_43_49': float(np.nanmedian(d.X[ec, j])) if len(ec) else np.nan, 'PSI_early_vs_late': psi(d.X[eb, j], d.X[ec, j]), 'early_n': len(eb), 'late_n': len(ec)})
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame(rows).sort_values('PSI_35_42_vs_43_49', ascending=False).to_csv(out / 'FEATURE_DRIFT.csv', index=False)
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame(strat).to_csv(out / 'FEATURE_DRIFT_BY_LABEL_DIAGNOSTIC.csv', index=False)
    pd.DataFrame(rows).set_index('feature')[['PSI_fit_vs_35_42', 'PSI_fit_vs_43_49']].nlargest(16, 'PSI_fit_vs_43_49').plot(kind='bar', figsize=(12, 5))
    plt.title('Feature drift: exploratory, NOT leakage proof')
    plt.tight_layout()
    plt.savefig(out / 'feature_drift.png', dpi=140)
    plt.close()


# ---------------------------------------------------------
# Dev Importance
# WHY: Permutation importance is a diagnostic on development periods only.
# ---------------------------------------------------------
# FUNCTION: dev_importance — Estimate permutation importance on development evaluation data only.
# WHY: Drift and importances are descriptive and can be distorted by selective labeling.
def dev_importance(d, args, out):
    # RESEARCH RATIONALE:
    # Permutation importance measures model reliance, not a feature causal effect.
    # Use development-only labels when deciding which features to inspect.
    tr = d.ix(1, 20)
    ev = d.ix(23, 25)
    x, _, _ = preprocess(d, 20, np.arange(93))
    # IMPORTANT: Keep controlled randomness reproducible across repeated experimental comparisons.
    model = LGBMClassifier(n_estimators=80 if args.smoke else 180, learning_rate=0.05, max_depth=5, num_leaves=20, min_child_samples=25, random_state=42, n_jobs=2, verbosity=-1)
    # IMPORTANT: Train the current model only on the selected historical known-label indices.
    model.fit(x[tr], d.y[tr])
    # IMPORTANT: Extract positive-class scores for ranking; thresholds are chosen separately.
    base = ap(d.y[ev], model.predict_proba(x[ev])[:, 1])
    # IMPORTANT: Use an explicit local random generator rather than hidden global stochastic state.
    rng = np.random.default_rng(2026)
    rows = []
    # LOOP (range(93)): Process every item using the same fixed input/output contract.
    for j in range(93):
        zz = x[ev].copy()
        # IMPORTANT: Break the targeted label/feature association for a negative-control comparison.
        zz[:, j] = rng.permutation(zz[:, j])
        # IMPORTANT: Extract positive-class scores for ranking; thresholds are chosen separately.
        perf = ap(d.y[ev], model.predict_proba(zz)[:, 1])
        rows.append({'feature': f'local_{j + 1:03}', 'gain': float(model.feature_importances_[j]), 'permutation_AP_drop': base - perf})
    dd = pd.DataFrame(rows).sort_values('permutation_AP_drop', ascending=False)
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    dd.to_csv(out / 'DEV_FEATURE_IMPORTANCE.csv', index=False)
    return [int(n.split('_')[1]) - 1 for n in dd.head(10).feature]


# ---------------------------------------------------------
# Plot Label Coverage
# WHY: Show the known/unknown class structure for selective-observation caveats.
# ---------------------------------------------------------
# FUNCTION: plot_label_coverage — Display how many transactions lack verified illicit/licit labels over time.
# WHY: Drift and importances are descriptive and can be distorted by selective labeling.
def plot_label_coverage(out):
    # RESEARCH RATIONALE:
    # Visualize how changing observed-label rates can distort AP interpretation.
    cov = pd.read_csv(out / 'LABEL_COVERAGE.csv')
    ax = cov.plot(x='time_step', y=['illicit', 'licit', 'unknown'], figsize=(11, 4))
    ax.set_title('Label coverage over time (unknown != negative)')
    plt.tight_layout()
    plt.savefig(out / 'label_coverage.png', dpi=140)
    plt.close()
