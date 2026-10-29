# MODULE: visualization.py
# PURPOSE: Publication plots from saved CSV outputs
# RESEARCH RULE: No retraining, threshold tuning or false-negative assignment to unknown labels.

"""Publication-oriented, non-interactive plots of already-saved experiment outputs.

Design constraints
------------------
* All figures are derived from CSV files, without re-training any model.
* Diagnostics for periods 35--49 are explicitly labeled POST-HOC / EXPOSED.
* Unknown labels are NEVER converted into known negatives.
* Figures use labeled-set metrics; they do not estimate deployment precision.

Each plot has a small, independent function so it can be reproduced or replaced
without modifying the training and evaluation code.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve
DPI = 160

# FUNCTION: _read — Load stored artifacts without invoking training.
# WHY: All plotting is derived from saved CSVs, never triggers training.
def _read(folder: Path, name: str) -> pd.DataFrame:
    """Read one required experiment artifact, failing with a useful path."""
    # RESEARCH RATIONALE:
    # Read only persisted CSVs so plotting never triggers expensive model fits.
    path = folder / name
    # IF (not path.is_file()): A missing report CSV must fail before drawing misleading empty plots.
    if not path.is_file():
        raise FileNotFoundError(f'Plot source missing: {path}')
    return pd.read_csv(path)

# FUNCTION: _save — Write plot files and release Matplotlib figure memory.
# WHY: All plotting is derived from saved CSVs, never triggers training.
def _save(fig, target: Path) -> None:
    # RESEARCH RATIONALE:
    # Explicitly close figures to avoid retaining multiple large figure objects.
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

# FUNCTION: _selection — Retain available requested models for subset-specific plots.
# WHY: All plotting is derived from saved CSVs, never triggers training.
def _selection(available, models):
    # RESEARCH RATIONALE:
    # A short candidate list is allowed without failing when a model was not requested.
    return [m for m in models if m in set(available)]

# FUNCTION: plot_dev_model_comparison — Rank development models only (not the exposed diagnostic window).
# WHY: All plotting is derived from saved CSVs, never triggers training.
def plot_dev_model_comparison(root: Path, target: Path) -> None:
    """Rank development models only (not the exposed diagnostic window)."""
    df = _read(root, 'DEV_SUMMARY.csv').sort_values('mean_AP')
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(df) + 1.5)))
    bars = ax.barh(df['model'], df['mean_AP'])
    # LOOP (zip(bars, df['mean_AP'])): Process every item using the same fixed input/output contract.
    for bar, val in zip(bars, df['mean_AP']):
        ax.text(min(0.96, bar.get_width() + 0.012), bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center', fontsize=8)
    ax.set(xlim=(0, 1), xlabel='Mean average precision across development folds', title='Development model comparison (pre-diagnostic folds)')
    ax.grid(axis='x', alpha=0.15)
    _save(fig, target)

# FUNCTION: plot_step_ap_and_baseline — Display AP and per-step known-positive prevalence side by side.
# WHY: All plotting is derived from saved CSVs, never triggers training.
def plot_step_ap_and_baseline(root: Path, target: Path, models) -> None:
    """Display AP and per-step known-positive prevalence side by side.

    AP cannot be interpreted without the prevalence of the labeled evaluation set.
    The vertical boundary indicates the *observed* breakdown, not a new split.
    """
    # RESEARCH RATIONALE:
    # Always show prevalence when comparing AP across changing positive-class rates.
    df = _read(root, 'DIAGNOSTIC_BY_STEP_ALL_MODELS.csv')
    # IF ('policy' in df): Avoid double-counting AP curves that are identical across threshold policies.
    if 'policy' in df:
        df = df[df['policy'] == 'F1_policy'].copy()
    chosen = _selection(df.model, models)
    fig, ax = plt.subplots(figsize=(12, 5))
    # LOOP (chosen): Process every item using the same fixed input/output contract.
    for name in chosen:
        sel = df[df.model == name].sort_values('time_step')
        ax.plot(sel.time_step, sel.AP, marker='o', ms=3, label=name)
    base = df.drop_duplicates('time_step').sort_values('time_step')
    ax.plot(base.time_step, base.prevalence, '--', label='Known-label prevalence', linewidth=2, color='black')
    ax.axvline(42.5, ls=':', color='gray', label='Exploratory boundary: 42 / 43')
    ax.set(xlim=(34.5, 49.5), ylim=(0, 1.02), ylabel='Average precision', xlabel='Time step', title='EXPOSED retrospective diagnostic: AP versus per-step prevalence')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.15)
    _save(fig, target)

# FUNCTION: plot_dev_paired_seed_differences — Paired model comparisons: fixed fold and seed, with/without graph.
# WHY: All plotting is derived from saved CSVs, never triggers training.
def plot_dev_paired_seed_differences(root: Path, target: Path) -> None:
    """Paired model comparisons: fixed fold and seed, with/without graph."""
    source = root / 'DEV_GRAPH_SEED_STABILITY.csv'
    # IF (not source.exists()): Represent an unrequested paired control clearly rather than inventing results.
    if not source.exists():
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.text(0.5, 0.5, 'Paired graph comparison not requested in this run', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        _save(fig, target)
        return
    df = _read(root, 'DEV_GRAPH_SEED_STABILITY.csv')
    pairs = [('SAGE_BATCH', 'SAGE_NO_GRAPH'), ('BIDIR_SAGE', 'BIDIR_NO_GRAPH')]
    fig, ax = plt.subplots(figsize=(9, 5))
    made = 0
    # LOOP (pairs): Process every item using the same fixed input/output contract.
    for name, zero in pairs:
        subset = df[df.model.isin([name, zero])]
        # IF (subset.model.nunique() != 2): A paired graph comparison needs both architecture variants.
        if subset.model.nunique() != 2:
            continue
        pivot = subset.pivot_table(index=['fold', 'seed'], columns='model', values='AP', aggfunc='first').dropna()
        # IF (pivot.empty): Skip a pair when no matched fold/seed observations remain.
        if pivot.empty:
            continue
        delta = pivot[name] - pivot[zero]
        x = np.arange(len(delta)) + made * 0.14
        ax.scatter(x, delta, label=f'{name} minus {zero}', alpha=0.8)
        made += 1
    ax.axhline(0, ls='--', color='gray')
    ax.set(ylabel='Paired AP difference', xlabel='Fold / seed observation index', title='Development: observed paired graph advantage (descriptive)')
    # IF (made): Add a legend only when a graph pair was actually plotted.
    if made:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    _save(fig, target)

# FUNCTION: plot_negative_controls — Controls must be labeled by *what was randomized*, not collapsed as one null.
# WHY: All plotting is derived from saved CSVs, never triggers training.
def plot_negative_controls(root: Path, target: Path) -> None:
    """Controls must be labeled by *what was randomized*, not collapsed as one null."""
    df = _read(root, 'DEV_NEGATIVE_CONTROL_SUMMARY.csv').sort_values('mean_AP')
    fig, ax = plt.subplots(figsize=(11, max(5, len(df) * 0.38 + 1)))
    ax.barh(df['type'], df.mean_AP)
    ax.set(xlim=(0, 1), xlabel='Mean AP across available repetitions/folds', title='Development controls: different null hypotheses (NOT interchangeable)')
    ax.grid(axis='x', alpha=0.15)
    _save(fig, target)


# ---------------------------------------------------------
# Plot Label Coverage
# WHY: Show the known/unknown class structure for selective-observation caveats.
# ---------------------------------------------------------
# FUNCTION: plot_label_coverage — Show all transactions separated into known illicit, known licit, and unknown.
# WHY: All plotting is derived from saved CSVs, never triggers training.
def plot_label_coverage(root: Path, target: Path) -> None:
    """Show all transactions separated into known illicit, known licit, and unknown."""
    df = _read(root, 'LABEL_COVERAGE.csv').sort_values('time_step')
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.stackplot(df.time_step, df.licit, df.illicit, df.unknown, labels=['Known licit', 'Known illicit', 'Unknown/unverified'], alpha=0.85)
    ax.set(xlabel='Time step', ylabel='Transactions', title='Label coverage by period (unknown ≠ licit)')
    ax.legend(loc='upper right', fontsize=8)
    _save(fig, target)

# FUNCTION: plot_feature_drift — Largest distribution changes by PSI; an alert, not proof of leakage.
# WHY: All plotting is derived from saved CSVs, never triggers training.
def plot_feature_drift(root: Path, target: Path, n=20) -> None:
    """Largest distribution changes by PSI; an alert, not proof of leakage."""
    df = _read(root, 'FEATURE_DRIFT.csv').nlargest(n, 'PSI_35_42_vs_43_49').sort_values('PSI_35_42_vs_43_49')
    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.34)))
    ax.barh(df.feature, df.PSI_35_42_vs_43_49)
    ax.set(xlabel='PSI (bins estimated from reference)', title='EXPOSED: feature drift (35–42 vs 43–49)')
    ax.grid(axis='x', alpha=0.15)
    _save(fig, target)

# FUNCTION: plot_feature_drift_by_label — Compare drift separately for known label groups (not causal concept-drift proof).
# WHY: All plotting is derived from saved CSVs, never triggers training.
def plot_feature_drift_by_label(root: Path, target: Path, n=12) -> None:
    """Compare drift separately for known label groups (not causal concept-drift proof)."""
    df = _read(root, 'FEATURE_DRIFT_BY_LABEL_DIAGNOSTIC.csv')
    # IF (not set([0, 1]).issubset(set(df.known_label))): Per-class drift needs both known-label groups in the recorded data.
    if not set([0, 1]).issubset(set(df.known_label)):
        raise ValueError('Expected known labels 0 and 1')
    piv = df[df.known_label.isin([0, 1])].pivot_table(index='feature', columns='known_label', values='PSI_early_vs_late')
    chosen = piv.max(axis=1).nlargest(n).index
    d = piv.loc[chosen].sort_values(1)
    fig, ax = plt.subplots(figsize=(10, max(5, len(d) * 0.45)))
    y = np.arange(len(d))
    ax.barh(y - 0.18, d[0], height=0.36, label='Known licit')
    ax.barh(y + 0.18, d[1], height=0.36, label='Known illicit')
    ax.set(yticks=y, yticklabels=d.index, xlabel='Conditional PSI', title='EXPOSED: shift conditioned on OBSERVED label (selection-biased)')
    ax.legend()
    ax.grid(axis='x', alpha=0.15)
    _save(fig, target)

# FUNCTION: plot_known_label_pr_roc — Curves use *only* confirmed labels in the previously exposed diagnostic window.
# WHY: All plotting is derived from saved CSVs, never triggers training.
def plot_known_label_pr_roc(root: Path, pr_target: Path, roc_target: Path, models) -> None:
    """Curves use *only* confirmed labels in the previously exposed diagnostic window."""
    df = _read(root, 'LABELED_DIAGNOSTIC_PREDICTIONS.csv')
    chosen = _selection(df.model, models)
    # LOOP ([('PR', pr_target), ('ROC', roc_target)]): Process every item using the same fixed input/output contract.
    for kind, target in [('PR', pr_target), ('ROC', roc_target)]:
        fig, ax = plt.subplots(figsize=(7, 6))
        # LOOP (chosen): Process every item using the same fixed input/output contract.
        for name in chosen:
            sub = df[df.model == name]
            y = sub.known_label.to_numpy(dtype=int)
            # IF (not set(np.unique(y)) == {0, 1}): Skip one-class slices with undefined binary ROC/PR comparison.
            if not set(np.unique(y)) == {0, 1}:
                continue
            score = sub.calibrated_score.to_numpy(dtype=float)
            # IF (kind == 'PR'): Use precision-recall axes for the PR plot rather than ROC axes.
            if kind == 'PR':
                precision, recall, _ = precision_recall_curve(y, score)
                ax.plot(recall, precision, label=name, alpha=0.9)
            else:
                fpr, tpr, _ = roc_curve(y, score)
                ax.plot(fpr, tpr, label=name, alpha=0.9)
        # IF (kind == 'PR'): Use precision-recall axes for the PR plot rather than ROC axes.
        if kind == 'PR':
            base = float(df[df.model == chosen[0]].known_label.mean()) if chosen else 0.0
            ax.axhline(base, ls=':', color='black', label=f'Labeled prevalence: {base:.3f}')
            ax.set(xlabel='Recall', ylabel='Precision', title='EXPOSED: precision–recall (labeled subset only)')
        else:
            ax.plot([0, 1], [0, 1], 'k:', alpha=0.6)
            ax.set(xlabel='False positive rate', ylabel='True positive rate', title='EXPOSED: ROC (labeled subset only)')
        ax.legend(fontsize=8)
        ax.set(xlim=(0, 1), ylim=(0, 1.01))
        ax.grid(alpha=0.15)
        _save(fig, target)

# FUNCTION: plot_topk_status — Plot confirmation status of alerts; unknown is not a false positive.
# WHY: All plotting is derived from saved CSVs, never triggers training.
def plot_topk_status(root: Path, target: Path, models) -> None:
    """Plot confirmation status of alerts; unknown is not a false positive."""
    # RESEARCH RATIONALE:
    # Unknown alerts are unverified, not mislabeled negatives.
    df = _read(root, 'TOPK_ALL_TRANSACTIONS_BATCH.csv')
    sub = df[df.model.isin(models)].groupby('model')[['known_illicit_in_alerts', 'known_licit_in_alerts', 'UNKNOWN_unverified_in_alerts']].sum().reindex(models).dropna(how='all')
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.55 * len(sub) + 1.5)))
    left = np.zeros(len(sub))
    # LOOP ([('known_illicit_in_alerts', 'Confirmed illicit'), ('known_licit_in_alerts', 'Conf...): Process every item using the same fixed input/output contract.
    for key, name in [('known_illicit_in_alerts', 'Confirmed illicit'), ('known_licit_in_alerts', 'Confirmed licit'), ('UNKNOWN_unverified_in_alerts', 'Unverified')]:
        v = sub[key].to_numpy()
        ax.barh(sub.index, v, left=left, label=name)
        left += v
    ax.set(xlabel='Top-K alerts (all 15 diagnostic periods)', title='EXPOSED: verified vs unverified alert status')
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.12)
    _save(fig, target)

# FUNCTION: plot_calibration_metrics — Report calibration as Brier error, rather than mistaking PR-AUC for calibration.
# WHY: All plotting is derived from saved CSVs, never triggers training.
def plot_calibration_metrics(root: Path, target: Path, models) -> None:
    """Report calibration as Brier error, rather than mistaking PR-AUC for calibration."""
    df = _read(root, 'CALIBRATION_DIAGNOSTICS.csv')
    df = df[df.model.isin(models) & (df.period == 'diagnostic_test_EXPOSED')]
    piv = df.pivot_table(index='model', columns='method', values='brier').reindex(models).dropna(how='all')
    fig, ax = plt.subplots(figsize=(9, max(4, len(piv) * 0.4 + 1)))
    piv.plot(kind='barh', ax=ax)
    ax.set(xlabel='Brier score (lower is better)', title='EXPOSED: observed-label Brier score by calibration')
    ax.grid(axis='x', alpha=0.15)
    _save(fig, target)


# ---------------------------------------------------------
# Create All Figures
# WHY: Recreate all visual reports from CSV files with no model training.
# ---------------------------------------------------------
# FUNCTION: create_all_figures — Regenerate eleven fixed-name charts from saved CSVs, without retraining.
# WHY: All plotting is derived from saved CSVs, never triggers training.
def create_all_figures(results_dir: str | Path, models=None) -> list[Path]:
    """Generate a complete report from saved CSV files (does not modify model results).

    Returns every generated PNG path, including for automated artifact validation.
    """
    # RESEARCH RATIONALE:
    # Generate the same eleven named figures, supporting regression/audit scripts.
    root = Path(results_dir)
    # IF (models is None): Use a documented default model shortlist only when no selection is supplied.
    if models is None:
        available = _read(root, 'DEV_SUMMARY.csv').model.tolist()
        preferred = ['XGB', 'LGB', 'CAT', 'MLP', 'SAGE_BATCH', 'SAGE_NO_GRAPH', 'BIDIR_SAGE', 'BIDIR_NO_GRAPH']
        models = _selection(available, preferred)
    out = root / 'figures'
    plots = [(plot_dev_model_comparison, (root, out / '01_development_model_ranking.png')), (plot_step_ap_and_baseline, (root, out / '02_step_average_precision.png', models)), (plot_dev_paired_seed_differences, (root, out / '03_paired_graph_seeds.png')), (plot_negative_controls, (root, out / '04_negative_controls.png')), (plot_label_coverage, (root, out / '05_label_coverage.png')), (plot_feature_drift, (root, out / '06_feature_drift.png')), (plot_feature_drift_by_label, (root, out / '07_label_conditional_drift.png')), (plot_known_label_pr_roc, (root, out / '08_precision_recall.png', out / '09_roc.png', models)), (plot_topk_status, (root, out / '10_topk_alert_status.png', models)), (plot_calibration_metrics, (root, out / '11_calibration_brier.png', models))]
    # LOOP (plots): Process every item using the same fixed input/output contract.
    for fn, args in plots:
        fn(*args)
    files = sorted(out.glob('*.png'))
    # IF (len(files) != 11): Refuse to claim complete visualization if any required PNG is missing.
    if len(files) != 11:
        raise AssertionError(f'Expected 11 figures, got {len(files)}')
    return files
