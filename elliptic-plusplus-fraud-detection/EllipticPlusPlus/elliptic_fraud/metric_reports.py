# MODULE: metric_reports.py
# PURPOSE: Post-training exports for Recall, Precision, classification metrics, and curves.
# RESEARCH RULE: Never use exposed diagnostic labels to tune or refit a model or a policy.
"""Export audit-friendly binary classification reports from frozen model predictions.

Inputs are already-saved predictions and thresholds produced by final_all().  No
model is re-trained here, and no new decision threshold is chosen on periods 35-49.
Unknown labels are never counted as known licit or as confirmed false positives.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve, roc_curve,
    brier_score_loss, log_loss, matthews_corrcoef,
)


def _safe_ratio(numerator, denominator):
    """Return a missing value for an undefined denominator (not a fabricated zero)."""
    return float(numerator / denominator) if denominator else float('nan')


def _binary_metrics(y_true, scores, threshold):
    """Recalculate classification and ranking metrics from KNOWN binary labels.

    Important: threshold is frozen on an earlier POLICY slice.  This function
    evaluates it only; no search or optimization is performed on these labels.
    """
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError('Expected non-empty equal-length 1D labels and scores')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Unknown or invalid labels must not enter classification metrics')
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError('Calibrated scores must be finite probabilities in [0,1]')
    if not np.isfinite(threshold):
        raise ValueError('A frozen, finite classification threshold is required')

    pred = p >= threshold
    positive = y == 1
    negative = y == 0
    tp = int(np.sum(pred & positive))
    fp = int(np.sum(pred & negative))
    fn = int(np.sum(~pred & positive))
    tn = int(np.sum(~pred & negative))
    support_positive = tp + fn
    support_negative = tn + fp
    # Match the training pipeline's documented zero-division convention for PPV/TPR.
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, support_positive)
    specificity = _safe_ratio(tn, support_negative)
    f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    f2 = _safe_ratio(5 * tp, 5 * tp + 4 * fn + fp)
    both_classes = bool(support_positive and support_negative)
    # Note: MCC can be 0 in some degenerate prediction cases by sklearn's convention.
    mcc = float(matthews_corrcoef(y, pred)) if both_classes else float('nan')
    ap = float(average_precision_score(y, p)) if both_classes else float('nan')
    auc = float(roc_auc_score(y, p)) if both_classes else float('nan')
    return {
        'n_known': len(y), 'n_illicit': support_positive, 'n_licit': support_negative,
        'prevalence_known': _safe_ratio(support_positive, len(y)),
        'threshold': float(threshold), 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'precision': precision, 'recall': recall, 'sensitivity_TPR': recall,
        'specificity_TNR': specificity, 'F1': f1, 'F2': f2,
        'accuracy': _safe_ratio(tp + tn, len(y)),
        'balanced_accuracy': (recall + specificity) / 2 if both_classes else float('nan'),
        'NPV': _safe_ratio(tn, tn + fn),
        'FPR': _safe_ratio(fp, support_negative),
        'FNR': _safe_ratio(fn, support_positive),
        'false_discovery_rate': _safe_ratio(fp, tp + fp),
        'MCC': mcc, 'predicted_positive_n': tp + fp,
        'FP_per_10k_known_licit': 10000 * _safe_ratio(fp, support_negative),
        'AP': ap, 'ROC_AUC': auc,
        'Brier': float(brier_score_loss(y, p)),
        'log_loss': float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1])),
    }


def _reliability_bins(y, p, bins=10):
    """Describe observed-label calibration without assuming unknown labels are licit."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.minimum(np.searchsorted(edges, p, side='right') - 1, bins - 1)
    rows = []
    for b in range(bins):
        sel = ids == b
        count = int(sel.sum())
        rows.append({
            'bin': b + 1, 'score_low': float(edges[b]), 'score_high': float(edges[b + 1]),
            'count_known': count,
            'mean_score': float(np.mean(p[sel])) if count else float('nan'),
            'fraction_illicit_known': float(np.mean(y[sel] == 1)) if count else float('nan'),
        })
    ece = sum(row['count_known'] / len(y) * abs(row['mean_score'] - row['fraction_illicit_known'])
              for row in rows if row['count_known'])
    return rows, float(ece)


def _downsample_points(x, y, extra=None, max_points=1000):
    """Keep ordered curve points for manageable CSV/PNG exports; metrics use ALL rows."""
    positions = np.unique(np.linspace(0, len(x) - 1, min(len(x), max_points), dtype=int))
    for i in positions:
        yield (float(x[i]), float(y[i]), float(extra[i]) if extra is not None and i < len(extra) else float('nan'))


def _plot_confusion(row, target, label):
    """Render the four raw counts for each model and independently frozen policy."""
    arr = np.array([[row['TN'], row['FP']], [row['FN'], row['TP']]], dtype=int)
    fig, ax = plt.subplots(figsize=(5.1, 4.3))
    ax.imshow(arr, cmap='Blues')
    ax.set_xticks([0, 1], ['Predicted licit', 'Predicted illicit'])
    ax.set_yticks([0, 1], ['Known licit', 'Known illicit'])
    ax.set_title(label + ' | exposed 35–49')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{arr[i, j]:,}', ha='center', va='center',
                    color='white' if arr[i, j] > max(1, arr.max() / 2) else 'black')
    fig.tight_layout()
    fig.savefig(target, dpi=130)
    plt.close(fig)


def _check_original_rows(original, calculated):
    """Fail loudly if the frozen experiment summary disagrees with prediction-level recomputation."""
    for field in ('AP', 'ROC_AUC', 'precision', 'recall', 'F1', 'TP', 'FP', 'FN', 'TN'):
        old = original.get(field, np.nan)
        new = calculated[field]
        if pd.isna(old) and pd.isna(new):
            continue
        if not np.isclose(float(old), float(new), atol=1e-9, rtol=1e-8, equal_nan=True):
            raise AssertionError(f"Stored {field} disagrees with recomputed {field}: {old} != {new}")


def _topk_reports(predictions, topk_details, out_dir):
    """Report Recall@K using all known positives while excluding unknown labels from verified precision."""
    rows = []
    for (model, period), group in topk_details.groupby(['model', 'time_step'], sort=True):
        all_known = predictions.loc[(predictions.model == model) & (predictions.time_step == period)]
        total_illicit = int((all_known.known_label == 1).sum())
        ranked = group.sort_values('rank', kind='stable')
        if not np.array_equal(ranked['rank'].to_numpy(), np.arange(1, len(ranked) + 1)):
            raise ValueError(f'Non-consecutive Top-K ranks: {model}, step={period}')
        for k in [10, 25, 50, 100, 200, 500]:
            if k > len(ranked):
                continue  # Larger K cannot be reconstructed from a stored top-100 list.
            head = ranked.iloc[:k]
            illicit = int((head.known_label == 1).sum())
            licit = int((head.known_label == 0).sum())
            unknown = int((head.known_label == -1).sum())
            rows.append({
                'model': model, 'time_step': int(period), 'K': k, 'alerts': len(head),
                'known_illicit_at_K': illicit, 'known_licit_at_K': licit,
                'unknown_unverified_at_K': unknown, 'total_known_illicit_in_step': total_illicit,
                'recall_at_K_of_known_illicit': _safe_ratio(illicit, total_illicit),
                'precision_at_K_among_known_alerts': _safe_ratio(illicit, illicit + licit),
                'known_alerts_at_K': illicit + licit,
                'caution': 'Unknown is UNVERIFIED, not a true negative or false positive',
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(out_dir / 'RECALL_PRECISION_AT_K_BY_STEP.csv', index=False)
        aggreg = result.groupby(['model', 'K'], as_index=False).agg(
            total_illicit_in_selected_steps=('total_known_illicit_in_step', 'sum'),
            illicit_alerts=('known_illicit_at_K', 'sum'),
            licit_alerts=('known_licit_at_K', 'sum'),
            unknown_alerts=('unknown_unverified_at_K', 'sum'),
            steps=('time_step', 'nunique'),
        )
        aggreg['micro_recall_at_K'] = aggreg['illicit_alerts'] / aggreg['total_illicit_in_selected_steps'].replace(0, np.nan)
        aggreg['precision_among_verified_alerts'] = aggreg['illicit_alerts'] / (aggreg['illicit_alerts'] + aggreg['licit_alerts']).replace(0, np.nan)
        aggreg.to_csv(out_dir / 'RECALL_PRECISION_AT_K_SUMMARY.csv', index=False)
    return int(len(result))


def _plot_time_step_comparisons(by_step, plot_dir):
    """Plot both frozen policies per model across diagnostic periods without smoothing.

    Why: pooling the entire exposed window can conceal severe time drift. Missing
    metrics remain gaps rather than being interpreted as zero performance.
    """
    created = []
    for model, model_rows in by_step.groupby('model', sort=True):
        policies = sorted(model_rows['policy'].unique(), key=lambda p: (p != 'F1_policy', p))
        fig, axes = plt.subplots(1, len(policies), figsize=(max(7, 5.4 * len(policies)), 4.4), sharey=True, squeeze=False)
        for ax, policy in zip(axes[0], policies):
            rows = model_rows.loc[model_rows.policy == policy].sort_values('time_step')
            steps = rows.time_step.to_numpy(dtype=int)
            ax.plot(steps, rows.recall.to_numpy(dtype=float), marker='o', markersize=3,
                    label='Recall (known illicit)')
            ax.plot(steps, rows.precision.to_numpy(dtype=float), marker='s', markersize=3,
                    label='Precision (known alerts)')
            ax.set(title=str(policy), xlabel='Time step', ylim=(-0.03, 1.03))
            ax.set_xticks(steps)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, loc='best')
        axes[0][0].set_ylabel('Score (known labels only)')
        fig.suptitle(f'{model} | Frozen policies by step | EXPOSED 35–49', fontsize=11)
        fig.text(.5, .005, 'Diagnostic only: periods 35–49 were previously exposed; no retuning.',
                 ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, .94))
        path = plot_dir / f'TIME_SERIES_RECALL_PRECISION_{model}.png'
        fig.savefig(path, dpi=140)
        plt.close(fig)
        created.append(path)
    return created


def _plot_topk_comparison(topk_summary, plot_dir):
    """Compare verified Precision@K and known-positive Recall@K for all models.

    Why: a Top-K list may contain unverified alerts. Its precision denominator
    includes only *known* alerts; the chart must not imply that unknowns are licit.
    """
    if topk_summary.empty:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    metrics = [('micro_recall_at_K', 'Recall@K of known illicit'),
               ('precision_among_verified_alerts', 'Precision@K among verified alerts')]
    for index, (model, rows) in enumerate(topk_summary.groupby('model', sort=True)):
        data = rows.sort_values('K')
        markers = ('o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', '*', 'h')
        for ax, (column, _) in zip(axes, metrics):
            ax.plot(data.K, data[column], marker=markers[index % len(markers)],
                    markersize=4, label=model)
    for ax, (_, label) in zip(axes, metrics):
        ax.set(xlabel='Alerts per period (K)', ylabel=label, ylim=(-.03, 1.03))
        ax.set_xticks(sorted(topk_summary.K.unique()))
        ax.grid(alpha=.25)
    fig.suptitle('Top-K comparison | EXPOSED 35–49 | known-label definitions', fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .01),
               ncol=min(6, len(labels)), fontsize=8)
    fig.text(.5, .91, 'Unknown alerts are excluded from verified precision; they are NOT licit.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .18, 1, .86))
    path = plot_dir / 'COMPARE_PRECISION_RECALL_AT_K.png'
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return [path]


def _plot_threshold_sensitivity(sweep, summary, plot_dir):
    """Plot hypothetical threshold sensitivity, with historical thresholds marked.

    Why: inspecting hypothetical thresholds is permissible as an EXPOSED diagnostic,
    but choosing a new threshold from these plots would invalidate a blind-test claim.
    Duplicate score curves across policies are plotted once per model.
    """
    created = []
    for model, rows in sweep.groupby('model', sort=True):
        unique_rows = rows.sort_values('threshold').drop_duplicates('threshold')
        if not unique_rows['diagnostic_only_NOT_for_threshold_selection'].all():
            raise AssertionError('Threshold sweep must be marked diagnostic-only')
        fig, ax = plt.subplots(figsize=(8.5, 4.7))
        for col, label in [('recall', 'Recall'), ('precision', 'Precision'),
                           ('F1', 'F1'), ('F2', 'F2')]:
            ax.plot(unique_rows.threshold, unique_rows[col], label=label)
        policy_rows = summary.loc[summary.model == model].sort_values('policy')
        for _, row in policy_rows.iterrows():
            ax.axvline(float(row.threshold), linestyle='--', linewidth=1.2,
                       label=f"Frozen {row.policy}: {row.threshold:.4f}")
        ax.set(xlabel='Hypothetical decision threshold', ylabel='Known-label metric',
               xlim=(0, 1), ylim=(-.03, 1.03))
        ax.grid(alpha=.25)
        ax.legend(loc='best', fontsize=8)
        fig.suptitle(f'{model} | EXPOSED threshold sensitivity — DIAGNOSTIC ONLY', fontsize=11)
        fig.text(.5, .01, 'NEVER select or retune a threshold using exposed periods 35–49.',
                 ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .045, 1, .94))
        path = plot_dir / f'THRESHOLD_SENSITIVITY_{model}_DIAGNOSTIC_ONLY.png'
        fig.savefig(path, dpi=140)
        plt.close(fig)
        created.append(path)
    return created


def _plot_reliability_diagrams(calibration_bins, plot_dir):
    """Plot known-label reliability with bin counts to expose sparse calibration bins.

    Why: observed outcomes are selectively labeled and bins with no observations
    must not become fake zero-risk dots. Counts show where each comparison is weak.
    """
    created = []
    for model, rows in calibration_bins.groupby('model', sort=True):
        rows = rows.sort_values('bin')
        nonempty = rows.loc[rows.count_known > 0]
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5))
        ax = axes[0]
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal calibration')
        ax.scatter(nonempty.mean_score, nonempty.fraction_illicit_known,
                   s=26, zorder=3, label='Observed, nonempty bins')
        ax.plot(nonempty.mean_score, nonempty.fraction_illicit_known, alpha=.6)
        ax.set(xlabel='Mean predicted illicit probability',
               ylabel='Observed illicit fraction (known only)',
               xlim=(-.03, 1.03), ylim=(-.03, 1.03))
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        axes[1].bar(rows.bin.to_numpy(dtype=int), rows.count_known.to_numpy(dtype=int))
        axes[1].set(xlabel='Probability bin (equal width)', ylabel='Known transactions')
        axes[1].set_xticks(rows.bin)
        axes[1].grid(axis='y', alpha=.25)
        ece = float(rows['ECE_known_labeled_only'].iloc[0])
        fig.suptitle(f'{model} | EXPOSED reliability (known labels only) | ECE={ece:.4f}', fontsize=11)
        fig.text(.5, .005, 'Descriptive only: selectively observed outcomes; not population calibration.',
                 ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .045, 1, .94))
        path = plot_dir / f'RELIABILITY_{model}.png'
        fig.savefig(path, dpi=140)
        plt.close(fig)
        created.append(path)
    return created


def create_supplementary_figures(results_dir):
    """Generate four supplementary graphic types from saved CSVs, never refitting models.

    This function is also safe for a finished, extracted historical run: it
    reads only existing FULL_METRICS CSV reports and writes PNG files under plots/.
    """
    root = Path(results_dir)
    report_dir = root / 'FULL_METRICS'
    plot_dir = report_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        name: pd.read_csv(report_dir / name) for name in (
            'CLASSIFICATION_BY_TIME_STEP.csv', 'THRESHOLD_SWEEP_DIAGNOSTIC_ONLY.csv',
            'CALIBRATION_BINS.csv', 'CLASSIFICATION_REPORT_ALL_MODELS.csv')
    }
    # Fail closed when report subsets are incomplete: every evaluated model must
    # appear in all three supplementary chart sources and the frozen-policy summary.
    expected_models = set(sources['CLASSIFICATION_REPORT_ALL_MODELS.csv'].model)
    for name, frame in sources.items():
        if set(frame.model) != expected_models:
            raise ValueError(f'Incomplete or mismatched model set in {name}')
    outputs = []
    outputs += _plot_time_step_comparisons(sources['CLASSIFICATION_BY_TIME_STEP.csv'], plot_dir)
    topk_file = report_dir / 'RECALL_PRECISION_AT_K_SUMMARY.csv'
    if topk_file.is_file():
        outputs += _plot_topk_comparison(pd.read_csv(topk_file), plot_dir)
    outputs += _plot_threshold_sensitivity(sources['THRESHOLD_SWEEP_DIAGNOSTIC_ONLY.csv'],
                                           sources['CLASSIFICATION_REPORT_ALL_MODELS.csv'], plot_dir)
    outputs += _plot_reliability_diagrams(sources['CALIBRATION_BINS.csv'], plot_dir)
    expected_count = 3 * len(expected_models) + int(topk_file.is_file())
    if len(outputs) != expected_count or any(not path.is_file() or path.stat().st_size == 0 for path in outputs):
        raise AssertionError('Supplementary chart generation incomplete; refusing to seal run')
    print(f'SUPPLEMENTARY PLOTS COMPLETE: {len(outputs)} PNGs written to {plot_dir}', flush=True)
    return outputs

def generate_metric_reports(results_dir):
    """Build all supplementary report files from existing frozen final predictions.

    This is post-processing: the historical F1 and false-positive-budget policies
    are used verbatim, and all alternative-threshold curves are descriptive only.
    Return the destination directory, whose files will enter the run checksum manifest.
    """
    root = Path(results_dir)
    required = ('FINAL_DIAGNOSTIC_ALL_MODELS.csv', 'LABELED_DIAGNOSTIC_PREDICTIONS.csv')
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(f'Missing finished-run artifact: {name}')
    frozen = pd.read_csv(root / required[0])
    predictions = pd.read_csv(root / required[1])
    if not len(frozen) or not len(predictions):
        raise ValueError('Empty predictions or policy report')
    if predictions.duplicated(['model', 'tx_id']).any():
        raise ValueError('Duplicate diagnostic transaction prediction by model and tx_id')
    if not predictions.known_label.isin([0, 1]).all():
        raise ValueError('Only known labels may appear in diagnostic prediction CSV')
    if not (predictions.time_step.between(35, 49)).all():
        raise ValueError('Expected exclusively exposed diagnostic periods 35 through 49')
    if set(predictions.model) != set(frozen.model):
        raise ValueError('Prediction-model set differs from final policy-summary models')
    if frozen.duplicated(['model', 'policy']).any():
        raise ValueError('Duplicate frozen policy per model')
    if not np.isfinite(predictions.calibrated_score).all() or not predictions.calibrated_score.between(0,1).all():
        raise ValueError('Invalid calibrated prediction probabilities')

    target = root / 'FULL_METRICS'
    target.mkdir(exist_ok=True)
    (target/'plots').mkdir(exist_ok=True)
    summary, bystep, threshold_sweep, curves, bins, checks = [], [], [], [], [], []
    cached = {name: df.sort_values('time_step') for name, df in predictions.groupby('model')}
    for _, original in frozen.iterrows():
        model, policy = str(original['model']), str(original['policy'])
        subset = cached[model]
        y = subset.known_label.to_numpy(dtype=int)
        p = subset.calibrated_score.to_numpy(dtype=float)
        fixed_threshold = float(original['threshold'])
        actual = _binary_metrics(y, p, fixed_threshold)
        _check_original_rows(original, actual)
        summary.append({'model': model, 'policy': policy, 'data_role': 'EXPOSED_POSTHOC_DIAGNOSTIC', **actual})
        checks.append({'model': model, 'policy': policy, 'result': 'PASS', 'comparison': 'Original AP/ROC/Precision/Recall/F1/TP/FP/FN/TN independently reproduced'})
        _plot_confusion(actual, target/'plots'/f'CONFUSION_{model}_{policy}.png', f'{model} | {policy}')
        for period, group in subset.groupby('time_step', sort=True):
            res = _binary_metrics(group.known_label, group.calibrated_score, fixed_threshold)
            bystep.append({'model': model, 'policy': policy, 'time_step': int(period), **res})
        # Do not perform a threshold search to choose an optimal policy on exposed diagnostics.
        fixed_grid = np.r_[0.0, 0.001, 0.005, 0.01, 0.02, 0.05,
                          np.arange(.1, 1., .05), .975, .99, .995, .999, 1.0]
        for candidate in np.unique(np.r_[fixed_grid, fixed_threshold]):
            stats = _binary_metrics(y, p, float(candidate))
            threshold_sweep.append({'model': model, 'policy': policy,
                                    'is_original_frozen_policy_threshold': bool(candidate == fixed_threshold),
                                    'diagnostic_only_NOT_for_threshold_selection': True, **stats})

    for model, subset in cached.items():
        y = subset.known_label.to_numpy(dtype=int)
        p = subset.calibrated_score.to_numpy(dtype=float)
        reliability, ece = _reliability_bins(y, p)
        bins.extend([{'model': model, 'ECE_known_labeled_only': ece, **r} for r in reliability])
        if np.unique(y).size != 2:
            continue
        pre, rec, th = precision_recall_curve(y, p)
        fpr, tpr, roc_th = roc_curve(y, p)
        for x, z, t in _downsample_points(rec, pre, np.r_[th, np.nan]):
            curves.append({'model': model, 'curve': 'PR', 'x': x, 'y': z, 'threshold': t})
        for x, z, t in _downsample_points(fpr, tpr, roc_th):
            curves.append({'model': model, 'curve': 'ROC', 'x': x, 'y': z, 'threshold': t})
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].plot(rec, pre)
        axes[0].axhline(y.mean(), linestyle='--', color='gray', label='Known-label prevalence')
        axes[0].set(xlabel='Recall', ylabel='Precision', title=f'{model} PR | AP={average_precision_score(y, p):.4f}', xlim=(0, 1), ylim=(0, 1))
        axes[0].legend(fontsize=8)
        axes[1].plot(fpr, tpr)
        axes[1].plot([0,1],[0,1], linestyle='--', color='gray')
        axes[1].set(xlabel='False positive rate', ylabel='True positive rate', title=f'{model} ROC | AUC={roc_auc_score(y, p):.4f}', xlim=(0, 1), ylim=(0, 1))
        fig.suptitle('EXPOSED 35-49 | known labels only | NOT a new blind test', fontsize=10)
        fig.tight_layout()
        fig.savefig(target/'plots'/f'PR_ROC_{model}.png', dpi=140)
        plt.close(fig)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(target/'CLASSIFICATION_REPORT_ALL_MODELS.csv', index=False)
    pd.DataFrame(bystep).to_csv(target/'CLASSIFICATION_BY_TIME_STEP.csv', index=False)
    pd.DataFrame(threshold_sweep).to_csv(target/'THRESHOLD_SWEEP_DIAGNOSTIC_ONLY.csv', index=False)
    pd.DataFrame(curves).to_csv(target/'PR_ROC_CURVE_POINTS.csv', index=False)
    pd.DataFrame(bins).to_csv(target/'CALIBRATION_BINS.csv', index=False)
    pd.DataFrame(checks).to_csv(target/'INDEPENDENT_METRIC_RECOMPUTATION_CHECK.csv', index=False)
    if (root/'TOPK_ALERT_DETAILS.csv').exists():
        details = pd.read_csv(root/'TOPK_ALERT_DETAILS.csv')
        if len(details) and not details.known_label.isin([-1,0,1]).all():
            raise ValueError('Unrecognized label in Top-K data')
        if len(details):
            _topk_reports(predictions, details, target)

    # Display visual comparisons for the historical F1 policy without selecting a new winner.
    f1_rows = summary_df.loc[summary_df.policy == 'F1_policy'].sort_values('AP', ascending=True)
    if len(f1_rows):
        fig, ax = plt.subplots(figsize=(max(9, 0.72*len(f1_rows)), 5.4))
        x = np.arange(len(f1_rows))
        width = 0.22
        ax.bar(x - width, f1_rows['precision'], width, label='Precision (frozen policy)')
        ax.bar(x, f1_rows['recall'], width, label='Recall (frozen policy)')
        ax.bar(x + width, f1_rows['F1'], width, label='F1 (frozen policy)')
        ax.set_xticks(x, f1_rows.model, rotation=45, ha='right')
        ax.set_ylabel('Metric on known labeled transactions')
        ax.set_ylim(0, 1)
        ax.set_title('Model comparison | exposed retrospective diagnostics 35-49')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(target/'plots'/'COMPARE_RECALL_PRECISION_F1.png', dpi=140)
        plt.close(fig)

    # Postprocess already-frozen CSV metrics into additional descriptive PNGs.
    # The runner calls this before SHA-256 manifest creation and results ZIP packing.
    create_supplementary_figures(root)

    (target/'METRIC_DEFINITIONS.md').write_text("""# Full Elliptic++ classification metrics

**Data status:** periods 35–49 were previously exposed; these are post-hoc,
retrospective diagnostics, **not a new independent blind test**. This report
uses only transactions with verified labels 0 (licit) and 1 (illicit).
Unverified / unknown (`-1`) transactions are NOT assumed licit.

## Definitions and denominators

- **TP**: predicted illicit and known illicit.
- **FP**: predicted illicit and known licit.
- **FN**: predicted licit and known illicit.
- **TN**: predicted licit and known licit.
- **Recall / Sensitivity / TPR** = TP / (TP + FN).
- **Precision / PPV** = TP / (TP + FP).
- **F1** = 2TP / (2TP + FP + FN).
- **F2** = 5TP / (5TP + 4FN + FP); emphasizes recall.
- **Specificity / TNR** = TN / (TN + FP).
- **FPR** = FP / (FP + TN); **FNR** = FN / (FN + TP).
- **NPV** = TN / (TN + FN).
- **Accuracy** = (TP + TN) / (TP + FP + FN + TN).
- **Balanced accuracy** = (Recall + Specificity) / 2, when both classes exist.
- **MCC**: Matthews correlation coefficient, known labels only.
- **Average Precision (AP)**: scikit-learn's threshold-free average_precision_score,
  **not** trapezoidal PR area.
- **ROC-AUC**: rank discrimination across known labels.
- **Brier** and **Log loss**: calibration/probability quality conditional on observed labels.
- **ECE**: descriptive equal-width 10-bin calibration discrepancy on known labels.
- **FP_per_10k_known_licit** = 10,000 * FP / (FP + TN), NOT per all transactions.
- **Recall@K of known illicit** uses all known illicit transactions in each period
  as denominator even when the Top-K alert list includes unknowns.
- **Precision@K among known alerts** = illicit / (illicit + licit) among alerts
  with known outcomes only. Unknown alerts are counted separately; this is NOT
  population-wide precision.
- Precision and Recall use `zero_division=0` when their denominator is zero,
  matching the project's original `metrics()` convention. Other undefined
  ratios are stored as NaN.

## Policies and leakage protection

`F1_policy` and `FPR_5_per_10k` thresholds are frozen using period 32–34 labels.
Their outcomes for 35–49 are reported as observed and need not meet the earlier
false-positive target. `THRESHOLD_SWEEP_DIAGNOSTIC_ONLY.csv` shows hypothetical
outcomes at other thresholds to describe sensitivity, **not** to optimize or
select a replacement policy on 35–49. Curves are plotted from full-resolution
scores; stored curve points are thinned only for manageable CSV size.

## Files

- `CLASSIFICATION_REPORT_ALL_MODELS.csv`: all metrics for each model × historical policy.
- `CLASSIFICATION_BY_TIME_STEP.csv`: metrics for each model × policy × period.
- `THRESHOLD_SWEEP_DIAGNOSTIC_ONLY.csv`: purely descriptive threshold sensitivity.
- `PR_ROC_CURVE_POINTS.csv`: visualization coordinates (downsampled, NOT metric computations).
- `CALIBRATION_BINS.csv`: 10-bin reliability summary and ECE.
- `RECALL_PRECISION_AT_K_BY_STEP.csv`: K=10/25/50/100 where saved ranking allows.
- `RECALL_PRECISION_AT_K_SUMMARY.csv`: pooled Top-K known-label metrics.
- `INDEPENDENT_METRIC_RECOMPUTATION_CHECK.csv`: original vs recomputed checks.
- `plots/`: confusion matrices, PR/ROC, cross-model policy metrics, and:
  - `TIME_SERIES_RECALL_PRECISION_<MODEL>.png`: per-period precision/recall for each frozen policy.
  - `COMPARE_PRECISION_RECALL_AT_K.png`: cross-model Top-K on known-label definitions.
  - `THRESHOLD_SENSITIVITY_<MODEL>_DIAGNOSTIC_ONLY.png`: hypothetical thresholds (never for tuning).
  - `RELIABILITY_<MODEL>.png`: reliability bins and known-label bin support.
""", encoding='utf-8')
    print(f'METRICS COMPLETE: {len(summary)} model/policy rows, {len(bystep)} step rows in {target}', flush=True)
    return target
