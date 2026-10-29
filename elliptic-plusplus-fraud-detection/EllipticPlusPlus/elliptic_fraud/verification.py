# MODULE: verification.py
# PURPOSE: Independent saved-artifact auditor
# RESEARCH RULE: Metrics are recalculated from saved predictions instead of trusting summaries.

"""Independent reproducibility checks for finished Elliptic++ runs.

Purpose
-------
* This module NEVER fits a model and NEVER uses diagnostic data to tune one.
* It recomputes metrics from saved row-level predictions, verifies the ZIP's
  SHA-256 manifest, and checks the saved model reload status.
* It can compare a refactored run to the historical v3.1 result archive.

Why a separate module?
----------------------
A broken experiment runner should not be able to call its own checks "PASS".
The reviewer can run this entrypoint *after* a run has finished, from either
its directory or its ZIP archive, without executing the training code again.

Caveat: numerical equivalence of saved CSVs is not a proof that the original
features were available online, or that a previously exposed test was blind.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any
import json
import zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.special import expit, logit
REQUIRED = ('RUN_SUCCESS.json', 'SHA256_MANIFEST.csv', 'DEV_SUMMARY.csv', 'DEV_NEGATIVE_CONTROL_SUMMARY.csv', 'FINAL_DIAGNOSTIC_ALL_MODELS.csv', 'DIAGNOSTIC_BY_STEP_ALL_MODELS.csv', 'LABELED_DIAGNOSTIC_PREDICTIONS.csv', 'MODEL_RELOAD_VERIFICATION.csv', 'CALIBRATOR_MONOTONICITY.csv', 'TOPK_ALL_TRANSACTIONS_BATCH.csv', 'LABEL_COVERAGE.csv')

@dataclass
class RunReader:
    """Read a completed run from a directory or ZIP, without extraction."""
    path: Path

    # FUNCTION: RunReader.__post_init__ — Index ZIP/directory artifacts safely and reject ambiguous paths.
    # WHY: Saved artifacts are audited by a separate module, not trusted on success marker alone.
    def __post_init__(self):
        # RESEARCH RATIONALE:
        # Permit folder or ZIP audit; reject duplicate basenames and unsafe archive paths.
        self.path = Path(self.path)
        # IF (self.path.is_file()): Handle ZIP input separately from a result directory.
        if self.path.is_file():
            # IF (not zipfile.is_zipfile(self.path)): Reject files that are not real ZIP archives.
            if not zipfile.is_zipfile(self.path):
                raise ValueError('Expected a ZIP archive or results directory')
            with zipfile.ZipFile(self.path) as z:
                names = [n for n in z.namelist() if not n.endswith('/')]
                # LOOP (names): Process every item using the same fixed input/output contract.
                for name in names:
                    parts = PurePosixPath(name).parts
                    # IF (name.startswith('/') or '..' in parts): Reject unsafe ZIP path traversal elements.
                    if name.startswith('/') or '..' in parts:
                        raise ValueError(f'Unsafe ZIP path: {name}')
                self.map = {}
                # LOOP (names): Process every item using the same fixed input/output contract.
                for name in names:
                    short = name.split('/')[-1]
                    # IF (short in self.map): Duplicate basenames would make audit artifact lookup ambiguous.
                    if short in self.map:
                        raise ValueError(f'Duplicate artifact filename: {short}')
                    self.map[short] = name
        # IF (self.path.is_dir()): Index directory artifacts without modifying the run.
        elif self.path.is_dir():
            self.map = {}
            # LOOP (self.path.rglob('*')): Process every item using the same fixed input/output contract.
            for file in self.path.rglob('*'):
                # IF (file.is_file()): Handle missing/incomplete files explicitly; never silently substitute fake data or claim a run succeeded.
                if file.is_file():
                    # IF (file.name in self.map): Reject ambiguous artifacts in directory-mode audits.
                    if file.name in self.map:
                        raise ValueError(f'Duplicate artifact filename: {file.name}')
                    self.map[file.name] = str(file.relative_to(self.path))
        else:
            raise FileNotFoundError(self.path)

    # FUNCTION: RunReader.exists — Implement exists in the research workflow.
    # WHY: Saved artifacts are audited by a separate module, not trusted on success marker alone.
    def exists(self, name: str) -> bool:
        return name in self.map

    # FUNCTION: RunReader.bytes — Read preserved artifact bytes without mutation.
    # WHY: Saved artifacts are audited by a separate module, not trusted on success marker alone.
    def bytes(self, name: str) -> bytes:
        # RESEARCH RATIONALE:
        # Read stored bytes without editing original research outputs.
        # IF (not self.exists(name)): Fail if a required audit artifact is missing.
        if not self.exists(name):
            raise FileNotFoundError(f'Missing {name} in {self.path}')
        # IF (self.path.is_file()): Handle ZIP input separately from a result directory.
        if self.path.is_file():
            with zipfile.ZipFile(self.path) as z:
                return z.read(self.map[name])
        return (self.path / self.map[name]).read_bytes()

    # FUNCTION: RunReader.csv — Parse a persisted tabular artifact.
    # WHY: Saved artifacts are audited by a separate module, not trusted on success marker alone.
    def csv(self, name: str) -> pd.DataFrame:
        return pd.read_csv(BytesIO(self.bytes(name)))

    # FUNCTION: RunReader.json — Parse saved metadata in JSON format.
    # WHY: Saved artifacts are audited by a separate module, not trusted on success marker alone.
    def json(self, name: str) -> dict:
        return json.loads(self.bytes(name))

# FUNCTION: _check — Implement _check in the research workflow.
# WHY: Saved artifacts are audited by a separate module, not trusted on success marker alone.
def _check(rows: list[dict], check: str, passed: bool, detail: str='') -> None:
    rows.append({'check': check, 'passed': bool(passed), 'detail': str(detail)[:250]})


# ---------------------------------------------------------
# Audit Run
# WHY: Verify stored scores, confusion counts, calibrators, hashes and outputs.
# ---------------------------------------------------------
# FUNCTION: audit_run — Independently recompute saved-prediction metrics and validate integrity checks.
# WHY: Saved artifacts are audited by a separate module, not trusted on success marker alone.
def audit_run(source: str | Path, *, require_models: int | None=None) -> dict:
    """Validate finished model artifacts against independent recomputation.

    Rebuild AP, ROC-AUC, confusion matrix, and FPR for every model/policy.
    The validation uses *only* recorded predictions, not training-source values.
    Any failed invariant raises AssertionError with clear reports.
    """
    # RESEARCH RATIONALE:
    # Independently recompute summary metrics from saved per-transaction predictions.
    # Verify checksums, policy thresholds, calibrator coefficients and model reload reports.
    # A completed marker without matching artifacts is never sufficient.
    r = RunReader(Path(source))
    checks: list[dict] = []
    # LOOP (REQUIRED): Process every item using the same fixed input/output contract.
    for name in REQUIRED:
        _check(checks, f'required:{name}', r.exists(name))
    missing = [x for x in REQUIRED if not r.exists(x)]
    # IF (missing): Do not audit a partial or incomplete run as if it were successful.
    if missing:
        raise AssertionError(f'Missing required artifacts: {missing}')
    success = r.json('RUN_SUCCESS.json')
    _check(checks, 'run completed', success.get('status') == 'COMPLETE')
    manifest = r.csv('SHA256_MANIFEST.csv')
    seen = set()
    # LOOP (manifest.itertuples(index=False)): Process every item using the same fixed input/output contract.
    for item in manifest.itertuples(index=False):
        name = str(item.file)
        # IF (Path(name).name in seen): Reject duplicate names inside the recorded checksum list.
        if Path(name).name in seen:
            _check(checks, f'unique:{name}', False)
            continue
        seen.add(Path(name).name)
        # IF (Path(name).name not in r.map): Every manifest entry must point to an existing artifact.
        if Path(name).name not in r.map:
            _check(checks, f'hash:{name}', False, 'Absent')
            continue
        digest = sha256(r.bytes(Path(name).name)).hexdigest()
        _check(checks, f'hash:{name}', digest == item.sha256)
    actual_files = set(r.map) - {'SHA256_MANIFEST.csv', 'EXTERNAL_AUDIT.json', 'V31_COMPARISON.json'}
    manifest_files = set(manifest.file.map(lambda x: Path(str(x)).name))
    _check(checks, 'manifest_covers_every_artifact', actual_files == manifest_files, f'untracked={sorted(actual_files - manifest_files)}; missing={sorted(manifest_files - actual_files)}')
    preds = r.csv('LABELED_DIAGNOSTIC_PREDICTIONS.csv')
    summary = r.csv('FINAL_DIAGNOSTIC_ALL_MODELS.csv')
    by_step = r.csv('DIAGNOSTIC_BY_STEP_ALL_MODELS.csv')
    models = sorted(preds.model.unique())
    _check(checks, 'model_set_matches_summary', set(models) == set(summary.model.unique()))
    _check(checks, 'model_set_matches_steps', set(models) == set(by_step.model.unique()))
    # IF (require_models is not None): Check that exactly the requested number of models was evaluated.
    if require_models is not None:
        _check(checks, 'required_model_count', len(models) == require_models, str(models))
    _check(checks, 'labels_only_0_1', set(preds.known_label.unique()) == {0, 1})
    _check(checks, 'prediction_scores_finite', np.isfinite(preds[['raw_score', 'calibrated_score']].to_numpy()).all())
    _check(checks, 'calibrated_scores_in_range', preds.calibrated_score.between(0, 1).all())
    _check(checks, 'only_exposed_steps', preds.time_step.between(35, 49).all())
    _check(checks, 'prediction_row_unique', not preds.duplicated(['model', 'tx_id', 'time_step']).any())
    numeric_tol = 1e-07
    calib = r.csv('CALIBRATOR_MONOTONICITY.csv').set_index('model')
    # LOOP (models): Process every item using the same fixed input/output contract.
    for model in models:
        p = preds[preds.model == model]
        y = p.known_label.to_numpy(dtype=int)
        s = p.calibrated_score.to_numpy(dtype=float)
        ap = float(average_precision_score(y, s))
        cc = calib.loc[model]
        reconstructed = expit(float(cc.intercept) + float(cc.slope) * logit(np.clip(p.raw_score.to_numpy(dtype=float), 1e-06, 1 - 1e-06)))
        _check(checks, f'calibration_roundtrip:{model}', np.max(np.abs(reconstructed - s)) <= 1e-07)
        _check(checks, f'model_artifacts:{model}', r.exists(f'{model}_pipeline.joblib') and (r.exists(f'{model}_model.joblib') or r.exists(f'{model}_weights.pt')))
        auc = float(roc_auc_score(y, s))
        # LOOP (summary[summary.model == model].itertuples(index=False)): Process every item using the same fixed input/output contract.
        for row in summary[summary.model == model].itertuples(index=False):
            label = f'{model}/{row.policy}'
            _check(checks, f'AP:{label}', abs(ap - row.AP) <= numeric_tol)
            _check(checks, f'ROC:{label}', abs(auc - row.ROC_AUC) <= numeric_tol)
            _check(checks, f'count:{label}', len(p) == row.n and int(y.sum()) == row.illicit)
            pred = s >= row.threshold
            tp = int(np.sum(pred & (y == 1)))
            fp = int(np.sum(pred & (y == 0)))
            fn = int(np.sum(~pred & (y == 1)))
            tn = int(np.sum(~pred & (y == 0)))
            _check(checks, f'confusion:{label}', (tp, fp, fn, tn) == (row.TP, row.FP, row.FN, row.TN), f'actual {(tp, fp, fn, tn)}')
            fpr = 10000 * fp / max(1, fp + tn)
            _check(checks, f'FPR_per_10k:{label}', abs(fpr - row.FP_per_10k_licit) <= numeric_tol)
        steps = by_step[(by_step.model == model) & (by_step.policy == 'F1_policy')]
        # LOOP (steps.itertuples(index=False)): Process every item using the same fixed input/output contract.
        for rr in steps.itertuples(index=False):
            j = p[p.time_step == rr.time_step]
            yy = j.known_label.to_numpy(dtype=int)
            ss = j.calibrated_score.to_numpy(dtype=float)
            exact_ap = float(average_precision_score(yy, ss)) if len(set(yy)) == 2 else np.nan
            close = np.isnan(exact_ap) and np.isnan(rr.AP) or abs(exact_ap - rr.AP) <= numeric_tol
            _check(checks, f'period_AP:{model}/{rr.time_step}', close)
    reload = r.csv('MODEL_RELOAD_VERIFICATION.csv')
    _check(checks, 'reload_covers_models', set(reload.model) == set(models))
    _check(checks, 'all_model_reload_status_pass', reload.status.eq('PASS').all())
    _check(checks, 'all_reloaded_prediction_deltas_small', reload.raw_prediction_max_abs_diff.le(1e-05).all())
    _check(checks, 'all_reloaded_feature_deltas_small', reload.feature_preprocessing_max_abs_diff.le(1e-06).all())
    calibrators = r.csv('CALIBRATOR_MONOTONICITY.csv')
    _check(checks, 'calibrators_cover_models', set(calibrators.model) == set(models))
    _check(checks, 'positive_calibration_slope', calibrators.slope.gt(0).all())
    _check(checks, 'calibration_optimizer_success', calibrators.optimizer_success.eq(True).all())
    topk = r.csv('TOPK_ALL_TRANSACTIONS_BATCH.csv')
    status_cols = ['known_illicit_in_alerts', 'known_licit_in_alerts', 'UNKNOWN_unverified_in_alerts']
    _check(checks, 'topk_all_statuses_sum_to_alerts', topk[status_cols].sum(axis=1).eq(topk.alerts).all())
    _check(checks, 'topk_covers_models', set(topk.model) == set(models))
    # IF (r.exists('TOPK_ALERT_DETAILS.csv')): Check individual Top-K records against the period summary.
    if r.exists('TOPK_ALERT_DETAILS.csv'):
        details = r.csv('TOPK_ALERT_DETAILS.csv')
        _check(checks, 'topk_details_match_alert_counts', details.groupby(['model', 'time_step']).size().sort_index().equals(topk.set_index(['model', 'time_step']).alerts.sort_index()))
        _check(checks, 'topk_scores_descending_by_rank', all(((group.sort_values('rank').score.diff().dropna() <= 1e-12).all() for _, group in details.groupby(['model', 'time_step']))))
        _check(checks, 'topk_unknown_not_negative', set(details.known_label.unique()).issubset({-1, 0, 1}))
    # IF (r.exists('DATA_QA.json')): Cross-check aggregated label coverage against the source quality report.
    if r.exists('DATA_QA.json'):
        q = r.json('DATA_QA.json')
        coverage = r.csv('LABEL_COVERAGE.csv')
        _check(checks, 'label_coverage_exact_totals', int(coverage['all'].sum()) == q['n'] and int(coverage['illicit'].sum()) == q['label_illicit'] and (int(coverage['licit'].sum()) == q['label_licit']) and (int(coverage['unknown'].sum()) == q['unknown']))
    failed = [c for c in checks if not c['passed']]
    report = {'source': str(source), 'model_count': len(models), 'models': models, 'checks_total': len(checks), 'checks_passed': len(checks) - len(failed), 'checks_failed': len(failed), 'passed': not failed, 'failed': failed, 'checks': checks, 'meaning': 'Audit of saved artifacts; not a new blind test.'}
    return report


# ---------------------------------------------------------
# Compare Runs
# WHY: Require identical source hashes and splits before comparing numerical metrics.
# ---------------------------------------------------------
# FUNCTION: compare_runs — Compare full real runs only when data hashes and time protocol agree.
# WHY: Saved artifacts are audited by a separate module, not trusted on success marker alone.
def compare_runs(new_source: str | Path, old_source: str | Path, *, metric_atol: float=0.02, threshold_atol: float=0.05) -> dict:
    """Compare v3.1 vs modular runs on equal data/splits/model sets.

    Metric tolerance addresses stochastic numerical/device variability. A
    mismatch is always reported, never ignored or silently rescaled.
    A valid comparison requires BOTH full runs to use real Elliptic++.
    """
    # RESEARCH RATIONALE:
    # Match real-dataset file hashes and chronological protocol before comparing model performance.
    # The numeric tolerance documents variability; do not increase it to conceal genuine differences.
    current, historical = (RunReader(Path(new_source)), RunReader(Path(old_source)))
    nmeta = current.json('RUN_SUCCESS.json')
    ometa = historical.json('RUN_SUCCESS.json')
    # IF (nmeta.get('dataset_kind') != ometa.get('dataset_kind')): Reject a synthetic-to-real comparison as scientifically invalid.
    if nmeta.get('dataset_kind') != ometa.get('dataset_kind'):
        raise ValueError('Cannot compare synthetic and real datasets')
    # IF (not str(nmeta.get('dataset_kind', '')).startswith('official')): Reject a synthetic-to-real comparison as scientifically invalid.
    if not str(nmeta.get('dataset_kind', '')).startswith('official'):
        raise ValueError('Reference comparison requires REAL Elliptic++ runs')
    # LOOP (('DATA_QA.json', 'TIME_PROTOCOL.csv', 'DEV_SUMMARY.csv')): Process every item using the same fixed input/output contract.
    for file in ('DATA_QA.json', 'TIME_PROTOCOL.csv', 'DEV_SUMMARY.csv'):
        # IF (not current.exists(file) or not historical.exists(file)): Verify both runs contain the same required provenance files.
        if not current.exists(file) or not historical.exists(file):
            raise ValueError(f'Missing required regression provenance: {file}')
    current_data = current.json('DATA_QA.json')
    historical_data = historical.json('DATA_QA.json')
    # LOOP (('n', 'p_local', 'label_illicit', 'label_licit', 'unknown', 'edges')): Process every item using the same fixed input/output contract.
    for key in ('n', 'p_local', 'label_illicit', 'label_licit', 'unknown', 'edges'):
        # IF (current_data.get(key) != historical_data.get(key)): Reject different dataset cardinalities before comparing metrics.
        if current_data.get(key) != historical_data.get(key):
            raise ValueError(f'Data cardinality differs: {key}')
    current_hashes = current_data.get('source', {}).get('sha256', {})
    historical_hashes = historical_data.get('source', {}).get('sha256', {})
    # IF (not current_hashes or current_hashes != historical_hashes): Identical input CSV SHA-256 digests are mandatory for parity claims.
    if not current_hashes or current_hashes != historical_hashes:
        raise ValueError('Underlying dataset CSV SHA-256 checksums do not match')
    left_protocol = current.csv('TIME_PROTOCOL.csv')
    right_protocol = historical.csv('TIME_PROTOCOL.csv')
    pd.testing.assert_frame_equal(left_protocol, right_protocol, check_dtype=False)
    new = current.csv('FINAL_DIAGNOSTIC_ALL_MODELS.csv')
    old = historical.csv('FINAL_DIAGNOSTIC_ALL_MODELS.csv')
    # IF (set(map(tuple, new[['model', 'policy']].to_numpy())) != set(map(tuple, old[['model', 'policy']].to_num...): Compare the same models and decision policies, never unmatched subsets.
    if set(map(tuple, new[['model', 'policy']].to_numpy())) != set(map(tuple, old[['model', 'policy']].to_numpy())):
        raise ValueError('Reference comparison requires the same models and policies')
    joined = new.merge(old, on=['model', 'policy'], suffixes=('_modular', '_reference'), validate='one_to_one')
    rows = []
    # LOOP (joined.itertuples(index=False)): Process every item using the same fixed input/output contract.
    for item in joined.itertuples(index=False):
        # LOOP ([('AP', metric_atol), ('ROC_AUC', metric_atol), ('threshold', threshold_atol), ('p...): Process every item using the same fixed input/output contract.
        for metric, tolerance in [('AP', metric_atol), ('ROC_AUC', metric_atol), ('threshold', threshold_atol), ('precision', metric_atol), ('recall', metric_atol)]:
            a = float(getattr(item, metric + '_modular'))
            b = float(getattr(item, metric + '_reference'))
            rows.append({'model': item.model, 'policy': item.policy, 'metric': metric, 'modular': a, 'reference': b, 'abs_difference': abs(a - b), 'tolerance': tolerance, 'passed': np.isfinite(a) and np.isfinite(b) and (abs(a - b) <= tolerance)})
    dev_new = current.csv('DEV_SUMMARY.csv')
    dev_old = historical.csv('DEV_SUMMARY.csv')
    # IF (set(dev_new.model) != set(dev_old.model)): Compare identical development candidate sets.
    if set(dev_new.model) != set(dev_old.model):
        raise ValueError('Development model sets differ')
    dev_merge = dev_new.merge(dev_old, on='model', suffixes=('_modular', '_reference'), validate='one_to_one')
    # LOOP (dev_merge.itertuples(index=False)): Process every item using the same fixed input/output contract.
    for item in dev_merge.itertuples(index=False):
        # LOOP (('mean_AP', 'min_AP')): Process every item using the same fixed input/output contract.
        for metric in ('mean_AP', 'min_AP'):
            a = float(getattr(item, metric + '_modular'))
            b = float(getattr(item, metric + '_reference'))
            rows.append({'model': item.model, 'policy': 'DEV', 'metric': metric, 'modular': a, 'reference': b, 'abs_difference': abs(a - b), 'tolerance': metric_atol, 'passed': abs(a - b) <= metric_atol})
    dd = pd.DataFrame(rows)
    return {'source_modular': str(new_source), 'source_reference': str(old_source), 'passed': bool(dd.passed.all()), 'n_checks': len(dd), 'n_failed': int((~dd.passed).sum()), 'comparison': dd.to_dict('records'), 'warning': 'No equivalence claim unless all compared metrics pass; environments should match.'}
