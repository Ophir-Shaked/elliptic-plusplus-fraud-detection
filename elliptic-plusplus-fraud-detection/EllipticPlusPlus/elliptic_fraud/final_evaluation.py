# MODULE: final_evaluation.py
# PURPOSE: Retrospective scoring, persistence and model reload
# RESEARCH RULE: Previously exposed periods 35-49 are exploratory diagnostics, not a new blind test.

"""Post-exposure diagnostic, save/reload checks, policy and feature ablations.

Period 35-49 is *not* a new blind test. All model decisions occur before it.
"""
import time
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from lightgbm import LGBMClassifier
from .config import FOLDS, FINAL
from .models import MLP, SAGE, BidirSAGE, fit_tab, fit_predict
from .preprocessing import preprocess
from .evaluation import ap, metrics, threshold_f1, threshold_fpr, platt_fit, platt_apply
_ROUNDTRIP_FEATURES = {}
_NN_MODES = {'MLP': 'no_graph', 'SAGE_BATCH': 'batch', 'SAGE_NO_GRAPH': 'no_graph', 'SAGE_REWIRED': 'rewired', 'BIDIR_SAGE': 'bidir', 'BIDIR_NO_GRAPH': 'no_graph', 'BIDIR_REWIRED': 'bidir_rewired'}


# ---------------------------------------------------------
#  Validate Model Roundtrip
# WHY: Saved estimators and their preprocessing pipeline must reproduce predictions.
# ---------------------------------------------------------
# FUNCTION: _validate_model_roundtrip — Reload saved preprocessors and models and compare exact transaction scores.
# RESEARCH RULE: 35–49 are exposed; persisted scores and reload checks must remain independently auditable.
def _validate_model_roundtrip(name, model, d, x, preproc, all_pred, out, cache, args):
    """Assert loading a saved model reproduces raw scores, before declaring success."""
    # RESEARCH RATIONALE:
    # Replay frozen preprocessing and reload saved model weights before accepting outputs.
    # Compare raw transaction scores, not only aggregate metrics, to catch serialization mistakes.
    # IMPORTANT: Use an explicit local random generator rather than hidden global stochastic state.
    rng = np.random.default_rng(42)
    chosen = rng.choice(d.n, size=min(120, d.n), replace=False)
    imp, sc = preproc
    pipeline = joblib.load(out / f'{name}_pipeline.joblib')
    # INVARIANT: Fail immediately if this research or data-integrity assumption is broken.
    assert pipeline['model_name'] == name
    # INVARIANT: Fail immediately if this research or data-integrity assumption is broken.
    assert np.array_equal(pipeline['feature_indices'], _ROUNDTRIP_FEATURES[name])
    imp2, sc2 = (pipeline['imputer'], pipeline['scaler'])
    x2 = np.clip(sc2.transform(imp2.transform(d.X[:, _ROUNDTRIP_FEATURES[name]])), -20, 20).astype('float32')
    max_feature_diff = float(np.max(np.abs(x2[chosen] - x[chosen])))
    # IF (max_feature_diff > 1e-06): Handle this alternate branch explicitly so that the experiment remains deterministic and auditable.
    if max_feature_diff > 1e-06:
        raise AssertionError('Preprocessor reload mismatch ' + name)
    # IF (name in ('LOGIT', 'XGB', 'LGB', 'CAT')): Select a specific model implementation without altering the common fit/predict contract.
    if name in ('LOGIT', 'XGB', 'LGB', 'CAT'):
        loaded = joblib.load(out / f'{name}_model.joblib')
        # IMPORTANT: Extract positive-class scores for ranking; thresholds are chosen separately.
        pr = loaded.predict_proba(x2[chosen])[:, 1]
    else:
        saved = torch.load(out / f'{name}_weights.pt', map_location='cpu', weights_only=True)
        loaded = MLP(x.shape[1]) if name == 'MLP' else BidirSAGE(x.shape[1]) if name.startswith('BIDIR_') else SAGE(x.shape[1])
        # IMPORTANT: Restore the best or saved model parameters before inference.
        loaded.load_state_dict(saved['state_dict'])
        loaded.eval()
        mode = _NN_MODES[name]
        device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
        loaded = loaded.to(device)
        with torch.no_grad():
            pred = torch.sigmoid(loaded(torch.tensor(x2, device=device), cache[mode])).cpu().numpy()
        pr = pred[chosen]
    maxpred = float(np.max(np.abs(pr - all_pred[chosen])))
    # IF (maxpred > 1e-05): Handle this alternate branch explicitly so that the experiment remains deterministic and auditable.
    if maxpred > 1e-05:
        raise AssertionError(f'Saved model {name} failed reload test: {maxpred}')
    return {'model': name, 'raw_prediction_max_abs_diff': maxpred, 'feature_preprocessing_max_abs_diff': max_feature_diff, 'pipeline_loaded_from_disk': True, 'n_checked': len(chosen), 'status': 'PASS'}


# ---------------------------------------------------------
# Final All
# WHY: Separate fit, stop, calibrate, threshold and exposed assessment.
# ---------------------------------------------------------
# FUNCTION: final_all — Train, calibrate, choose policy thresholds, save exposed diagnostics, and verify artifacts.
# RESEARCH RULE: 35–49 are exposed; persisted scores and reload checks must remain independently auditable.
def final_all(d, args, out, keep, models):
    # RESEARCH RATIONALE:
    # Frozen order: FIT (1-26), STOP (27-29), CAL (30-31), POLICY (32-34), EXPOSED (35-49).
    # Top-K includes unverified nodes; they must not be counted as confirmed false positives.
    # Raw and calibrated scores are both saved for independent validation.
    fit = d.ix(*FINAL['fit'])
    stop = d.ix(*FINAL['stop'])
    cal = d.ix(*FINAL['calibration'])
    pol = d.ix(*FINAL['policy'])
    test = d.ix(*FINAL['diagnostic_test'])
    test_all = d.ix(*FINAL['diagnostic_test'], labeled=False)
    x, imp, sc = preprocess(d, FINAL['fit'][1], keep)
    cache = {}
    results = []
    by_step = []
    topk = []
    topk_details = []
    pred_rows = []
    cal_rows = []
    score_drift = []
    calibrator_rows = []
    reload_rows = []
    # LOOP (models): Repeat the same reproducible procedure for every selected fold, model or period.
    for name in models:
        start = time.time()
        # IMPORTANT: Use the shared interface so tabular and neural models follow the same time protocol.
        model, pall, info = fit_predict(name, d, x, fit, stop, np.arange(d.n), args, 42, cache)
        # IMPORTANT: Fit calibration on the calibration slice only, never on exposed diagnostics.
        pl = platt_fit(d.y[cal], pall[cal])
        pcal = platt_apply(pl, pall)
        calibrator_rows.append({'model': name, 'slope': pl.slope, 'intercept': pl.intercept, 'slope_strictly_positive': pl.slope >= 0.01 - 1e-09, 'slope_at_floor': pl.slope <= 0.010001, 'optimizer_success': pl.optimizer_success})
        # IMPORTANT: Derive classification threshold from the earlier policy period.
        policies = {'F1_policy': threshold_f1(d.y[pol], pcal[pol]), 'FPR_5_per_10k': threshold_fpr(d.y[pol], pcal[pol])}
        # LOOP (policies.items()): Repeat the same reproducible procedure for every selected fold, model or period.
        for pname, thr in policies.items():
            results.append({'model': name, 'policy': pname, 'fit_iterations': info['iterations'], 'runtime_s': time.time() - start, 'evaluation_role': 'EXPOSED_POSTHOC_DIAGNOSTIC', **metrics(d.y[test], pcal[test], thr)})
            # LOOP (range(35, 50)): Repeat the same reproducible procedure for every selected fold, model or period.
            for t in range(35, 50):
                sel = test[d.time[test] == t]
                # IF (len(sel)): Guard against missing or invalid inputs before numerical operations and reported metrics.
                if len(sel):
                    by_step.append({'model': name, 'policy': pname, 'time_step': t, **metrics(d.y[sel], pcal[sel], thr)})
        # LOOP ([('calibration', cal), ('policy', pol), ('diagnostic_test_EXPOSED', test)]): Repeat the same reproducible procedure for every selected fold, model or period.
        for split, idx in [('calibration', cal), ('policy', pol), ('diagnostic_test_EXPOSED', test)]:
            # LOOP ([('raw', pall[idx]), ('platt', pcal[idx])]): Repeat the same reproducible procedure for every selected fold, model or period.
            for typ, pr in [('raw', pall[idx]), ('platt', pcal[idx])]:
                cal_rows.append({'model': name, 'period': split, 'method': typ, 'brier': brier_score_loss(d.y[idx], pr), 'logloss': log_loss(d.y[idx], np.clip(pr, 1e-08, 1 - 1e-08), labels=[0, 1]), 'average_prediction': float(np.mean(pr)), 'known_positive_rate': float(np.mean(d.y[idx] == 1))})
        # LOOP (range(35, 50)): Repeat the same reproducible procedure for every selected fold, model or period.
        for t in range(35, 50):
            step = test_all[d.time[test_all] == t]
            # IF (not len(step)): Enforce the chronological scope here; validation and final diagnostics must not feed earlier fitting stages.
            if not len(step):
                continue
            ids = step[np.argsort(-pcal[step], kind='stable')[:min(len(step), int(args.top_k))]]
            labels = d.y[ids]
            # LOOP (enumerate(ids, 1)): Repeat the same reproducible procedure for every selected fold, model or period.
            for rank, idx in enumerate(ids, 1):
                topk_details.append({'model': name, 'time_step': t, 'rank': rank, 'tx_id': d.ids[idx], 'known_label': int(d.y[idx]), 'score': float(pcal[idx]), 'role': 'RETROSPECTIVE_BATCH'})
            npositive = int(np.sum((d.time == t) & (d.y == 1)))
            topk.append({'model': name, 'time_step': t, 'alert_universe_size': len(step), 'alerts': len(ids), 'known_illicit_in_alerts': int(np.sum(labels == 1)), 'known_licit_in_alerts': int(np.sum(labels == 0)), 'UNKNOWN_unverified_in_alerts': int(np.sum(labels == -1)), 'recall_of_known_illicit': float(np.sum(labels == 1) / npositive) if npositive else np.nan, 'note': 'retrospective BATCH top-k; labels selection-biased; unknown is NOT a false positive'})
        # LOOP (test): Repeat the same reproducible procedure for every selected fold, model or period.
        for idx in test:
            pred_rows.append({'model': name, 'tx_id': d.ids[idx], 'time_step': int(d.time[idx]), 'known_label': int(d.y[idx]), 'raw_score': float(pall[idx]), 'calibrated_score': float(pcal[idx])})
        # LOOP ((0, 1, -1)): Repeat the same reproducible procedure for every selected fold, model or period.
        for lab in (0, 1, -1):
            early = d.ix(35, 42, False)
            late = d.ix(43, 49, False)
            a = pcal[early[d.y[early] == lab]]
            b = pcal[late[d.y[late] == lab]]
            score_drift.append({'model': name, 'known_label': lab, 'early_n': len(a), 'late_n': len(b), 'early_median': float(np.median(a)) if len(a) else np.nan, 'late_median': float(np.median(b)) if len(b) else np.nan, 'KS': float(ks_2samp(a, b).statistic) if len(a) > 1 and len(b) > 1 else np.nan})
        # IMPORTANT: Write structured experiment metadata for independent audit.
        joblib.dump({'imputer': imp, 'scaler': sc, 'calibrator': pl, 'policies': policies, 'feature_indices': list(map(int, keep)), 'model_name': name}, out / f'{name}_pipeline.joblib')
        # IF (name in _NN_MODES): Select a specific model implementation without altering the common fit/predict contract.
        if name in _NN_MODES:
            # IMPORTANT: Snapshot the exact learned parameter tensors for reproducible reload.
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'feature_count': len(keep), 'model_type': name, 'seed': 42}, out / f'{name}_weights.pt')
        else:
            # IMPORTANT: Write structured experiment metadata for independent audit.
            joblib.dump(model, out / f'{name}_model.joblib')
        _ROUNDTRIP_FEATURES[name] = np.asarray(keep, dtype=int)
        reload_rows.append(_validate_model_roundtrip(name, model, d, x, (imp, sc), pall, out, cache, args))
        print(f'FINAL DIAGNOSTIC {name}: AP={ap(d.y[test], pcal[test]):.5f} runtime={time.time() - start:.1f}s', flush=True)
        del model
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame(reload_rows).to_csv(out / 'MODEL_RELOAD_VERIFICATION.csv', index=False)
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame(results).to_csv(out / 'FINAL_DIAGNOSTIC_ALL_MODELS.csv', index=False)
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame(by_step).to_csv(out / 'DIAGNOSTIC_BY_STEP_ALL_MODELS.csv', index=False)
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame(topk).to_csv(out / 'TOPK_ALL_TRANSACTIONS_BATCH.csv', index=False)
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame(topk_details).to_csv(out / 'TOPK_ALERT_DETAILS.csv', index=False)
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame(calibrator_rows).to_csv(out / 'CALIBRATOR_MONOTONICITY.csv', index=False)
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame(pred_rows).to_csv(out / 'LABELED_DIAGNOSTIC_PREDICTIONS.csv', index=False)
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame(cal_rows).to_csv(out / 'CALIBRATION_DIAGNOSTICS.csv', index=False)
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame(score_drift).to_csv(out / 'SCORE_DRIFT_BY_LABEL.csv', index=False)
    pivot = pd.DataFrame(by_step)
    pivot = pivot[pivot.policy == 'F1_policy'].pivot(index='time_step', columns='model', values='AP')
    pivot.plot(figsize=(11, 5), marker='o')
    plt.axvline(42.5, ls='--', color='gray', alpha=0.7)
    plt.title('EXPOSED posthoc test (35-49): per-step PR-AUC, known labels only')
    plt.ylabel('PR-AUC')
    plt.xlabel('time_step')
    plt.tight_layout()
    plt.savefig(out / 'diagnostic_ap_by_step.png', dpi=150)
    plt.close()
    return pd.DataFrame(results)


# ---------------------------------------------------------
# Ablations
# WHY: Select feature removals on historical folds; no diagnostic leakage into choice.
# ---------------------------------------------------------
# FUNCTION: ablations — Compare restricted feature sets on historical development folds.
# RESEARCH RULE: 35–49 are exposed; persisted scores and reload checks must remain independently auditable.
def ablations(d, args, out):
    # RESEARCH RATIONALE:
    # Feature exclusion is a sensitivity diagnostic; no hidden winner selection on later periods.
    rows = []
    selected = []
    # LOOP (enumerate(FOLDS, 1)): Repeat the same reproducible procedure for every selected fold, model or period.
    for fold, (a, b, c, e, f, g) in enumerate(FOLDS, 1):
        tr = d.ix(a, b)
        st = d.ix(c, e)
        ev = d.ix(f, g)
        raw, _, _ = preprocess(d, b, np.arange(93))
        # IMPORTANT: Keep controlled randomness reproducible across repeated experimental comparisons.
        probe = LGBMClassifier(n_estimators=80 if args.smoke else 150, learning_rate=0.05, max_depth=5, num_leaves=23, min_child_samples=25, random_state=42, n_jobs=2, verbosity=-1)
        probe.fit(raw[tr], d.y[tr])
        top10 = np.argsort(-probe.feature_importances_, kind='stable')[:10]
        selected.append({'fold': fold, 'fit_end': b, 'train_only_selected_top10': ','.join((f'local_{int(j) + 1:03}' for j in top10))})
        variants = {'all': np.arange(93), 'drop_053_055': np.array([i for i in range(93) if i not in (52, 54)]), 'drop_002': np.array([i for i in range(93) if i != 1]), 'drop_003': np.array([i for i in range(93) if i != 2]), 'drop_002_003': np.array([i for i in range(93) if i not in (1, 2)]), 'drop_train_only_gain_top10': np.array([i for i in range(93) if i not in top10])}
        # LOOP (variants.items()): Repeat the same reproducible procedure for every selected fold, model or period.
        for label, keep in variants.items():
            x, _, _ = preprocess(d, b, keep)
            m, meta = fit_tab('LGB', x, d, tr, st, args, 42)
            # IMPORTANT: Extract positive-class scores for ranking; thresholds are chosen separately.
            pp = m.predict_proba(x[ev])[:, 1]
            rows.append({'variant': label, 'fold': fold, 'eval_AP': ap(d.y[ev], pp), 'eval_prevalence': float(np.mean(d.y[ev] == 1)), 'features': len(keep), 'role': 'DEVELOPMENT_DIAGNOSTIC'})
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame(rows).to_csv(out / 'DEV_LGB_FEATURE_ABLATIONS.csv', index=False)
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame(selected).to_csv(out / 'DEV_TRAIN_ONLY_FEATURE_RANKING.csv', index=False)


# ---------------------------------------------------------
# Exposed Ablation Diagnostics
# WHY: Report post-hoc sensitivity analyses as exploratory only.
# ---------------------------------------------------------
# FUNCTION: exposed_ablation_diagnostics — Describe previously exposed feature-removal sensitivity; never treat it as blind validation.
# RESEARCH RULE: 35–49 are exposed; persisted scores and reload checks must remain independently auditable.
def exposed_ablation_diagnostics(d, args, out):
    """Exploratory diagnosis of exposed 35-49; NEVER use for model selection."""
    # RESEARCH RATIONALE:
    # Previously exposed feature ablations are post hoc and cannot establish new test accuracy.
    tr = d.ix(*FINAL['fit'])
    st = d.ix(*FINAL['stop'])
    cal = d.ix(*FINAL['calibration'])
    diagnostic = d.ix(*FINAL['diagnostic_test'])
    early = d.ix(35, 42)
    late = d.ix(43, 49)
    variants = {'all_93': np.arange(93), 'drop_local_053_055': np.array([j for j in range(93) if j not in (52, 54)]), 'drop_local_002': np.array([j for j in range(93) if j != 1]), 'drop_local_003': np.array([j for j in range(93) if j != 2]), 'drop_local_002_003': np.array([j for j in range(93) if j not in (1, 2)])}
    rows = []
    # LOOP (variants.items()): Repeat the same reproducible procedure for every selected fold, model or period.
    for label, keep in variants.items():
        x, _, _ = preprocess(d, FINAL['fit'][1], keep)
        model, meta = fit_tab('LGB', x, d, tr, st, args, 42)
        # IMPORTANT: Extract positive-class scores for ranking; thresholds are chosen separately.
        calibrated = platt_fit(d.y[cal], model.predict_proba(x[cal])[:, 1])
        # LOOP ([('early_35_42', early), ('late_43_49', late), ('all_35_49', diagnostic)]): Repeat the same reproducible procedure for every selected fold, model or period.
        for name, idx in [('early_35_42', early), ('late_43_49', late), ('all_35_49', diagnostic)]:
            # IMPORTANT: Extract positive-class scores for ranking; thresholds are chosen separately.
            scores = platt_apply(calibrated, model.predict_proba(x[idx])[:, 1])
            rows.append({'variant': label, 'period': name, 'n': len(idx), 'n_illicit': int(np.sum(d.y[idx] == 1)), 'AP': ap(d.y[idx], scores), 'prevalence': float(np.mean(d.y[idx] == 1)), 'ROC_AUC': float(roc_auc_score(d.y[idx], scores)) if len(np.unique(d.y[idx])) == 2 else np.nan, 'median_positive_score': float(np.median(scores[d.y[idx] == 1])) if np.any(d.y[idx] == 1) else np.nan, 'role': 'EXPOSED_POSTHOC_DIAGNOSIS_NOT_NEW_TEST', 'stop_iterations': meta['iterations']})
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame(rows).to_csv(out / 'EXPOSED_FEATURE_ABLATION_DIAGNOSIS.csv', index=False)
