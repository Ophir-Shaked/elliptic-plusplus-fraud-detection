"""Fast deterministic tests of extra Recall/Precision reporting, without real-data downloads."""

import numpy as np
import pandas as pd
import pytest

from elliptic_fraud.evaluation import metrics
from elliptic_fraud.metric_reports import _binary_metrics, generate_metric_reports


def test_new_metric_counts_and_denominators():
    y = np.array([1, 1, 0, 0, 0, 1])
    scores = np.array([0.9, 0.2, 0.8, 0.1, 0.3, 0.6])
    d = _binary_metrics(y, scores, 0.5)
    assert (d['TP'], d['FP'], d['FN'], d['TN']) == (2, 1, 1, 2)
    assert d['recall'] == pytest.approx(2 / 3)
    assert d['precision'] == pytest.approx(2 / 3)
    assert d['F1'] == pytest.approx(2 / 3)
    assert d['F2'] == pytest.approx(2 / 3)
    assert d['specificity_TNR'] == pytest.approx(2 / 3)
    assert d['FP_per_10k_known_licit'] == pytest.approx(10000 / 3)


def test_zero_predictions_matches_original_policy_convention():
    y = np.array([0, 1])
    scores = np.array([0.2, 0.4])
    d = _binary_metrics(y, scores, 1.0)
    orig = metrics(y, scores, 1.0)
    assert d['TP'] == d['FP'] == 0
    assert d['recall'] == orig['recall'] == 0
    assert d['precision'] == orig['precision'] == 0
    assert d['F1'] == orig['F1'] == 0
    assert d['NPV'] == pytest.approx(0.5)  # All predictions are negative.


def test_unknown_labels_rejected_for_binary_metrics():
    with pytest.raises(ValueError, match='Unknown'):
        _binary_metrics([1, -1, 0], [0.8, 0.4, 0.2], 0.5)


def _write_fixture(folder):
    folder.mkdir()
    prediction_rows = []
    frozen_rows = []
    rank_rows = []
    for model, bump in [('XGB', 0.0), ('LOGIT', 0.01)]:
        y = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1] * 2)
        scores = np.array([.9,.2,.6,.8,.1,.7,.4,.3,.05,.85,.25,.54] * 2) + bump
        steps = np.array([35]*12 + [36]*12)
        for i, (label, score, step) in enumerate(zip(y,scores,steps)):
            prediction_rows.append({'model':model, 'tx_id':str(model)+'-'+str(i),
                                    'time_step': int(step), 'known_label': int(label),
                                    'raw_score':float(score), 'calibrated_score':float(score)})
        for policy, threshold in [('F1_policy',.5), ('FPR_5_per_10k',.95)]:
            frozen_rows.append({'model': model, 'policy': policy, 'evaluation_role': 'EXPOSED_POSTHOC_DIAGNOSTIC',
                                **metrics(y,scores,threshold)})
        # Rank 10 per time step and one unknown alert at the end of the list.
        for step in (35,36):
            positions=np.flatnonzero(steps==step)
            order=positions[np.argsort(-scores[positions])[:10]]
            for rank, i in enumerate(order, 1):
                rank_rows.append({'model':model,'time_step':step,'rank':rank,
                                  'tx_id':str(model)+'-'+str(i),'known_label':int(y[i]),
                                  'score':float(scores[i])})
    pd.DataFrame(prediction_rows).to_csv(folder/'LABELED_DIAGNOSTIC_PREDICTIONS.csv',index=False)
    pd.DataFrame(frozen_rows).to_csv(folder/'FINAL_DIAGNOSTIC_ALL_MODELS.csv',index=False)
    pd.DataFrame(rank_rows).to_csv(folder/'TOPK_ALERT_DETAILS.csv',index=False)


def test_full_export_matches_frozen_predictions(tmp_path):
    run_dir=tmp_path/'EXPOSED_DIAGNOSTIC_fixture'
    _write_fixture(run_dir)
    out=generate_metric_reports(run_dir)
    results=pd.read_csv(out/'CLASSIFICATION_REPORT_ALL_MODELS.csv')
    assert len(results)==4
    assert set(results['policy'])=={'F1_policy','FPR_5_per_10k'}
    assert (results['n_known']==24).all()
    assert (pd.read_csv(out/'INDEPENDENT_METRIC_RECOMPUTATION_CHECK.csv').result=='PASS').all()
    assert len(pd.read_csv(out/'CLASSIFICATION_BY_TIME_STEP.csv'))==8
    assert len(pd.read_csv(out/'RECALL_PRECISION_AT_K_BY_STEP.csv'))==4
    assert len(list((out/'plots').glob('CONFUSION_*.png')))==4
    assert len(list((out/'plots').glob('PR_ROC_*.png')))==2
    # All four supplementary chart types must be generated automatically.
    assert len(list((out/'plots').glob('TIME_SERIES_RECALL_PRECISION_*.png')))==2
    assert len(list((out/'plots').glob('THRESHOLD_SENSITIVITY_*_DIAGNOSTIC_ONLY.png')))==2
    assert len(list((out/'plots').glob('RELIABILITY_*.png')))==2
    assert (out/'plots'/'COMPARE_PRECISION_RECALL_AT_K.png').is_file()
    assert all(p.stat().st_size > 1_000 for p in (out/'plots').glob('*.png'))


def test_mismatched_stored_counts_fail_closed(tmp_path):
    run_dir=tmp_path/'EXPOSED_DIAGNOSTIC_tampered'
    _write_fixture(run_dir)
    file=run_dir/'FINAL_DIAGNOSTIC_ALL_MODELS.csv'
    rows=pd.read_csv(file)
    rows.loc[0,'TP'] += 1
    rows.to_csv(file,index=False)
    with pytest.raises(AssertionError, match='TP'):
        generate_metric_reports(run_dir)


def test_postprocess_is_repeatable_and_never_changes_saved_metrics(tmp_path):
    """Chart regeneration must leave the input metrics and historical decisions intact."""
    from elliptic_fraud.metric_reports import create_supplementary_figures
    import hashlib
    run_dir = tmp_path / 'EXPOSED_DIAGNOSTIC_repeatable'
    _write_fixture(run_dir)
    report = generate_metric_reports(run_dir)
    table = report / 'CLASSIFICATION_REPORT_ALL_MODELS.csv'
    before = hashlib.sha256(table.read_bytes()).hexdigest()
    output_paths = create_supplementary_figures(run_dir)
    assert len(output_paths) == 7  # Two plots per model in three categories, plus Top-K.
    assert all(path.is_file() and path.stat().st_size > 1000 for path in output_paths)
    assert hashlib.sha256(table.read_bytes()).hexdigest() == before
