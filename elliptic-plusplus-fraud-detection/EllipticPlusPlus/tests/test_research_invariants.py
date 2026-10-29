# Unit tests for chronological research invariants
# ---------------------------------------------------------------
# PURPOSE: Tests cover source syntax, future-data noninterference and degree-preserving rewiring.
# RESEARCH RULE: These tests catch software mistakes; they are not a scientific evaluation dataset.
# ---------------------------------------------------------------

"""Scientific invariants that must remain true across refactors.

Run from repository root: python -m pytest -q
"""
from dataclasses import replace
from pathlib import Path
import ast
import numpy as np
import pandas as pd

from elliptic_fraud.config import FOLDS, FINAL, MODELS
from elliptic_fraud.data import make_synthetic
from elliptic_fraud.preprocessing import preprocess
from elliptic_fraud.graph import within_step_edges
from elliptic_fraud.controls import _shuffle_by_period, _cohort_signature
from elliptic_fraud.visualization import plot_dev_model_comparison


# ---- Regression test: dev and exposed periods do not overlap ----
# WHY: A failing assertion stops verification; do not downgrade failures to warnings.
# TEST CONTRACT: test_dev_and_exposed_periods_do_not_overlap — Development eval periods precede final fit/stop/calibration/policy/test.
# WHY: This invariant must fail loudly when implementation behavior changes.
def test_dev_and_exposed_periods_do_not_overlap():
    """Development eval periods precede final fit/stop/calibration/policy/test."""
    assert all(a <= b < c <= e < f <= g <= 25 for a,b,c,e,f,g in FOLDS)
    assert FINAL['fit'][1] < FINAL['stop'][0]
    assert FINAL['stop'][1] < FINAL['calibration'][0]
    assert FINAL['calibration'][1] < FINAL['policy'][0]
    assert FINAL['policy'][1] < FINAL['diagnostic_test'][0]


# ---- Regression test: known classes only in supervised splits ----
# WHY: A failing assertion stops verification; do not downgrade failures to warnings.
# TEST CONTRACT: test_known_classes_only_in_supervised_splits — Implement test_known_classes_only_in_supervised_splits in the research workflow.
# WHY: This invariant must fail loudly when implementation behavior changes.
def test_known_classes_only_in_supervised_splits():
    d = make_synthetic()
    # LOOP (FOLDS): Process every item using the same fixed input/output contract.
    for a,b,c,e,f,g in FOLDS:
        assert np.all(d.y[d.ix(a,b)] >= 0)
        assert np.all(d.y[d.ix(f,g)] >= 0)
    assert np.sum(d.y == -1) > 0


# ---- Regression test: future feature mutation does not change training preprocessing ----
# WHY: A failing assertion stops verification; do not downgrade failures to warnings.
# TEST CONTRACT: test_future_feature_mutation_does_not_change_training_preprocessing — Implement test_future_feature_mutation_does_not_change_training_preprocessing in the research workflow.
# WHY: This invariant must fail loudly when implementation behavior changes.
def test_future_feature_mutation_does_not_change_training_preprocessing():
    d=make_synthetic()
    x,imputer,scaler=preprocess(d,20,np.arange(d.p))
    other=d.X.copy()
    other[d.time>=21]+=1000
    x2,imp2,sc2=preprocess(replace(d,X=other),20,np.arange(d.p))
    assert np.array_equal(x[d.time<=20],x2[d.time<=20])
    assert np.array_equal(imputer.statistics_,imp2.statistics_)
    assert np.array_equal(scaler.mean_,sc2.mean_)


# ---- Regression test: period shuffle preserves each period labels ----
# WHY: A failing assertion stops verification; do not downgrade failures to warnings.
# TEST CONTRACT: test_period_shuffle_preserves_each_period_labels — Implement test_period_shuffle_preserves_each_period_labels in the research workflow.
# WHY: This invariant must fail loudly when implementation behavior changes.
def test_period_shuffle_preserves_each_period_labels():
    rng=np.random.default_rng(17)
    ts=np.repeat(np.arange(1,6),30)
    x=(np.arange(len(ts))%4==0).astype(int)
    y=_shuffle_by_period(x,ts,rng)
    assert not np.array_equal(x,y)
    # LOOP (np.unique(ts)): Process every item using the same fixed input/output contract.
    for period in np.unique(ts):
        mask=ts==period
        assert np.array_equal(np.sort(x[mask]),np.sort(y[mask]))


# ---- Regression test: directed rewire exact degrees and periods ----
# WHY: A failing assertion stops verification; do not downgrade failures to warnings.
# TEST CONTRACT: test_directed_rewire_exact_degrees_and_periods — Implement test_directed_rewire_exact_degrees_and_periods in the research workflow.
# WHY: This invariant must fail loudly when implementation behavior changes.
def test_directed_rewire_exact_degrees_and_periods():
    d=make_synthetic()
    s,v=within_step_edges(d)
    sr,vr=within_step_edges(d,rewire=True)
    assert np.array_equal(s,sr)
    assert np.array_equal(np.bincount(v,minlength=d.n),np.bincount(vr,minlength=d.n))
    assert np.array_equal(np.bincount(s,minlength=d.n),np.bincount(sr,minlength=d.n))
    assert np.all(d.time[sr]==d.time[vr])
    assert np.all(sr!=vr)
    assert len(set(zip(sr.tolist(),vr.tolist()))) == len(sr)


# ---- Regression test: cohort signature not affected by later periods ----
# WHY: A failing assertion stops verification; do not downgrade failures to warnings.
# TEST CONTRACT: test_cohort_signature_not_affected_by_later_periods — Implement test_cohort_signature_not_affected_by_later_periods in the research workflow.
# WHY: This invariant must fail loudly when implementation behavior changes.
def test_cohort_signature_not_affected_by_later_periods():
    d=make_synthetic()
    original=_cohort_signature(d)
    x=d.X.copy();x[d.time>=40]+=50
    updated=_cohort_signature(replace(d,X=x))
    assert np.array_equal(original[:40],updated[:40])


# ---- Regression test: source modules parse and all models defined ----
# WHY: A failing assertion stops verification; do not downgrade failures to warnings.
# TEST CONTRACT: test_source_modules_parse_and_all_models_defined — Implement test_source_modules_parse_and_all_models_defined in the research workflow.
# WHY: This invariant must fail loudly when implementation behavior changes.
def test_source_modules_parse_and_all_models_defined():
    root=Path(__file__).resolve().parents[1]
    files=list((root/'elliptic_fraud').glob('*.py'))
    assert len(files)>=12
    # LOOP (files): Process every item using the same fixed input/output contract.
    for file in files:
        ast.parse(file.read_text(encoding='utf-8'))
    assert len(MODELS)==11


# ---- Regression test: plotting is independent of training ----
# WHY: A failing assertion stops verification; do not downgrade failures to warnings.
# TEST CONTRACT: test_plotting_is_independent_of_training — Implement test_plotting_is_independent_of_training in the research workflow.
# WHY: This invariant must fail loudly when implementation behavior changes.
def test_plotting_is_independent_of_training(tmp_path):
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame({'model':['XGB','BIDIR_SAGE'],'mean_AP':[.8,.7],
                  'min_AP':[.7,.6],'sd_AP':[.1,.1],'folds':[3,3]}).to_csv(tmp_path/'DEV_SUMMARY.csv',index=False)
    output=tmp_path/'overview.png'
    plot_dev_model_comparison(tmp_path,output)
    assert output.stat().st_size>1000


# ---- Regression test: configured models are unique ----
# WHY: A failing assertion stops verification; do not downgrade failures to warnings.
# TEST CONTRACT: test_configured_models_are_unique — Avoid accidentally scoring or saving the same model twice.
# WHY: This invariant must fail loudly when implementation behavior changes.
def test_configured_models_are_unique():
    """Avoid accidentally scoring or saving the same model twice."""
    assert len(MODELS) == len(set(MODELS))

# ---- Regression test: exposed never overlaps training ----
# WHY: A failing assertion stops verification; do not downgrade failures to warnings.
# TEST CONTRACT: test_exposed_test_never_overlaps_training — The historical fit/stop/calibrate/policy ranges must finish first.
# WHY: This invariant must fail loudly when implementation behavior changes.
def test_exposed_test_never_overlaps_training():
    """The historical fit/stop/calibrate/policy ranges must finish first."""
    assert FINAL['fit'][1] < FINAL['stop'][0]
    assert FINAL['stop'][1] < FINAL['calibration'][0]
    assert FINAL['calibration'][1] < FINAL['policy'][0]
    assert FINAL['policy'][1] < FINAL['diagnostic_test'][0]

# ---- Regression test: folds are strictly chronological ----
# WHY: A failing assertion stops verification; do not downgrade failures to warnings.
# TEST CONTRACT: test_folds_are_strictly_chronological — Implement test_folds_are_strictly_chronological in the research workflow.
# WHY: This invariant must fail loudly when implementation behavior changes.
def test_folds_are_strictly_chronological():
    # LOOP (FOLDS): Process every item using the same fixed input/output contract.
    for a,b,c,d,e,f in FOLDS:
        assert a <= b < c <= d < e <= f
        assert f < FINAL['diagnostic_test'][0]
