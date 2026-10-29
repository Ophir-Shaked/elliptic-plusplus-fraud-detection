# Adversarial regression tests for leakage and artifacts
# ---------------------------------------------------------------
# PURPOSE: Intentionally perturb the future inputs and labels to verify historical noninterference.
# RESEARCH RULE: The tests should fail if an artifact or provenance constraint is violated.
# ---------------------------------------------------------------

"""Regression tests for the modular refactor and its independent audit.

Run: python -m pytest -q
No external network, Git LFS or private financial data is needed.
All generated transactions in these tests are strictly synthetic fixtures.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import average_precision_score

from elliptic_fraud.config import FINAL
from elliptic_fraud.data import make_synthetic, load_real
from elliptic_fraud.evaluation import metrics, platt_fit, platt_apply, threshold_f1, threshold_fpr
from elliptic_fraud.graph import to_sparse_torch, within_step_edges
from elliptic_fraud.models import SAGE, BidirSAGE, fit_predict
from elliptic_fraud.preprocessing import preprocess
from elliptic_fraud.final_evaluation import final_all
from elliptic_fraud.verification import RunReader, audit_run, compare_runs
from elliptic_fraud.utils import sha


# ---- Encapsulate args behavior ----
# WHY: See its detailed docstring and expected inputs/outputs.
# FUNCTION: args — Minimal deterministic fixture flags used by model training.
# WHY: Keep input validation, business logic, and output handling explicit for reproducibility.
def args(**kwargs):
    """Minimal deterministic fixture flags used by model training."""
    cfg = dict(smoke=True, quick=True, gpu=False, top_k=10, no_download=True,
               data_dir='/__this_dataset_does_not_exist__')
    cfg.update(kwargs)
    return SimpleNamespace(**cfg)


@pytest.mark.parametrize('name,mode', [('SAGE','batch'),('SAGE','rewired'),
                                        ('BidirSAGE','bidir'),('BidirSAGE','bidir_rewired')])
# ---- Regression test: future features do not change past graph logits ----
# WHY: A failing assertion stops verification; do not downgrade failures to warnings.
# TEST CONTRACT: test_future_features_do_not_change_past_graph_logits — Direct noninterference test of BOTH graph architectures and rewiring modes.
# WHY: This invariant must fail loudly when implementation behavior changes.
def test_future_features_do_not_change_past_graph_logits(name,mode):
    """Direct noninterference test of BOTH graph architectures and rewiring modes.

    The forward pass sees all node features, but each time step's graph component
    must be disconnected from future time steps. Model weights are fixed here.
    """
    torch.set_num_threads(1)
    d = make_synthetic(seed=111)
    x,_,_ = preprocess(d,20,np.arange(93))
    x2 = x.copy()
    x2[d.time>=35] += 25.0
    # IMPORTANT: Build sparse graph context consistent with the chosen model ablation.
    adj = to_sparse_torch(d,torch.device('cpu'),mode)
    torch.manual_seed(10)
    net = BidirSAGE(93).eval() if name=='BidirSAGE' else SAGE(93).eval()
    with torch.no_grad():
        p = net(torch.from_numpy(x),adj).numpy()
        q = net(torch.from_numpy(x2),adj).numpy()
    early = d.time<=10
    # BLAS may select a different GEMM kernel when later inputs change.
    # Differences below float32 rounding tolerance are not information flow.
    assert np.max(np.abs(p[early]-q[early])) < 1e-6
    assert np.max(np.abs(p[d.time>=35]-q[d.time>=35]))>1e-6


# ---- Regression test: unknown and future labels have no effect on historical training ----
# WHY: A failing assertion stops verification; do not downgrade failures to warnings.
# TEST CONTRACT: test_unknown_and_future_labels_have_no_effect_on_historical_training — Train a GNN twice after changing ONLY labels after period 34.
# WHY: This invariant must fail loudly when implementation behavior changes.
def test_unknown_and_future_labels_have_no_effect_on_historical_training():
    """Train a GNN twice after changing ONLY labels after period 34.

    Future labels must never influence fitted parameters or stop-period scores.
    This is a dynamic training test in addition to source-code protocol checks.
    """
    torch.set_num_threads(1)
    d = make_synthetic(seed=333)
    train = d.ix(1,10)
    stop = d.ix(11,12)
    x,_,_ = preprocess(d,10,np.arange(93))
    # IMPORTANT: Use the shared interface so tabular and neural models follow the same time protocol.
    _,p,meta = fit_predict('BIDIR_SAGE',d,x,train,stop,stop,args(),42,{})
    y = d.y.copy()
    y[d.time>=35] = np.where(y[d.time>=35]==1,0,1)
    other = replace(d,y=y)
    # IMPORTANT: Use the shared interface so tabular and neural models follow the same time protocol.
    _,q,other_meta = fit_predict('BIDIR_SAGE',other,x,train,stop,stop,args(),42,{})
    np.testing.assert_allclose(p,q,rtol=0,atol=0)
    assert meta['iterations']==other_meta['iterations']


# ---- Regression test: policy calibration and model fit ignore future diagnostic labels ----
# WHY: A failing assertion stops verification; do not downgrade failures to warnings.
# TEST CONTRACT: test_policy_calibration_and_model_fit_ignore_future_diagnostic_labels — Poison labels in 35-49, then refit history and recompute policies.
# WHY: This invariant must fail loudly when implementation behavior changes.
def test_policy_calibration_and_model_fit_ignore_future_diagnostic_labels():
    """Poison labels in 35-49, then refit history and recompute policies.

    This checks stage independence at runtime, not just calendar constants.
    """
    d=make_synthetic(seed=345)
    poisoned = d.y.copy()
    poisoned[d.time>=35] = 1 - np.maximum(0,poisoned[d.time>=35])
    d2=replace(d,y=poisoned)
    fit=d.ix(*FINAL['fit']); stop=d.ix(*FINAL['stop'])
    cal=d.ix(*FINAL['calibration']); policy=d.ix(*FINAL['policy'])
    x,_,_=preprocess(d,FINAL['fit'][1],np.arange(93))
    # LOOP (('XGB','BIDIR_SAGE')): Process every item using the same fixed input/output contract.
    for name in ('XGB','BIDIR_SAGE'):
        # IMPORTANT: Use the shared interface so tabular and neural models follow the same time protocol.
        _,p,_=fit_predict(name,d,x,fit,stop,np.arange(d.n),args(),42,{})
        # IMPORTANT: Use the shared interface so tabular and neural models follow the same time protocol.
        _,q,_=fit_predict(name,d2,x,fit,stop,np.arange(d2.n),args(),42,{})
        np.testing.assert_allclose(p,q,rtol=0,atol=1e-7)
        # IMPORTANT: Fit calibration on the calibration slice only, never on exposed diagnostics.
        ca=platt_fit(d.y[cal],p[cal]); cb=platt_fit(d2.y[cal],q[cal])
        np.testing.assert_allclose([ca.intercept,ca.slope],[cb.intercept,cb.slope],rtol=0,atol=1e-7)
        xpol=platt_apply(ca,p[policy]);ypol=platt_apply(cb,q[policy])
        assert threshold_f1(d.y[policy],xpol)==threshold_f1(d2.y[policy],ypol)
        assert threshold_fpr(d.y[policy],xpol)==threshold_fpr(d2.y[policy],ypol)


# ---- Regression test: known label metrics require known classes ----
# WHY: A failing assertion stops verification; do not downgrade failures to warnings.
# TEST CONTRACT: test_known_label_metrics_require_known_classes — Check standard AP and threshold accounting on a tiny transparent fixture.
# WHY: This invariant must fail loudly when implementation behavior changes.
def test_known_label_metrics_require_known_classes():
    """Check standard AP and threshold accounting on a tiny transparent fixture."""
    y=np.array([1,0,1,0,0,1,0])
    s=np.array([.9,.5,.8,.4,.1,.7,.2])
    m=metrics(y,s,threshold=.6)
    assert m['TP']==3 and m['FP']==0 and m['FN']==0 and m['TN']==4
    assert m['AP']==pytest.approx(average_precision_score(y,s))
    assert m['FP_per_10k_licit']==0


# ---- Regression test: real dataset is not silently replaced with synthetic ----
# WHY: A failing assertion stops verification; do not downgrade failures to warnings.
# TEST CONTRACT: test_real_dataset_is_not_silently_replaced_with_synthetic — --no-download with no real CSVs must fail, never fall back to smoke.
# WHY: This invariant must fail loudly when implementation behavior changes.
def test_real_dataset_is_not_silently_replaced_with_synthetic(tmp_path):
    """--no-download with no real CSVs must fail, never fall back to smoke."""
    with pytest.raises(FileNotFoundError):
        load_real(args(data_dir=str(tmp_path)))


# ---- Regression test: synthetic integration saves and reloads four model types ----
# WHY: A failing assertion stops verification; do not downgrade failures to warnings.
# TEST CONTRACT: test_synthetic_integration_saves_and_reloads_four_model_types — End-to-end final stage covers a tree, MLP, SAGE and bidirectional SAGE.
# WHY: This invariant must fail loudly when implementation behavior changes.
def test_synthetic_integration_saves_and_reloads_four_model_types(tmp_path):
    """End-to-end final stage covers a tree, MLP, SAGE and bidirectional SAGE.

    The exhaustive 11-model test is the Colab/full-smoke integration gate.
    """
    torch.set_num_threads(1)
    d=make_synthetic(seed=777)
    names=['XGB','MLP','SAGE_BATCH','BIDIR_SAGE']
    df=final_all(d,args(),tmp_path,np.arange(93),names)
    assert set(df.model)==set(names)
    reload=pd.read_csv(tmp_path/'MODEL_RELOAD_VERIFICATION.csv')
    assert set(reload.model)==set(names) and reload.status.eq('PASS').all()
    assert reload.raw_prediction_max_abs_diff.max()<1e-5
    # LOOP (names): Process every item using the same fixed input/output contract.
    for name in names:
        assert (tmp_path/f'{name}_pipeline.joblib').exists()
        suffix='model.joblib' if name=='XGB' else 'weights.pt'
        assert (tmp_path/f'{name}_{suffix}').exists()


# ---- Regression test: verifier rejects incomplete artifact ----
# WHY: A failing assertion stops verification; do not downgrade failures to warnings.
# TEST CONTRACT: test_verifier_rejects_incomplete_artifact — An empty results folder cannot be declared a completed experiment.
# WHY: This invariant must fail loudly when implementation behavior changes.
def test_verifier_rejects_incomplete_artifact(tmp_path):
    """An empty results folder cannot be declared a completed experiment."""
    with pytest.raises(AssertionError,match='Missing required'):
        audit_run(tmp_path)


# ---- Regression test: comparator rejects synthetic vs real ----
# WHY: A failing assertion stops verification; do not downgrade failures to warnings.
# TEST CONTRACT: test_comparator_rejects_synthetic_vs_real — Prevent misleading equivalence checks across different datasets.
# WHY: This invariant must fail loudly when implementation behavior changes.
def test_comparator_rejects_synthetic_vs_real(tmp_path):
    """Prevent misleading equivalence checks across different datasets."""
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    (a/'RUN_SUCCESS.json').write_text('{"dataset_kind":"SYNTHETIC SMOKE ONLY"}')
    (b/'RUN_SUCCESS.json').write_text('{"dataset_kind":"official Elliptic++ transaction data"}')
    with pytest.raises(ValueError,match='Cannot compare synthetic'):
        compare_runs(a,b)


REFERENCE_REAL = (Path(__file__).resolve().parents[1] / "reference" / "v3_1_real")
REFERENCE_REAL_AVAILABLE = (REFERENCE_REAL / "DATA_QA.json").is_file()



# ---- Regression test: locked reference compares to itself ----
# WHY: A failing assertion stops verification; do not downgrade failures to warnings.
# TEST CONTRACT: test_locked_reference_compares_to_itself — The published historical baseline must be a usable regression fixture.
# WHY: This invariant must fail loudly when implementation behavior changes.
@pytest.mark.skipif(
    not REFERENCE_REAL_AVAILABLE,
    reason="Code-only ZIP intentionally excludes frozen historical reference data",
)
def test_locked_reference_compares_to_itself():
    """The published historical baseline must be a usable regression fixture."""
    base=Path(__file__).resolve().parents[1]/'reference'/'v3_1_real'
    result=compare_runs(base,base,metric_atol=0,threshold_atol=0)
    assert result['passed'] and result['n_failed']==0
    assert result['n_checks']>=60


# ---- Regression test: refactor comparison rejects changed source hash ----
# WHY: A failing assertion stops verification; do not downgrade failures to warnings.
# TEST CONTRACT: test_refactor_comparison_rejects_changed_source_hash — A different data release is not a valid numerical refactor comparison.
# WHY: This invariant must fail loudly when implementation behavior changes.
@pytest.mark.skipif(
    not REFERENCE_REAL_AVAILABLE,
    reason="Code-only ZIP intentionally excludes frozen historical reference data",
)
def test_refactor_comparison_rejects_changed_source_hash(tmp_path):
    """A different data release is not a valid numerical refactor comparison."""
    import json, shutil
    base=Path(__file__).resolve().parents[1]/'reference'/'v3_1_real'
    target=tmp_path/'changed'; shutil.copytree(base,target)
    data=json.loads((target/'DATA_QA.json').read_text())
    data['source']['sha256']['txs_features.csv']='0'*64
    (target/'DATA_QA.json').write_text(json.dumps(data))
    with pytest.raises(ValueError,match='SHA-256'):
        compare_runs(target,base)


# ---- Regression test: refactor comparison flags changed metric ----
# WHY: A failing assertion stops verification; do not downgrade failures to warnings.
# TEST CONTRACT: test_refactor_comparison_flags_changed_metric — The comparator must FAIL, not silently tolerate arbitrary score drift.
# WHY: This invariant must fail loudly when implementation behavior changes.
@pytest.mark.skipif(
    not REFERENCE_REAL_AVAILABLE,
    reason="Code-only ZIP intentionally excludes frozen historical reference data",
)
def test_refactor_comparison_flags_changed_metric(tmp_path):
    """The comparator must FAIL, not silently tolerate arbitrary score drift."""
    import shutil
    base=Path(__file__).resolve().parents[1]/'reference'/'v3_1_real'
    target=tmp_path/'changed'; shutil.copytree(base,target)
    m=pd.read_csv(target/'FINAL_DIAGNOSTIC_ALL_MODELS.csv')
    m.loc[m.model=='XGB','AP']=0.01
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    m.to_csv(target/'FINAL_DIAGNOSTIC_ALL_MODELS.csv',index=False)
    result=compare_runs(target,base)
    assert not result['passed']
    assert any((x['model']=='XGB' and x['metric']=='AP' and not x['passed'])
               for x in result['comparison'])
