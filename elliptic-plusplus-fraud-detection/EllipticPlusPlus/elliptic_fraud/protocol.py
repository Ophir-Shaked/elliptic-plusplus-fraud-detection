# MODULE: protocol.py
# PURPOSE: Dataset schema checks and graph noninterference
# RESEARCH RULE: Structural graph disconnection does not establish source feature availability.

"""Chronological splits, source-data QA and structural noninterference checks."""
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from .config import FOLDS, FINAL
from .utils import dump
from .graph import within_step_edges
from .models import SAGE


# ---------------------------------------------------------
# Qa
# WHY: Fail early on inconsistent official dataset counts or feature groups.
# ---------------------------------------------------------
# FUNCTION: qa — Reject malformed datasets and save label-coverage/schema checks.
# WHY: Protect fixed time windows, known-label masks and graph-noninterference contracts.
def qa(d, out):
    # RESEARCH RATIONALE:
    # For real Elliptic++, exact source counts are frozen as a schema-integrity test.
    # Coverage by time period reveals label-selection changes missed by global metrics.
    stats = {'n': d.n, 'p_local': d.p, 'label_illicit': int((d.y == 1).sum()), 'label_licit': int((d.y == 0).sum()), 'unknown': int((d.y < 0).sum()), 'edges': len(d.src), 'same_step_edges': int(np.sum(d.time[d.src] == d.time[d.dst])), 'strict_past_edges': int(np.sum(d.time[d.src] < d.time[d.dst])), 'reverse_step_edges': int(np.sum(d.time[d.src] > d.time[d.dst])), 'source': d.source, 'protocol_version': '3.1', 'additional_features_available': d.X_extra is not None}
    # IF (len(d.src) == 0): A transaction graph comparison requires observed source edges.
    if len(d.src) == 0:
        raise AssertionError('Graph has zero raw edges')
    # IF (stats['same_step_edges'] == 0): End-of-period graph inference requires within-period edges.
    if stats['same_step_edges'] == 0:
        raise AssertionError('No within-step graph: cannot run batch comparison')
    # IF (d.source['kind'].startswith('official')): Apply fixed official dataset counts only to genuine official data.
    if d.source['kind'].startswith('official'):
        expected = {'n': 203769, 'label_illicit': 4545, 'label_licit': 42019, 'unknown': 157205, 'edges': 234355}
        # LOOP (expected.items()): Repeat the same reproducible procedure for every selected fold, model or period.
        for field, value in expected.items():
            # IF (stats[field] != value): Abort when official dataset cardinalities differ from locked expectations.
            if stats[field] != value:
                raise AssertionError(f'Official data count mismatch in {field}: {stats[field]} != {value}')
    # IF (d.p != 93): Prevent silent changes to the 93-dimensional main baseline.
    if d.p != 93:
        raise AssertionError('Expected 93 local transaction features')
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame([{'time_step': int(t), 'all': int(np.sum(d.time == t)), 'licit': int(np.sum((d.time == t) & (d.y == 0))), 'illicit': int(np.sum((d.time == t) & (d.y == 1))), 'unknown': int(np.sum((d.time == t) & (d.y == -1))), 'labeled_fraction': float(np.mean(d.y[d.time == t] >= 0))} for t in range(1, 50)]).to_csv(out / 'LABEL_COVERAGE.csv', index=False)
    # IF (d.X_extra is not None and d.X_extra.shape != (d.n, 17)): Verify that the optional additional feature group has exactly 17 columns.
    if d.X_extra is not None and d.X_extra.shape != (d.n, 17):
        raise AssertionError('Incorrect supplemental 17-feature matrix')
    # IMPORTANT: Write structured experiment metadata for independent audit.
    dump(out / 'DATA_QA.json', stats)
    # IMPORTANT: Write structured experiment metadata for independent audit.
    dump(out / 'FEATURE_GROUP_SCHEMA.json', {'main_model_93': 'Columns 2:95 (0-based, stop exclusive)', 'excluded_72_aggregate': 'Columns 95:167; 72 graph-derived columns excluded', 'supplementary_17': 'Columns 167:184; evaluated on development windows only', 'excluded_time': 'Column 1; temporal grouping used only in chronological protocol', 'source_semantics': 'Source describes groups; construction timing of raw local features remains unverified', 'prediction_task': 'end-of-step BATCH for graph; no online deployment claim'})
    return stats


# ---------------------------------------------------------
# Leakage Contract
# WHY: Perturb future component features to test past-logit noninterference.
# ---------------------------------------------------------
# FUNCTION: leakage_contract — Perturb future node features and verify earlier graph logits stay unchanged.
# WHY: Protect fixed time windows, known-label masks and graph-noninterference contracts.
def leakage_contract(d, out):
    """Check noninterference and nonempty graph; structural tests, not timing provenance."""
    # RESEARCH RATIONALE:
    # Perturb all late features and hold model weights fixed; early logits must not change.
    # This test proves graph isolation only, not raw feature timestamp availability.
    # IMPORTANT: Use an explicit local random generator rather than hidden global stochastic state.
    rng = np.random.default_rng(2027)
    before, after = (np.flatnonzero(d.time <= 10), np.flatnonzero(d.time >= 35))
    # INVARIANT: Fail immediately if this research or data-integrity assumption is broken.
    assert len(before) > 0 and len(after) > 0
    s, v = within_step_edges(d)
    # INVARIANT: Fail immediately if this research or data-integrity assumption is broken.
    assert len(s) > 0 and np.all(d.time[s] == d.time[v])
    early_edges = np.flatnonzero((d.time[s] <= 10) & (d.time[v] <= 10))
    late_edges = np.flatnonzero((d.time[s] >= 35) & (d.time[v] >= 35))
    # IF (len(early_edges) == 0 or len(late_edges) == 0): The leakage perturbation test needs nonempty early and late graph components.
    if len(early_edges) == 0 or len(late_edges) == 0:
        raise AssertionError('Cannot test graph noninterference: missing early/late edges')
    src_e = s[early_edges[:min(200, len(early_edges))]]
    dst_e = v[early_edges[:min(200, len(early_edges))]]
    src_l = s[late_edges[:min(200, len(late_edges))]]
    dst_l = v[late_edges[:min(200, len(late_edges))]]
    idx = np.unique(np.r_[src_e, dst_e, src_l, dst_l])
    local = np.full(d.n, -1, dtype='int32')
    local[idx] = np.arange(len(idx))
    valid = (local[s] >= 0) & (local[v] >= 0)
    a = sparse.coo_matrix((np.ones(np.sum(valid), dtype='float32'), (local[v[valid]], local[s[valid]])), shape=(len(idx), len(idx))).tocsr()
    a.data[:] = 1
    deg = np.asarray(a.sum(axis=1)).ravel()
    a = (sparse.diags(1 / np.maximum(1, deg)) @ a).tocoo()
    adj = torch.sparse_coo_tensor(torch.tensor(np.vstack([a.row, a.col]), dtype=torch.long), torch.tensor(a.data.copy(), dtype=torch.float32), (len(idx), len(idx))).coalesce()
    values = np.nan_to_num(d.X[idx, :], nan=0, posinf=0, neginf=0)
    values = np.clip(values, -20, 20)
    x = torch.tensor(values, dtype=torch.float32)
    altered = x.clone()
    late_mask = torch.tensor(d.time[idx] >= 35, dtype=torch.bool)
    noise = rng.normal(0, 10, size=altered[late_mask].shape).astype('float32')
    altered[late_mask] += torch.from_numpy(noise)
    torch.manual_seed(2027)
    model = SAGE(d.p).eval()
    with torch.no_grad():
        orig = model(x, adj)
        other = model(altered, adj)
        delta = torch.max(torch.abs(orig[d.time[idx] <= 10] - other[d.time[idx] <= 10])).item()
    # INVARIANT: Fail immediately if this research or data-integrity assumption is broken.
    assert delta < 1e-06, f'Future input perturbed past predictions: {delta}'
    results = {'within_step_edges': len(s), 'checked_induced_edges': int(adj._nnz()), 'nonempty_batch_adjacency': bool(adj._nnz() > 0), 'future_perturbation_max_diff_early_logits': delta, 'past_future_graph_disconnected': True, 'only_batch_mode_claimed': True, 'limits': 'Cannot verify whether local source features were computed using future information.'}
    # IMPORTANT: Write structured experiment metadata for independent audit.
    dump(out / 'LEAKAGE_CONTRACT_TESTS.json', results)
    return results


# ---------------------------------------------------------
# Check Splits
# WHY: Reject overlapping or one-class chronological folds.
# ---------------------------------------------------------
# FUNCTION: check_splits — Enforce chronological nonoverlapping windows and two-class evaluation slices.
# WHY: Protect fixed time windows, known-label masks and graph-noninterference contracts.
def check_splits(d):
    # RESEARCH RATIONALE:
    # Every evaluation/stop slice needs both observed binary classes.
    # Final diagnostic time periods are deliberately disjoint from fitting, stop, calibration and policy.
    rows = []
    defs = []
    # LOOP (enumerate(FOLDS, 1)): Repeat the same reproducible procedure for every selected fold, model or period.
    for k, (a, b, c, e, f, g) in enumerate(FOLDS, 1):
        # INVARIANT: Fail immediately if this research or data-integrity assumption is broken.
        assert a <= b < c <= e < f <= g <= 34
        defs.extend([(f'DEV{k}', 'fit', a, b), (f'DEV{k}', 'stop', c, e), (f'DEV{k}', 'eval', f, g)])
    # INVARIANT: Fail immediately if this research or data-integrity assumption is broken.
    assert FINAL['fit'][1] < FINAL['stop'][0] and FINAL['stop'][1] < FINAL['calibration'][0]
    # INVARIANT: Fail immediately if this research or data-integrity assumption is broken.
    assert FINAL['calibration'][1] < FINAL['policy'][0] and FINAL['policy'][1] < FINAL['diagnostic_test'][0]
    defs.extend([('FINAL', label, *interval) for label, interval in FINAL.items()])
    # LOOP (defs): Repeat the same reproducible procedure for every selected fold, model or period.
    for fold, stage, lo, hi in defs:
        idx = d.ix(lo, hi)
        pos = int(np.sum(d.y[idx] == 1))
        neg = int(np.sum(d.y[idx] == 0))
        # IF (pos < 1 or neg < 1): Binary loss/calibration/AP require both observed classes per selected period.
        if pos < 1 or neg < 1:
            raise AssertionError(f'{fold}/{stage} has only one known class')
        rows.append({'fold': fold, 'stage': stage, 'start': lo, 'end': hi, 'labeled': len(idx), 'illicit': pos if stage != 'diagnostic_test' else np.nan, 'licit': neg if stage != 'diagnostic_test' else np.nan})
    return pd.DataFrame(rows)
