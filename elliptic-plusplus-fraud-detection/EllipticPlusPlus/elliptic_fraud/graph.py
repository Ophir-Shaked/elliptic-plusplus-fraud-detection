"""Directed within-period graphs and degree-preserving structural controls.

Research contract
-----------------
* Transactions within the same step may exchange messages at *period close*.
* This does NOT imply streaming inference at individual transaction timestamps.
* The rewired null preserves each node's in-degree and out-degree exactly.
* Sparse matrices are row-normalized to compute mean-neighbor messages.
"""

import numpy as np
import torch
from scipy import sparse

from .utils import dump


# Keep one rewired graph per dataset instance and random seed within a run.
_REWIRE_CACHE = {}


# FUNCTION: within_step_edges — Remove cross-period links/self-loops; optionally rewire while exactly preserving directed node degrees.
# WHY: Directed graph context is within-period and assumes period-close batch availability.
def within_step_edges(d, rewire=False, seed=142):
    """Return a simple directed graph with no cross-period edges.

    A directed edge (source, target) denotes a message from source to target.
    The structural null swaps targets within the same period, while rejecting
    self-loops and duplicate edges.
    """
    # RESEARCH RATIONALE:
    # Keep directed links only inside the SAME period; no cross-period message path.
    # Degree-preserving target swaps are structural null controls, not realistic new payment flows.
    same_period = d.time[d.src] == d.time[d.dst]
    non_self_loop = d.src != d.dst
    mask = same_period & non_self_loop

    # Deduplicate before rewiring to match the simple-graph sparse semantics.
    pairs = np.unique(
        np.stack((d.src[mask], d.dst[mask]), axis=1), axis=0,
    )
    sources = pairs[:, 0].astype("int32", copy=True)
    targets = pairs[:, 1].astype("int32", copy=True)
    # IF (not rewire): Return the observed graph unchanged for the real-edge comparator.
    if not rewire:
        return sources, targets

    key = (id(d), seed)
    # IF (key in _REWIRE_CACHE): Reuse a deterministic structural control for repeated model fits.
    if key in _REWIRE_CACHE:
        return _REWIRE_CACHE[key]

    # IMPORTANT: Use an explicit local random generator rather than hidden global stochastic state.
    rng = np.random.default_rng(seed)
    original_sources = sources.copy()
    original_targets = targets.copy()

    # LOOP (np.unique(d.time)): Apply identical edge-validity and degree constraints to each graph item.
    for period in np.unique(d.time):
        positions = np.flatnonzero(d.time[sources] == period)
        m = len(positions)
        # IF (m < 3): There are too few edges to perform useful double-edge swaps.
        if m < 3:
            continue

        edge_set = {
            (int(sources[j]), int(targets[j])) for j in positions
        }
        # A candidate swap keeps all source nodes fixed:
        # (u1 -> v1, u2 -> v2) -> (u1 -> v2, u2 -> v1).
        # LOOP (range(3 * m)): Apply identical edge-validity and degree constraints to each graph item.
        for _ in range(3 * m):
            a, b = positions[rng.integers(0, m, size=2)]
            # IF (a == b): Do not attempt to exchange one directed edge with itself.
            if a == b:
                continue
            u1, u2 = int(sources[a]), int(sources[b])
            v1, v2 = int(targets[a]), int(targets[b])
            # IF (u1 == u2 or v1 == v2 or u1 == v2 or u2 == v1): Reject duplicate endpoints and newly introduced self-loops.
            if u1 == u2 or v1 == v2 or u1 == v2 or u2 == v1:
                continue
            # IF ((u1, v2) in edge_set or (u2, v1) in edge_set): Reject a swap if it would create an already-existing directed edge.
            if (u1, v2) in edge_set or (u2, v1) in edge_set:
                continue

            edge_set.remove((u1, v1))
            edge_set.remove((u2, v2))
            edge_set.add((u1, v2))
            edge_set.add((u2, v1))
            targets[a], targets[b] = v2, v1

    # These are *exact* degree invariants, not distributional approximations.
    # INVARIANT: Fail immediately if this research or data-integrity assumption is broken.
    assert np.array_equal(original_sources, sources)
    # INVARIANT: Fail immediately if this research or data-integrity assumption is broken.
    assert np.array_equal(
        np.bincount(original_targets, minlength=d.n),
        np.bincount(targets, minlength=d.n),
    )
    # INVARIANT: Fail immediately if this research or data-integrity assumption is broken.
    assert np.all(d.time[sources] == d.time[targets])
    # INVARIANT: Fail immediately if this research or data-integrity assumption is broken.
    assert len(set(zip(sources.tolist(), targets.tolist()))) == len(sources)
    # INVARIANT: Fail immediately if this research or data-integrity assumption is broken.
    assert np.all(sources != targets)

    _REWIRE_CACHE[key] = (sources, targets)
    return sources, targets


# FUNCTION: graph_rewire_audit — Persist a structural-control audit that a reviewer can inspect.
# WHY: Directed graph context is within-period and assumes period-close batch availability.
def graph_rewire_audit(d, out):
    """Write an explicit verification report for the rewired null graph."""
    # RESEARCH RATIONALE:
    # Check both incoming and outgoing degrees for every node, not just total edge count.
    # Save the audit so graph control assumptions can be inspected independently.
    original_sources, original_targets = within_step_edges(d)
    rewired_sources, rewired_targets = within_step_edges(d, rewire=True)

    # INVARIANT: Fail immediately if this research or data-integrity assumption is broken.
    assert len(original_sources) == len(rewired_sources)
    # INVARIANT: Fail immediately if this research or data-integrity assumption is broken.
    assert np.array_equal(
        np.bincount(original_sources, minlength=d.n),
        np.bincount(rewired_sources, minlength=d.n),
    )
    # INVARIANT: Fail immediately if this research or data-integrity assumption is broken.
    assert np.array_equal(
        np.bincount(original_targets, minlength=d.n),
        np.bincount(rewired_targets, minlength=d.n),
    )

    report = {
        "edge_count": len(original_sources),
        "destination_fraction_changed": float(
            np.mean(original_targets != rewired_targets)
        ),
        "exact_source_outdegree_preserved": True,
        "exact_target_indegree_preserved": True,
        "no_duplicate_or_self_loop_edges_after_swap": True,
        "control": "within-step directed double-edge swaps; no class labels used",
    }
    # IMPORTANT: Write structured experiment metadata for independent audit.
    dump(out / "GRAPH_REWIRE_DEGREE_AUDIT.json", report)
    return report


# FUNCTION: _adjacency — Compute directed row-normalized sparse adjacency A[target,source] for incoming mean-neighbor messages.
# WHY: Directed graph context is within-period and assumes period-close batch availability.
def _adjacency(d, device, sources, targets):
    """Create row-stochastic A[target, source] for incoming-message means."""
    # RESEARCH RATIONALE:
    # Rows receive messages and columns send them: A[target, source] = 1.
    # Row normalization implements a mean rather than a sum over neighbors.
    # Zero-degree nodes get a zero neighbor message, not NaN.
    # SciPy handles sparse normalization; PyTorch performs sparse matrix
    # multiplication on the same CPU/GPU device as the neural model.
    matrix = sparse.coo_matrix(
        (np.ones(len(sources), dtype="float32"), (targets, sources)),
        shape=(d.n, d.n),
    ).tocsr()
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    matrix.data[:] = 1

    degree = np.asarray(matrix.sum(axis=1)).ravel()
    # An isolated node has degree zero and receives a zero message vector.
    normalized = (
        sparse.diags(1 / np.maximum(1, degree)) @ matrix
    ).tocoo()
    coordinates = np.vstack((normalized.row, normalized.col)).astype("int64")

    return torch.sparse_coo_tensor(
        torch.tensor(coordinates, device=device),
        torch.tensor(normalized.data.astype("float32"), device=device),
        (d.n, d.n),
    ).coalesce()


# FUNCTION: to_sparse_torch — Choose incoming-only, bidirectional, degree-rewired, or no-graph message context.
# WHY: Directed graph context is within-period and assumes period-close batch availability.
def to_sparse_torch(d, device, mode):
    """Choose real, rewired, empty or two-direction sparse adjacency."""
    # RESEARCH RATIONALE:
    # Use the SAME architecture with no_graph to measure the added value of edges.
    # Bidirectional aggregation treats incoming and outgoing links separately.
    # IF (mode == "no_graph"): Provide zero neighbor messages to the architecture-matched no-graph control.
    if mode == "no_graph":
        return None

    rewired = mode in ("rewired", "bidir_rewired")
    sources, targets = within_step_edges(d, rewire=rewired)
    incoming = _adjacency(d, device, sources, targets)

    # Direction matters: the second sparse matrix aggregates outgoing neighbors.
    # IF (mode in ("bidir", "bidir_rewired")): Return separate incoming and outgoing sparse adjacency matrices.
    if mode in ("bidir", "bidir_rewired"):
        outgoing = _adjacency(d, device, targets, sources)
        return incoming, outgoing
    return incoming


# FUNCTION: agg — Multiply a sparse adjacency by node representations; use zero messages for no-graph ablations.
# WHY: Directed graph context is within-period and assumes period-close batch availability.
def agg(adjacency, hidden):
    """Mean-neighbor message, or an all-zero message for no-graph baselines."""
    # RESEARCH RATIONALE:
    # Sparse multiplication scales with observed edges rather than a dense N-by-N matrix.
    # The no-graph baseline feeds zero neighbor messages without shrinking parameter count.
    # IF (adjacency is None): Isolate the effect of connectivity by supplying zero neighbor messages.
    if adjacency is None:
        return torch.zeros_like(hidden)
    # IMPORTANT: Aggregate neighbor messages with sparse multiplication to avoid a dense N-by-N graph.
    return torch.sparse.mm(adjacency, hidden)
