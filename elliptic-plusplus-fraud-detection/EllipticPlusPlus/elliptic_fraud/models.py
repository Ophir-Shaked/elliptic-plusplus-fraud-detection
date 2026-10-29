"""Tabular baselines, MLP, GraphSAGE, and shared training interfaces.

Research contract
-----------------
* Known labeled historical examples contribute to supervised loss.
* Unlabeled nodes may supply graph context in retrospective batch mode.
* Tree budgets and neural early stopping use *historical stop windows* only.
* Matched no-graph comparisons retain the same neural architecture.
"""

import random

import numpy as np
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn
from xgboost import XGBClassifier

from .evaluation import ap
from .graph import agg, to_sparse_torch


# -----------------------------------------------------------------------------
# Neural network architectures
# -----------------------------------------------------------------------------
# CLASS: SAGE — Two-layer incoming-edge GraphSAGE with mean-neighbor aggregation.
class SAGE(nn.Module):
    """Two-layer incoming-edge GraphSAGE with mean-neighbor aggregation."""

    # FUNCTION: SAGE.__init__ — Define network layers with dimensions matching the architecture’s node/neighbor concatenation.
    # WHY: Use architecture-matched graph controls and train loss only on known historical labels.
    def __init__(self, p, hidden=48, dropout=.2):
        # RESEARCH RATIONALE:
        # Layer widths double because root and mean-neighbor features are concatenated.
        super().__init__()
        self.l1 = nn.Linear(2 * p, hidden)
        self.l2 = nn.Linear(2 * hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.dropout = dropout

    # FUNCTION: SAGE.forward — Compute one logit per transaction; keep architecture consistent with graph ablations.
    # WHY: Use architecture-matched graph controls and train loss only on known historical labels.
    def forward(self, x, adjacency):
        # A node receives both its own features and its neighbors' mean.
        # RESEARCH RATIONALE:
        # First pass sees one-hop incoming neighbors; second pass can see two-hop paths.
        # Dropout acts only in train() mode; eval() produces deterministic inference.
        h = torch.cat((x, agg(adjacency, x)), dim=1)
        h = torch.nn.functional.dropout(
            torch.relu(self.l1(h)), self.dropout, training=self.training,
        )
        # The second pass may transmit information along two-hop paths.
        h = torch.cat((h, agg(adjacency, h)), dim=1)
        h = torch.nn.functional.dropout(
            torch.relu(self.l2(h)), self.dropout, training=self.training,
        )
        return self.out(h).squeeze(-1)


# CLASS: BidirSAGE — Two-layer GraphSAGE with separate incoming/outgoing mean messages.
class BidirSAGE(nn.Module):
    """Two-layer GraphSAGE with separate incoming/outgoing mean messages."""

    # FUNCTION: BidirSAGE.__init__ — Define network layers with dimensions matching the architecture’s node/neighbor concatenation.
    # WHY: Use architecture-matched graph controls and train loss only on known historical labels.
    def __init__(self, p, hidden=48, dropout=.2):
        # RESEARCH RATIONALE:
        # Three concatenated streams: root, incoming mean, and outgoing mean.
        # The root skip connection can preserve transaction-specific information.
        super().__init__()
        self.l1 = nn.Linear(3 * p, hidden)
        self.l2 = nn.Linear(3 * hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.root_skip = nn.Linear(p, 1, bias=False)
        self.dropout = dropout

    # FUNCTION: BidirSAGE.forward — Compute one logit per transaction; keep architecture consistent with graph ablations.
    # WHY: Use architecture-matched graph controls and train loss only on known historical labels.
    def forward(self, x, adjacencies):
        # RESEARCH RATIONALE:
        # Use separate directed adjacency matrices; reversing an edge changes the message path.
        incoming, outgoing = (
            (None, None) if adjacencies is None else adjacencies
        )
        # Retain own features and separate the two transaction directions.
        h = torch.cat((x, agg(incoming, x), agg(outgoing, x)), dim=1)
        h = torch.nn.functional.dropout(
            torch.relu(self.l1(h)), self.dropout, training=self.training,
        )
        h = torch.cat((h, agg(incoming, h), agg(outgoing, h)), dim=1)
        h = torch.nn.functional.dropout(
            torch.relu(self.l2(h)), self.dropout, training=self.training,
        )
        # Root-skip preserves direct original-feature information at output.
        return self.out(h).squeeze(-1) + self.root_skip(x).squeeze(-1)


# CLASS: MLP — Feature-only two-layer neural control; no transaction adjacency.
class MLP(nn.Module):
    """Feature-only two-layer neural control; no transaction adjacency."""

    # FUNCTION: MLP.__init__ — Define network layers with dimensions matching the architecture’s node/neighbor concatenation.
    # WHY: Use architecture-matched graph controls and train loss only on known historical labels.
    def __init__(self, p, hidden=48, dropout=.2):
        super().__init__()
        self.l1 = nn.Linear(p, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.dropout = dropout

    # FUNCTION: MLP.forward — Compute one logit per transaction; keep architecture consistent with graph ablations.
    # WHY: Use architecture-matched graph controls and train loss only on known historical labels.
    def forward(self, x, adjacency=None):
        # RESEARCH RATIONALE:
        # Ignore adjacency completely: this is the feature-only neural reference.
        x = torch.nn.functional.dropout(
            torch.relu(self.l1(x)), self.dropout, training=self.training,
        )
        x = torch.nn.functional.dropout(
            torch.relu(self.l2(x)), self.dropout, training=self.training,
        )
        return self.out(x).squeeze(-1)


# -----------------------------------------------------------------------------
# Conventional tabular models: LOGIT, XGB, LGB, CAT
# -----------------------------------------------------------------------------
# FUNCTION: fit_tab — Fit tabular classifiers and choose a tree budget on historical stop data only.
# WHY: Use architecture-matched graph controls and train loss only on known historical labels.
def fit_tab(name, xx, d, train, stop, args, seed):
    """Fit a tabular model using historical known labels only.

    Hyperparameter budget is chosen according to full/quick/smoke mode.
    Candidate tree counts are selected by AP on historical stop periods,
    never by the previously exposed diagnostic period 35-49.
    """
    # RESEARCH RATIONALE:
    # Tune tree budget only on the earlier stop window; never the exposed diagnostics.
    # The sqrt class-imbalance weight is bounded to avoid extreme positive-class leverage.
    # CatBoost stays on CPU for the retained reference configuration.
    labels = d.y[train]
    n0 = int(sum(labels == 0))
    n1 = int(sum(labels == 1))
    # Temper class-imbalance weighting rather than using a huge raw ratio.
    weight = float(np.clip(np.sqrt(n0 / max(n1, 1)), 1, 25))

    # IF (name == "LOGIT"): Fit the linear baseline without tree-budget optimization.
    if name == "LOGIT":
        # IMPORTANT: Keep controlled randomness reproducible across repeated experimental comparisons.
        model = LogisticRegression(
            max_iter=600, C=.15, class_weight="balanced", random_state=seed,
        )
        # IMPORTANT: Train the current model only on the selected historical known-label indices.
        model.fit(xx[train], labels)
        # IMPORTANT: Extract positive-class scores for ranking; thresholds are chosen separately.
        stop_score = ap(d.y[stop], model.predict_proba(xx[stop])[:, 1])
        return model, {"iterations": 1, "stop_ap": stop_score}

    # IF (args.smoke): Use the smallest compute budget only for synthetic software integration.
    if args.smoke:
        choices = [12, 25]
    # IF (args.quick): Use the reduced development budget, not the full reference budget.
    elif args.quick:
        choices = [60, 130]
    else:
        choices = [80, 150, 280]

    best = (-np.inf, None, None)
    # LOOP (choices): Process every item using the same fixed input/output contract.
    for trees in choices:
        # IF (name == "XGB"): Choose the XGBoost configuration from the fixed candidate registry.
        if name == "XGB":
            # IMPORTANT: Keep controlled randomness reproducible across repeated experimental comparisons.
            model = XGBClassifier(
                n_estimators=trees, max_depth=4, learning_rate=.05,
                colsample_bytree=.85, subsample=.85, reg_lambda=5,
                min_child_weight=5, scale_pos_weight=weight,
                random_state=seed, n_jobs=2, tree_method="hist",
                device="cuda" if args.gpu and torch.cuda.is_available() else "cpu",
                eval_metric="logloss",
            )
        # IF (name == "LGB"): Choose LightGBM with its fixed historical model parameters.
        elif name == "LGB":
            # IMPORTANT: Keep controlled randomness reproducible across repeated experimental comparisons.
            model = LGBMClassifier(
                n_estimators=trees, learning_rate=.045, num_leaves=23,
                max_depth=6, min_child_samples=40, colsample_bytree=.85,
                reg_lambda=5, scale_pos_weight=weight, random_state=seed,
                n_jobs=2, verbosity=-1,
            )
        # IF (name == "CAT"): Choose the CPU CatBoost configuration for reproducibility.
        elif name == "CAT":
            # CPU choice is deliberate for more reproducible cross-GPU results.
            model = CatBoostClassifier(
                iterations=trees, depth=5, learning_rate=.045,
                l2_leaf_reg=5, loss_function="Logloss",
                scale_pos_weight=weight, random_seed=seed,
                verbose=False, thread_count=2, task_type="CPU",
            )
        else:
            raise ValueError(name)

        # IMPORTANT: Train the current model only on the selected historical known-label indices.
        model.fit(xx[train], labels)
        # IMPORTANT: Extract positive-class scores for ranking; thresholds are chosen separately.
        value = ap(d.y[stop], model.predict_proba(xx[stop])[:, 1])
        # IF (np.isfinite(value) and value > best[0]): Keep the model with the best finite historical stop AP.
        if np.isfinite(value) and value > best[0]:
            best = (value, model, trees)

    # IF (best[1] is None): Fail rather than return an unfitted tabular model.
    if best[1] is None:
        raise RuntimeError(f"{name}: no valid tree fit")
    return best[1], {"iterations": best[2], "stop_ap": best[0]}


# -----------------------------------------------------------------------------
# Full-batch training for MLP / GraphSAGE variants
# -----------------------------------------------------------------------------
# FUNCTION: fit_nn — Train supervised loss on known historical labels while allowing period-close graph context.
# WHY: Use architecture-matched graph controls and train loss only on known historical labels.
def fit_nn(name, xx, d, train, stop, args, seed, adj_cache):
    """Train a neural model with a supervised mask for historical known labels.

    The full graph forward pass is allowed only under period-end batch inference.
    Unknown labels NEVER contribute to binary cross-entropy. Validation labels
    influence only early-stopping choice inside historical stop periods.
    """
    # RESEARCH RATIONALE:
    # A forward pass can use unlabeled graph context; cross-entropy cannot use unknown labels.
    # The stop split chooses an epoch without using any labels from diagnostic periods 35-49.
    # Restore a CPU copy of the best state; the last optimizer epoch is not necessarily best.
    device = torch.device(
        "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # IF (device.type == "cuda"): Seed CUDA explicitly when GPU training is enabled.
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    graph_mode = {
        "MLP": "no_graph",
        "SAGE_BATCH": "batch",
        "SAGE_NO_GRAPH": "no_graph",
        "SAGE_REWIRED": "rewired",
        "BIDIR_SAGE": "bidir",
        "BIDIR_NO_GRAPH": "no_graph",
        "BIDIR_REWIRED": "bidir_rewired",
    }[name]
    # IF (graph_mode not in adj_cache): Construct each expensive sparse adjacency once per device and graph mode.
    if graph_mode not in adj_cache:
        # IMPORTANT: Build sparse graph context consistent with the chosen model ablation.
        adj_cache[graph_mode] = to_sparse_torch(d, device, graph_mode)
    adjacency = adj_cache[graph_mode]

    # IF (name == "MLP"): Use a feature-only MLP with no graph messages.
    if name == "MLP":
        model = MLP(xx.shape[1])
    # IF (name.startswith("BIDIR_")): Use the three-channel incoming/outgoing directed architecture.
    elif name.startswith("BIDIR_"):
        model = BidirSAGE(xx.shape[1])
    else:
        model = SAGE(xx.shape[1])
    model = model.to(device)

    features = torch.tensor(xx, device=device)
    train_indices = torch.tensor(train, dtype=torch.long, device=device)
    train_labels = torch.tensor(
        d.y[train].astype("float32"), device=device,
    )
    weight = float(np.clip(
        np.sqrt(sum(d.y[train] == 0) / max(1, sum(d.y[train] == 1))), 1, 25,
    ))
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([weight], device=device),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=.002, weight_decay=1e-4,
    )

    epochs = 8 if args.smoke else (24 if args.quick else 65)
    patience = 4 if args.smoke else (8 if args.quick else 12)
    best_ap = -np.inf
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    # LOOP (range(1, epochs + 1)): Process every item using the same fixed input/output contract.
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(features, adjacency)
        # This mask enforces supervised training on known historical labels.
        loss = loss_function(logits[train_indices], train_labels)
        # IMPORTANT: Propagate gradients from known historical supervised labels.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        # IMPORTANT: Update trainable weights after the current batch’s gradients were computed.
        optimizer.step()

        model.eval()
        with torch.no_grad():
            stop_scores = torch.sigmoid(
                model(features, adjacency)[stop]
            ).cpu().numpy()
        score = ap(d.y[stop], stop_scores)

        # IF (np.isfinite(score) and score > best_ap + 1e-5): Update the best checkpoint using historical validation AP only.
        if np.isfinite(score) and score > best_ap + 1e-5:
            best_ap = score
            # IMPORTANT: Snapshot the exact learned parameter tensors for reproducible reload.
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        # IF (bad_epochs >= patience): Stop training after repeated validation non-improvement.
        if bad_epochs >= patience:
            break

    # IF (best_state is None): Reject a run without a valid early-stop checkpoint.
    if best_state is None:
        raise RuntimeError(f"{name}: no valid historical stop AP")

    # IMPORTANT: Restore the best or saved model parameters before inference.
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        all_scores = torch.sigmoid(model(features, adjacency)).cpu().numpy()
    return model, all_scores, {"iterations": best_epoch, "stop_ap": best_ap}


# FUNCTION: fit_predict — Expose one training/scoring interface shared by all tabular and neural models.
# WHY: Use architecture-matched graph controls and train loss only on known historical labels.
def fit_predict(name, d, xx, train, stop, requested, args, seed, adj_cache):
    """Uniform model interface; `requested` selects which scores to return."""
    # RESEARCH RATIONALE:
    # The caller sees one interface across sklearn trees, logistic regression and PyTorch.
    # Neural models score all graph nodes first, then return only requested positions.
    # IF (name in ("LOGIT", "XGB", "LGB", "CAT")): Dispatch tabular models through predict_proba; neural models need batch inference.
    if name in ("LOGIT", "XGB", "LGB", "CAT"):
        model, info = fit_tab(name, xx, d, train, stop, args, seed)
        # IMPORTANT: Extract positive-class scores for ranking; thresholds are chosen separately.
        scores = model.predict_proba(xx[requested])[:, 1]
        return model, scores, info

    model, all_scores, info = fit_nn(
        name, xx, d, train, stop, args, seed, adj_cache,
    )
    return model, all_scores[requested], info
