"""Load official Elliptic++ transactions or a deterministic synthetic fixture.

Research contract
-----------------
* Align feature rows, labels and edges through transaction IDs, never row order.
* The class -1 is UNKNOWN: it is not a negative class.
* The main experiment uses 93 positional features for v3.1 comparability.
* The synthetic fixture is for software testing, not scientific conclusions.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FNS, REPO
from .utils import sha


# -----------------------------------------------------------------------------
# CSV parsing and repository acquisition
# -----------------------------------------------------------------------------
# FUNCTION: load_csv — Read official CSV tables without accidentally dropping a genuine first transaction row.
# WHY: Transaction IDs and labels must be aligned before graph construction; unknown=-1 is not a negative.
def load_csv(path):
    """Read a headerless CSV, optionally discarding one recognized header row."""
    # RESEARCH RATIONALE:
    # Keep original column positions: a shifted header changes the 93-feature baseline.
    # Do not coerce transaction IDs into floating point before aligning tables.
    frame = pd.read_csv(path, header=None, low_memory=False)
    first = str(frame.iat[0, 0]).strip().lower().replace("_", "")
    # IF (first in ("txid", "txid1", "source", "transactionid")): Remove a recognized header row; otherwise preserve the first real transaction.
    if first in ("txid", "txid1", "source", "transactionid"):
        frame = frame.iloc[1:].reset_index(drop=True)
    return frame


# FUNCTION: valid_dir — Reject incomplete downloads, empty files, and unresolved Git LFS pointers.
# WHY: Transaction IDs and labels must be aligned before graph construction; unknown=-1 is not a negative.
def valid_dir(directory):
    """Require the three real CSVs, not tiny Git LFS pointer files."""
    # RESEARCH RATIONALE:
    # Git LFS may download a small pointer instead of the actual CSV payload.
    # Validating cache contents prevents apparently successful runs on placeholder text.
    directory = Path(directory)
    # LOOP (FNS): Process every item using the same fixed input/output contract.
    for filename in FNS:
        path = directory / filename
        # IF (not path.is_file() or path.stat().st_size <= 500): Reject missing files and tiny Git LFS placeholder payloads.
        if not path.is_file() or path.stat().st_size <= 500:
            return False
        with path.open("rb") as stream:
            # IF (b"git-lfs.github.com/spec" in stream.read(150)): Reject Git LFS pointer text instead of treating it as real CSV data.
            if b"git-lfs.github.com/spec" in stream.read(150):
                return False
    return True


# FUNCTION: get_data_dir — Reuse local CSVs or fetch only needed official dataset files using Git LFS.
# WHY: Transaction IDs and labels must be aligned before graph construction; unknown=-1 is not a negative.
def get_data_dir(args):
    """Reuse cached data; otherwise fetch the public repository using Git LFS.

    PyCharm/Windows: install Git and Git LFS before the first real run.
    Colab/Linux: the missing git-lfs package can be installed via apt-get.
    """
    # RESEARCH RATIONALE:
    # Prefer already-cached official CSV files to save Colab bandwidth.
    # Fail in no-download mode rather than silently falling back to fake data.
    root = Path(args.data_dir)
    candidates = [
        root,
        root / "Transactions Dataset",
        root / "EllipticPlusPlus" / "Transactions Dataset",
    ]
    # LOOP (candidates): Process every item using the same fixed input/output contract.
    for candidate in candidates:
        # IF (valid_dir(candidate)): Reuse a cached verified three-file official dataset before downloading.
        if valid_dir(candidate):
            return candidate

    # IF (args.no_download): Fail explicitly when the user forbids downloading missing official data.
    if args.no_download:
        raise FileNotFoundError(f"Missing official CSV files under {root}")

    root.mkdir(parents=True, exist_ok=True)
    destination = root / "EllipticPlusPlus"

    # Git LFS is essential: without it Git may download pointer text instead of CSVs.
    # IF (shutil.which("git-lfs") is None): Install or require Git LFS before retrieving large transaction CSVs.
    if shutil.which("git-lfs") is None:
        # IF (shutil.which("apt-get") is None): On Windows, fail with clear installation steps instead of attempting Linux commands.
        if shutil.which("apt-get") is None:
            raise RuntimeError("Install Git LFS first (Windows: git-lfs install)")
        subprocess.run(["apt-get", "-qq", "update"], check=True)
        subprocess.run(["apt-get", "-qq", "install", "-y", "git-lfs"], check=True)

    subprocess.run(["git", "lfs", "install", "--skip-repo"], check=True)

    # Sparse clone prevents fetching unnecessary repository directories.
    # IF (not (destination / ".git").exists()): Clone only if a local checkout does not exist.
    if not (destination / ".git").exists():
        environment = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
        subprocess.run(
            [
                "git", "clone", "--depth", "1", "--filter=blob:none",
                "--sparse", REPO, str(destination),
            ],
            env=environment,
            check=True,
        )

    subprocess.run(
        [
            "git", "-C", str(destination), "sparse-checkout", "set",
            "Transactions Dataset",
        ],
        check=True,
    )
    included = ",".join("Transactions Dataset/" + filename for filename in FNS)
    subprocess.run(
        [
            "git", "-C", str(destination), "lfs", "pull",
            "--include=" + included, "--exclude=",
        ],
        check=True,
    )

    result = destination / "Transactions Dataset"
    # IF (not valid_dir(result)): Refuse to proceed if the repository still holds pointers or incomplete files.
    if not valid_dir(result):
        raise RuntimeError("Incomplete CSV download or unresolved Git LFS pointers")
    return result


# -----------------------------------------------------------------------------
# Aligned dataset container
# -----------------------------------------------------------------------------
@dataclass
class Data:
    """Transaction features, known/unknown labels, time periods and directed edges."""

    ids: np.ndarray                 # Transaction IDs, in feature-row order.
    time: np.ndarray                # Period index (1..49).
    X: np.ndarray                   # Main model features: shape (n, 93).
    y: np.ndarray                   # 1=illicit; 0=licit; -1=unknown.
    src: np.ndarray                 # Edge source positions in X.
    dst: np.ndarray                 # Edge destination positions in X.
    source: dict                    # Provenance and original file hashes.
    X_extra: np.ndarray = None      # Additional 17 development-only features.

    @property
    def n(self):
        return len(self.time)

    @property
    def p(self):
        return self.X.shape[1]

    # FUNCTION: Data.ix — Select indices in inclusive chronological periods; optionally mask unknown labels.
    # WHY: Transaction IDs and labels must be aligned before graph construction; unknown=-1 is not a negative.
    def ix(self, start, end, labeled=True):
        """Return row positions in an *inclusive* chronological window.

        `labeled=True` excludes -1. Graph construction can still use all rows.
        """
        # RESEARCH RATIONALE:
        # The default index selection hides label=-1 from supervised losses and metrics.
        # Set labeled=False only for features/graph context and unverified Top-K alerts.
        mask = (self.time >= start) & (self.time <= end)
        # IF (labeled): Include unknown nodes only for graph context; exclude them from supervised labels.
        if labeled:
            mask &= self.y >= 0
        return np.flatnonzero(mask)


# -----------------------------------------------------------------------------
# Official transaction / label / edge loading
# -----------------------------------------------------------------------------
# FUNCTION: load_real — Align IDs, classes, features, and edges; retain unknown=-1 separately.
# WHY: Transaction IDs and labels must be aligned before graph construction; unknown=-1 is not a negative.
def load_real(args):
    """Load the exact official data schema used by the v3.1 study."""
    # RESEARCH RATIONALE:
    # ID/time/feature alignment determines every downstream graph and label.
    # Class 3 is unknown: it must NEVER become a known licit example.
    # The fixed first-93 feature policy is for v3.1 comparability, not proof of online availability.
    directory = get_data_dir(args)
    features, classes, edges = (load_csv(directory / name) for name in FNS)
    # IF (features.shape[1] < 95): The fixed 93-feature baseline requires ID, period and 93 feature columns.
    if features.shape[1] < 95:
        raise ValueError("Feature CSV must have at least 95 columns")

    # Normalize IDs in ALL tables before joining; don't rely on CSV row order.
    ids = (
        features.iloc[:, 0].astype(str).str.strip()
        .str.replace(r"(?<=\d)\.0+$", "", regex=True).to_numpy()
    )
    # IF (pd.Index(ids).has_duplicates): Duplicate transaction IDs make label/edge alignment ambiguous.
    if pd.Index(ids).has_duplicates:
        raise AssertionError("Duplicate transaction IDs in feature CSV")

    periods = pd.to_numeric(
        features.iloc[:, 1], errors="raise"
    ).to_numpy(dtype=np.int16)

    # Fixed positional baseline for old-results comparability. Exact original
    # feature availability at prediction time is NOT independently established.
    X = np.array(
        features.iloc[:, 2:95].apply(pd.to_numeric, errors="coerce"),
        dtype="float32", copy=True, order="C",
    )
    X[~np.isfinite(X)] = np.nan
    # IF (periods.min() != 1 or periods.max() != 49 or X.shape[1] != 93): Reject data inconsistent with the 49-step original study and 93-feature baseline.
    if periods.min() != 1 or periods.max() != 49 or X.shape[1] != 93:
        raise ValueError("Unexpected official time or local-feature schema")
    # IF (features.shape[1] != 184): Require the official full CSV schema before selecting positional feature groups.
    if features.shape[1] != 184:
        raise ValueError("Expected 184 columns in the official feature CSV")

    # Keep the extra 17 columns separate so graph-vs-tabular comparisons remain
    # comparable to the original 93-feature setup.
    extra = np.array(
        features.iloc[:, 167:184].apply(pd.to_numeric, errors="coerce"),
        dtype="float32", copy=True, order="C",
    )
    extra[~np.isfinite(extra)] = np.nan

    class_ids = (
        classes.iloc[:, 0].astype(str).str.strip()
        .str.replace(r"(?<=\d)\.0+$", "", regex=True)
    )
    label_text = classes.iloc[:, 1].astype(str).str.strip().str.lower()
    label_map = {
        "1": 1, "1.0": 1, "illicit": 1,
        "2": 0, "2.0": 0, "licit": 0,
        "3": -1, "3.0": -1, "unknown": -1,
    }
    mapped = label_text.map(label_map)
    # IF (mapped.isna().any() or class_ids.duplicated().any()): Reject unrecognized class codes and repeated class-table IDs.
    if mapped.isna().any() or class_ids.duplicated().any():
        raise ValueError("Unknown class values or duplicate class IDs")
    labels = pd.Series(mapped.to_numpy(dtype="int8"), index=class_ids).reindex(ids)
    # IF (labels.isna().any()): Every feature-row transaction must have a source class-row entry.
    if labels.isna().any():
        raise ValueError("Some transactions have no class-list entry")

    # Edges also reference transaction IDs, not row numbers.
    index_by_id = pd.Series(np.arange(len(ids), dtype="int32"), index=ids)
    source_ids = (
        edges.iloc[:, 0].astype(str).str.strip()
        .str.replace(r"(?<=\d)\.0+$", "", regex=True)
    )
    destination_ids = (
        edges.iloc[:, 1].astype(str).str.strip()
        .str.replace(r"(?<=\d)\.0+$", "", regex=True)
    )
    mapped_source = source_ids.map(index_by_id)
    mapped_destination = destination_ids.map(index_by_id)
    valid = mapped_source.notna() & mapped_destination.notna()
    src = mapped_source[valid].to_numpy(dtype="int32")
    dst = mapped_destination[valid].to_numpy(dtype="int32")

    # Keep the historical metadata contract: downstream QA and regression tests
    # use these fields to detect a different data release or feature selection.
    # IMPORTANT: Record content integrity so changed artifacts can be detected.
    provenance = {
        "kind": "official Elliptic++ transaction data",
        "dataset_dir": str(directory),
        "sha256": {name: sha(directory / name) for name in FNS},
        "raw_edges": len(edges),
        "resolved_edges": len(src),
        "feature_csv_columns": int(features.shape[1]),
        "feature_policy": (
            "ID/time excluded; FIRST 93 numerical columns used for original-"
            "protocol comparability, not proven complete local-feature schema"
        ),
        "excluded_csv_column_count": int(features.shape[1] - 95),
        "feature_groups": {
            "local_93": "CSV positions 2..94",
            "neighbor_aggregate_72_EXCLUDED": "CSV positions 95..166",
            "additional_17": "CSV positions 167..183 (development-only auxiliary analysis)",
            "time": "CSV position 1 EXCLUDED from main model",
        },
        "availability_limits": (
            "Exact source feature timing cannot be established from CSV alone"
        ),
    }
    return Data(ids, periods, X, labels.to_numpy(dtype="int8"), src, dst,
                provenance, extra)


# -----------------------------------------------------------------------------
# Synthetic fixture: never cite its AP as financial evidence
# -----------------------------------------------------------------------------
# FUNCTION: make_synthetic — Generate deterministic artificial data for software tests only.
# WHY: Transaction IDs and labels must be aligned before graph construction; unknown=-1 is not a negative.
def make_synthetic(seed=777):
    """Return deterministic 49-step dummy data for unit/smoke tests ONLY."""
    # RESEARCH RATIONALE:
    # Introduce controlled feature drift, label drift and unknown labels for regression testing.
    # A synthetic AP score is NOT evidence about illicit Bitcoin transactions.
    # IMPORTANT: Use an explicit local random generator rather than hidden global stochastic state.
    rng = np.random.default_rng(seed)
    nodes_per_period, feature_count = 75, 93
    n = 49 * nodes_per_period
    periods = np.repeat(np.arange(1, 50, dtype="int16"), nodes_per_period)
    X = rng.normal(0, 1, size=(n, feature_count)).astype("float32")

    # Intentional distribution and concept shifts exercise drift diagnostics.
    late = periods >= 43
    X[late, 1] += 3.0
    X[late, 2] -= 2.0
    logits = 1.4 * X[:, 0] + .6 * X[:, 1] + rng.normal(0, .9, n) - 1.5
    logits[late] = (
        -1.2 * X[late, 0] + .9 * X[late, 3]
        + rng.normal(0, .9, np.sum(late)) - 1.5
    )
    probability = 1 / (1 + np.exp(-logits))
    labels = (rng.uniform(size=n) < probability).astype("int8")
    labels[rng.uniform(size=n) < .35] = -1

    src, dst = [], []
    # LOOP (range(1, 50)): Process every item using the same fixed input/output contract.
    for step in range(1, 50):
        nodes = np.flatnonzero(periods == step)
        sources = rng.choice(nodes, 120)
        targets = rng.choice(nodes, 120)
        not_self_loop = sources != targets
        src.extend(sources[not_self_loop])
        dst.extend(targets[not_self_loop])

    extra = rng.normal(0, 1, size=(n, 17)).astype("float32")
    return Data(
        np.array([str(i) for i in range(n)]), periods, X, labels,
        np.array(src, dtype="int32"), np.array(dst, dtype="int32"),
        {"kind": "SYNTHETIC SMOKE ONLY"}, extra,
    )
