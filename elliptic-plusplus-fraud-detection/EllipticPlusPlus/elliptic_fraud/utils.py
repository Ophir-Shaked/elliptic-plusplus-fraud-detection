# -----------------------------------------------------------------------------
# MODULE: utils.py -- consistent research artifact saving
# PURPOSE: JSON serialization and streaming SHA-256 checksums.
# LIMIT: Hashes establish byte integrity, not statistical correctness.
# -----------------------------------------------------------------------------
"""Small utilities shared by all experiment and verification modules."""

import hashlib
import json
from pathlib import Path

import numpy as np


# FUNCTION: tojson — Convert common NumPy values to JSON-compatible Python types.
# WHY: Keep input validation, business logic, and output handling explicit for reproducibility.
def tojson(value):
    """Convert common NumPy values to JSON-compatible Python types."""
    # RESEARCH RATIONALE:
    # Convert numpy scalars and arrays to JSON-safe types without changing numerical meaning.
    # IF (isinstance(value, np.integer)): Handle this alternate branch explicitly so that the experiment remains deterministic and auditable.
    if isinstance(value, np.integer):
        return int(value)
    # IF (isinstance(value, np.floating)): Handle this alternate branch explicitly so that the experiment remains deterministic and auditable.
    if isinstance(value, np.floating):
        return float(value)
    # IF (isinstance(value, np.ndarray)): Handle this alternate branch explicitly so that the experiment remains deterministic and auditable.
    if isinstance(value, np.ndarray):
        return value.tolist()
    # IF (isinstance(value, Path)): Handle this alternate branch explicitly so that the experiment remains deterministic and auditable.
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


# FUNCTION: dump — Write human-readable JSON; use the same encoding across all reports.
# WHY: Keep input validation, business logic, and output handling explicit for reproducibility.
def dump(path, obj):
    """Write human-readable JSON; use the same encoding across all reports."""
    # RESEARCH RATIONALE:
    # Persist reviewer-readable provenance, reports and protocol settings.
    Path(path).write_text(
        json.dumps(obj, indent=2, default=tojson, allow_nan=True),
        encoding="utf-8",
    )


# FUNCTION: sha — Compute SHA-256 in 1 MiB chunks, avoiding large in-memory copies.
# WHY: Keep input validation, business logic, and output handling explicit for reproducibility.
def sha(path):
    """Compute SHA-256 in 1 MiB chunks, avoiding large in-memory copies."""
    # RESEARCH RATIONALE:
    # Compute source/artifact SHA-256 incrementally to avoid loading huge files in RAM.
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        # LOOP (iter(lambda: stream.read(2 ** 20), b"")): Process every item using the same fixed input/output contract.
        for chunk in iter(lambda: stream.read(2 ** 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
