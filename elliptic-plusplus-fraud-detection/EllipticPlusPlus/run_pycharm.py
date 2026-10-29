# %% [markdown]
# Elliptic++ — PyCharm / Jupyter-compatible launcher
# ---------------------------------------------------------------
# WHY: Running cells here launches the same run.py used by Google Colab.
# NOTE: In PyCharm use Run File or "Run Cell" for one # %% section.
# ---------------------------------------------------------------

# %% 1 — Central local settings
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
PYTHON = sys.executable                         # Always use PyCharm's selected interpreter.
DATA_DIR = PROJECT / "data"                   # Official CSV cache; not included in GitHub.
RESULTS_DIR = PROJECT / "results"             # New run directories and ZIP reports.
SMOKE_DIR = PROJECT / "smoke_results"           # Isolate artificial from real results.

# Edit these switches without touching your research implementation.
USE_GPU = False                               # Set True only if CUDA works.
MODEL_SELECTION = ""                           # Empty: all 11; example: "XGB,BIDIR_SAGE".
CONTROL_REPS = 120                            # Number of development null repetitions.
SEED_REPEATS = 2                              # Paired GNN initialization seeds.
TOP_K = 100                                   # Alerts per period (all labels including unknown).
RUN_TESTS = True                              # Quick local unit tests.
RUN_SMOKE = False                             # Set True for an end-to-end synthetic test.
RUN_REAL = False                              # Set True to download and train on official data.
RUN_VERIFY_LATEST = False                     # Validate latest completed results.


# FUNCTION: execute — Run a subprocess with PyCharm’s selected Python interpreter and fail on error.
# WHY: Keep input validation, business logic, and output handling explicit for reproducibility.
def execute(*arguments: str) -> None:
    """Run a command with the active Python interpreter and fail on errors."""
    command = [PYTHON, *map(str, arguments)]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


# FUNCTION: latest_completed — Find the newest run containing a success marker; reject partial runs.
# WHY: Keep input validation, business logic, and output handling explicit for reproducibility.
def latest_completed(folder: Path, pattern: str) -> Path:
    """Find a completed experiment; never mistake partial output for success."""
    candidates = sorted(p for p in folder.glob(pattern) if p.is_dir()
                        and (p / "RUN_SUCCESS.json").exists())
    # IF (not candidates): Handle this alternate branch explicitly so that the experiment remains deterministic and auditable.
    if not candidates:
        raise FileNotFoundError(f"No completed run in {folder}")
    return candidates[-1]


# %% 2 — Optional: install dependencies in the CURRENT PyCharm environment
# Run this cell once (or use PyCharm Terminal: python -m pip install -r requirements.txt).
# For safety it is disabled during ordinary Run File execution.
INSTALL_REQUIREMENTS = False
# IF (INSTALL_REQUIREMENTS): Handle this alternate branch explicitly so that the experiment remains deterministic and auditable.
if INSTALL_REQUIREMENTS:
    execute("-m", "pip", "install", "-r", "requirements.txt")


# %% 3 — Unit / research protocol tests (no dataset download)
# IF (RUN_TESTS): Handle this alternate branch explicitly so that the experiment remains deterministic and auditable.
if RUN_TESTS:
    execute("-m", "pytest", "-q", "tests")


# %% 4 — Synthetic integration test (cheap; NOT empirical fraud evidence)
# IF (RUN_SMOKE): Keep lightweight test budgets distinct from the full research configuration.
if RUN_SMOKE:
    args = ["run.py", "--smoke", "--quick", "--no-download",
            "--data-dir", str(DATA_DIR), "--output", str(SMOKE_DIR),
            "--control-reps", "4", "--seed-repeats", "1", "--top-k", str(TOP_K)]
    # IF (MODEL_SELECTION.strip()): Select a specific model implementation without altering the common fit/predict contract.
    if MODEL_SELECTION.strip():
        args += ["--models", MODEL_SELECTION.strip()]
    # IF (USE_GPU): Only select accelerated execution when requested and available; keep CPU behavior usable.
    if USE_GPU:
        args.append("--gpu")
    execute(*args)


# %% 5 — Full REAL Elliptic++ experiment (requires Git + Git LFS)
# This performs actual training. Only enable it after the unit/smoke tests pass.
# IF (RUN_REAL): Handle this alternate branch explicitly so that the experiment remains deterministic and auditable.
if RUN_REAL:
    args = ["run.py", "--data-dir", str(DATA_DIR), "--output", str(RESULTS_DIR),
            "--control-reps", str(CONTROL_REPS),
            "--seed-repeats", str(SEED_REPEATS), "--top-k", str(TOP_K)]
    # IF (MODEL_SELECTION.strip()): Select a specific model implementation without altering the common fit/predict contract.
    if MODEL_SELECTION.strip():
        args += ["--models", MODEL_SELECTION.strip()]
    # IF (USE_GPU): Only select accelerated execution when requested and available; keep CPU behavior usable.
    if USE_GPU:
        args.append("--gpu")
    execute(*args)


# %% 6 — Recreate plots + audit a completed REAL run without retraining
# IF (RUN_VERIFY_LATEST): Enforce the chronological scope here; validation and final diagnostics must not feed earlier fitting stages.
if RUN_VERIFY_LATEST:
    current = latest_completed(RESULTS_DIR, "EXPOSED_DIAGNOSTIC_*")
    execute("plot_results.py", "--results", str(current))
    # This audit verifies saved predictions and metadata, independent of model fitting.
    model_count = 11 if not MODEL_SELECTION.strip() else len(MODEL_SELECTION.split(","))
    execute("verify_results.py", "audit", str(current), "--models", str(model_count))
    print("Verified:", current)


# %% 7 — Print working paths
print(json.dumps({
    "project": str(PROJECT),
    "interpreter": PYTHON,
    "data_directory": str(DATA_DIR),
    "results_directory": str(RESULTS_DIR),
    "note": "Steps 35-49 are exposed retrospective diagnostics, not a blind test",
}, indent=2))
