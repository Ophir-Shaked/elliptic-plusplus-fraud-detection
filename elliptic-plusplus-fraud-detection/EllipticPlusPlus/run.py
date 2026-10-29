# Small command-line entry point
# ---------------------------------------------------------------
# PURPOSE: Every run option is parsed centrally in elliptic_fraud/runner.py.
# ---------------------------------------------------------------

"""Run the modular Elliptic++ research pipeline from the repository root."""
from elliptic_fraud.runner import main

# IF (__name__ == "__main__"): Select a specific model implementation without altering the common fit/predict contract.
if __name__ == "__main__":
    main()
