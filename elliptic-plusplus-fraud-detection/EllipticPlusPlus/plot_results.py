# Regenerate saved plots without retraining
# ---------------------------------------------------------------
# PURPOSE: CSV artifacts are the inputs; only PNG figures are recomputed.
# RESEARCH RULE: Use it to restyle figures without changing scientific results.
# ---------------------------------------------------------------

"""Create visual reports from a completed Elliptic++ run, without retraining.

Example:
    python plot_results.py --results ./results/EXPOSED_DIAGNOSTIC_20261028T114152Z
"""
import argparse
from pathlib import Path
from elliptic_fraud.visualization import create_all_figures
from elliptic_fraud.metric_reports import create_supplementary_figures


# ---- Command-line orchestration with strict nonzero failures ----
# FUNCTION: main — Regenerate figures using results already on disk.
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',required=True,help='Directory holding run CSV/JSON files')
    parser.add_argument('--models',default=None,help='Comma-separated set to plot (default: informative subset)')
    args=parser.parse_args()
    chosen=args.models.split(',') if args.models else None
    files=create_all_figures(args.results,chosen)
    # IF (saved FULL_METRICS tables exist): Regenerate supplementary charts from
    # frozen reports; no model training and no decision-policy selection occurs.
    if (Path(args.results) / 'FULL_METRICS' / 'CLASSIFICATION_BY_TIME_STEP.csv').is_file():
        files += create_supplementary_figures(args.results)
    # LOOP (files): Process every item using the same fixed input/output contract.
    for p in files:print(p)


# IF (__name__=='__main__'): Select a specific model implementation without altering the common fit/predict contract.
if __name__=='__main__':
    main()
