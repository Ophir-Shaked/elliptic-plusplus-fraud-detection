
# ---------------------------------------------------------------
# PURPOSE: An audit re-checks stored results independently of the training code.
# RESEARCH RULE: The process returns a nonzero exit code whenever a required check fails.
# ---------------------------------------------------------------

"""Independent artifact audit and v3.1-to-modular comparison.

Examples:
    python verify_results.py audit /content/results/RUN.zip --models 11
    python verify_results.py compare /content/modular/RUN.zip /content/v31/RUN.zip
"""
import argparse
import json
from pathlib import Path

from elliptic_fraud.verification import audit_run, compare_runs


# ---- Command-line orchestration with strict nonzero failures ----
# WHY: The exact behavior depends on the entry-point file.
# FUNCTION: main — Expose independent audit and frozen-reference comparison through CLI.
# WHY: Keep input validation, business logic, and output handling explicit for reproducibility.
def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='operation', required=True)
    a = sub.add_parser('audit', help='Independently re-evaluate a finished run')
    a.add_argument('source', help='Results directory or ZIP')
    a.add_argument('--models', type=int, default=None, help='Require exactly N models (11 for full real run)')
    c = sub.add_parser('compare', help='Compare a new real run with v3.1 historical archive')
    c.add_argument('modular', help='New modular run ZIP or directory')
    c.add_argument('reference', help='Historical real v3.1 ZIP or directory')
    c.add_argument('--metric-atol', type=float, default=0.02)
    c.add_argument('--threshold-atol', type=float, default=0.05)
    args=p.parse_args()
    # IF (args.operation=='audit'): Validate this conditional path before continuing with the selected workflow.
    if args.operation=='audit':
        result=audit_run(args.source,require_models=args.models)
        output=Path(args.source).with_suffix('.audit.json') if Path(args.source).is_file() else Path(args.source)/'EXTERNAL_AUDIT.json'
    else:
        # IF (args.metric_atol<0 or args.threshold_atol<0): Use a prior, explicitly permitted selection/calibration criterion, not exposed diagnostic labels.
        if args.metric_atol<0 or args.threshold_atol<0:
            p.error('Tolerance must be nonnegative')
        result=compare_runs(args.modular,args.reference,metric_atol=args.metric_atol,
                            threshold_atol=args.threshold_atol)
        output=Path(args.modular).with_suffix('.comparison.json') if Path(args.modular).is_file() else Path(args.modular)/'V31_COMPARISON.json'
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Saved:',output)
    print('STATUS:', 'PASS' if result['passed'] else 'FAIL')
    # IF (args.operation == 'audit'): Validate this conditional path before continuing with the selected workflow.
    if args.operation == 'audit':
        print('CHECKS:',result['checks_passed'],'/',result['checks_total'])
    else:
        print('REGRESSION COMPARISONS:',result['n_checks']-result['n_failed'],'/',result['n_checks'])
    # IF (not result['passed']): Handle this alternate branch explicitly so that the experiment remains deterministic and auditable.
    if not result['passed']:
        print('FAILED:', json.dumps(result.get('failed',[]),ensure_ascii=False)[:5000])
        raise SystemExit(1)


# IF (__name__ == '__main__'): Select a specific model implementation without altering the common fit/predict contract.
if __name__ == '__main__':
    main()
