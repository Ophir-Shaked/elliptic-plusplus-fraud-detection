# Export additional metrics from an existing extracted run (without retraining).
# Usage: python export_all_metrics.py /path/to/EXPOSED_DIAGNOSTIC_...
from pathlib import Path
import argparse
from elliptic_fraud.metric_reports import generate_metric_reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path, help='Extracted result folder')
    args = parser.parse_args()
    generate_metric_reports(args.run_dir)


if __name__ == '__main__':
    main()
