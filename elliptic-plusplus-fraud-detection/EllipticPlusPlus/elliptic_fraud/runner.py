# MODULE: runner.py
# PURPOSE: Complete research experiment orchestration
# RESEARCH RULE: Do not mark a run successful until required outputs and model roundtrips pass.

"""Top-level reproducible experiment orchestration, configured through CLI flags."""
import argparse
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .config import FOLDS, FINAL, MODELS, GRAPH_PAIRS
from .utils import dump, sha
from .data import load_real, make_synthetic
from .protocol import qa, check_splits, leakage_contract
from .graph import graph_rewire_audit
from .analysis import diagnostics, plot_label_coverage, dev_importance
from .controls import neg_control, feature_group_development, label_coverage_diagnostic
from .development import run_dev, paired_graph_boot, graph_seed_stability
from .final_evaluation import final_all, ablations, exposed_ablation_diagnostics
from .visualization import create_all_figures
from .metric_reports import generate_metric_reports


# ---------------------------------------------------------
# Main
# WHY: Expose all workflow options through one reproducible CLI.
# ---------------------------------------------------------
# FUNCTION: main — Parse settings and orchestrate all mandatory stages, fail closed on missing artifacts.
# RESEARCH RULE: One reproducible orchestrator controls the same pipeline from PyCharm and Colab.
def main():
    # RESEARCH RATIONALE:
    # One CLI orchestrates the same research modules in Colab, PyCharm and a terminal.
    # No success marker is written until all expected artifacts and model roundtrips complete.
    # A synthetic smoke run must never be presented as evidence about real transactions.
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true', help='Generate fake data only for SOFTWARE tests')
    p.add_argument('--quick', action='store_true', help='Use smaller fixed compute budget (not publication run)')
    p.add_argument('--no-download', action='store_true')
    p.add_argument('--gpu', action='store_true', help='Use Tesla T4 for torch and XGBoost (CPU fallback)')
    project_root = Path(__file__).resolve().parents[1]
    p.add_argument('--data-dir', default=str(project_root / 'data'))
    p.add_argument('--output', default=str(project_root / 'results'))
    p.add_argument('--models', default=','.join(MODELS), help='Comma-separated model names (default: all 11)')
    p.add_argument('--top-k', type=int, default=100)
    p.add_argument('--disable-rewired', action='store_true')
    p.add_argument('--seed-repeats', type=int, default=2, help='Paired SAGE seeds in development folds')
    p.add_argument('--control-reps', type=int, default=120, help='Conditional permutation repetitions; early development only')
    args = p.parse_args()
    # IF (args.top_k < 1 or args.seed_repeats < 1 or args.control_reps < 2): Validate experiment sizes before any expensive data download or training.
    if args.top_k < 1 or args.seed_repeats < 1 or args.control_reps < 2:
        raise ValueError('top-k/seed-repeats must be positive; control-reps >=2')
    requested = [x.strip().upper() for x in args.models.split(',') if x.strip()]
    # IF (not requested or len(set(requested)) != len(requested) or (not set(requested).issubset(set(MODELS)))): Reject unknown or repeated model names rather than silently dropping candidates.
    if not requested or len(set(requested)) != len(requested) or (not set(requested).issubset(set(MODELS))):
        raise ValueError('Invalid model selection. Choose unique names from: ' + ', '.join(MODELS))
    # IF (args.smoke): Limit synthetic testing resources; synthetic AP is not a research result.
    if args.smoke:
        args.seed_repeats = min(args.seed_repeats, 2)
    # IF (args.gpu and (not torch.cuda.is_available())): Explicitly report a CPU fallback when CUDA is unavailable.
    if args.gpu and (not torch.cuda.is_available()):
        print('GPU NOT available, proceeding on CPU. Check Runtime > Change runtime type.', flush=True)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    out = Path(args.output) / ('SMOKE_SYNTHETIC_' + stamp if args.smoke else 'EXPOSED_DIAGNOSTIC_' + stamp)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print('START', out, 'GPU', args.gpu and torch.cuda.is_available(), flush=True)
    print('RESEARCH NOTE: previous test periods 35-49 are EXPOSED; results are descriptive.', flush=True)
    shutil.make_archive(str(out / 'SOURCE_SNAPSHOT'), 'zip', root_dir=Path(__file__).resolve().parent.parent, base_dir='elliptic_fraud')
    shutil.copy2(Path(__file__).resolve().parent.parent / 'run.py', out / 'RUN_ENTRYPOINT.py')
    d = make_synthetic() if args.smoke else load_real(args)
    report = qa(d, out)
    print('DATA', report, flush=True)
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    check_splits(d).to_csv(out / 'TIME_PROTOCOL.csv', index=False)
    # IMPORTANT: Write structured experiment metadata for independent audit.
    dump(out / 'ANALYSIS_CONFIG_SNAPSHOT.json', {'documentation_status': 'Retrospective v3.1 correction to v3.0; NOT preregistered, previous test was exposed', 'fit_stop_calibration_policy_diagnostic': FINAL, 'dev_folds': FOLDS, 'models': requested, 'feature_subset': 'first 93 numeric columns after ID/time; plus development-only study of 17 additional on-chain columns; exact availability timestamps not known', 'graph_task': 'end-of-time-step retrospective batch only', 'no_blind_test': 'periods 35-49 previously analyzed', 'top_k': args.top_k, 'seed_repeats': args.seed_repeats, 'control_reps': args.control_reps, 'data_source': d.source, 'model_subset': requested})
    leakage_contract(d, out)
    rewire_audit = graph_rewire_audit(d, out)
    diagnostics(d, out)
    plot_label_coverage(out)
    feature_group_development(d, args, out)
    label_coverage_diagnostic(d, args, out)
    top10 = dev_importance(d, args, out)
    negative = neg_control(d, args, out)
    models = [m for m in requested if not (args.disable_rewired and m in ('SAGE_REWIRED', 'BIDIR_REWIRED'))]
    pairs = [(g, z) for g, z in GRAPH_PAIRS if g in models and z in models]
    dev, oof = run_dev(d, args, out, np.arange(93), models)
    paired = [paired_graph_boot(oof, args, out, g, z, '_' + g) for g, z in pairs]
    paired_seeds = graph_seed_stability(d, args, out, pairs=pairs) if pairs else []
    print('PAIRED_GRAPH_DEV', paired, flush=True)
    ablations(d, args, out)
    exposed_ablation_diagnostics(d, args, out)
    winner = str(dev.iloc[0].model)
    # IMPORTANT: Write structured experiment metadata for independent audit.
    dump(out / 'EXPLORATORY_ANALYSIS_PROTOCOL.json', {'data_role': 'EXPOSED / EXPLORATORY; NO NEW BLIND TEST EXISTS', 'winner_development_only': winner, 'features': 'first 93 numerical feature positions excluding tx/time; full feature semantics not validated', 'batch_definition': 'All graph edges in the same step assumed accessible at period close', 'online_claim': 'NOT SUPPORTED: same-step within-transaction timestamps unavailable', 'split_windows': FINAL, 'development_folds': FOLDS, 'supervised_training_labeled_only': True, 'unlabeled_graph_context_allowed': True, 'imputer_scaler_fit_only_through_fit_end': True, 'platt_fit_calibration_only': FINAL['calibration'], 'decision_threshold_fit_policy_only': FINAL['policy'], 'model_candidates': models, 'graph_comparator': pairs, 'graph_degree_preserving_rewire': rewire_audit, 'negative_controls': ['train_global_shuffle', 'train_within_period_shuffle_group_probe', 'eval_global_label_null', 'eval_within_period_label_null', 'eval_within_period_score_null', 'cohort_signature_BATCH', 'time_logistic_trend', 'constant_baseline'], 'ablation_note': 'local_002/003 flagged on a previously exposed test; diagnostic only', 'source_and_known_limits': d.source, 'seed': seed, 'seed_repeats': args.seed_repeats, 'smoke_synthetic': args.smoke})
    final = final_all(d, args, out, np.arange(93), models)
    # Export independent recall/precision/confusion reports before the success marker and ZIP manifest.
    # The existing policy thresholds remain frozen; exposed labels are NEVER used to tune them.
    generate_metric_reports(out)
    figure_paths = create_all_figures(out, models)
    print(f'GENERATED {len(figure_paths)} figures', flush=True)
    report = {'environment': {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'platform': platform.platform()}, 'data_quality': report, 'paired_graph_development': paired, 'paired_graph_seed_stability': paired_seeds, 'development_selected_model': winner, 'diagnostic_comparison_summary': final[['model', 'policy', 'AP', 'ROC_AUC', 'precision', 'recall']].to_dict('records'), 'limitations': ['NO prospective blind evaluation: 35-49 were exposed in the previous project.', 'No guaranteed online availability for raw local features or same-step edges.', 'Same-step batch graphs may include future-within-step information relative to an individual tx.', 'Unknown labels must not be treated as licit or counted as confirmed false positives.', 'Graph transduction using unlabeled features is allowed only for period-end batch setting.', 'First 93 positional features are a restricted baseline and not a confirmed complete inventory of locally computable Elliptic++ features.', 'Calibration slope constrained to >=0.01 to preserve score ranking; values at the floor indicate weak or inverse calibration association.', 'AP is conditional on labeled sample; coverage is time-dependent.', 'Directed degree-preserving within-step rewiring is a negative control, not a realistic financial graph.', 'Feature groups are based on published structure; local feature timestamp provenance not independently verified.', 'A classifier can predict label observation, but its performance does not establish the missing-label causal mechanism.', 'Prior exposed tests drove some hypotheses; all conclusions about 35-49 remain post hoc.', 'Previously exposed 35-49 feature ablations are strictly descriptive and may not be used to select a new model.', 'Training-only within-period label shuffle preserves cohort prevalence; cannot be interpreted as a strict null.', 'Within-period evaluation permutation is conditional on observed labels and cannot resolve selection bias.', 'Period bootstrap is descriptive; does not establish significance under selective labeling.']}
    # IMPORTANT: Write structured experiment metadata for independent audit.
    dump(out / 'RESEARCH_SUMMARY.json', report)
    needed = ['DEV_SUMMARY.csv', 'DEV_NEGATIVE_CONTROLS.csv', 'DEV_CONDITIONAL_SCORE_NULL.csv', 'DEV_COHORT_ONLY_BASELINE.csv', 'DEV_FEATURE_GROUP_COMPARISON.csv', 'DEV_LABEL_OBSERVATION_BIAS.csv', 'GRAPH_REWIRE_DEGREE_AUDIT.json', 'MODEL_RELOAD_VERIFICATION.csv', 'FINAL_DIAGNOSTIC_ALL_MODELS.csv', 'FULL_METRICS/CLASSIFICATION_REPORT_ALL_MODELS.csv', 'FULL_METRICS/INDEPENDENT_METRIC_RECOMPUTATION_CHECK.csv']
    # IF (pairs): Run paired seed controls only when both architecture variants were selected.
    if pairs:
        needed.append('DEV_GRAPH_SEED_STABILITY.csv')
    # LOOP (needed): Repeat the same reproducible procedure for every selected fold, model or period.
    for name in needed:
        # IF (not (out / name).exists() or (out / name).stat().st_size < 20): Refuse to mark the run complete unless required output artifacts exist.
        if not (out / name).exists() or (out / name).stat().st_size < 20:
            raise AssertionError('Required output missing or empty: ' + name)
    # IMPORTANT: Write structured experiment metadata for independent audit.
    dump(out / 'RUN_SUCCESS.json', {'status': 'COMPLETE', 'dataset_kind': d.source['kind'], 'diagnostic_only': not args.smoke, 'elapsed_seconds': round(time.time() - start, 2)})
    # IMPORTANT: Record content integrity so changed artifacts can be detected.
    files = [(str(p.relative_to(out)), sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'SHA256_MANIFEST.csv']
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame(files, columns=['file', 'sha256']).to_csv(out / 'SHA256_MANIFEST.csv', index=False)
    zip_path = shutil.make_archive(str(out), 'zip', root_dir=out.parent, base_dir=out.name)
    print('\nCOMPLETE', out, 'ZIP', zip_path, 'SECONDS', round(time.time() - start, 1), flush=True)
    # IF (args.smoke): Limit synthetic testing resources; synthetic AP is not a research result.
    if args.smoke:
        print('SMOKE ONLY: synthetic data. NEVER cite these AP values as research results.', flush=True)
