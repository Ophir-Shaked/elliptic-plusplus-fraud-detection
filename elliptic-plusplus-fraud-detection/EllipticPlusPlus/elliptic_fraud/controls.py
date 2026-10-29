# MODULE: controls.py
# PURPOSE: Negative controls and label-selection analysis
# RESEARCH RULE: Within-period train-label shuffle preserves period class prevalence and is not a clean global null.

"""Null tests, feature-group and observed-label selection diagnostics."""
import numpy as np
import pandas as pd
from dataclasses import replace
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from .config import FOLDS
from .evaluation import ap
from .models import fit_tab
from .preprocessing import preprocess
from .utils import dump


# ---------------------------------------------------------
#  Shuffle By Period
# WHY: Keep period class counts fixed while breaking transaction-label correspondence.
# ---------------------------------------------------------
# FUNCTION: _shuffle_by_period — Destroy within-period assignment while preserving period-level prevalence/distribution.
# RESEARCH RULE: Different shuffles test different hypotheses; preserve interpretation labels.
def _shuffle_by_period(values, periods, rng):
    """Independently permute the given vector *within each time period*."""
    # RESEARCH RATIONALE:
    # Permutation destroys within-period assignments but preserves each period distribution.
    result = np.array(values, copy=True)
    # LOOP (np.unique(periods)): Repeat the same reproducible procedure for every selected fold, model or period.
    for t in np.unique(periods):
        loc = np.flatnonzero(periods == t)
        # IMPORTANT: Break the targeted label/feature association for a negative-control comparison.
        result[loc] = rng.permutation(result[loc])
    return result


# ---------------------------------------------------------
#  Cohort Signature
# WHY: Construct an unsupervised baseline that gives every transaction in one period the same score.
# ---------------------------------------------------------
# FUNCTION: _cohort_signature — Construct group-level unsupervised time-step summaries for cohort-only controls.
# RESEARCH RULE: Different shuffles test different hypotheses; preserve interpretation labels.
def _cohort_signature(d):
    """Unsupervised, current-period-only feature summaries for end-of-period BATCH.

    Contains no class labels; n and first ten feature means/stds per period.
    Summaries do not mix rows from different periods, including future periods.
    """
    # RESEARCH RATIONALE:
    # Cohort-level summaries cannot rank transactions differently inside one period.
    # This is a batch-only comparator, not an online transaction scoring signal.
    raw = np.zeros((50, 22), dtype=np.float64)
    # LOOP (range(1, 50)): Repeat the same reproducible procedure for every selected fold, model or period.
    for t in range(1, 50):
        subset = np.asarray(d.X[d.time == t, :10], dtype=np.float64)
        # IF (len(subset) == 0): Guard against missing or invalid inputs before numerical operations and reported metrics.
        if len(subset) == 0:
            raise AssertionError(f'Missing period {t}')
        subset = np.nan_to_num(subset, nan=0.0, posinf=0.0, neginf=0.0)
        raw[t] = np.r_[np.log1p(len(subset)), t / 49.0, subset.mean(0), subset.std(0)]
    return raw


# ---------------------------------------------------------
# Neg Control
# WHY: Use multiple distinct null hypotheses to separate cohort signatures from transaction ranking.
# ---------------------------------------------------------
# FUNCTION: neg_control — Run distinct null tests for label alignment, period effects, and within-period ranking.
# RESEARCH RULE: Different shuffles test different hypotheses; preserve interpretation labels.
def neg_control(d, args, out):
    """Separates true null checks from period-prevalence / batch-cohort probes.

    Shuffling train labels only within time preserves EACH period's class rate.
    High AP on unshuffled evaluation labels is not a clean null failure.
    Compare also independent evaluation shuffles, and permutations of *scores*
    within evaluation period, which retain its prevalence while destroying the
    transaction-specific association. Nothing from 35..49 enters this function.
    """
    # RESEARCH RATIONALE:
    # Global training-label shuffle and within-period shuffle test DIFFERENT confounding hypotheses.
    # A high within-period shuffled-label AP can still reflect period class prevalence.
    # Evaluation-score shuffling is the targeted test of within-period ranking signal.
    # IMPORTANT: Use an explicit local random generator rather than hidden global stochastic state.
    rng = np.random.default_rng(991)
    rows = []
    within_eval = []
    period_only = []
    null_rows = []
    reps = 4 if args.smoke else 16 if args.quick else args.control_reps
    train_reps = 2 if args.smoke else 4 if args.quick else 5
    cohort = _cohort_signature(d)
    # LOOP (enumerate(FOLDS, 1)): Repeat the same reproducible procedure for every selected fold, model or period.
    for fold, (a, b, c, e, f, g) in enumerate(FOLDS, 1):
        tr = d.ix(a, b)
        ev = d.ix(f, g)
        train_y = d.y[tr].copy()
        eval_y = d.y[ev].copy()
        train_t = d.time[tr]
        eval_t = d.time[ev]
        xx, _, _ = preprocess(d, b, np.arange(93))
        prevalence = float(np.mean(eval_y == 1))

        # FUNCTION: fit_fixed — Implement fit_fixed in the research workflow.
        # RESEARCH RULE: Different shuffles test different hypotheses; preserve interpretation labels.
        def fit_fixed(y_train, features, seed):
            # IMPORTANT: Keep controlled randomness reproducible across repeated experimental comparisons.
            model = LGBMClassifier(n_estimators=25 if args.smoke else 120, learning_rate=0.05, num_leaves=20, max_depth=5, min_child_samples=25, random_state=seed, verbosity=-1, n_jobs=2)
            # IMPORTANT: Train the current model only on the selected historical known-label indices.
            model.fit(features[tr], y_train)
            # IMPORTANT: Extract positive-class scores for ranking; thresholds are chosen separately.
            return model.predict_proba(features[ev])[:, 1]

        # FUNCTION: record — Implement record in the research workflow.
        # RESEARCH RULE: Different shuffles test different hypotheses; preserve interpretation labels.
        def record(kind, repeat, pred, truth=eval_y, interpretation=''):
            result = ap(truth, pred)
            rows.append({'fold': fold, 'type': kind, 'repeat': repeat, 'AP': result, 'eval_prevalence': float(np.mean(truth == 1)), 'AP_over_prevalence': float(result / max(1e-12, np.mean(truth == 1))), 'interpretation': interpretation, 'role': 'DEVELOPMENT_ONLY_NOT_EXPOSED_TEST'})
            return result
        real_scores = fit_fixed(train_y, xx, 101)
        real_ap = record('real_labels', 0, real_scores, interpretation='actual transaction-level model')
        record('constant_training_prevalence', 0, np.full(len(ev), float(np.mean(train_y))), interpretation='constant score; AP equals evaluation positive prevalence')
        ts = d.time.astype('float32')[:, None]
        record('time_only_TREE_extrapolation_limited', 0, fit_fixed(train_y, ts, 101), interpretation='tree extrapolation limitation; NOT decisive evidence against time effects')
        time_x = np.column_stack((d.time / 49.0, (d.time / 49.0) ** 2)).astype('float32')
        # IMPORTANT: Keep controlled randomness reproducible across repeated experimental comparisons.
        time_logit = LogisticRegression(C=0.1, max_iter=500, random_state=101)
        time_logit.fit(time_x[tr], train_y)
        # IMPORTANT: Extract positive-class scores for ranking; thresholds are chosen separately.
        record('time_only_LOGISTIC_extrapolating', 0, time_logit.predict_proba(time_x[ev])[:, 1], interpretation='time-only extrapolating parametric baseline')
        ts_train = np.unique(train_t)
        rates = np.array([float(np.mean(train_y[train_t == t])) for t in ts_train])
        # IMPORTANT: Create a scaler whose fit is restricted to the historical training window.
        cohort_scaler = StandardScaler().fit(cohort[ts_train])
        # IMPORTANT: Keep controlled randomness reproducible across repeated experimental comparisons.
        ridge = Ridge(alpha=20.0, random_state=101)
        ridge.fit(cohort_scaler.transform(cohort[ts_train]), rates)
        cohort_predictions = np.clip(ridge.predict(cohort_scaler.transform(cohort[eval_t])), 1e-06, 1 - 1e-06)
        cohort_ap = record('cohort_signature_BATCH', 0, cohort_predictions, interpretation='only current period unlabeled feature summaries; one score per time-step')
        period_only.append({'fold': fold, 'method': 'cohort_signature_BATCH', 'AP': cohort_ap, 'reference_AP': real_ap, 'prevalence': prevalence})
        # LOOP (range(1, reps + 1)): Repeat the same reproducible procedure for every selected fold, model or period.
        for rep in range(1, reps + 1):
            # IF (rep <= train_reps): Handle this alternate branch explicitly so that the experiment remains deterministic and auditable.
            if rep <= train_reps:
                # IMPORTANT: Break the targeted label/feature association for a negative-control comparison.
                train_shuf_global = rng.permutation(train_y)
                score_global = fit_fixed(train_shuf_global, xx, 101 + rep)
                record('shuffled_training_labels_GLOBAL', rep, score_global, interpretation='training-only global null; evaluation is real')
                train_shuf_period = _shuffle_by_period(train_y, train_t, rng)
                score_period = fit_fixed(train_shuf_period, xx, 101 + rep)
                record('shuffled_training_labels_WITHIN_PERIOD', rep, score_period, interpretation='period class rates preserved; group confounding probe, NOT a strict null')
            # IMPORTANT: Break the targeted label/feature association for a negative-control comparison.
            y_eval_global = rng.permutation(eval_y)
            y_eval_period = _shuffle_by_period(eval_y, eval_t, rng)
            record('real_model_EVAL_LABELS_GLOBAL_NULL', rep, real_scores, y_eval_global, interpretation='global score-label null in evaluation; NOT for threshold/model selection')
            record('real_model_EVAL_LABELS_PERIOD_NULL', rep, real_scores, y_eval_period, interpretation='conditional within-period null; preserves actual evaluation class rate by step')
            p_period = _shuffle_by_period(real_scores, eval_t, rng)
            null_ap = record('real_model_EVAL_SCORES_PERIOD_NULL', rep, p_period, interpretation='preserves period score distribution & class rate, destroys within-period ranking')
            within_eval.append({'fold': fold, 'repeat': rep, 'real_AP': real_ap, 'within_period_score_permutation_AP': null_ap, 'AP_lift_over_conditional_null': real_ap - null_ap, 'eval_prevalence': prevalence})
            # IF (rep <= train_reps): Handle this alternate branch explicitly so that the experiment remains deterministic and auditable.
            if rep <= train_reps:
                score_joint = ap(y_eval_period, score_period)
                null_rows.append({'fold': fold, 'repeat': rep, 'joint_conditional_null_AP': score_joint, 'train_labels_permuted_within_period': True, 'eval_labels_permuted_independently_within_period': True, 'eval_prevalence': prevalence})
        print(f'NEG_CONTROL DEV{fold}: real_AP={real_ap:.4f}; batch_cohort_AP={cohort_ap:.4f}', flush=True)
    df = pd.DataFrame(rows)
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    df.to_csv(out / 'DEV_NEGATIVE_CONTROLS.csv', index=False)
    agg = df.groupby('type').agg(mean_AP=('AP', 'mean'), min_AP=('AP', 'min'), max_AP=('AP', 'max'), n=('AP', 'size')).reset_index()
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    agg.to_csv(out / 'DEV_NEGATIVE_CONTROL_SUMMARY.csv', index=False)
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame(within_eval).to_csv(out / 'DEV_CONDITIONAL_SCORE_NULL.csv', index=False)
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame(period_only).to_csv(out / 'DEV_COHORT_ONLY_BASELINE.csv', index=False)
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame(null_rows).to_csv(out / 'DEV_TRAIN_AND_EVAL_CONDITIONAL_NULL.csv', index=False)
    cautions = {'purpose': 'separate transaction-specific signal from time/selection confounding', 'not_a_p_value': 'These permutations and overlapping development fits are descriptive; do not claim formal independent p-values.', 'why_within_training_shuffle_can_be_high': 'Class prevalence by period is preserved; model can learn cohort fingerprints and rank periods.', 'proper_within_period_null': 'real_model_EVAL_SCORES_PERIOD_NULL destroys transaction-specific score alignment within each evaluation period, preserving cohort score distribution.', 'cohort_baseline': 'Current-period batch unsupervised means/stds; a single class-rate score per period, fit on TRAIN periods only.', 'no_exposed_period_used': 'All training and evaluation here terminate in period 25', 'evaluation_null_repetitions': reps, 'training_shuffle_repetitions': train_reps}
    # IMPORTANT: Write structured experiment metadata for independent audit.
    dump(out / 'NEGATIVE_CONTROL_INTERPRETATION.json', cautions)
    return agg


# ---------------------------------------------------------
# Feature Group Development
# WHY: Compare different feature sets using only non-exposed development folds.
# ---------------------------------------------------------
# FUNCTION: feature_group_development — Evaluate feature-group baselines on development periods only.
# RESEARCH RULE: Different shuffles test different hypotheses; preserve interpretation labels.
def feature_group_development(d, args, out):
    """Supplemental 17 blockchain variables vs 93 local, train/evaluate ONLY dev folds.

    Never include inherited 72 pre-aggregated neighbor features, which could
    defeat a clean graph/no-graph attribution comparison.
    """
    # RESEARCH RATIONALE:
    # Vary feature groups only on development folds; do not tune on exposed diagnostics.
    # IF (d.X_extra is None): Guard against missing or invalid inputs before numerical operations and reported metrics.
    if d.X_extra is None:
        raise AssertionError('Missing 17 additional Elliptic++ variables')
    from dataclasses import replace
    full = np.concatenate([d.X, d.X_extra], axis=1)
    variants = {'93_local': np.arange(93), '110_local_plus_17_extra': np.arange(110), '17_extra_only': np.arange(93, 110)}
    rows = []
    # LOOP (enumerate(FOLDS, 1)): Repeat the same reproducible procedure for every selected fold, model or period.
    for fold, (a, b, c, e, f, g) in enumerate(FOLDS, 1):
        tr = d.ix(a, b)
        st = d.ix(c, e)
        ev = d.ix(f, g)
        dfull = replace(d, X=full)
        # LOOP (variants.items()): Repeat the same reproducible procedure for every selected fold, model or period.
        for label, keep in variants.items():
            x, _, _ = preprocess(dfull, b, keep)
            model, info = fit_tab('LGB', x, dfull, tr, st, args, 42)
            # IMPORTANT: Extract positive-class scores for ranking; thresholds are chosen separately.
            pp = model.predict_proba(x[ev])[:, 1]
            rows.append({'fold': fold, 'group': label, 'AP': ap(d.y[ev], pp), 'baseline_prevalence': float(np.mean(d.y[ev] == 1)), 'feature_count': len(keep), 'stop_iterations': info['iterations'], 'role': 'DEVELOPMENT_ONLY_DO_NOT_PICK_FROM_EXPOSED_TEST'})
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame(rows).to_csv(out / 'DEV_FEATURE_GROUP_COMPARISON.csv', index=False)
    return rows


# ---------------------------------------------------------
# Label Coverage Diagnostic
# WHY: Measure predictability of label observation, not evidence of the observation mechanism.
# ---------------------------------------------------------
# FUNCTION: label_coverage_diagnostic — Study whether a transaction is labeled, not whether it is illicit.
# RESEARCH RULE: Different shuffles test different hypotheses; preserve interpretation labels.
def label_coverage_diagnostic(d, args, out):
    """Classified vs unknown is an observed selection mechanism, NOT fraud class."""
    # RESEARCH RATIONALE:
    # Observation vs unknown is a different target from illicit vs licit.
    # A predictive observation model indicates selection, not its causal mechanism.
    from dataclasses import replace
    new = replace(d, y=(d.y >= 0).astype('int8'))
    rows = []
    # LOOP (enumerate(FOLDS, 1)): Repeat the same reproducible procedure for every selected fold, model or period.
    for fold, (a, b, c, e, f, g) in enumerate(FOLDS, 1):
        tr = np.flatnonzero((d.time >= a) & (d.time <= b))
        ev = np.flatnonzero((d.time >= f) & (d.time <= g))
        x, _, _ = preprocess(new, b, np.arange(93))
        # IMPORTANT: Keep controlled randomness reproducible across repeated experimental comparisons.
        mod = LGBMClassifier(n_estimators=30 if args.smoke else 100, random_state=42, learning_rate=0.05, num_leaves=20, max_depth=5, n_jobs=2, verbosity=-1)
        mod.fit(x[tr], new.y[tr])
        # IMPORTANT: Extract positive-class scores for ranking; thresholds are chosen separately.
        scores = mod.predict_proba(x[ev])[:, 1]
        yy = new.y[ev]
        rows.append({'fold': fold, 'n': len(yy), 'known_label_fraction': float(yy.mean()), 'AP': ap(yy, scores), 'ROC_AUC': float(roc_auc_score(yy, scores)), 'interpretation': 'Predictability of label observation != proof of missing-not-at-random mechanism'})
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame(rows).to_csv(out / 'DEV_LABEL_OBSERVATION_BIAS.csv', index=False)
    return rows
