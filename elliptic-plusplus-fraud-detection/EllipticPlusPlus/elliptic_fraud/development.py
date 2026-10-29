# MODULE: development.py
# PURPOSE: Chronological walk-forward development folds
# RESEARCH RULE: Development ranking never uses previously exposed periods 35-49.

"""Chronological development folds and paired graph comparisons."""
import time
import numpy as np
import pandas as pd
import torch
from .config import FOLDS, MODELS, GRAPH_PAIRS
from .preprocessing import preprocess
from .evaluation import ap
from .models import fit_predict


# ---------------------------------------------------------
# Run Dev
# WHY: Evaluate fit/early-stop/test windows in temporal order.
# ---------------------------------------------------------
# FUNCTION: run_dev — Evaluate candidates on three walk-forward historical folds; save paired transaction scores.
# WHY: Historical walk-forward evaluation is separate from the exposed diagnostic window.
def run_dev(d, args, out, keep, models=MODELS, prefix=''):
    """Walk-forward development evaluation, keeping each train/stop/eval disjoint.

    The saved OOF table can later be paired by transaction ID, fold and period.
    """
    # RESEARCH RATIONALE:
    # Fit preprocessing independently for each development fold to prevent future-statistic leakage.
    # Save transaction-level out-of-fold scores for paired comparisons on the SAME nodes.
    rows = []
    oof = []
    # LOOP (enumerate(FOLDS, 1)): Repeat the same reproducible procedure for every selected fold, model or period.
    for fold, (a, b, c, e, f, g) in enumerate(FOLDS, 1):
        tr = d.ix(a, b)
        stop = d.ix(c, e)
        ev = d.ix(f, g)
        x, _, _ = preprocess(d, b, keep)
        cache = {}
        # LOOP (models): Repeat the same reproducible procedure for every selected fold, model or period.
        for name in models:
            start = time.time()
            # IMPORTANT: Use the shared interface so tabular and neural models follow the same time protocol.
            mod, pr, info = fit_predict(name, d, x, tr, stop, ev, args, 42, cache)
            rows.append({'fold': fold, 'model': name, 'early_stop_AP': info['stop_ap'], 'eval_AP': ap(d.y[ev], pr), 'prevalence': float(np.mean(d.y[ev] == 1)), 'eval_n': len(ev), 'iterations': info['iterations'], 'seconds': time.time() - start})
            oof.extend([{'id': d.ids[idx], 'time_step': int(d.time[idx]), 'fold': fold, 'model': name, 'known_label': int(d.y[idx]), 'score': float(score)} for idx, score in zip(ev, pr)])
            print(f"DEV{fold} {prefix}{name}: AP={ap(d.y[ev], pr):.5f} stop={info['stop_ap']:.5f}", flush=True)
            del mod
        # IF (torch.cuda.is_available()): Only select accelerated execution when requested and available; keep CPU behavior usable.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    df.to_csv(out / (prefix + 'DEV_FOLDS.csv'), index=False)
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame(oof).to_csv(out / (prefix + 'DEV_OOF_PREDICTIONS.csv'), index=False)
    agg = df.groupby('model').agg(folds=('fold', 'nunique'), mean_AP=('eval_AP', 'mean'), min_AP=('eval_AP', 'min'), sd_AP=('eval_AP', 'std')).reset_index().sort_values('mean_AP', ascending=False)
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    agg.to_csv(out / (prefix + 'DEV_SUMMARY.csv'), index=False)
    # IF (not (agg.folds == 3).all()): Enforce the chronological scope here; validation and final diagnostics must not feed earlier fitting stages.
    if not (agg.folds == 3).all():
        raise AssertionError('Missing development folds')
    return (agg, pd.DataFrame(oof))


# ---------------------------------------------------------
# Paired Graph Boot
# WHY: Descriptive interval; overlapping training histories break independence assumptions.
# ---------------------------------------------------------
# FUNCTION: paired_graph_boot — Perform descriptive block-level bootstrap on matched graph/no-graph predictions.
# WHY: Historical walk-forward evaluation is separate from the exposed diagnostic window.
def paired_graph_boot(oof, args, out, graph='SAGE_BATCH', zero='SAGE_NO_GRAPH', suffix=''):
    """Resample evaluation periods to describe paired AP difference.

    Fold training sets overlap, so this interval is DESCRIPTIVE and should
    never be advertised as an independent prospective significance test.
    """
    # RESEARCH RATIONALE:
    # Resample fold/period blocks rather than transactions because samples in a period are dependent.
    # Intervals are descriptive: overlapping fit histories and observed-label bias remain.
    a = oof[oof.model == graph].set_index(['id', 'time_step', 'fold'])
    b = oof[oof.model == zero].set_index(['id', 'time_step', 'fold'])
    df = a[['known_label', 'score']].join(b[['score']].rename(columns={'score': 'score_zero'}), how='inner')
    # IF (len(df) != len(a) or len(df) != len(b)): Guard against missing or invalid inputs before numerical operations and reported metrics.
    if len(df) != len(a) or len(df) != len(b):
        raise AssertionError('Unpaired graph comparison')
    df = df.reset_index()
    period = np.array([f'{f}:{t}' for f, t in zip(df.fold, df.time_step)])
    keys = np.unique(period)
    # IMPORTANT: Use an explicit local random generator rather than hidden global stochastic state.
    rng = np.random.default_rng(777)
    reps = 24 if args.smoke else 80 if args.quick else 250
    rows = []
    # LOOP (range(reps)): Repeat the same reproducible procedure for every selected fold, model or period.
    for i in range(reps):
        selected = rng.choice(keys, size=len(keys), replace=True)
        sel = np.concatenate([np.flatnonzero(period == key) for key in selected])
        y = df.known_label.to_numpy()[sel]
        # IF (len(np.unique(y)) < 2): Guard against missing or invalid inputs before numerical operations and reported metrics.
        if len(np.unique(y)) < 2:
            continue
        p = df.score.to_numpy()[sel]
        q = df.score_zero.to_numpy()[sel]
        rows.append({'rep': i, 'AP_graph': ap(y, p), 'AP_zero': ap(y, q), 'AP_difference': ap(y, p) - ap(y, q)})
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    pd.DataFrame(rows).to_csv(out / ('DEV_GRAPH_PAIRED_PERIOD_BOOTSTRAP' + suffix + '.csv'), index=False)
    return {'graph': graph, 'zero': zero, 'n_oof_paired': len(df), 'AP_graph': ap(df.known_label, df.score), 'AP_no_graph': ap(df.known_label, df.score_zero), 'difference': ap(df.known_label, df.score) - ap(df.known_label, df.score_zero), 'boot_95pct': np.quantile([r['AP_difference'] for r in rows], [0.025, 0.975]).tolist() if rows else [], 'caution': 'OOF folds share overlapping train windows; period bootstrap is descriptive'}


# ---------------------------------------------------------
# Graph Seed Stability
# WHY: Compare graph/no-graph models using identical seeds and fold boundaries.
# ---------------------------------------------------------
# FUNCTION: graph_seed_stability — Compare architecture-matched models with identical folds and random seeds.
# WHY: Historical walk-forward evaluation is separate from the exposed diagnostic window.
def graph_seed_stability(d, args, out, pairs=None):
    """Run paired graph/no-graph models at matched fold and random seed."""
    # RESEARCH RATIONALE:
    # Paired seeds isolate graph-input effects from part of neural initialization noise.
    # Compare matching fold and seed rather than unrelated AP runs.
    pairs = GRAPH_PAIRS if pairs is None else pairs
    rows = []
    seeds = list(range(41, 41 + args.seed_repeats))
    # LOOP (enumerate(FOLDS, 1)): Repeat the same reproducible procedure for every selected fold, model or period.
    for fold, (a, b, c, e, f, g) in enumerate(FOLDS, 1):
        train = d.ix(a, b)
        stop = d.ix(c, e)
        ev = d.ix(f, g)
        x, _, _ = preprocess(d, b, np.arange(93))
        cache = {}
        # LOOP (seeds): Repeat the same reproducible procedure for every selected fold, model or period.
        for seed in seeds:
            # LOOP (pairs): Repeat the same reproducible procedure for every selected fold, model or period.
            for graph, zero in pairs:
                # LOOP ((graph, zero)): Repeat the same reproducible procedure for every selected fold, model or period.
                for model in (graph, zero):
                    # IMPORTANT: Use the shared interface so tabular and neural models follow the same time protocol.
                    net, pr, info = fit_predict(model, d, x, train, stop, ev, args, seed, cache)
                    rows.append({'fold': fold, 'seed': seed, 'model': model, 'AP': ap(d.y[ev], pr), 'n_evaluation': len(ev), 'prevalence': float(np.mean(d.y[ev] == 1)), 'early_stop_AP': info['stop_ap']})
                    del net
        # IF (torch.cuda.is_available()): Only select accelerated execution when requested and available; keep CPU behavior usable.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
    df.to_csv(out / 'DEV_GRAPH_SEED_STABILITY.csv', index=False)
    result = []
    # LOOP (pairs): Repeat the same reproducible procedure for every selected fold, model or period.
    for graph, zero in pairs:
        dd = df[df.model.isin([graph, zero])].pivot(index=['fold', 'seed'], columns='model', values='AP')
        dd['AP_difference'] = dd[graph] - dd[zero]
        # IMPORTANT: Persist a reviewer-readable table for independent analysis and validation.
        dd.reset_index().to_csv(out / f'DEV_{graph}_PAIRED_SEED_DIFFERENCES.csv', index=False)
        result.append({'graph': graph, 'without_graph': zero, 'paired_mean_difference': float(dd.AP_difference.mean()), 'paired_min_difference': float(dd.AP_difference.min()), 'paired_max_difference': float(dd.AP_difference.max()), 'n_pairs': len(dd)})
    return result
