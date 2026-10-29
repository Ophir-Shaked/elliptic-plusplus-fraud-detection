# MODULE: evaluation.py
# PURPOSE: Metrics, thresholds and monotone calibration
# RESEARCH RULE: Calibration fits on 30-31, policy on 32-34; classification performance uses known labels only.

"""Classification metrics, pre-test policy selection and monotone calibration."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve


# ---------------------------------------------------------
# Ap
# WHY: Avoid reporting binary AP on a one-class slice.
# ---------------------------------------------------------
# FUNCTION: ap — Calculate average precision only for a meaningful two-class labeled sample.
# WHY: AP is threshold-free; calibration and policies use separate historical windows.
def ap(y, p):
    # RESEARCH RATIONALE:
    # A one-class slice has no meaningful binary average-precision comparison.
    return float(average_precision_score(y, p)) if len(y) > 0 and np.unique(y).size == 2 else np.nan


# ---------------------------------------------------------
# Metrics
# WHY: Recompute measures from known-label predictions and a fixed threshold.
# ---------------------------------------------------------
# FUNCTION: metrics — Report ranking and threshold-dependent metrics separately on observed labels.
# WHY: AP is threshold-free; calibration and policies use separate historical windows.
def metrics(y, p, threshold=np.nan):
    # RESEARCH RATIONALE:
    # Keep ranking metrics separate from threshold-dependent counts.
    # All denominators refer to KNOWN labels, not the entire transaction population.
    y = np.asarray(y)
    p = np.asarray(p, dtype='float64')
    n = len(y)
    # IF (n == 0): Avoid division-by-zero and undefined metrics for an empty labeled slice.
    if n == 0:
        return {'n': 0}
    prevalence = float(np.mean(y == 1))
    res = {'n': n, 'illicit': int(sum(y == 1)), 'licit': int(sum(y == 0)), 'prevalence': prevalence, 'AP': ap(y, p), 'AP_over_prevalence': ap(y, p) / prevalence if prevalence > 0 else np.nan, 'ROC_AUC': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan, 'median_positive_score': float(np.median(p[y == 1])) if any(y == 1) else np.nan, 'median_negative_score': float(np.median(p[y == 0])) if any(y == 0) else np.nan}
    # IF (np.isfinite(threshold)): Confusion counts require an explicit policy threshold; AP does not.
    if np.isfinite(threshold):
        pred = p >= threshold
        tp = int(sum(pred & (y == 1)))
        fp = int(sum(pred & (y == 0)))
        fn = int(sum(~pred & (y == 1)))
        tn = int(sum(~pred & (y == 0)))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        res.update(TP=tp, FP=fp, FN=fn, TN=tn, precision=prec, recall=rec, F1=2 * prec * rec / (prec + rec) if prec + rec > 0 else 0, FP_per_10k_licit=10000 * fp / max(1, fp + tn), threshold=float(threshold))
    return res


# ---------------------------------------------------------
# Threshold F1
# WHY: Select threshold using only policy-window labels.
# ---------------------------------------------------------
# FUNCTION: threshold_f1 — Choose the F1 policy threshold exclusively on its historical policy slice.
# WHY: AP is threshold-free; calibration and policies use separate historical windows.
def threshold_f1(y, p):
    # RESEARCH RATIONALE:
    # Choose the F1 maximizing threshold using the historical POLICY slice only.
    prec, rec, th = precision_recall_curve(y, p)
    ix = np.argmax(2 * prec[:-1] * rec[:-1] / np.maximum(1e-12, prec[:-1] + rec[:-1]))
    return float(th[ix])


# ---------------------------------------------------------
# Threshold Fpr
# WHY: Choose a historical false-positive budget; do not assume it holds later.
# ---------------------------------------------------------
# FUNCTION: threshold_fpr — Choose an observed-negative false-positive budget threshold (not a future guarantee).
# WHY: AP is threshold-free; calibration and policies use separate historical windows.
def threshold_fpr(y, p, rate=5.0):
    # RESEARCH RATIONALE:
    # The negative budget is per 10,000 known licit transactions, NOT all alerts.
    # A bound observed in the policy slice is not guaranteed after temporal shift.
    a = np.sort(np.asarray(p)[np.asarray(y) == 0])[::-1]
    # IF (len(a) == 0): Do not choose a false-positive threshold without known negative examples.
    if len(a) == 0:
        raise ValueError('No known negatives')
    allowed = int(np.floor(rate * len(a) / 10000))
    # IF (allowed >= len(a)): When all known negatives fit the budget, use the permissive threshold.
    if allowed >= len(a):
        return 0.0
    return float(np.nextafter(a[allowed], np.inf))

@dataclass
class MonotonePlatt:
    """Constrained Platt fit. Positive slope preserves score ordering."""
    intercept: float
    slope: float
    optimizer_success: bool

    # FUNCTION: MonotonePlatt.predict — Apply positive-slope Platt calibration without changing rank ordering.
    # WHY: AP is threshold-free; calibration and policies use separate historical windows.
    def predict(self, p):
        # RESEARCH RATIONALE:
        # Logit-transform clipped input scores before applying fitted affine calibration.
        z = logit(np.clip(np.asarray(p, dtype=float), 1e-06, 1 - 1e-06))
        return expit(self.intercept + self.slope * z)


# ---------------------------------------------------------
# Platt Fit
# WHY: Fit a positive-slope calibrator that preserves rank ordering.
# ---------------------------------------------------------
# FUNCTION: platt_fit — Optimize monotone logistic calibration using the separate historical calibration window.
# WHY: AP is threshold-free; calibration and policies use separate historical windows.
def platt_fit(y, p):
    # RESEARCH RATIONALE:
    # Strictly positive slope preserves score ordering even when probabilities are recalibrated.
    # Calibration uses a distinct historical period after early stopping.
    y = np.asarray(y, dtype=float)
    # IF (len(np.unique(y)) != 2): Calibration requires both known classes.
    if len(np.unique(y)) != 2:
        raise ValueError('Calibration needs both known classes')
    z = logit(np.clip(np.asarray(p, dtype=float), 1e-06, 1 - 1e-06))
    baseline = float(np.clip(y.mean(), 1e-06, 1 - 1e-06))

    # FUNCTION: objective — Implement objective in the research workflow.
    # WHY: AP is threshold-free; calibration and policies use separate historical windows.
    def objective(ab):
        scores = ab[0] + ab[1] * z
        loss = float(np.mean(np.logaddexp(0, scores) - y * scores) + 0.0001 * (ab[1] - 1) ** 2)
        delta = expit(scores) - y
        grad = np.array([np.mean(delta), np.mean(delta * z) + 0.0002 * (ab[1] - 1)])
        return (loss, grad)
    opt = minimize(objective, [float(logit(baseline)), 1.0], jac=True, method='L-BFGS-B', bounds=[(None, None), (0.01, 50.0)], options={'maxiter': 500, 'ftol': 1e-12})
    # IF (not opt.success or not np.isfinite(opt.x).all()): Fail closed if the monotone calibrator did not converge to finite parameters.
    if not opt.success or not np.isfinite(opt.x).all():
        raise RuntimeError(f'Monotone calibration optimization failed: {opt.message}')
    return MonotonePlatt(float(opt.x[0]), float(opt.x[1]), bool(opt.success))


# ---------------------------------------------------------
# Platt Apply
# WHY: Apply frozen calibration learned in a permitted historical window.
# ---------------------------------------------------------
# FUNCTION: platt_apply — Reuse a frozen calibrator on later raw scores without refitting.
# WHY: AP is threshold-free; calibration and policies use separate historical windows.
def platt_apply(platt, p):
    # RESEARCH RATIONALE:
    # Re-use the previously fitted calibrator rather than re-fitting it on new outcomes.
    return platt.predict(p)
