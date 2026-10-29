# Historical-only imputation and standardization
# ---------------------------------------------------------------
# PURPOSE: Fitting preprocessing on future features leaks information through the statistics.
# RESEARCH RULE: Fit on permitted historical nodes; transform the other periods without refitting.
# ---------------------------------------------------------------

"""Fit imputers and scalers only on allowed historical periods."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ---- Fit SimpleImputer and StandardScaler on the historical fit range ----
# WHY: Feature transformations for later periods reuse these fitted statistics.
# FUNCTION: preprocess — Fit median imputation/scaling only on historical feature rows; transform other periods without refitting.
# WHY: Fit historical imputer/scaler statistics before transforming future periods.
def preprocess(d,end,keep):
    # RESEARCH RATIONALE:
    # Fit median imputation and feature scaling through the historical fit cutoff.
    # Applying learned transforms to later periods is allowed; re-fitting them is leakage.
    # Keep empty columns so network input dimensions remain stable.
    train=d.ix(1,end,labeled=False)
    # IMPORTANT: Impute missing numerical features without leaking later-period statistics.
    imp=SimpleImputer(strategy='median',keep_empty_features=True)
    # IMPORTANT: Fit a transformation on the permitted historical feature rows only.
    a=imp.fit_transform(d.X[train][:,keep])
    # IF (a.shape[1]!=len(keep)): Retain every selected feature column, including all-missing historical columns.
    if a.shape[1]!=len(keep):raise AssertionError('Imputation dropped features')
    # IMPORTANT: Create a scaler whose fit is restricted to the historical training window.
    scl=StandardScaler().fit(a)
    xx=np.clip(scl.transform(imp.transform(d.X[:,keep])),-20,20).astype('float32')
    # IF (not np.isfinite(xx).all()): Reject NaNs/Infs before passing features to classifiers.
    if not np.isfinite(xx).all():raise AssertionError('Nonfinite model input')
    return xx,imp,scl
