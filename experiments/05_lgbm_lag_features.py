# %% [markdown]
# # Experiment 05 — Lag features + upper-tail aggregates + poly2 smoothing
#
# **Date:** 2025-01-07
# **Result:** RMSE 3.739 (poly2 smoothed) / 3.939 (raw) — best so far
#   Skore report: https://skore.probabl.ai/bob-slay/bobathon-esilv/cross-validations/51301
#
# ## New features (added to features.py)
#
# ### Lag features — chronological within each patient's fold
# - ``off_corrected_prev``: ``off_corrected`` at the preceding visit.
#   Corr(target)=0.88, ~52% coverage. Fills the gap when current ``off``
#   is missing (those rows had RMSE 4.81 in prior experiments).
# - ``off_corrected_next``: ``off_corrected`` at the next visit. Corr=0.86.
# - ``on_corrected_prev``: ``on_corrected`` at the preceding visit.
# - ``off_cummax``: running max of ``off_corrected`` up to (and including)
#   the current visit, sorted by age. Corr(target)=0.95 — the single
#   strongest feature discovered. Encodes the monotone-progression constraint:
#   the disease never improves, so the peak ``off`` seen so far is a hard
#   lower bound on the true current OFF score.
#
# ### Upper-tail patient aggregates
# - ``patient_off_max``: global max of ``off_corrected`` per patient.
# - ``patient_off_q75``: 75th percentile. Upper quantiles are less affected
#   by occasional under-measurement than the mean.
#
# ## New post-processing (added to postprocess.py)
#
# ### Polynomial fit on age (``smooth_by_age_poly``)
# Fits a quadratic ``pred ~ age`` per patient and replaces raw predictions
# with the fitted curve. Respects irregular visit spacing (mean ~1yr, std
# ~0.5yr) unlike Savitzky-Golay which treats consecutive visits as
# equally spaced. CV RMSE comparison:
#
# | Method            | RMSE  | Delta vs raw |
# |-------------------|-------|-------------|
# | Raw               | 3.939 |      —      |
# | SG window=5       | 3.820 |    -0.119   |
# | SG window=7       | 3.834 |    -0.105   |
# | **Poly2 (age)**   |**3.739**| **-0.200** |
# | Poly3 (age)       | 3.794 |    -0.145   |
# | SG5 + Poly2       | 3.763 |    -0.176   |
#
# Poly2 wins cleanly on all 5 folds.

# %%
import numpy as np
import pandas as pd
import skore
from sklearn.metrics import root_mean_squared_error

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.evaluate import get_cv_folds
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_lgbm_learner
from parkinsons.postprocess import (
    smooth_patient_predictions,
    smooth_by_age_poly,
    smooth_combined,
)

# %% [markdown]
# ## Project connection

# %%
cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)

# %% [markdown]
# ## Data

# %%
DATA_DIR = PROJECT_ROOT / "data"
X_train, y_train = load_dataset(DATA_DIR)
print(f"Train: {X_train.shape}")
print(f"Target: mean={y_train.mean():.2f}, std={y_train.std():.2f}")

# %% [markdown]
# ## Fixed folds

# %%
folds = get_cv_folds(X_train, y_train)

# %% [markdown]
# ## Cross-validation — LightGBM + lag features

# %%
learner = build_lgbm_learner(
    max_iter=800,
    learning_rate=0.02,
    min_samples_leaf=40,
    l2_regularization=2.0,
    max_leaf_nodes=127,
)

all_y_true = []
all_y_pred_raw = []
fold_rmses_raw = []
fold_data = []   # (X_va, y_va, preds_raw)

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    X_tr = X_train.iloc[tr_idx].reset_index(drop=True)
    y_tr = y_train.iloc[tr_idx].reset_index(drop=True)
    X_va = X_train.iloc[va_idx].reset_index(drop=True)
    y_va = y_train.iloc[va_idx].reset_index(drop=True)

    learner.fit(X_tr, y_tr)
    preds = learner.predict(X_va)

    rmse = root_mean_squared_error(y_va.values, preds)
    fold_rmses_raw.append(rmse)
    all_y_true.append(y_va.values)
    all_y_pred_raw.append(preds)
    fold_data.append((X_va, y_va, preds))
    print(f"  fold {fold_idx}: RMSE (raw) = {rmse:.4f}")

overall_rmse_raw = root_mean_squared_error(
    np.concatenate(all_y_true), np.concatenate(all_y_pred_raw)
)
print(f"\nRaw  — Overall RMSE : {overall_rmse_raw:.4f}  "
      f"| mean: {np.mean(fold_rmses_raw):.4f} +/- {np.std(fold_rmses_raw):.4f}")

# %% [markdown]
# ## Smoothing comparison

# %%
def eval_smooth(fold_data, all_y_true, fn, label):
    smoothed = [fn(p, X_va) for X_va, _, p in fold_data]
    overall = root_mean_squared_error(np.concatenate(all_y_true), np.concatenate(smoothed))
    per_fold = [root_mean_squared_error(y_va.values, s) for (_, y_va, _), s in zip(fold_data, smoothed)]
    print(f"  {label:30s}: {overall:.4f}  | {[f'{r:.4f}' for r in per_fold]}")
    return smoothed, overall

print("\nSmoothing comparison:")
_, rmse_sg5   = eval_smooth(fold_data, all_y_true,
    lambda p, X: smooth_patient_predictions(p, X, window=5),          "SG window=5")
_, rmse_poly2 = eval_smooth(fold_data, all_y_true,
    lambda p, X: smooth_by_age_poly(p, X, degree=2),                  "Poly2(age)")
_, rmse_comb  = eval_smooth(fold_data, all_y_true,
    lambda p, X: smooth_combined(p, X, sg_window=5, poly_degree=2),   "SG5 + Poly2")

print(f"\nBest: Poly2 (RMSE={rmse_poly2:.4f})")
print(f"Dummy RMSE: {y_train.std():.4f}")

# %% [markdown]
# ## Persist report to skore (raw predictions)

# %%
report = skore.CrossValidationReport(
    build_lgbm_learner(max_iter=800, learning_rate=0.02, min_samples_leaf=40,
                       l2_regularization=2.0, max_leaf_nodes=127),
    X=X_train, y=y_train, splitter=folds, n_jobs=-1,
)
project.put("05_lgbm_lag_features", report)
print("Report saved: 05_lgbm_lag_features")
print(report.metrics.rmse())

# %% [markdown]
# ## Submission: refit on full training data + poly2 smoothing

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset(DATA_DIR)

final_model = build_lgbm_learner(
    max_iter=800, learning_rate=0.02, min_samples_leaf=40,
    l2_regularization=2.0, max_leaf_nodes=127,
)
final_model.fit(X_train, y_train)
test_preds_raw   = final_model.predict(X_test)
test_preds_poly2 = smooth_by_age_poly(test_preds_raw, X_test, degree=2)

sub = X_test[["Index"]].copy()
sub["target"] = test_preds_poly2
sub.to_csv(submissions_dir / "05_lgbm_lag_poly2.csv", index=False)
print(f"Submission written: {len(sub)} rows -> 05_lgbm_lag_poly2.csv")
print(sub.head())
