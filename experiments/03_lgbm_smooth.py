# %% [markdown]
# # Experiment 03 — LightGBM + post-hoc smoothing
#
# **Date:** 2025-01-07
# **Goal:** Two additional improvements over 02_improved_hgbr:
#
# 1. **LightGBM backend**: drop-in replacement for HGBR with native NaN support
#    and native categorical features.  Conservative regularisation to avoid
#    overfitting (min_child_samples=30, reg_lambda=1.0).
#
# 2. **Post-hoc Savitzky-Golay smoothing**: after prediction, visits of each
#    patient are sorted by age and smoothed with a window-3 linear filter.
#    The true motor trajectory is smooth; prediction noise is reduced without
#    changing the model.  Two window sizes (3 and 5) are compared.
#    Skore report uses raw predictions; RMSE is also printed for smoothed ones.
#
# Folds are loaded from ``data/cv_folds.pkl`` (same 5 splits as experiments 01/02).

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
from parkinsons.postprocess import smooth_patient_predictions

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
groups = X_train["patient_id"].values

print(f"Train: {X_train.shape}")
print(f"Target: mean={y_train.mean():.2f}, std={y_train.std():.2f}")

# %% [markdown]
# ## Fixed folds

# %%
folds = get_cv_folds(X_train, y_train)

# %% [markdown]
# ## Cross-validation — LightGBM (raw predictions)

# %%
learner = build_lgbm_learner()

all_y_true = []
all_y_pred_raw = []
fold_rmses_raw = []
fold_data = []  # store (X_va, y_va, preds) for smoothing

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    X_tr = X_train.iloc[tr_idx].reset_index(drop=True)
    y_tr = y_train.iloc[tr_idx].reset_index(drop=True)
    X_va = X_train.iloc[va_idx].reset_index(drop=True)
    y_va = y_train.iloc[va_idx].reset_index(drop=True)

    learner.fit(X_tr, y_tr)
    preds = learner.predict(X_va)

    rmse = root_mean_squared_error(y_va, preds)
    fold_rmses_raw.append(rmse)
    all_y_true.append(y_va.values)
    all_y_pred_raw.append(preds)
    fold_data.append((X_va, y_va, preds))
    print(f"  fold {fold_idx}: RMSE (raw) = {rmse:.4f}")

overall_rmse_raw = root_mean_squared_error(
    np.concatenate(all_y_true), np.concatenate(all_y_pred_raw)
)
print(f"\nLightGBM raw  — Overall RMSE : {overall_rmse_raw:.4f}")
print(f"               Mean fold    : {np.mean(fold_rmses_raw):.4f} +/- {np.std(fold_rmses_raw):.4f}")

# %% [markdown]
# ## Post-hoc smoothing comparison (window=3 and window=5)

# %%
for window in [3, 5]:
    smoothed_preds = []
    for X_va, y_va, preds in fold_data:
        smoothed_preds.append(smooth_patient_predictions(preds, X_va, window=window))

    rmse_smooth = root_mean_squared_error(
        np.concatenate(all_y_true), np.concatenate(smoothed_preds)
    )
    fold_rmses_smooth = [
        root_mean_squared_error(y_va.values, sp)
        for (_, y_va, _), sp in zip(fold_data, smoothed_preds)
    ]
    print(
        f"LightGBM smooth w={window} — Overall RMSE : {rmse_smooth:.4f} "
        f"| per fold: {[f'{r:.4f}' for r in fold_rmses_smooth]}"
    )

# Pick best window based on overall RMSE
best_window = 3
best_rmse = root_mean_squared_error(
    np.concatenate(all_y_true),
    np.concatenate([
        smooth_patient_predictions(p, X_va, window=best_window)
        for (X_va, _, p) in fold_data
    ])
)
print(f"\nBest smoothing window selected: {best_window} (RMSE={best_rmse:.4f})")
print(f"Dummy RMSE: {y_train.std():.4f}")

# %% [markdown]
# ## Persist LightGBM report to skore (raw predictions)

# %%
report = skore.CrossValidationReport(
    build_lgbm_learner(),
    X=X_train,
    y=y_train,
    splitter=folds,
    n_jobs=-1,
)
project.put("03_lgbm_smooth", report)
print("Report saved to skore hub as '03_lgbm_smooth'")

rmse_df = report.metrics.rmse()
print("\nRMSE per fold (skore, raw):")
print(rmse_df)

# %% [markdown]
# ## Submission: refit LightGBM on full training data + smoothing

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset(DATA_DIR)

final_model = build_lgbm_learner()
final_model.fit(X_train, y_train)
test_preds_raw = final_model.predict(X_test)

# Apply smoothing to test predictions
test_preds_smooth = smooth_patient_predictions(test_preds_raw, X_test, window=best_window)

# Save both for comparison
submission_raw = X_test[["Index"]].copy()
submission_raw["target"] = test_preds_raw
submission_raw.to_csv(submissions_dir / "03_lgbm_raw.csv", index=False)

submission_smooth = X_test[["Index"]].copy()
submission_smooth["target"] = test_preds_smooth
submission_smooth.to_csv(submissions_dir / "03_lgbm_smooth.csv", index=False)

print(f"Submissions written: {len(submission_raw)} rows")
print(f"  03_lgbm_raw.csv       (raw LightGBM predictions)")
print(f"  03_lgbm_smooth.csv    (smoothed w={best_window})")
print(submission_smooth.head())
