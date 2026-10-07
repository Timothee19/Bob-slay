# %% [markdown]
# # Experiment 01 — HGBR with full feature engineering
#
# **Date:** 2025-01-07
# **Goal:** Establish a strong baseline using HistGradientBoostingRegressor with:
#   - Rater-bias correction (multiplicative b_r per rater from off/target ratio)
#   - Pharmacokinetic correction of `on` score (time-dependent ratio curve)
#   - LOO intra-split patient aggregates (patient_off_mean, patient_slope, etc.)
#   - Missingness flags (MNAR mechanism preserved)
#   - Native NaN handling by HGBR (no imputation)
# **Result:** GroupKFold RMSE ~4.42 vs dummy ~16.50 (73% reduction)

# %%
import numpy as np
import pandas as pd
import skore
from sklearn.model_selection import GroupKFold

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_learner

# %% [markdown]
# ## Project connection

# %%
cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(
    name="ibm-hackaton",
    mode="hub",
    workspace=cfg["workspace"],
)

# %% [markdown]
# ## Data

# %%
DATA_DIR = PROJECT_ROOT / "data"
X_train, y_train = load_dataset(DATA_DIR)
groups = X_train["patient_id"].values

print(f"Train: {X_train.shape}, Test: {load_test_dataset(DATA_DIR).shape}")
print(f"Target: mean={y_train.mean():.2f}, std={y_train.std():.2f}")

# %% [markdown]
# ## Cross-validation with GroupKFold
#
# GroupKFold ensures all visits of a patient land in the same fold,
# preventing patient-level leakage. The splitter is passed as a pre-computed
# generator so skore.CrossValidationReport receives the correct grouped splits.

# %%
gkf = GroupKFold(n_splits=5)

# Pre-compute split indices so skore gets the grouped splits
split_indices = list(gkf.split(X_train, y_train, groups=groups))

learner = build_learner()

report = skore.CrossValidationReport(
    learner,
    X=X_train,
    y=y_train,
    splitter=split_indices,
    n_jobs=-1,
)
report

# %% [markdown]
# ## RMSE summary

# %%
rmse_df = report.metrics.rmse()
r2_df = report.metrics.r2()
print("RMSE per fold (GroupKFold):")
print(rmse_df)
print("\nR² per fold:")
print(r2_df)
print(f"\nDummy RMSE (mean prediction): {y_train.std():.4f}")

# %% [markdown]
# ## Persist report

# %%
project.put("01_hgbr_full_fe", report)
print("Report saved to skore hub as '01_hgbr_full_fe'")

# %% [markdown]
# ## Submission: refit on all training data, predict test set

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset(DATA_DIR)

# Refit on the full training set
final_model = build_learner()
final_model.fit(X_train, y_train)
test_predictions = final_model.predict(X_test)

submission = X_test[["Index"]].copy()
submission["target"] = test_predictions
submission.to_csv(submissions_dir / "01_hgbr_full_fe.csv", index=False)
print(f"Submission written: {len(submission)} rows")
print(submission.head())
