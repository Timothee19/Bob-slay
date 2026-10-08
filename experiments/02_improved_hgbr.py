# %% [markdown]
# # Experiment 02 — HGBR improved: no-LOO patient aggregates + native categoricals
#   + trend features + on_off_ratio
#
# **Date:** 2025-01-07
# **Goal:** Stack four improvements on top of the 01_hgbr_full_fe baseline:
#
# 1. **No LOO**: patient aggregates (mean, std, slope, intercept) now use all
#    visits of the patient within the fold — not leave-one-out.  Consistent
#    with the test regime where all patient visits are available.
#
# 2. **Native categoricals**: ``gene``, ``cohort``, and ``rater_id`` passed as
#    ``category`` dtype so HGBR finds optimal splits without ordinal forcing.
#
# 3. **Trend features**: ``patient_trend_at_age`` (personalised linear forecast
#    at this visit's age) and ``off_residual_from_trend`` (deviation from trend).
#    ``visit_rel_index`` = visit_index / patient_n_visits (normalised position).
#
# 4. **on_off_ratio instead of on_off_mean**: the mean is not useful when both
#    ON and OFF are biased in the same direction at the same visit.  The ratio
#    captures motor fluctuation independently of the absolute level.
#
# Folds are loaded from ``data/cv_folds.pkl`` (same 5 splits as exp 01).

# %%
import numpy as np
import pandas as pd
import skore
from sklearn.metrics import root_mean_squared_error

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.evaluate import get_cv_folds
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_learner

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
# ## Fixed folds (loaded from cache)

# %%
folds = get_cv_folds(X_train, y_train)
print(f"Folds loaded: {len(folds)} splits")

# %% [markdown]
# ## Cross-validation — HGBR with all improvements
#
# Manual loop so we can inspect per-fold RMSE and later apply post-processing.

# %%
learner = build_learner(backend="hgbr")

all_y_true = []
all_y_pred = []
fold_rmses = []

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    X_tr = X_train.iloc[tr_idx].reset_index(drop=True)
    y_tr = y_train.iloc[tr_idx].reset_index(drop=True)
    X_va = X_train.iloc[va_idx].reset_index(drop=True)
    y_va = y_train.iloc[va_idx].reset_index(drop=True)

    learner.fit(X_tr, y_tr)
    preds = learner.predict(X_va)

    rmse = root_mean_squared_error(y_va, preds)
    fold_rmses.append(rmse)
    all_y_true.append(y_va.values)
    all_y_pred.append(preds)
    print(f"  fold {fold_idx}: RMSE = {rmse:.4f}")

overall_rmse = root_mean_squared_error(
    np.concatenate(all_y_true), np.concatenate(all_y_pred)
)
print(f"\nOverall RMSE : {overall_rmse:.4f}")
print(f"Mean fold RMSE: {np.mean(fold_rmses):.4f} +/- {np.std(fold_rmses):.4f}")
print(f"Dummy RMSE    : {y_train.std():.4f}")

# %% [markdown]
# ## Persist report to skore

# %%
report = skore.CrossValidationReport(
    build_learner(backend="hgbr"),
    X=X_train,
    y=y_train,
    splitter=folds,
    n_jobs=-1,
)
project.put("02_improved_hgbr", report)
print("Report saved to skore hub as '02_improved_hgbr'")

rmse_df = report.metrics.rmse()
print("\nRMSE per fold (skore):")
print(rmse_df)

# %% [markdown]
# ## Submission: refit on all training data, predict test set

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset(DATA_DIR)

final_model = build_learner(backend="hgbr")
final_model.fit(X_train, y_train)
test_predictions = final_model.predict(X_test)

submission = X_test[["Index"]].copy()
submission["target"] = test_predictions
submission.to_csv(submissions_dir / "02_improved_hgbr.csv", index=False)
print(f"Submission written: {len(submission)} rows")
print(submission.head())
