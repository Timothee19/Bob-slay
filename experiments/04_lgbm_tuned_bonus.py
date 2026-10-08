# %% [markdown]
# # Experiment 04 — LightGBM bonus: ledd_velocity + tuned hyperparams + smoothing w=5
#
# **Date:** 2025-01-07
# **Goal:** Push further with two additional improvements on top of 03_lgbm_smooth:
#
# 1. **ledd_velocity**: per-patient rate of LEDD increase (mg/year) computed as
#    the slope of ``ledd ~ age``.  Captures the pace of dose escalation, which
#    reflects disease progression beyond what the current LEDD value shows.
#    NaN for untreated patients or single-visit — handled natively by LightGBM.
#
# 2. **Tuned LightGBM hyperparams**: increase ``num_leaves`` to 127 (more expressive
#    trees) while keeping strong regularisation (``min_child_samples=40``,
#    ``reg_lambda=2.0``) to avoid overfitting.  Lower ``learning_rate`` to 0.02
#    with more trees (800) to let the model converge smoothly.
#
# 3. **Smoothing w=5**: exp 03 showed w=5 slightly better than w=3 on all folds.
#
# Folds loaded from ``data/cv_folds.pkl``.

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
print(f"Train: {X_train.shape}")
print(f"Target: mean={y_train.mean():.2f}, std={y_train.std():.2f}")

# %% [markdown]
# ## Fixed folds

# %%
folds = get_cv_folds(X_train, y_train)

# %% [markdown]
# ## Cross-validation — LightGBM tuned (raw)

# %%
# Tuned: more leaves + stronger reg + lower lr + more trees
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
fold_data = []

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
print(f"\nLGBM tuned raw   — Overall RMSE : {overall_rmse_raw:.4f}")
print(f"                   Mean fold    : {np.mean(fold_rmses_raw):.4f} +/- {np.std(fold_rmses_raw):.4f}")

# %% [markdown]
# ## Post-hoc smoothing (w=5)

# %%
SMOOTH_WINDOW = 5

smoothed_preds_all = [
    smooth_patient_predictions(preds, X_va, window=SMOOTH_WINDOW)
    for (X_va, _, preds) in fold_data
]
overall_rmse_smooth = root_mean_squared_error(
    np.concatenate(all_y_true), np.concatenate(smoothed_preds_all)
)
fold_rmses_smooth = [
    root_mean_squared_error(y_va.values, sp)
    for (_, y_va, _), sp in zip(fold_data, smoothed_preds_all)
]
print(f"LGBM tuned smooth w={SMOOTH_WINDOW} — Overall RMSE : {overall_rmse_smooth:.4f}")
print(f"Per fold: {[f'{r:.4f}' for r in fold_rmses_smooth]}")
print(f"\nDummy RMSE: {y_train.std():.4f}")

# %% [markdown]
# ## Persist report to skore (raw predictions)

# %%
report = skore.CrossValidationReport(
    build_lgbm_learner(
        max_iter=800,
        learning_rate=0.02,
        min_samples_leaf=40,
        l2_regularization=2.0,
        max_leaf_nodes=127,
    ),
    X=X_train,
    y=y_train,
    splitter=folds,
    n_jobs=-1,
)
project.put("04_lgbm_tuned_bonus", report)
print("Report saved to skore hub as '04_lgbm_tuned_bonus'")

rmse_df = report.metrics.rmse()
print("\nRMSE per fold (skore, raw):")
print(rmse_df)

# %% [markdown]
# ## Submission: refit on all training data + smoothing w=5

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset(DATA_DIR)

final_model = build_lgbm_learner(
    max_iter=800,
    learning_rate=0.02,
    min_samples_leaf=40,
    l2_regularization=2.0,
    max_leaf_nodes=127,
)
final_model.fit(X_train, y_train)
test_preds_raw = final_model.predict(X_test)
test_preds_smooth = smooth_patient_predictions(test_preds_raw, X_test, window=SMOOTH_WINDOW)

submission = X_test[["Index"]].copy()
submission["target"] = test_preds_smooth
submission.to_csv(submissions_dir / "04_lgbm_tuned_smooth.csv", index=False)
print(f"Submission written: {len(submission)} rows")
print(submission.head())
