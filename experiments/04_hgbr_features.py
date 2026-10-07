# %% [markdown]
# # Experiment 04 — HGBR with full feature engineering
#
# HistGradientBoostingRegressor with:
# - All engineered features from FeatureEngineer (visit structure, patient
#   aggregates, lag/lead, rater-bias correction, pharmacodynamic proxies)
# - GroupKFold(5) on patient_id to match the Kaggle patient holdout
# - Report pushed to Hub under key `04_hgbr_features`
# - Submission written to `submissions/04_hgbr_features.csv`

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parkinson.features import FeatureEngineer

X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test  = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

y = y_train["target"]
groups = X_train["patient_id"]

{"train_shape": X_train.shape, "test_shape": X_test.shape}  # noqa: B018

# %% [markdown]
# ## Build the pipeline
#
# FeatureEngineer needs y to learn rater biases (off - target per rater).
# We pass it explicitly via set_output and a FunctionTransformer wrapper
# so the full pipeline stays sklearn-compatible.
#
# Note: HistGradientBoostingRegressor handles NaN natively and supports
# categorical features declared via categorical_features=.

# %%
CATEGORICAL_COLS_AFTER_FE = ["cohort", "gene", "rater_id"]

hgbr = HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.05,
    max_depth=6,
    min_samples_leaf=20,
    l2_regularization=0.1,
    categorical_features=CATEGORICAL_COLS_AFTER_FE,
    random_state=42,
    verbose=0,
)

# %% [markdown]
# ## Cross-validate with GroupKFold
#
# We cannot use skore.evaluate directly here because FeatureEngineer.fit()
# needs y (for rater bias). We manually run GroupKFold and collect RMSE,
# then build a final EstimatorReport on the last fold for the Hub.

# %%
from sklearn.metrics import root_mean_squared_error

cv = GroupKFold(n_splits=5)
fold_rmses = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y, groups)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx],       y.iloc[val_idx]

    fe = FeatureEngineer()
    fe.fit(X_tr, y_tr)
    X_tr_feat  = fe.transform(X_tr)
    X_val_feat = fe.transform(X_val)

    hgbr_fold = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=0.1,
        categorical_features=CATEGORICAL_COLS_AFTER_FE,
        random_state=42,
        verbose=0,
    )
    hgbr_fold.fit(X_tr_feat, y_tr)
    preds_val = hgbr_fold.predict(X_val_feat)
    rmse = root_mean_squared_error(y_val, preds_val)
    fold_rmses.append(rmse)
    print(f"Fold {fold+1}/5  RMSE = {rmse:.4f}")

cv_rmse_mean = float(np.mean(fold_rmses))
cv_rmse_std  = float(np.std(fold_rmses))
print(f"\nCV RMSE: {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}  (dummy floor = 16.48)")
{"cv_rmse_mean": cv_rmse_mean, "cv_rmse_std": cv_rmse_std, "fold_rmses": fold_rmses}  # noqa: B018

# %% [markdown]
# ## Build a single EstimatorReport for the Hub
#
# We use skore.evaluate with splitter="prefit" on a held-out fold so we
# get the full diagnostics (prediction-error plot, feature importances…).
# The model is already fit on train_idx from the last fold above.

# %%
from skore import EstimatorReport

# Use the last fold's splits (train_idx / val_idx still in scope after the loop).
# EstimatorReport lets us pass X_train + X_test explicitly so the Hub serialises
# the report correctly (project.put requires train data to be present).
fe_last = FeatureEngineer()
fe_last.fit(X_train.iloc[train_idx], y.iloc[train_idx])
X_tr_last_feat  = fe_last.transform(X_train.iloc[train_idx])
X_val_last_feat = fe_last.transform(X_train.iloc[val_idx])

# Pass an UNFITTED estimator — EstimatorReport fits it internally on X_train/y_train.
hgbr_for_report = HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.05,
    max_depth=6,
    min_samples_leaf=20,
    l2_regularization=0.1,
    categorical_features=CATEGORICAL_COLS_AFTER_FE,
    random_state=42,
    verbose=0,
)

report = EstimatorReport(
    hgbr_for_report,
    X_train=X_tr_last_feat, y_train=y.iloc[train_idx],
    X_test=X_val_last_feat,  y_test=y.iloc[val_idx],
)
print("Report RMSE (last fold val):", report.metrics.rmse())

# %% [markdown]
# ## Push report to Skore Hub

# %%
from parkinson.hub import load_skore_credentials
from skore import Project, login

cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("04_hgbr_features", report)

# %% [markdown]
# ## Generate submission file
#
# Final model is fit on the full train set.

# %%
fe_full = FeatureEngineer()
fe_full.fit(X_train, y)
X_train_full_feat = fe_full.transform(X_train)
X_test_feat       = fe_full.transform(X_test)

hgbr_full = HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.05,
    max_depth=6,
    min_samples_leaf=20,
    l2_regularization=0.1,
    categorical_features=CATEGORICAL_COLS_AFTER_FE,
    random_state=42,
    verbose=0,
)
hgbr_full.fit(X_train_full_feat, y)
preds_test = hgbr_full.predict(X_test_feat)

submission = pd.DataFrame({"Index": X_test["Index"], "target": preds_test})
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "04_hgbr_features.csv", index=False)

{"submission_shape": submission.shape, "pred_mean": round(float(preds_test.mean()), 3), "pred_std": round(float(preds_test.std()), 3)}  # noqa: B018
