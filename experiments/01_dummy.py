# %% [markdown]
# # Experiment 01 — Dummy mean baseline
#
# The laziest possible model: always predict the mean `target` from train.
# This establishes the error floor. Any real model must beat it.
#
# Reference key on Skore Hub: `01_dummy`
# Submission file: `submissions/01_dummy.csv`

# %%
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

visits = X_train.merge(y_train, on="Index")

FEATURE_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]

X = visits[FEATURE_COLS]
y = visits["target"]

{"train_shape": X.shape, "test_shape": X_test.shape}  # noqa: B018

# %% [markdown]
# ## Evaluate the dummy regressor
#
# `skore.evaluate` with default splitter=0.2 (random 20% holdout).
# The RMSE here is the floor: it equals the standard deviation of `target`
# when the model knows nothing about the features.

# %%
from sklearn.dummy import DummyRegressor
from skore import evaluate

dummy = DummyRegressor(strategy="mean")
report = evaluate(dummy, X, y)
report.metrics.rmse()

# %% [markdown]
# ## Push report to Skore Hub

# %%
from parkinson.hub import load_skore_credentials
from skore import Project, login

cfg = load_skore_credentials()
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
project.put("01_dummy", report)

# %% [markdown]
# ## Generate the submission file
#
# Fit the dummy on the full train set, predict on X_test,
# write `submissions/01_dummy.csv` matching `sample_submission.csv` format.

# %%
dummy_full = DummyRegressor(strategy="mean")
dummy_full.fit(X, y)

X_test_features = X_test[FEATURE_COLS]
preds = dummy_full.predict(X_test_features)

submission = pd.DataFrame({"Index": X_test["Index"], "target": preds})

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission.to_csv(submissions_dir / "01_dummy.csv", index=False)

{"submission_shape": submission.shape, "head": submission.head().to_dict(orient="records")}  # noqa: B018
