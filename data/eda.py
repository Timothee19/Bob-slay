# %% [markdown]
# # EDA: Parkinson's Disease True-OFF Motor Score
#
# Exploratory data analysis of the bobathon-esilv competition dataset,
# run before designing a model.
#
# - **Raw data** is read-only — `data/X_train.csv`, `data/y_train.csv`,
#   `data/X_test.csv` are never modified here.
# - **Outputs** go under `EDA_DIR` (the repo's `data/`): an
#   `eda_<table>.html` report per table, summarized in `eda.md`.
#
# Context: each row is a patient visit. The goal is to predict the
# debiased true OFF MDS-UPDRS motor score (`target`) from clinical
# features including drug timing, measured ON/OFF scores, and demographics.

# %%
import json
from pathlib import Path

import pandas as pd
import skrub

# This file lives in data/, so parents[1] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# EDA outputs always land here (created if missing); the raw data may
# live elsewhere.
EDA_DIR = PROJECT_ROOT / "data"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load the raw data
#
# Three tables: features for train visits, targets for train visits,
# and features for test visits (held-out patients).

# %%
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

# Merge features + target into a single train table for analysis
RAW = X_train.merge(y_train, on="Index")

{"X_train_shape": X_train.shape, "y_train_shape": y_train.shape, "X_test_shape": X_test.shape, "merged_train_shape": RAW.shape}  # noqa: B018

# %% [markdown]
# ## Table overview — merged train
#
# Per-table report (column types, distributions, associations) saved to
# `data/eda_train.html`, plus a compact per-column summary.

# %%
report = skrub.TableReport(RAW, title="train (features + target)", verbose=0)
report.write_html(EDA_DIR / "eda_train.html")

summary = json.loads(report.json())
n_rows = summary.get("n_rows")
overview = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary.get("columns", [])
]
{"n_rows": n_rows, "n_columns": len(overview), "columns": overview}  # noqa: B018

# %% [markdown]
# ## Table overview — X_test
#
# Profile of the held-out test patients.

# %%
report_test = skrub.TableReport(X_test, title="test (features only)", verbose=0)
report_test.write_html(EDA_DIR / "eda_test.html")

summary_test = json.loads(report_test.json())
n_rows_test = summary_test.get("n_rows")
overview_test = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary_test.get("columns", [])
]
{"n_rows": n_rows_test, "n_columns": len(overview_test), "test_columns": overview_test}  # noqa: B018

# %% [markdown]
# ## Target
#
# Distribution of the true OFF MDS-UPDRS motor score (`target`).
# This shapes the metric and regression modelling approach.

# %%
TARGET = "target"
target_col = next((col for col in summary.get("columns", []) if col.get("name") == TARGET), None)  # noqa: B018
target_col  # noqa: B018

# %% [markdown]
# ## Patients per group
#
# Count of distinct patients and visits distribution — key because
# the Kaggle holdout is by patient, driving the splitter choice.

# %%
patient_stats = {
    "n_patients_train": int(X_train["patient_id"].nunique()),
    "n_patients_test": int(X_test["patient_id"].nunique()),
    "n_visits_train": int(X_train.shape[0]),
    "n_visits_test": int(X_test.shape[0]),
    "visits_per_patient_train": {
        "mean": round(float(X_train.groupby("patient_id").size().mean()), 2),
        "median": float(X_train.groupby("patient_id").size().median()),
        "max": int(X_train.groupby("patient_id").size().max()),
    },
}
patient_stats  # noqa: B018

# %% [markdown]
# ## Structure signals
#
# Datetime columns and high-cardinality id / group-like columns
# (which point to grouped validation, to avoid leaking patients across folds).

# %%
datetime_cols = [
    col.get("name")
    for col in summary.get("columns", [])
    if "date" in str(col.get("dtype", "")).lower()
]
unique_ratio = sorted(
    (
        {
            "column": col.get("name"),
            "unique_ratio": (col.get("n_unique") or 0) / n_rows if n_rows else None,
        }
        for col in summary.get("columns", [])
    ),
    key=lambda r: (r["unique_ratio"] is not None, r["unique_ratio"]),
    reverse=True,
)
{"datetime_cols": datetime_cols, "top_unique_ratio": unique_ratio[:10]}  # noqa: B018

# %% [markdown]
# ## Associations
#
# Strongest pairwise column associations. Strong feature↔target links
# are candidate predictors; an implausibly perfect one is a possible
# leakage flag to call out explicitly.

# %%
assoc = skrub.column_associations(RAW)
# Same library as RAW: pandas has to_dict, polars has to_dicts.
rows = assoc.to_dicts() if hasattr(assoc, "to_dicts") else assoc.to_dict(orient="records")
target_links = [
    row
    for row in rows
    if row["left_column_name"] == TARGET or row["right_column_name"] == TARGET
]
{"with_target": target_links[:15], "strongest": rows[:10]}  # noqa: B018

# %% [markdown]
# ## Summary
#
# The findings and their modelling implications are written up in
# `data/eda.md`.
