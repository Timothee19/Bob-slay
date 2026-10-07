# %% [markdown]
# # EDA: Parkinson's UPDRS dataset
#
# Exploratory data analysis of X_train / y_train (+ X_test), run before
# designing a model.
#
# - **Raw data** is read-only — `RAW` points at `data/X_train.csv` merged with
#   `data/y_train.csv`. This file never cleans or modifies the raw data.
# - **Outputs** go under `EDA_DIR` (the repo's `data/`): `eda_train.html`,
#   `eda_test.html`, and `eda.md`.

# %%
import json
from pathlib import Path

import pandas as pd
import skrub

# This file lives in data/, so parents[1] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# EDA outputs always land here (created if missing); the raw data lives here
# too for this dataset, but they are separate concerns.
EDA_DIR = PROJECT_ROOT / "data"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load the raw data
#
# Merge X_train with y_train on the `Index` column so the target is
# visible alongside the features. X_test is profiled separately.

# %%
X_train = pd.read_csv(PROJECT_ROOT / "data" / "X_train.csv")
y_train = pd.read_csv(PROJECT_ROOT / "data" / "y_train.csv")
X_test = pd.read_csv(PROJECT_ROOT / "data" / "X_test.csv")

RAW = X_train.merge(y_train, on="Index")
RAW.shape  # noqa: B018

# %% [markdown]
# ## Table overview — training set (features + target)
#
# Per-column report: dtype, fraction missing, number of unique values.

# %%
report = skrub.TableReport(RAW, title="train", verbose=0)
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
# ## Table overview — test set (features only)
#
# Quick profile of X_test to check for distribution shifts vs train.

# %%
report_test = skrub.TableReport(X_test, title="test", verbose=0)
report_test.write_html(EDA_DIR / "eda_test.html")

summary_test = json.loads(report_test.json())
overview_test = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("n_unique"),
    }
    for col in summary_test.get("columns", [])
]
{"n_rows": summary_test.get("n_rows"), "n_columns": len(overview_test), "columns": overview_test}  # noqa: B018

# %% [markdown]
# ## Target
#
# Distribution of the continuous `target` column — spread, skew, and
# range shape the metric and whether a target transform is warranted.

# %%
TARGET = "target"
next((col for col in summary.get("columns", []) if col.get("name") == TARGET), None)  # noqa: B018

# %% [markdown]
# ## Structure signals
#
# Datetime columns (which point to time-based validation) and
# high-cardinality id / group-like columns (which point to grouped
# validation, to avoid leaking an entity across folds).

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
# leakage flag.

# %%
assoc = skrub.column_associations(RAW)
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
