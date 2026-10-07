# JOURNAL

<!--
Durable index of every experiment in this workspace. Four sections,
in order: Status, Data understanding (EDA), History, Backlog - keep
them so the file stays quick to scan. Each journal/NN_short_name.md
design note pairs one-to-one with experiments/NN_short_name.py (same
stem).
-->

## Status

- **Project / dataset:** Parkinson's UPDRS regression (train 44 590 rows × 13 features)
- **Goal:** Predict the continuous UPDRS score (`target`) for Parkinson's patients, minimising RMSE on held-out data.
- **Last experiment:** 01_hgbr_full_fe — 2025-01-07
- **Last result:** GroupKFold RMSE 4.42 ± 0.05 (dummy 16.50, R² 0.928); published → https://skore.probabl.ai/bob-slay/ibm-hackaton/cross-validations/49686

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2025-01-07
  - env manager: pip+venv - recorded: 2025-01-07
  - agent feature: installed - recorded: 2025-01-07
  - optional features: none - recorded: 2025-01-07
  - package name (`src/<pkg>/`): <pkg> - recorded: <YYYY-MM-DD>
  - skore mode: hub - recorded: 2025-01-07
  - skore hub workspace: bob-slay - recorded: 2025-01-07
  - skore mlflow tracking uri: n/a - recorded: 2025-01-07
  - student prior: <beginner | some-sklearn | comfortable> - recorded: <YYYY-MM-DD>
  - CV splitter family: GroupKFold (patient_id) - recorded: 2025-01-07

## Data understanding (EDA)

<!--
Short index entry - the full analysis lives in data/eda.md. If the
data exploration was skipped, keep just the Status: skipped line.
-->

- **Status:** done - 2025-01-07
- **Summary:** 44 590 train rows × 13 features; continuous regression target (UPDRS score, range 0–109.5, mean ~37). Key finding: `patient_id` repeats ~8 times per patient → `GroupKFold` is required to avoid leakage. `off` (r=0.87) and `on` (r=0.67) are the two dominant predictors but carry 42 % and 30 % missing values respectively. Heavy missingness throughout (`ledd` 37 %, `gene` 32 %, `time_since_intake_off` 79 %) demands robust imputation or NaN-native estimators.
- **Report:** [data/eda.md](../data/eda.md)

## History

<!--
One row per experiment, in chronological order. Newest at the bottom.
Status values: planned | approved | running | done | abandoned.
-->

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| `01_hgbr_full_fe` | HGBR + biais rater + correction PK + agrégats LOO patient | done | RMSE 4.42 ± 0.05 (vs dummy 16.50) | n/a |

## Backlog

<!--
Ideas not yet committed to a journal/NN_*.md design note. Each row
has a stable B<N> index so it can be picked by number ("go with B2").
-->

| # | Item | Source |
|---|---|---|
