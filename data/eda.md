# EDA: Parkinson's UPDRS dataset

_Generated from `data/eda.py` on 2025-01-07._

## Dataset at a glance

- **Tables:** 4 CSV files — `X_train.csv`, `y_train.csv`, `X_test.csv`, `sample_submission.csv`
- **Shape:** train 44 590 × 14 (13 features + target); test 11 013 × 13
- **Target:** `target` — continuous float, regression task (UPDRS score)
- **Rich reports:** [eda_train.html](eda_train.html) · [eda_test.html](eda_test.html)

## Per-column findings

| Column | dtype | Null % | N unique | Notes |
|---|---|---|---|---|
| `Index` | Int64 | 0 % | 44 590 | Row key — unique identifier |
| `patient_id` | string | 0 % | 5 576 | Patient key; 8 rows per patient on average (44 590 / 5 576) |
| `cohort` | string | 0 % | 2 | Binary cohort label |
| `sexM` | Int64 | 0 % | 2 | Binary sex indicator |
| `gene` | string | **32.4 %** | 4 | High missingness; 4 categories |
| `age_at_diagnosis` | Float64 | **5.2 %** | 563 | Small missingness |
| `age` | Float64 | 0 % | 1 082 | Current age at visit |
| `ledd` | Float64 | **36.6 %** | 1 320 | Levodopa-equivalent daily dose; notably missing |
| `time_since_intake_on` | Float64 | **46.4 %** | 64 | Time since last "ON" intake; heavily missing |
| `time_since_intake_off` | Float64 | **78.8 %** | 178 | Time since last "OFF" intake; most missing column |
| `rater_id` | string | 0 % | 50 | Rater who scored the patient |
| `on` | Float64 | **29.6 %** | 83 | UPDRS score under medication; strong predictor |
| `off` | Float64 | **42.4 %** | 101 | UPDRS score off medication; strongest predictor |
| `target` | Float64 | 0 % | 867 | No missing values |

**Highlights:**

- `time_since_intake_off` is missing in ~79 % of rows — functionally semi-sparse.
- `time_since_intake_on` (~46 %), `ledd` (~37 %), and `gene` (~32 %) are also substantially missing. Imputation will be needed.
- `on` and `off` are also partially missing, despite being the two strongest predictors of `target`.
- `age_at_diagnosis` and `age` are highly correlated (Pearson 0.94) — near-duplicate; consider dropping one or using the difference.
- `Index` is a unique row key — not a feature.

## Target

- **Range:** 0.0 – 109.5
- **Mean:** 37.47 · **Std:** 16.50 · **Median:** 37.3
- **IQR:** 25.6 → 49.3 (range of 23.7)
- **Distribution:** roughly bell-shaped centered near 37, right-skewed tail above ~65:
  - Bin counts: 0–11 → 2 477, 11–22 → 5 641, 22–33 → 9 764, 33–44 → 10 703, 44–55 → 8 844, 55–66 → 5 203, 66–77 → 1 757, 77–88 → 180, 88–99 → 13, 99–109 → 8
  - A long right tail: very few observations above 77 (< 0.5 %).

This is a continuous regression target. No severe skew that would demand a log-transform, but the right tail could inflate RMSE for high-severity patients.

## Structure

- **No datetime columns** detected by skrub's dtype inference.
- **`patient_id`** has unique ratio 0.125 → ~8 rows per patient on average. Each patient appears multiple times (longitudinal / repeated-measures structure). Rows are NOT independent — the same patient contributes many rows, and leaking patient identity across train/validation folds will inflate validation scores.
- **`rater_id`** — 50 raters; moderate cardinality. Rater effects might inflate train-test generalization if raters are split naively.
- **`Index`** is a unique row key (ratio 1.0) — drop from features.

## Associations

Top feature↔target links (Pearson correlation / Cramér V):

| Feature | Pearson r | Cramér V |
|---|---|---|
| `off` | **0.871** | 0.406 |
| `on` | **0.669** | 0.368 |
| `ledd` | 0.298 | 0.236 |
| `age` | 0.310 | 0.114 |
| `age_at_diagnosis` | 0.133 | 0.051 |
| `cohort` | n/a | 0.090 |
| `gene` | n/a | 0.052 |
| `time_since_intake_on` | ~0 | 0.177 |
| `rater_id` | n/a | 0.019 |
| `sexM` | ~0 | 0.023 |

**Notable feature↔feature links:**

- `age_at_diagnosis` ↔ `age`: Pearson 0.94 — near-collinear. One can likely be dropped or replaced by `age - age_at_diagnosis` (disease duration).
- `on` ↔ `off`: Pearson 0.872 — measuring the same UPDRS scale in two states.
- `on` ↔ `target` and `off` ↔ `target`: both very high. No implausible-leakage flag (they are conceptually the inputs that predict the combined/other UPDRS score), but the definition of `target` relative to `on`/`off` should be confirmed.

## Modelling implications

- **Repeated-measures (patient_id) → `GroupKFold` on `patient_id`.** The most important structural finding. Rows are not independent — a patient appears ~8 times. A standard KFold will leak patient info into the validation fold, producing overly optimistic scores. Use `GroupKFold(groups=patient_id)` or `GroupShuffleSplit` to ensure all rows of a given patient land in the same fold.
- **Regression task → RMSE / MAE as primary metrics.** No stratification needed; GroupKFold handles the grouping.
- **Heavy missingness in key columns.** `off` (the best predictor) is missing in ~42 % of rows; `ledd` in ~37 %; `gene` in ~32 %. Any pipeline must include imputation or use estimators / encoders that handle NaN natively (e.g. `HistGradientBoostingRegressor`, or skrub's `tabular_pipeline`).
- **Near-collinear pair.** `age` and `age_at_diagnosis` (r = 0.94). Consider engineering `disease_duration = age - age_at_diagnosis` and dropping one.
- **`on`/`off` partial missingness risk.** These two are the strongest predictors but are absent in 30–42 % of rows. A model trained with them benefits greatly when present; when both are missing the problem is much harder. Check if missingness is MAR/MCAR.
- **`time_since_intake_on`/`off`.** Near-zero Pearson with target (0.0002 / 0.008) but non-trivial Cramér V (~0.18 / 0.09). Mostly a noise feature unless domain context clarifies.
- **`rater_id`.** 50 raters with no linear correlation to target (Cramér V 0.019). Likely ignorable, but rater bias could exist; a learner with categorical encoding may pick it up.
- **Right tail.** Very few very-high-severity cases (target > 77). RMSE may be inflated by them; consider monitoring MAE alongside RMSE.

## Open questions

1. **Target definition:** Is `target` a composite UPDRS (III?) score, or a different subscale than `on`/`off`? Knowing this clarifies whether `on`/`off` are leakage risks or legitimate features.
2. **`on`/`off` missingness mechanism:** Is the UPDRS "on" / "off" measurement simply not collected for some visits (MAR), or does missingness encode something (e.g. patient not on medication)? This affects imputation strategy.
3. **`gene` missingness:** Is it genuinely unknown for 32 % of patients, or was it not tested in one cohort? The cohort indicator may already capture this.
4. **Test/train patient overlap:** Do the 1 395 patients in X_test overlap with the 5 576 in X_train? If so, GroupKFold remains correct. If not, it is a cold-start problem.
5. **`ledd` units and meaning:** Levodopa-equivalent daily dose — confirm units are consistent across both cohorts.
