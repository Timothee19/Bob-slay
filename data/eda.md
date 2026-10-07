# EDA: Parkinson's Disease True-OFF Motor Score

_Generated from `data/eda.py` on 2025-10-07._

## Dataset at a glance

- **Tables:** 3 (X_train, y_train, X_test — merged into a single 14-column train table for analysis)
- **Shape:** 44,590 visits × 14 columns (train), 11,013 visits × 13 columns (test)
- **Patients:** 5,576 distinct train patients, 1,395 distinct test patients (holdout is by patient — no overlap)
- **Target:** `target` — continuous regression (true debiased OFF MDS-UPDRS motor score)
- **Rich reports:** [eda_train.html](eda_train.html) · [eda_test.html](eda_test.html)

## Per-column findings

| Column | Dtype | Missing % | Unique |
|---|---|---|---|
| Index | Int64 | 0% | 44,590 (row id) |
| patient_id | String | 0% | 5,576 |
| cohort | String | 0% | 2 |
| sexM | Int64 | 0% | 2 |
| gene | String | **32.4%** | 4 values |
| age_at_diagnosis | Float64 | 5.2% | 563 |
| age | Float64 | 0% | 1,082 |
| ledd | Float64 | **36.6%** | 1,320 |
| time_since_intake_on | Float64 | **46.4%** | 64 |
| time_since_intake_off | Float64 | **78.8%** | 178 |
| rater_id | String | 0% | 50 |
| on | Float64 | **29.6%** | 83 |
| off | Float64 | **42.4%** | 101 |
| target | Float64 | 0% | 867 |

**Key missingness observations:**
- `time_since_intake_off` is missing in 78.8% of visits — the majority of visits never had an OFF exam timed.
- `time_since_intake_on` is missing in 46.4% — many visits lack ON timing data too.
- `ledd` (levodopa equivalent daily dose) is missing in 36.6%, `off` in 42.4%, `on` in 29.6%.
- `gene` is missing in ~32% — likely structural (some cohorts don't collect genotype).
- `target` has 0% missingness — it is always available in train.

Missing values in `on`, `off`, `ledd`, and timing columns should be treated as signal (the fact of missingness is informative: an absent `off` likely means no formal OFF exam was conducted that visit).

## Target

- **Task:** regression (continuous score)
- **Range in data:** 0.0 – 109.5 (theoretical max is 132; the upper tail is not reached in this dataset)
- **Mean:** 37.47, **Median:** 37.3, **SD:** 16.50
- **IQR:** 25.6 – 49.3 (range ≈ 23.7 points)
- **Distribution:** roughly symmetric, slightly right-skewed (histogram peaks around 30–50 and falls off toward higher scores). No extreme skew; a log transform is likely unnecessary but could be tested.
- **Histogram bin counts:** majority of visits fall in 21–54 range (the three largest bins sum to ~26,300 of 44,590 visits).

The standard deviation of 16.5 is the dummy-mean RMSE floor — any real model must beat it.

## Structure

- **No datetime columns** detected in the merged train table (no timestamps).
- **Visits per patient (train):** mean = 8.0, median = 7.0, max = 12. Patients contribute multiple rows; this is a repeated-measures / longitudinal structure.
- **`patient_id`** has a unique-ratio of 0.125 — 5,576 patients × ~8 visits = 44,590 rows. The Kaggle holdout is by patient, which means a random row split will leak patient-level information across folds. **`GroupKFold` on `patient_id` is required for proper CV.**
- **`Index`** is a unique row identifier (unique-ratio = 1.0). Must be excluded from features.
- **`rater_id`** has 50 unique values — the clinician who scored the visit. Potential source of systematic bias in measured scores; worth including as a categorical feature.

## Associations

**With `target` (Pearson):**

| Feature | Pearson r | Cramér V |
|---|---|---|
| `off` | **0.871** | 0.406 |
| `on` | **0.669** | 0.368 |
| `ledd` | 0.298 | 0.236 |
| `age` | 0.310 | 0.114 |
| `time_since_intake_on` | ~0.000 | 0.177 |
| `time_since_intake_off` | ~0.008 | 0.092 |
| `age_at_diagnosis` | 0.133 | 0.051 |

**Potential leakage flag:** `off` (Pearson r = 0.871) is very strongly correlated with `target`. This is expected clinically — the measured OFF score is a biased version of the true OFF — but the model needs `off` as a feature to debias it. This is **not leakage**: `off` is available at inference time. However, any model that simply learns `target ≈ off` will not generalise beyond the drug-timing and rater biases present in the train set. A useful model must learn to correct for these biases.

**Notable feature-feature links:**
- `age_at_diagnosis` ↔ `age`: Pearson r = 0.942, Cramér V = 0.562 — near-collinear (both measure disease timeline). `age - age_at_diagnosis` = years since diagnosis is a more informative derived feature.
- `on` ↔ `off`: r = 0.872 — strong co-movement; both present as noisy proxies of true disease severity.
- `time_since_intake_on` ↔ `on`: r = −0.218 — negative association: the longer since intake, the lower the ON score (drug wearing off).

## Modelling implications

- **Splitter:** `GroupKFold` on `patient_id` is non-negotiable for CV. Random splits will overestimate generalisation by leaking the same patient's visits into both train and validation sets. A group-aware split is the minimum for meaningful cross-validation.
- **Metric:** RMSE is appropriate for this continuous regression task. The dummy-mean RMSE ≈ 16.5 is the floor.
- **Missing data:** `on`, `off`, `ledd`, `time_since_intake_on`, `time_since_intake_off`, and `gene` have substantial missingness. A model must either impute them or use estimators that handle NaN natively (e.g. `HistGradientBoostingRegressor`). Missingness indicators for `off`, `on`, and timing columns are likely informative features.
- **Features to exclude:** `Index` (row identifier), `patient_id` (leaks the group identity as a numeric feature — must be used only as the grouping key). `rater_id` should be included as a categorical feature (rater-level bias).
- **Derived feature candidate:** `age - age_at_diagnosis` = time since diagnosis (years). Captures disease progression better than `age` alone, and avoids the collinearity between `age` and `age_at_diagnosis`.
- **Key features:** `off`, `on`, `ledd`, `age` (or `time_since_diagnosis`), and the timing delays (`time_since_intake_on`, `time_since_intake_off`).
- **`time_since_intake_on`** has near-zero Pearson r with `target` (0.00023) but non-trivial Cramér V (0.177) — the relationship with `target` is non-linear. A tree-based model will capture this.

## Open questions

1. **`time_since_intake_on` vs `time_since_intake_off`**: the 64 unique values of `time_since_intake_on` vs 178 for `time_since_intake_off` suggests the ON timing is more discretised (fixed protocol slots). Is this intentional in the synthetic data?
2. **`gene` categorical values**: 4 categories with 32% missingness. Are the missing `gene` values structurally absent (a cohort that didn't collect genotype) or randomly missing? The `ledd` missingness rate varies by `cohort` — similar structural missingness by cohort may apply to `gene`.
3. **Rater bias**: `rater_id` has 50 raters. Are rater effects large enough to warrant a per-rater offset or mixed-effects treatment?
4. **Temporal ordering**: visits are indexed by `Index` (a row ID), not a date. Is `Index` a proxy for visit chronology within a patient? If so, temporal features (visit number, time between visits) could be derived.
