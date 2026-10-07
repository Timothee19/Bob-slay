# Experiment Journal — bobathon-esilv

## Status

- **Workspace decisions:**
  - Tabular library: pandas (2025-10-07)
  - CV splitter: GroupKFold on `patient_id` (patient holdout, no temporal ordering)
  - Primary metric: RMSE

## Data understanding (EDA)

- **Status:** done - 2025-10-07
- **Summary:** 44,590 train visits across 5,576 patients (mean 8 visits/patient, max 12). Target `target` is a continuous true-OFF MDS-UPDRS score (mean ≈ 37.5, SD ≈ 16.5). `off` (r = 0.87) and `on` (r = 0.67) are the strongest predictors. Heavy missingness in `time_since_intake_off` (79%), `ledd` (37%), `off` (42%), `on` (30%). Kaggle holdout is by patient → `GroupKFold` required.
- **Report:** [data/eda.md](../data/eda.md)

## Experiments

| # | Key | Method | CV RMSE | Notes |
|---|---|---|---|---|
| 01 | `01_dummy` | DummyRegressor (mean) | ~16.5 | Floor baseline |
| 04 | `04_hgbr_features` | HGBR + FeatureEngineer (rater bias, PD proxy, lags, aggregates) | **4.47 ± 0.09** | Bug corrigé : sort_values dans transform() désalignait X et y. Hub: [49499](https://skore.probabl.ai/bob-slay/bobathon-esilv/estimators/49499) |
