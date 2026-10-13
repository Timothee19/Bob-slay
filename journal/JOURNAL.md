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
- **Last experiment:** 11_ensemble3_bateman — 2025-01-07
- **Last result:** GroupKFold RMSE TBD (Bateman PK model, 81 features, 3-model ensemble); published → https://skore.probabl.ai/bob-slay/bobathon-esilv — clé `10_ensemble3_bateman`

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
| `02_improved_hgbr` | HGBR: no-LOO + catégorielles natives + trend features + on_off_ratio | done | RMSE 4.276 ± 0.051 (-0.14 vs baseline) | improvement-plan.md |
| `03_lgbm_smooth` | LightGBM + lissage Savitzky-Golay post-hoc w=3 | done | RMSE 4.154 smooth / 4.266 raw (-0.27 vs baseline) | improvement-plan.md |
| `04_lgbm_tuned_bonus` | LightGBM tuné (lr=0.02, leaves=127) + ledd_velocity + smooth w=5 | done | RMSE 4.145 smooth / 4.252 raw (-0.28 vs baseline) | improvement-plan.md |
| `05_lgbm_lag_features` | LightGBM + lags (off_prev/next/cummax) + patient_off_max/q75 + poly2 smoothing | done | RMSE **3.739** poly2 / 3.939 raw (-0.68 vs baseline) | improvement-plan.md |
| `06_lgbm_enriched_features` | LightGBM 2000/0.01 + 15 nouvelles features (lags×2, ON agg, LEDD/OFF, IQR, cummin) | done | RMSE **3.532** poly2 / 3.706 raw (-0.21 vs exp05) — [hub](https://skore.probabl.ai/bob-slay/bobathon-esilv/cross-validations/60136) | CONTEXT.md |
| `07_lgbm_enriched_v2` | LightGBM + 12 features (quadratic slope, rater target stats, LEDD agg, off_accel) + isotonic post-proc | done | RMSE **3.490** poly2+isotonic / 3.633 raw (-0.04 vs exp06) — [hub](https://skore.probabl.ai/bob-slay/bobathon-esilv/cross-validations/60148) | CONTEXT.md |
| `08_ensemble_enriched` | Ensemble 2×LightGBM + 10 features (gene/cohort agg, obs_time, dd_ledd interactions, off_running_std) | done | RMSE **3.456** poly2+isotonic / 3.589 raw (-0.03 vs exp07) — [hub](https://skore.probabl.ai/bob-slay/bobathon-esilv/cross-validations/60198) | CONTEXT.md |
| `09_ensemble3_xgb` | Ensemble 3 modèles (2×LightGBM + XGBoost) + 7 features EWMA + patient_pk_ratio (80 features) | done | RMSE TBD — [hub](https://skore.probabl.ai/bob-slay/bobathon-esilv) clé `09_ensemble3_xgb` | CONTEXT.md |
| `11_ensemble3_bateman` | Exp 09 + remplacement patient_pk_ratio → modèle de Bateman (pk_concentration_on/off) (81 features) | done | RMSE TBD — [hub](https://skore.probabl.ai/bob-slay/bobathon-esilv) clé `10_ensemble3_bateman` | CONTEXT.md |
| `12_ensemble_RSearchSkrub` | skrub.tabular_pipeline (HistGBR) + HalvingRandomSearchCV 800 candidats, feature toggles EXP06/07/08/11, GroupKFold(5) | done | RMSE **3.801** ± 0.034 — [hub](https://skore.probabl.ai/bob-slay/bobathon-esilv/cross-validations/61871) clé `12_skrub_halvingsearch` | CONTEXT.md |
| `13_skrub_halving_tuned` | Exp 12 + lr bas [0.005,0.01,0.02,0.05], max_leaf_nodes(20-512), l2 élargie, max_resources=3000, min_resources=300 | done | RMSE **3.500** ± 0.030 — [hub](https://skore.probabl.ai/bob-slay/bobathon-esilv/cross-validations/64707) clé `13_skrub_halving_tuned` | CONTEXT.md |
| `14_skrub_drop_correlated` | Exp 13 + suppression manuelle colonnes catégorielles sources (rater_id/gene/cohort), pruning auto corr. Pearson \|r\|>0.98 (76 features après pruning), 100 candidats | done | RMSE **3.513** ± 0.030 — [hub](https://skore.probabl.ai/bob-slay/bobathon-esilv/cross-validations/65073) clé `14_skrub_drop_correlated` | CONTEXT.md |
| `15_final_dataops` | Exp 14 best params figés dans un graphe skrub DataOps pur (DAY2 §6) : `skrub.var` + `mark_as_X(GroupKFold)` + `TableVectorizer` + `HistGBR` — aucune recherche supplémentaire | done | RMSE **3.512** ± 0.041 — [hub](https://skore.probabl.ai/bob-slay/bobathon-esilv/cross-validations/65284) clé `15_final_dataops` | CONTEXT.md |

## Backlog

<!--
Ideas not yet committed to a journal/NN_*.md design note. Each row
has a stable B<N> index so it can be picked by number ("go with B2").
-->

| # | Item | Source |
|---|---|---|
