# Contexte Projet — Parkinson's UPDRS Regression

_Créé le : 2025-01-07. Ce fichier est la mémoire permanente du projet. Mettre à jour après chaque itération._

---

## Problème

Prédire le **score UPDRS continu** (`target`, 0–109.5, mean ~37.5, std ~16.5) pour des patients atteints de la maladie de Parkinson. C'est une **régression**, évaluée par **RMSE** (à minimiser).

- **Dummy RMSE (prédire la moyenne)** : ~16.5 → seuil de référence.
- **Train** : 44 590 lignes × 13 features (`X_train.csv` + `y_train.csv`)
- **Test** : 11 013 lignes × 13 features (`X_test.csv`) — aucun patient commun avec le train.

---

## Contrainte structurelle critique : données longitudinales

Chaque patient contribue **~8 visites**. Les lignes ne sont PAS indépendantes.
**Règle absolue : toute cross-validation doit utiliser `GroupKFold(groups=patient_id)`.**
Un `KFold` standard fuiterait les infos patient → RMSE de validation surestimé.

---

## Sémantique des variables clés

| Variable | Signification | Relation à `target` |
|---|---|---|
| `target` | Score UPDRS-OFF "débiaisé" (vérité terrain) | — |
| `off` | Score UPDRS mesuré hors médication | `off ≈ 0.85 × target` (bruité) |
| `on` | Score UPDRS mesuré sous médication | `on ≈ f(t) × target` où t = `time_since_intake_on` |
| `rater_id` | Évaluateur qui a scoré le patient | Biais multiplicatif b_r ∈ [0.7, 1.1] |
| `ledd` | Dose quotidienne lévodopa-équivalente | Manquant = patient non traité (début maladie) |
| `gene` | Mutation génétique | Manquant = non testé (pas aléatoire) |
| `patient_id` | Identifiant patient | ~8 visites par patient, jamais commun train/test |

### La courbe pharmacocinétique de `on`

`on/target` varie de ~0.85 (0 h après prise) à ~0.45 (2-4 h après prise).
Cette courbe est estimée empiriquement par binning de `time_since_intake_on`.

---

## Stratégie de missing values

**NE PAS imputer** les colonnes clés. Le missingness est informatif (non aléatoire) :
- `ledd` manquant → patient non traité (début maladie, score bas attendu)
- `on`/`off` manquants → mécanisme non aléatoire

**Solution** : utiliser `HistGradientBoostingRegressor` qui gère nativement les NaN.
Seule exception : `age_at_diagnosis` (5 % manquant, imputation médiane globale uniquement pour calculer `disease_duration`).

---

## Feature Engineering implémenté

### Features stateless (pas de dépendance au fold)

| Feature | Calcul | Justification |
|---|---|---|
| `disease_duration` | `age − age_at_diagnosis` | age et age_at_diagnosis corrélés à 0.94 |
| `ledd_missing_flag` | 1 si ledd est NaN | Signal informatif (non traité) |
| `on_missing_flag` | 1 si on est NaN | Signal informatif |
| `off_missing_flag` | 1 si off est NaN | Signal informatif |
| `time_on_missing_flag` | 1 si time_since_intake_on est NaN | |
| encodage cohort/gene/sexM | ordinal | HGBR adapté |

### Features fold-aware (calculées sur le fold train, appliquées à la validation)

#### 1. Biais rater (ST-1)

```python
b_r = median(off / target)  # groupé par rater_id, calculé sur fold train
on_corrected  = on  / b_r[rater_id]
off_corrected = off / b_r[rater_id]
```
Représente chaque rater par son facteur de biais `b_r` (feature numérique).
Fallback : b_r = 0.85 (valeur globale off/target) si le rater n'est pas dans le fold train.

#### 2. Correction temporelle de `on` (ST-2)

```python
# f(t) estimée par binning de time_since_intake_on sur le fold train
f_hat = interpolation(bins_t, median(on/target) par bin)
on_time_corrected = on / f_hat(time_since_intake_on)   # NaN si t manquant
```

#### 3. Agrégats longitudinaux par patient (ST-3) — sans LOO

Calculés sur **toutes les visites** du patient dans le fold (sans leave-one-out) :
- `patient_off_mean` : moyenne de `off_corrected`
- `patient_on_mean` : moyenne de `on_corrected`
- `patient_off_std` : écart-type de `off_corrected`
- `patient_slope` : pente de la régression linéaire `off_corrected ~ age`
- `patient_intercept` : ordonnée à l'origine de cette régression
- `patient_n_visits` : nombre de visites du patient dans le fold
- `visit_index` : rang de la visite courante triée par âge
- `visit_rel_index` : `visit_index / patient_n_visits` (position normalisée)
- `ledd_velocity` : pente de `ledd ~ age` (vitesse d'escalade de dose, NaN si non traité)

**Cohérence avec le test** : sur X_test toutes les visites d'un patient sont disponibles
→ utiliser toutes les visites (pas de LOO) est cohérent et donne un signal plus stable.

#### 4. Features dérivées (ST-4 suite)

- `on_off_diff` : `off_corrected − on_corrected` (fluctuation motrice absolue)
- `on_off_ratio` : `off_corrected / on_corrected` (fluctuation motrice normalisée — invariante au biais absolu)
- `patient_trend_at_age` : `patient_intercept + patient_slope × age` (prior personnalisé)
- `off_residual_from_trend` : `off_corrected − patient_trend_at_age` (fluctuation intra-patient)

**Décision** : `on_off_mean` retiré — si OFF est trop bas à une visite, ON l'est aussi (même
état du patient le même jour). Moyenner ne réduit pas le biais intra-visite.

#### 5. Catégorielles natives

`gene`, `cohort`, `rater_id` passés en dtype `category` — HGBR/LightGBM trouvent les meilleurs
splits sans ordinalisation forcée. `rater_bias_feature` (float) est conservé en complément.

---

## Modèle

Deux backends disponibles dans `ParkinsonsPipeline` :
- **HGBR** (`backend="hgbr"`) : `HistGradientBoostingRegressor` — NaN natif, catégorielles via indices
- **LightGBM** (`backend="lgbm"`) : `LGBMRegressor` — NaN natif, catégorielles via dtype category

**Post-processing** : lissage Savitzky-Golay (window=5, polyorder=1) sur la séquence
temporelle de chaque patient après prédiction → réduction du bruit intra-patient.

**CV** : folds fixés dans `data/cv_folds.pkl` — tous les modèles comparables sur les mêmes 5 splits.

---

## Workspace

- **skore mode** : hub
- **skore hub workspace** : bob-slay
- **project name** : ibm-hackaton
- **package** : `src/parkinsons/`
- **env manager** : pip + venv
- **tabular lib** : pandas

---

## Repères RMSE (résultats réels GroupKFold)

| Expérience | Configuration | RMSE brut | RMSE poly2 | Skore |
|---|---|---|---|---|
| `01_hgbr_full_fe` | HGBR + LOO + biais rater + PK | 4.42 | — | — |
| `02_improved_hgbr` | HGBR + no-LOO + catégorielles + trend + on_off_ratio | 4.27 | 4.276 | [lien](https://skore.probabl.ai/bob-slay/bobathon-esilv/cross-validations/51062) |
| `03_lgbm_smooth` | LightGBM + smooth w=3 | 4.27 | 4.154 | [lien](https://skore.probabl.ai/bob-slay/bobathon-esilv/cross-validations/51089) |
| `04_lgbm_tuned_bonus` | LightGBM tuné + ledd_velocity + smooth w=5 | 4.25 | 4.145 | [lien](https://skore.probabl.ai/bob-slay/bobathon-esilv/cross-validations/51153) |
| `05_lgbm_lag_features` | LightGBM + lags + upper-tail agg + poly2 smooth | 3.939 | **3.739** | [lien](https://skore.probabl.ai/bob-slay/bobathon-esilv/cross-validations/51301) |
| `06_lgbm_enriched_features` | LightGBM 2000/0.01 + 15 nouvelles features (lags×2, ON agg, LEDD/OFF, IQR, cummin) | 3.706 | **3.532** | [lien](https://skore.probabl.ai/bob-slay/bobathon-esilv/cross-validations/60136) |
| `07_lgbm_enriched_v2` | LightGBM 2000/0.01 + 12 features supplémentaires (quadratic slope, rater target stats, LEDD agg, off_accel) + subsample=0.85 + isotonic post-proc | 3.633 | **3.490** (poly2+isotonic) | [lien](https://skore.probabl.ai/bob-slay/bobathon-esilv/cross-validations/60148) |
| `08_ensemble_enriched` | Ensemble 2×LightGBM (3000 arbres, lr=0.008) + 10 features (gene/cohort agg, obs_time, visit_density, dd_ledd interactions, off_running_std) | 3.589 | **3.456** (poly2+isotonic) | [lien](https://skore.probabl.ai/bob-slay/bobathon-esilv/cross-validations/60198) |
| `09_ensemble3_xgb` | Ensemble 3 modèles (2×LightGBM + XGBoost) + 7 features (off_ewma, on_ewma, patient_pk_ratio, off_vs_gene/cohort_mean, visit_rev_index, ledd_ewma) — 80 features total | done | TBD | [hub](https://skore.probabl.ai/bob-slay/bobathon-esilv) clé `09_ensemble3_xgb` |
| `11_ensemble3_bateman` | Exp 09 + remplacement `patient_pk_ratio` → équation de Bateman C(t)=(ledd·ka/(ka−ke))·(e^{-ke·t}−e^{-ka·t}) (ka=1.5, ke=0.46). Deux nouvelles features : `pk_concentration_on` et `pk_concentration_off`. 81 features total. Soumission : `submissions/10_ensemble3_bateman.csv` | done | TBD | [hub](https://skore.probabl.ai/bob-slay/bobathon-esilv) clé `10_ensemble3_bateman` |
| `12_ensemble_RSearchSkrub` | skrub.tabular_pipeline(HistGBR, NaN-natif) + HalvingRandomSearchCV 800 candidats (resource=max_iter jusqu'à 500), feature toggles EXP06/07/08/11, GroupKFold(5) | done | RMSE **3.801** ± 0.034 | [hub](https://skore.probabl.ai/bob-slay/bobathon-esilv/cross-validations/61871) clé `12_skrub_halvingsearch` |
| `13_skrub_halving_tuned` | Exp 12 + lr restreint [0.005,0.01,0.02,0.05], max_leaf_nodes randint(20,512), l2 loguniform(1e-3,30), max_resources=3000, min_resources=300. Best: lr=0.01, max_iter=2700, max_leaf_nodes=237 | done | RMSE **3.500** ± 0.030 | [hub](https://skore.probabl.ai/bob-slay/bobathon-esilv/cross-validations/64707) clé `13_skrub_halving_tuned` |
| `14_skrub_drop_correlated` | Exp 13 + drop colonnes catégorielles sources + pruning auto \|r\|>0.98 (76 features). Best: lr=0.02, max_iter=2700, max_leaf_nodes=93, min_samples_leaf=67, l2=1.45, max_features=0.57. Tous groupes actifs. | done | RMSE **3.513** ± 0.030 | [hub](https://skore.probabl.ai/bob-slay/bobathon-esilv/cross-validations/65073) clé `14_skrub_drop_correlated` |
| Dummy (prédire la moyenne) | — | 16.50 | — | — |

---

## Fichiers importants

| Fichier | Rôle |
|---|---|
| `data/eda.md` | Analyse exploratoire complète |
| `feature-engineering-modelling-plan.md` | Plan détaillé FE + modélisation |
| `CONTEXT.md` | Ce fichier — mémoire permanente |
| `src/parkinsons/features.py` | Tout le feature engineering |
| `src/parkinsons/pipeline.py` | Déclaration du learner HGBR |
| `src/parkinsons/data.py` | Chargement des données |
| `src/parkinsons/evaluate.py` | CV strategy (GroupKFold) |
| `experiments/01_hgbr_full_fe.py` | Expérience principale |
| `journal/JOURNAL.md` | Index des expériences |

---

## Décisions prises et rationale

1. **GroupKFold obligatoire** — données longitudinales, ~8 visites/patient, patients test ≠ patients train.
2. **Pas d'imputation médiane** — missingness non aléatoire (MNAR). NaN HGBR/LightGBM natif.
3. **Biais rater via ratio médian** — plus robuste que target encoding car capte le biais multiplicatif.
4. **Correction pharmacocinétique de `on`** — estimée empiriquement, pas de modèle PK imposé.
5. **`rater_id` catégoriel + `rater_bias_feature` float** — les deux apportent des signaux complémentaires.
6. **Agrégats patient sans LOO** — cohérent avec le test (toutes visites disponibles) ; signal plus stable.
7. **`on_off_mean` retiré** — biais intra-visite : OFF et ON sont biaisés dans le même sens le même jour.
8. **Lissage post-hoc poly2 sur l'âge** — ajuste une parabole `pred ~ age` par patient. Respecte l'irrégularité des visites (SG traite chaque visite comme équidistante). Gain : -0.20 RMSE vs SG.
9. **Folds fixés (`data/cv_folds.pkl`)** — comparabilité stricte entre toutes les expériences.
10. **Pas de target encoding patient** — les patients test sont inconnus → inapplicable.
11. **Lags `off_cummax` / `off_prev` / `off_next`** — exploitent la structure longitudinale dans le fold. `off_cummax` (corr=0.95 avec target) est la feature la plus puissante du projet ; `off_prev` aide particulièrement les visites sans OFF courant (RMSE -0.87 dans ce sous-groupe).
12. **`patient_off_max` / `patient_off_q75`** — upper-tail de off_corrected par patient ; moins biaisé que la moyenne sur les mesures sous-estimées.
13. **Exp 06 — Enrichissement feature (2025-01-07)** : 15 nouvelles features ajoutées dans `add_extra_features()` dans l'expérience elle-même. Gain de -0.207 RMSE poly2 (3.739 → 3.532). Nouvelles features : `off_corrected_prev2`, `on_corrected_prev2`, `off_lag_delta`, `inter_visit_gap`, `off_lag_delta_per_year`, `off_corrected_next2`, `patient_on_max`, `patient_on_q75`, `patient_on_std`, `patient_on_slope`, `ledd_off_ratio`, `ledd_per_off_trend`, `patient_off_iqr`, `off_cummin`, `on_time_corrected_prev`. LightGBM plus profond (2000 arbres, lr=0.01). Top features : `rater_id_cat`, `ledd_velocity`, `patient_on_slope`, `patient_off_max`, `patient_off_std`, `patient_on_max`.
14. **Exp 07 — Features v2 + isotonic post-proc (2025-01-07)** : 12 nouvelles features : `patient_slope2` (terme quadratique de la progression), `off_residual_prev`, `on_time_corrected_next`, `ledd_prev/next`, `off_lag_accel`, `patient_ledd_mean/max`, `ledd_zscore`, `rater_target_mean/std/count`. Subsampling LightGBM (subsample=0.85, colsample_bytree=0.85). Nouveau post-processing **isotonic** (PAVA monotone non-décroissant par patient) — seul il est pire que poly2 (RMSE 3.60), mais **poly2 puis isotonic = 3.490** (-0.042 vs exp06). Soumission : `07_lgbm_enrichedv2_poly2+isotonic.csv`.
15. **Exp 08 — Ensemble 2×LightGBM + gene/cohort agg (2025-01-07)** : Ensemble de 2 modèles LightGBM (3000 arbres, lr=0.008, seeds différents, num_leaves différents) pour réduire la variance. 10 nouvelles features : `gene_off_mean`, `gene_slope_mean`, `cohort_off_mean/std`, `cohort_slope_mean`, `patient_observation_time`, `visits_per_year`, `dd_ledd_interaction`, `dd_ledd_velocity`, `off_running_std`. Total : 73 features. RMSE poly2+isotonic = **3.456** (-0.034 vs exp07). Soumissions : `08_ensemble_enriched_poly2.csv` (3.459) et `08_ensemble_enriched_poly2iso.csv` (3.456). Top nouvelles features : `off_running_std`, `visits_per_year`, `patient_observation_time`.
16. **Exp 09 — Ensemble 3 modèles (LightGBM×2 + XGBoost) + features EWMA (2025-01-07)** : Ajout de XGBoost (3000 arbres, max_depth=6, tree_method=hist) comme 3ème modèle. 7 nouvelles features : `off_ewma`, `on_ewma` (moyennes exponentielles avec shift -1), `patient_pk_ratio` (biais PK par patient), `off_vs_gene_mean`, `off_vs_cohort_mean` (résidus relatifs au groupe), `visit_rev_index`, `ledd_ewma`. Total : 80 features. Soumissions : `09_ensemble3_poly2iso.csv` (meilleure), `09_ensemble3_poly2.csv`. Skore : en cours de push.
