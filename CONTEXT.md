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

| Expérience | Configuration | RMSE (best post-proc) |
|---|---|---|
| `01_hgbr_full_fe` | HGBR + LOO + biais rater + PK | 4.42 |
| `02_improved_hgbr` | HGBR + no-LOO + catégorielles + trend + on_off_ratio | 4.276 |
| `03_lgbm_smooth` | LightGBM + smooth w=3 | 4.154 |
| `04_lgbm_tuned_bonus` | LightGBM tuné + ledd_velocity + smooth w=5 | 4.145 |
| `05_lgbm_lag_features` | LightGBM + lags + upper-tail agg + poly2 smooth | **3.739** |
| Dummy (prédire la moyenne) | — | 16.50 |

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
