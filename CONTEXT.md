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

#### 3. Agrégats longitudinaux par patient (ST-3)

Calculés depuis les visites du patient dans le **fold train uniquement** :
- `patient_off_mean` : moyenne de `off_corrected` (proxy du niveau absolu du patient)
- `patient_on_mean` : moyenne de `on_corrected`
- `patient_off_std` : écart-type de `off_corrected`
- `patient_slope` : pente de la régression linéaire `off_corrected ~ age` (vitesse progression)
- `patient_intercept` : ordonnée à l'origine de cette régression
- `patient_n_visits` : nombre de visites dans le fold train
- `visit_index` : rang de la visite courante triée par âge

**Anti-fuite critique** : pour une visite en validation, les agrégats proviennent
exclusivement du fold train. Sur X_test : agrégats depuis X_train complet.

#### 4. Features dérivées (ST-4 suite)

- `on_off_diff` : `off_corrected − on_corrected` (fluctuation motrice)
- `on_off_mean` : `(off_corrected + on_corrected) / 2`

---

## Modèle

Un seul `HistGradientBoostingRegressor` (HGBR) :
- **Avantage clé** : gestion native des NaN → pas d'imputation nécessaire pour `on`/`off`/`ledd`
- **Garde contre overfitting** : `early_stopping=True`, `min_samples_leaf=20`, `l2_regularization=0.1`
- **CV** : `GroupKFold(n_splits=5, groups=patient_id)`

---

## Workspace

- **skore mode** : hub
- **skore hub workspace** : bob-slay
- **project name** : ibm-hackaton
- **package** : `src/parkinsons/`
- **env manager** : pip + venv
- **tabular lib** : pandas

---

## Repères RMSE attendus

| Étape | RMSE estimé |
|---|---|
| Dummy (moyenne globale) | ~16.5 |
| HGBR brut sans FE | ~9–11 |
| + Biais rater | ~8–10 |
| + Agrégats patient | ~6–8 |
| + Correction temporelle on | ~5–7 |
| + Tuning HGBR | ~4.5–6.5 |

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
2. **Pas d'imputation médiane** — missingness non aléatoire (MNAR). NaN HGBR natif.
3. **HGBR unique (pas de stacking)** — suffisant pour ce premier modèle ; complexité non justifiée.
4. **Biais rater via ratio médian** — plus robuste que target encoding car capte le biais multiplicatif.
5. **Correction pharmacocinétique de `on`** — estimée empiriquement, pas de modèle PK imposé.
6. **`rater_id` représenté par `b_r`** — la feature numérique b_r est meilleure que l'identifiant catégoriel.
7. **Agrégats patient calculés dans chaque fold** — anti-fuite stricte.
