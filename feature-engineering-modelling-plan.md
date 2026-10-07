# Feature Engineering & Modelling Plan — Parkinson's UPDRS Regression

## Overview

**Goal:** Prédire le score UPDRS `target` continu (0–109.5, mean ~37.5, RMSE dummy ~16.5)
depuis 13 features cliniques enregistrées sur plusieurs visites par patient.
**Métrique principale :** RMSE (à minimiser).
**Architecture finale :** un seul `HistGradientBoostingRegressor` (HGBR) évalué en
`GroupKFold(n_splits=5, groups=patient_id)`.

### Domaine : ce que `target`, `on`, `off` signifient

- `target` = score UPDRS-OFF "débiaisé" (vérité terrain).
- `off` ≈ 0.85 × target — mesure bruitée hors-médicament; très proche de target.
- `on` ≈ f(time_since_intake_on) × target — mesure sous médicament; le ratio varie
  de ~0.85 (juste après la prise) à ~0.45 (2-4 h après).
- `rater_id` — chaque rater applique un facteur multiplicatif b_r ∈ [0.7, 1.1]
  à ses scores. Ce biais doit être estimé et corrigé dans chaque fold de CV.

---

## Sous-tâche 0 — Infrastructure de validation

**Intent :** Définir le cadre de CV strict avant tout développement de feature.
Tout score rapporté doit être reproduit sur ce cadre.

**Expected outcomes :**
- `GroupKFold(n_splits=5)` groupé sur `patient_id` utilisé partout.
- Toutes les features "patient-level" et "rater-level" calculées UNIQUEMENT sur le
  fold train, puis appliquées au fold validation → pas de fuite.
- Pipeline skrub `X.skb.apply_func(...)` ou sklearn `Pipeline` avec un
  `FunctionTransformer` custom wrappant chaque étape de feature engineering.

**Steps :**
1. Charger `X_train`, `y_train`. Maintenir `patient_id` en dehors des features
   pures (utilisé uniquement comme `groups` pour GroupKFold).
2. Implémenter une fonction `compute_fold_features(X_train_fold, y_train_fold,
   X_val_fold)` qui sera appelée à chaque fold pour calculer les features
   nécessitant la connaissance du train (rater bias, agrégats patient).
3. Wrapper le HGBR final dans un `Pipeline` sklearn qui inclut cette logique.

**Relevant context :** `data/X_train.csv` — `patient_id` col 1; `data/eda.md`
§Structure — 5 576 patients × ~8 visites.

**Status :** [ ] pending

---

## Sous-tâche 1 — Correction du biais rater

**Intent :** Chaque rater a un facteur multiplicatif b_r sur ses scores `on` et `off`.
Corriger ce biais rend `on_corrected` et `off_corrected` plus proches de `target`,
réduisant directement l'erreur résiduelle du modèle.

### Pourquoi cette approche

Si `score_obs = b_r × target`, alors `score_obs / b_r ≈ target`.
On estime b_r pour chaque rater r comme :
```
b_r = median_over_visits_of_rater_r( off_obs / target )
```
- On utilise `off` plutôt que `on` pour l'estimation car `off ≈ 0.85 × target`
  avec moins de bruit temporel (pas de dépendance à `time_since_intake`).
- Le 0.85 global est absorbé dans b_r (b_r estimé englobe le facteur 0.85 moyen).
- Pour les raters sans données dans le fold train : fallback b_r = 1.0 (pas de
  correction). Ces raters sont rares et leur biais n'est pas estimable.

### Calcul dans chaque fold

```
fold_train: pour chaque (rater_id, Index) où off et target sont tous les deux non-NaN
  ratio_r = off / target
  b_r = median(ratio_r) groupé par rater_id
```

Puis sur train ET validation :
```
on_corrected  = on  / b_r[rater_id]   (NaN si on est NaN)
off_corrected = off / b_r[rater_id]   (NaN si off est NaN)
```

### Attention

- Calculer b_r UNIQUEMENT sur le fold train → pas de fuite vers la validation.
- Sur X_test (inférence finale) : b_r calculé sur l'intégralité de X_train + y_train.
- Les colonnes `on` et `off` originales sont GARDÉES en plus des versions corrigées
  (le HGBR peut exploiter les deux).

**Expected outcomes :**
- 2 nouvelles features : `on_corrected`, `off_corrected`.
- Corrélation `off_corrected` ↔ `target` supérieure à `off` brut.

**Steps :**
1. Dans `compute_fold_features`, calculer le dictionnaire `{rater_id: b_r}` depuis
   le fold train.
2. Mapper sur toutes les lignes (train + val) via `rater_id`.
3. Créer `on_corrected`, `off_corrected`.
4. Vérifier : scatter plot `off_corrected` vs `target` moins dispersé que `off` vs
   `target`.

**Relevant context :** `eda.md` §Associations — `off` Pearson 0.871 avec target;
50 raters avec Cramér V 0.019 mais le biais multiplicatif n'est pas capturé par
Pearson/Cramér V.

**Status :** [ ] pending

---

## Sous-tâche 2 — Correction temporelle de `on`

**Intent :** `on ≈ f(t) × target` où `t = time_since_intake_on` et f(t) décroît
de ~0.85 à ~0.45 entre 0 et ~3 h. En estimant f(t) on peut calculer
`on_time_corrected = on / f(t)` ≈ `target`, une feature très puissante.

### Modélisation de f(t)

f(t) est une courbe pharmacocinétique empirique. Approche :

1. **Estimation non-paramétrique depuis les données :**
   - Sur les lignes où `on`, `target`, et `time_since_intake_on` sont tous non-NaN :
     calculer `ratio = on / target`.
   - Binning de `t` en tranches de 0.5 h → médiane de `ratio` par bin.
   - Lisser avec une spline ou LOWESS pour obtenir une fonction continue f̂(t).
   - Clamper f̂(t) ∈ [0.3, 1.0] pour éviter les divisions explosives.

2. **Application :**
   ```
   on_time_corrected = on / f̂(time_since_intake_on)   si t est non-NaN
   on_time_corrected = NaN                              sinon
   ```
   Le HGBR gère nativement les NaN — pas de problème si t est absent.

3. **Calcul dans chaque fold :** f̂ est estimée sur le fold train uniquement.
   Sur X_test : f̂ estimée sur l'intégralité du train.

### Feature dérivée supplémentaire

- `on_residual_time` = `on_corrected` − `on_time_corrected` : quantifie le bruit
  résiduel entre les deux corrections (utile signal pour le HGBR).

**Expected outcomes :**
- Feature `on_time_corrected` avec corrélation target supérieure à `on` brut.
- La courbe f̂(t) montre bien la décroissance attendue de ~0.85 à ~0.45.

**Steps :**
1. Calculer les ratios `on/target` vs `time_since_intake_on` sur le fold train.
2. Estimer f̂(t) par binning + interpolation (scipy `interp1d` ou numpy `polyfit`
   degré 2 sur les médiane de chaque bin).
3. Créer `on_time_corrected`.
4. Garder `on_corrected` (ST-1) et `on_time_corrected` comme deux features
   distinctes — le HGBR choisira la pondération optimale.

**Relevant context :** `eda.md` §Modelling implications — `on/off` partial
missingness; `time_since_intake_on` near-zero Pearson mais non-trivial Cramér V 0.177.

**Status :** [ ] pending

---

## Sous-tâche 3 — Agrégats longitudinaux par patient

**Intent :** Pour chaque patient, le score `target` suit approximativement une
droite dans le temps (`target ≈ α_p + β_p × age`). Les paramètres de cette droite
(et des agrégats simples) sont des features très informatives et calculables sur
X_test depuis les autres visites du patient présentes dans X_test.

### Features à calculer par patient

Depuis les visites du patient dans le fold train (ou dans X_test pour l'inférence) :

| Feature | Calcul | Pourquoi |
|---|---|---|
| `patient_off_mean` | mean(off_corrected) où off non-NaN | Niveau moyen du patient |
| `patient_off_std` | std(off_corrected) | Variabilité intra-patient |
| `patient_on_mean` | mean(on_corrected) où on non-NaN | Niveau sous médication |
| `patient_slope` | β_p de la régression linéaire off_corrected ~ age | Vitesse de progression |
| `patient_intercept` | α_p de cette régression | Niveau à l'origine |
| `patient_n_visits` | count(visites dans le fold train) | Fiabilité de l'agrégat |
| `visit_index` | rang de la visite courante triée par age | Position longitudinale |

### Règle anti-fuite (critical)

Pour la visite i d'un patient dans le fold VALIDATION :
- Les agrégats sont calculés depuis les visites du MÊME patient dans le fold TRAIN
  uniquement. Les visites validation du patient ne contribuent PAS.
- Si le patient n'a AUCUNE visite dans le fold train : fallback aux moyennes
  globales (cohort-level mean pour `patient_off_mean`, slope = 0).

Pour X_test (inférence) :
- Les agrégats sont calculés depuis X_train complet (on a accès à tout le train).
- Pour les patients test : calculer les agrégats depuis les AUTRES visites du même
  patient dans X_test (pas de target → utiliser `off_corrected` ou `on_corrected`).
  La régression linéaire utilise `age` et `off_corrected` / `on_corrected` comme
  proxy de target.

### Régression linéaire par patient

```
Pour les patients avec ≥ 3 visites avec off non-NaN dans le fold train :
  β_p, α_p = linregress(age, off_corrected)
  patient_slope    = β_p
  patient_intercept = α_p

Pour les patients avec < 3 visites :
  patient_slope    = cohort_median_slope
  patient_intercept = cohort_median_intercept
```

**Expected outcomes :**
- `patient_off_mean` devient probablement la feature la plus importante après
  `off_corrected` (elle encapsule l'information de toutes les visites du patient).
- `patient_slope` capture la trajectoire de progression — différenciateur fort
  entre patients stables et en dégradation rapide.

**Steps :**
1. Dans `compute_fold_features`, grouper par `patient_id` sur le fold train.
2. Calculer les 7 features de la table ci-dessus.
3. Merger sur le fold validation via `patient_id`.
4. Implémenter le fallback pour les patients sans historique train.
5. Implémenter la logique X_test (agrégats depuis X_train + proxy X_test).

**Relevant context :** `data/X_train.csv` — visite IPLP5212 montre bien la
progression linéaire : on passe de 7 à 29 sur ~13 ans; `eda.md` §Structure —
~8 visites par patient.

**Status :** [ ] pending

---

## Sous-tâche 4 — Feature Engineering de base (hors agrégats)

**Intent :** Traitement des features simples avant passage au HGBR. Ces
transformations sont stateless et peuvent être appliquées sans connaissance
du fold train (pas de fuite possible).

### Features à créer

| Feature | Calcul | Justification |
|---|---|---|
| `disease_duration` | age − age_at_diagnosis | r=0.94 entre age et age_at_diagnosis; durée maladie = signal clinique |
| `on_off_diff` | off_corrected − on_corrected | Fluctuation motrice; effet du médicament |
| `on_off_mean` | (off_corrected + on_corrected) / 2 | Estimateur moyen quand les deux sont présents |
| `ledd_missing_flag` | 1 si ledd est NaN | ledd manquant = patient non traité (début maladie) — informatif |
| `on_missing_flag` | 1 si on est NaN | Mécanisme de missingness non aléatoire |
| `off_missing_flag` | 1 si off est NaN | Idem |
| `time_on_missing_flag` | 1 si time_since_intake_on est NaN | |

### NaN management

- NE PAS imputer `on`, `off`, `ledd`, `time_since_intake_on`, `time_since_intake_off`.
  Le HGBR gère nativement les NaN — c'est son avantage clé ici.
- `age_at_diagnosis` manquant à 5 % : imputation globale par médiane UNIQUEMENT
  pour calculer `disease_duration`; l'original reste NaN pour le HGBR.
- `gene` : pas d'imputation; catégorie "Unknown" explicite (le HGBR avec ordinal
  encoding peut gérer une catégorie NaN).

### Encodage catégoriel

- `cohort` (2 cat.), `gene` (4 cat. + Unknown), `sexM` (binaire) : ordinal encoding
  (adapté au HGBR qui travaille en ordre).
- `rater_id` : garder comme feature numérique = `b_r` estimé (ST-1) — le facteur
  de biais lui-même est la meilleure représentation du rater.

**Expected outcomes :**
- Feature matrix numérique complète sans string, compatible HGBR.
- `disease_duration` remplace `age_at_diagnosis`.
- Les flags de missingness ajoutent le signal du mécanisme MAR.

**Steps :**
1. Calculer `disease_duration`, `on_off_diff`, `on_off_mean`.
2. Créer les 4 flags de missingness.
3. Encoder `cohort`, `gene`, `sexM` en ordinal.
4. Remplacer `rater_id` (string) par le `b_r` calculé en ST-1.
5. Dropper `Index`, `age_at_diagnosis`, `patient_id` (déjà utilisé pour grouping).

**Relevant context :** `eda.md` §FE-2 — collinéarité age/age_at_diagnosis;
§FE-3 — stratégie de missingness.

**Status :** [ ] pending

---

## Sous-tâche 5 — Tuning du HGBR

**Intent :** Optimiser les hyperparamètres du HGBR pour minimiser le RMSE en
GroupKFold, en évitant l'overfitting (early stopping + régularisation) et
l'underfitting (capacité suffisante).

### Architecture du pipeline final

```
Input features (après ST-0 à ST-4)
  │
  ▼
HistGradientBoostingRegressor(
    max_iter=500,            ← borne haute; early stopping actif
    early_stopping=True,
    validation_fraction=0.1, ← inner holdout dans chaque fold
    n_iter_no_change=20,     ← patience
    learning_rate=0.05,      ← faible pour profiter du max_iter haut
    max_depth=None,          ← laisse les leaves décider
    min_samples_leaf=20,     ← évite les feuilles trop petites (44k lignes)
    l2_regularization=0.1,   ← régularisation légère
    max_bins=255,            ← maximum pour la résolution
    categorical_features=None ← on encode avant
)
```

### Stratégie de tuning (si temps disponible)

Utiliser `skore.evaluate` (via cross-validation GroupKFold) comme boucle externe.
Tune via `RandomizedSearchCV` avec GroupKFold sur :

| Param | Range |
|---|---|
| `learning_rate` | [0.01, 0.05, 0.1] |
| `min_samples_leaf` | [10, 20, 40] |
| `l2_regularization` | [0, 0.05, 0.1, 0.5] |
| `max_leaf_nodes` | [31, 63, 127] |

**Garde-fous overfitting :**
- Early stopping avec `n_iter_no_change=20` et `validation_fraction=0.1` dans chaque
  fold — le HGBR arrête automatiquement quand le gain stagne.
- `min_samples_leaf ≥ 20` avec 44k lignes — évite les splits sur du bruit.
- `learning_rate` faible (0.05) avec `max_iter` élevé (500).

**Garde-fous underfitting :**
- `max_bins=255` (résolution maximale des histogrammes).
- `max_depth=None` ou `max_depth=6` — suffisant pour capturer les interactions.
- Toutes les features engineerées (ST-1→ST-4) sont dans la feature matrix.

**Expected outcomes :**
- RMSE GroupKFold < 10 (objectif ambitieux, ~40 % de réduction vs dummy 16.5).
- L'early stopping converge entre 100 et 400 itérations selon les folds.

**Steps :**
1. Définir la pipeline complète (ST-0 → HGBR).
2. Évaluer avec RMSE en GroupKFold sur les hyperparamètres par défaut ci-dessus.
3. Si RMSE > 11 : lancer RandomizedSearchCV (n_iter=30) sur la grille ci-dessus.
4. Reporter le RMSE par fold (mean ± std) pour détecter la variance inter-fold.

**Relevant context :** `eda.md` §Modelling implications — HGBR native NaN; 44 590
lignes × ~20 features engineerées.

**Status :** [ ] pending

---

## Résumé du flux de features

```
X_train / X_test
       │
       ▼ ST-4 (stateless)
  disease_duration, on_off_diff, on_off_mean, flags missingness, encoding catégoriels
       │
       ▼ ST-1 (fold-aware)
  b_r par rater → on_corrected, off_corrected, rater_bias_feature
       │
       ▼ ST-2 (fold-aware)
  f̂(t) → on_time_corrected, on_residual_time
       │
       ▼ ST-3 (fold-aware)
  patient_off_mean, patient_off_std, patient_on_mean,
  patient_slope, patient_intercept, patient_n_visits, visit_index
       │
       ▼ ST-5
  HistGradientBoostingRegressor → prédiction target
```

---

## Ordre d'implémentation recommandé

| Ordre | Sous-tâche | Impact RMSE estimé | Risque fuite |
|---|---|---|---|
| 1 | ST-0 — Infrastructure CV | Bloque tout | Oui si mal fait |
| 2 | ST-4 — FE de base | Moyen | Non |
| 3 | ST-1 — Biais rater | Élevé | Oui si fold-aware mal fait |
| 4 | ST-3 — Agrégats patient | Très élevé | Oui si fold-aware mal fait |
| 5 | ST-2 — Correction temporelle on | Élevé | Oui si fold-aware mal fait |
| 6 | ST-5 — Tuning HGBR | Moyen | Non |

---

## Repères RMSE attendus (progression)

| Étape | RMSE attendu |
|---|---|
| Dummy (mean) | ~16.5 |
| HGBR brut sans FE | ~9–11 |
| + Biais rater (ST-1) | ~8–10 |
| + Agrégats patient (ST-3) | ~6–8 |
| + Correction temporelle on (ST-2) | ~5–7 |
| + Tuning (ST-5) | ~4.5–6.5 |

---

## Questions résolues

- `target` ≠ combinaison de `on`/`off` → pas de leakage, on garde les deux.
- Patients test ≠ patients train → pas de `patient_mean_target` (cold start).
- Missingness non aléatoire → pas d'imputation par médiane, NaN natif HGBR.
- `rater_id` → représenté par son facteur de biais b_r (pas de target encoding).
- Un seul modèle : HGBR. Pas de RF ni stacking pour ce premier modèle.
