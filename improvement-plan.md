# Plan: Amélioration du modèle UPDRS Parkinson (baseline RMSE 4.42)

_Créé le : 2025-01-07. Baseline: GroupKFold RMSE 4.42 ± 0.05._

---

## Objectif

Faire baisser le RMSE GroupKFold sans overfitting, en évaluant chaque idée
indépendamment sur exactement les mêmes 5 folds fixés. Chaque sous-tâche est une
expérience numérotée (`02_*`, `03_*`, etc.) qui s'appuie sur la baseline `01_hgbr_full_fe`.

**Ordre d'exécution :**
```
Prérequis (folds fixés)
  → Exp 02 (biais rater séparé ON/OFF)
  → Exp 03 (tendance personnalisée)
  → Exp 04 (on_off_mean vs on_off_ratio)
  → Exp 05 (lissage post-hoc)
  → Exp 06 (LightGBM — comparaison finale)
  → Exp 07 (prior patient k-NN — si temps)
```

Règle de rétention : une idée est conservée seulement si le gain RMSE est **net sur
les 5 folds** (pas uniquement la moyenne). Si le gain n'est pas net, l'idée est
abandonnée et documentée dans le journal.

---

## Corrections apportées au plan (révision 2025-01-07)

1. **Exp 02 — Retirer le LOO :** L'agrégation patient actuelle exclut la visite
   courante (leave-one-out). Sur le test on a toutes les visites → utiliser toutes
   les visites (y compris la courante) est cohérent et plus informatif.
   **Catégorielles natives :** passer `cohort`, `gene`, `rater_id` en dtype `category`
   pour que HGBR/LightGBM les traite nativement (meilleur split, pas d'ordinal forcé).

2. **Exp 04 — on/off same visit :** La corrélation entre le biais de OFF et le biais de ON
   est intra-visite, pas intra-rater. Si OFF est trop bas à une visite donnée, ON l'est
   aussi ce jour-là (même état du patient, même conditions d'évaluation). Moyenner ON et
   OFF ne réduit pas ce biais systématique. On retire `on_off_mean` et on teste `on_off_ratio`.

3. **Backlog — Pas de target encoding patient :** les patients test sont inconnus →
   le target encoding patient nécessiterait la cible → inapplicable.

---

## Prérequis transversal — Fixer les folds une seule fois

**Intent :** Toutes les comparaisons RMSE doivent être faites sur les mêmes 5 splits
GroupKFold. Actuellement les folds sont recalculés à chaque run via `.split()`, ce qui
est déterministe (GroupKFold n'est pas aléatoire) mais implicite. On matérialise les
indices une fois et on les sauvegarde dans `data/cv_folds.pkl` pour que chaque
expérience charge le même objet.

**Relevant context:**
- [`src/parkinsons/evaluate.py`](src/parkinsons/evaluate.py) — splitter défini mais pas fixé sur disque
- [`experiments/01_hgbr_full_fe.py`](experiments/01_hgbr_full_fe.py) — appelle `.split()` à chaque run
- La liste `split_indices = list(gkf.split(X_train, y_train, groups=groups))` est passée
  à `skore.CrossValidationReport`

**Todo:**
- [ ] Ajouter dans `src/parkinsons/evaluate.py` (ou `data.py`) une fonction
  `get_cv_folds(X, y, groups, cache_path)` qui charge `data/cv_folds.pkl` si présent
  sinon le crée
- [ ] Chaque nouvelle expérience charge les folds depuis le cache plutôt que de
  recalculer

**Expected outcomes:** Un fichier `data/cv_folds.pkl` stable ; toutes les expériences
impriment le même `fold 0 train size: N` pour que la comparabilité soit vérifiable.

**Status:** [ ] pending

---

## Idée 1 — Biais rater séparé pour ON et OFF (Expérience 02)

**Intent :** Actuellement `b_r` est estimé depuis `off / target` et appliqué de la même
façon à `on` et `off`. Mais le rater peut avoir un biais différent sur `on` (il évalue
un patient sous médicament, état cliniquement distinct). Si `b_r_on ≠ b_r_off` pour les
mêmes raters, appliquer une correction séparée améliorera `on_time_corrected` et donc le
signal global.

**Analyse préalable (dans le script de l'expérience) :**
- Calculer pour chaque rater `b_r_off = median(off / target)` et
  `b_r_on_raw = median(on / target)` (sans correction PK pour l'instant)
- Calculer la corrélation rang entre `b_r_off` et `b_r_on_raw`
- Si corrélation > 0.85 → biais identique, pas de gain attendu
- Si corrélation < 0.85 → estimer `b_r_on` séparément et appliquer

**Implémentation :**
- Dans `_estimate_rater_bias`, ajouter un paramètre `score_col` (`'off'` ou `'on'`)
  pour estimer un biais par type de score
- Dans `_apply_rater_correction`, utiliser `rater_bias_off` pour `off_corrected` et
  `rater_bias_on` pour `on_corrected`
- Nouvelle feature numérique `rater_bias_on_feature` (bias facteur ON du rater)
- Garder `rater_bias_feature` existant (= bias OFF)

**Relevant context:**
- [`src/parkinsons/features.py:_estimate_rater_bias`](src/parkinsons/features.py) L121–153
- [`src/parkinsons/features.py:_apply_rater_correction`](src/parkinsons/features.py) L156–182
- `FEATURE_COLS` L471–505 — liste à mettre à jour si on ajoute une feature

**Expected outcomes:** RMSE ≤ 4.35 ou abandon si biais identiques (|corr| > 0.9).

**Status:** [ ] pending

---

## Idée 2 — Supprimer le LOO + catégorielles natives (Expérience 02 — ÉTENDUE)

**Intent :** Deux corrections de fond :

**A) Retirer le leave-one-out sur les agrégats patient :**
Actuellement `_loo_mean`, `_loo_std`, `_patient_slope_intercept` excluent la visite
courante dans le calcul des agrégats. En GroupKFold, toutes les visites d'un patient
sont dans le même fold. Sur le **test**, on a toutes ses visites disponibles.
→ Utiliser **toutes** les visites du patient (y compris la courante) pour calculer
slope/intercept/mean/std est cohérent et donne un signal plus stable.

En CV : pour la **validation**, utiliser toutes les visites du fold de validation du
patient (pas en LOO). Pour le **train** : idem, toutes les visites du patient dans le
fold train. C'est déjà sans anti-fuite car les agrégats utilisent uniquement des
features X, pas y.

**B) Catégorielles natives :**
Actuellement `gene` et `cohort` sont ordinalement encodés en entiers, et `rater_id`
est représenté par `rater_bias_feature` (un float). LightGBM et HGBR supportent les
dtypes `category` pandas nativement — le modèle trouve lui-même les meilleurs splits
catégoriels.
→ Passer `cohort`, `gene` et `rater_id` en `dtype='category'` plutôt qu'en entier.

**Implémentation :**
- Supprimer `_loo_mean` et `_loo_std` → remplacer par `_group_mean` / `_group_std`
  (sans exclusion de la ligne courante)
- Supprimer `_patient_slope_intercept` LOO → utiliser simplement `np.polyfit` sur
  toutes les visites du patient
- Dans `_stateless_features`, ajouter `gene_cat`, `cohort_cat`, `rater_id_cat` en dtype
  `category` au lieu de l'encodage ordinal (ou en plus)
- Mettre à jour `FEATURE_COLS` (ajouter les catégorielles, garder `rater_bias_feature`)

**Relevant context:**
- [`src/parkinsons/features.py:_loo_mean`](src/parkinsons/features.py) L297–329
- [`src/parkinsons/features.py:_loo_std`](src/parkinsons/features.py) L332–353
- [`src/parkinsons/features.py:_patient_slope_intercept`](src/parkinsons/features.py) L356–396
- [`src/parkinsons/features.py:_add_patient_loo_features`](src/parkinsons/features.py) L399–429

**Expected outcomes:** RMSE ≤ 4.20 (le LOO avec ~8 visites par patient est bruyant ;
l'agrégat complet est plus stable et plus cohérent avec l'usage en test).

**Status:** [ ] pending

---

## Idée 3 — Tendance linéaire personnalisée comme feature explicite (Expérience 03)

**Intent :** La cible d'un patient évolue quasi-linéairement avec l'âge.
`patient_slope` et `patient_intercept` existent déjà, mais le modèle doit inférer lui-
même que `patient_intercept + patient_slope × age` est la prévision personnalisée. En
l'exposant comme feature directe, on lui donne un signal déjà assemblé.

**Nouvelles features à ajouter dans `_derived_features` ou une nouvelle fonction :**
- `patient_trend_at_age = patient_intercept + patient_slope × age`
  (valeur prédite par la droite linéaire personnalisée à l'âge de la visite)
- `off_residual_from_trend = off_corrected − patient_trend_at_age`
  (écart de la mesure brute par rapport à la tendance — capte les fluctuations
  intra-patient au-delà de la progression)
- `visit_rel_index = visit_index / patient_n_visits`
  (position normalisée de la visite dans la séquence du patient)

**Anti-fuite :** Ces features utilisent `patient_slope` et `patient_intercept` qui sont
déjà calculés fold-aware (estimés sur le fold train uniquement, appliqués au val).
Pas de nouvelle source de fuite.

**Sur X_test :** Comme aujourd'hui, les agrégats patient sont calculés depuis X_test
complet (toutes visites du patient disponibles). Ne pas appliquer LOO sur test — la
fonction `_add_patient_loo_features` est appelée sur X_val (= X_test) séparément et
tous les visiteurs test sont dans la même "partition" (toutes leurs visites sont dans
X_test), donc le LOO retire correctement la visite courante.

**Relevant context:**
- [`src/parkinsons/features.py:_add_patient_loo_features`](src/parkinsons/features.py) L399–429
- [`src/parkinsons/features.py:_derived_features`](src/parkinsons/features.py) L448–464
- `FEATURE_COLS` L471–505

**Expected outcomes:** RMSE ≤ 4.30 (le modèle a déjà slope + intercept, le gain sera
modeste mais mesurable). Gain attendu plus fort si l'on vérifie que
`patient_trend_at_age` a une importance SHAP élevée.

**Status:** [ ] pending

---

## Idée 3 — Lissage post-hoc des prédictions par patient (Expérience 04)

**Intent :** Le modèle prédit chaque visite indépendamment. La vraie trajectoire est
lisse. Un lissage sur la séquence temporelle de chaque patient (tri par âge) peut
réduire l'erreur sur les points "aberrants" de la prédiction.

**Approche :**
- Après `predict(X_val)`, grouper les prédictions par `patient_id`, trier par `age`,
  puis appliquer un filtre `savgol_filter(window=3, polyorder=1)` (= moyenne glissante
  robuste) ou un simple rolling mean de taille 3
- Comparer RMSE avec/sans lissage sur chaque fold
- Si le patient n'a que 1–2 visites dans le fold, ne pas lisser (pas de signal)

**Implémentation :**
- Ajouter une fonction `smooth_patient_predictions(preds, X_val, window=3)` dans un
  nouveau fichier `src/parkinsons/postprocess.py`
- Dans l'expérience, appliquer après chaque `predict` dans la boucle CV manuelle
  (remplace l'appel direct à `skore.CrossValidationReport` qui ne supporte pas ce
  post-processing)

**Note :** `skore.CrossValidationReport` ne permet pas d'injecter un post-processing.
Il faut passer en boucle CV manuelle (accumuler `(y_true_fold, y_pred_fold)` sur les 5
folds, lisser, calculer `root_mean_squared_error` global). Cette boucle existe déjà
en prototype dans `01_hgbr_full_fe.py` via `split_indices`.

**Relevant context:**
- [`experiments/01_hgbr_full_fe.py`](experiments/01_hgbr_full_fe.py) — structure de la boucle
- [`src/parkinsons/pipeline.py`](src/parkinsons/pipeline.py) — `fit` / `predict` exposés

**Expected outcomes:** RMSE ≤ 4.30 ou pas de gain si les prédictions sont déjà assez
lisses. Risque : sur-lissage si window trop grande — tester window=3 et window=5.

**Status:** [ ] pending

---

## Idée 4 — LightGBM à la place de HGBR (Expérience 05)

**Intent :** LightGBM supporte nativement les NaN, produit souvent de meilleures
performances que HGBR avec un temps d'entraînement réduit. Les hyperparamètres par
défaut de LightGBM sont plus agressifs (min_child_samples, num_leaves), ce qui nécessite
une régularisation explicite pour éviter l'overfitting.

**Implémentation :**
- Ajouter `LGBMRegressor` comme backend optionnel dans `ParkinsonsPipeline` via un
  paramètre `model='hgbr'` | `'lgbm'`
- Ou plus simple : créer `build_lgbm_learner()` dans `pipeline.py` qui instancie
  `ParkinsonsPipeline(backend='lgbm', ...)` avec des hyperparamètres de départ
  conservateurs (`min_child_samples=30`, `num_leaves=63`, `reg_lambda=1.0`,
  `n_estimators=500`, `learning_rate=0.05`, `early_stopping_rounds=20`)
- LightGBM gère les NaN différemment : il peut traiter chaque bin en décidant si les
  NaN vont à gauche ou à droite — avantage sur HGBR

**Prérequis :** `lightgbm` doit être installé dans le venv (`pip install lightgbm`).

**Relevant context:**
- [`src/parkinsons/pipeline.py`](src/parkinsons/pipeline.py) — `_make_hgbr()` à dupliquer
- [`src/parkinsons/pipeline.py:build_learner`](src/parkinsons/pipeline.py) L143–168

**Expected outcomes:** RMSE ≤ 4.20 (LightGBM dépasse généralement HGBR de ~0.1–0.3
sur des datasets tabulaires de cette taille).

**Status:** [ ] pending

---

## Idée 4b — on_off_ratio à la place de on_off_mean (Expérience 04)

**Intent :** Si `off` est trop bas à une visite, `on` est aussi trop bas ce jour-là
(même patient, même état, même conditions). Le biais est intra-visite, pas intra-rater.
Moyenner ON et OFF **ne réduit pas** ce biais systématique — les deux sont biaisés dans
le même sens à la même visite.

`on_off_ratio = off_corrected / on_corrected` est plus pertinent : il capte la
fluctuation motrice normalisée indépendamment du niveau absolu. Si le rater sous-évalue
les deux de 10 %, le ratio reste le même. La différence absolue `on_off_diff` est déjà
présente — le ratio apporte une info complémentaire.

Deux choses à tester dans la même expérience :
1. Supprimer `on_off_mean`
2. Ajouter `on_off_ratio = off_corrected / on_corrected`

**Deux variantes à comparer dans la même expérience :**
- Variante A : supprimer `on_off_mean` de `FEATURE_COLS` (garder `on_off_diff`)
- Variante B : remplacer `on_off_mean` par `on_off_ratio = off_corrected / on_corrected`
  (avec clamping si `on_corrected` ≈ 0)

**Implémentation :**
- Dans `_derived_features`, ajouter `on_off_ratio` avec guard `on_corrected.replace(0, NaN)`
- Tester les deux variantes vs baseline (3 runs sur les mêmes folds fixés)
- Conserver la meilleure variante si gain > 0.02 RMSE

**Relevant context:**
- [`src/parkinsons/features.py:_derived_features`](src/parkinsons/features.py) L448–464
- `FEATURE_COLS` L471–505

**Expected outcomes:** Gain modeste (0.02–0.05). Si aucune variante ne bat la baseline,
`on_off_mean` est conservé tel quel.

**Status:** [ ] pending

---

## Idée 5 — Lissage post-hoc des prédictions par patient (Expérience 05)

**Intent :** Le modèle prédit chaque visite indépendamment. La vraie trajectoire est
lisse. Un lissage sur la séquence temporelle de chaque patient (tri par âge) peut
réduire l'erreur sur les points "aberrants" de la prédiction.

**Approche :**
- Après `predict(X_val)`, grouper les prédictions par `patient_id`, trier par `age`,
  puis appliquer un filtre `savgol_filter(window=3, polyorder=1)` (= moyenne glissante
  robuste) ou un simple rolling mean de taille 3
- Comparer RMSE avec/sans lissage sur chaque fold
- Si le patient n'a que 1–2 visites dans le fold, ne pas lisser (pas de signal)

**Rapport double :**
- `skore.CrossValidationReport` sur les prédictions brutes (non lissées) → sauvegardé
- RMSE manuel calculé sur les prédictions lissées → imprimé côte à côte dans le script

**Implémentation :**
- Ajouter une fonction `smooth_patient_predictions(preds, X_val, window=3)` dans un
  nouveau fichier `src/parkinsons/postprocess.py`
- La boucle CV est manuelle (accumulation de `(y_true, y_pred)` sur les 5 folds)
  pour permettre le post-processing

**Relevant context:**
- [`experiments/01_hgbr_full_fe.py`](experiments/01_hgbr_full_fe.py) — structure de la boucle
- [`src/parkinsons/pipeline.py`](src/parkinsons/pipeline.py) — `fit` / `predict` exposés

**Expected outcomes:** RMSE ≤ 4.30 ou pas de gain si les prédictions sont déjà assez
lisses. Risque : sur-lissage si window trop grande — tester window=3 et window=5.

**Status:** [ ] pending

---

## Idée 6 — LightGBM à la place de HGBR (Expérience 06)

**Intent :** LightGBM supporte nativement les NaN, produit souvent de meilleures
performances que HGBR avec un temps d'entraînement réduit. Les hyperparamètres par
défaut de LightGBM sont plus agressifs (min_child_samples, num_leaves), ce qui nécessite
une régularisation explicite pour éviter l'overfitting.

On compare LightGBM avec les **meilleures features retenues** des expériences 02–05
pour mesurer le gain du modèle seul (à features fixées).

**Implémentation :**
- Ajouter `build_lgbm_learner()` dans [`src/parkinsons/pipeline.py`](src/parkinsons/pipeline.py)
  qui instancie `ParkinsonsPipeline` avec un `LGBMRegressor(...)` backend
- Paramètres de départ conservateurs : `min_child_samples=30`, `num_leaves=63`,
  `reg_lambda=1.0`, `n_estimators=500`, `learning_rate=0.05`, `early_stopping_rounds=20`
- Ou plus simple : créer un `ParkinsonsPipelineLGBM` qui hérite de `ParkinsonsPipeline`
  et surcharge uniquement `_make_hgbr()` par `_make_lgbm()`

**Prérequis :** `lightgbm` doit être installé dans le venv (`pip install lightgbm`).

**Relevant context:**
- [`src/parkinsons/pipeline.py`](src/parkinsons/pipeline.py) — `_make_hgbr()` à dupliquer
- [`src/parkinsons/pipeline.py:build_learner`](src/parkinsons/pipeline.py) L143–168

**Expected outcomes:** RMSE ≤ 4.20 (LightGBM dépasse généralement HGBR de ~0.1–0.3
sur des datasets tabulaires de cette taille).

**Status:** [ ] pending

---

## Idée 7 — Prior patient depuis les k-plus-proches voisins train (Expérience 07)

**Intent :** Pour les patients test ayant < 3 visites, leur slope/intercept sont peu
fiables (régression sur 1–2 points). Enrichir avec un prior issu des k=5 patients train
les plus similaires.

**Implémentation :**
- Calculer la similarité patient-à-patient via distance euclidienne sur les features
  agrégées (age médian, disease_duration médian, ledd médian, gene_enc)
- Pour chaque patient test ayant `patient_n_visits < 3`, remplacer son `patient_slope`
  et `patient_intercept` par la moyenne pondérée des k=5 patients train les plus proches
- Appliquer uniquement quand `patient_n_visits < 3` (sinon la propre régression est
  fiable)

**Relevant context:**
- [`src/parkinsons/features.py:_patient_slope_intercept`](src/parkinsons/features.py) L356–396

**Expected outcomes:** Gain marginal sur RMSE CV (les patients val ont en moyenne ~8
visites dans leur fold). Bénéfice possible sur la soumission finale (patients test avec
peu de visites).

**Status:** [ ] pending

---

## Idée 8 — Agrégats patient depuis le train sur les visites de validation (Expérience — obsolète / remplacée)

GroupKFold place **toutes** les visites d'un patient dans un seul fold, donc un patient
val n'a aucune visite dans le fold train. Cette idée est entièrement couverte par
l'Idée 7 (k-NN prior). Cette section est conservée comme note d'analyse.

---

## Idées bonus à explorer si les idées 1–5 gagnent

| # | Idée | Rationale | Risque overfitting |
|---|---|---|---|
| B1 | Target encoding de `patient_id` (LOO RMSE par patient sur le train) | Capte le niveau absolu par patient | Fort — à faire en fold-aware strict |
| B2 | Feature `age²` | Trajectoire légèrement quadratique | Faible |
| B3 | Feature `ledd_velocity` (variation LEDD entre visites) | Augmentation dose ↔ progression | Faible si calculé fold-aware |
| B4 | Isotonic regression post-hoc (enforce monotone croissant par patient) | La trajectoire ne devrait pas descendre | Peut dégrader si tendance non-monotone |
| B5 | Stacking HGBR + LGBM avec meta-learner ridge | Diversité modèle | Modéré — à tester dernier |

---

## Idées bonus backlog

| # | Idée | Rationale | Risque overfitting |
|---|---|---|---|
| B1 | Target encoding `patient_id` LOO RMSE par patient | Capte le niveau absolu par patient | Fort — fold-aware strict requis |
| B2 | Feature `age²` | Trajectoire légèrement quadratique | Faible |
| B3 | Feature `ledd_velocity` (variation LEDD entre visites) | Augmentation dose ↔ progression | Faible si calculé fold-aware |
| B4 | Isotonic regression post-hoc (enforce monotone croissant par patient) | La trajectoire ne devrait pas descendre | Peut dégrader si tendance non-monotone |
| B5 | Stacking HGBR + LGBM avec meta-learner ridge | Diversité modèle | Modéré — tester en dernier |

---

## Fichiers à créer / modifier

| Fichier | Action |
|---|---|
| `src/parkinsons/evaluate.py` | Ajouter `get_cv_folds()` + cache `data/cv_folds.pkl` |
| `src/parkinsons/features.py` | Modifier `_estimate_rater_bias`, `_apply_rater_correction`; ajouter features tendance et `on_off_ratio` |
| `src/parkinsons/pipeline.py` | Ajouter `build_lgbm_learner()` |
| `src/parkinsons/postprocess.py` | Nouveau — lissage post-hoc des prédictions |
| `experiments/02_rater_bias_split.py` | Biais rater séparé ON/OFF |
| `experiments/03_trend_features.py` | Features tendance personnalisée |
| `experiments/04_on_off_ratio.py` | Test on_off_mean vs on_off_ratio |
| `experiments/05_smooth_preds.py` | Lissage post-hoc |
| `experiments/06_lgbm.py` | LightGBM (sur les meilleures features retenues) |
| `experiments/07_patient_knn_prior.py` | Prior k-NN patient (si temps) |
| `data/cv_folds.pkl` | Folds fixés (généré au premier run) |
| `CONTEXT.md` | Mise à jour après chaque idée retenue |
| `journal/JOURNAL.md` | Mise à jour après chaque idée retenue |
