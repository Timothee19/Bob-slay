# %% [markdown]
# # Experiment 11c — Courbe ON_time+OFF par patient
#
# **Date:** 2025-01-07
# **Base:** Exp 11b (RMSE 3.381, courbe ON+OFF corrigée)
#
# ## Différence vs exp 11b
#
# Dans exp 11b la courbe combinait `on_corrected` et `off_corrected`.
# Or `on_corrected` n'est pas corrigé de l'heure de prise du médicament :
# il est systématiquement plus bas que `off_corrected`, ce qui fausse la
# courbe de progression.
#
# **Correction** : on remplace `on_corrected` par `on_time_corrected`
# (déjà corrigé de la pharmacocinétique dans `build_features`).  Les
# visites sans `on_time_corrected` valide ne contribuent pas à la courbe
# (dropna), ce qui est cohérent avec le reste du pipeline.
#
# ## Nouvelle feature
#
# **Une seule courbe par patient** construite en concaténant les points
# `(age, on_time_corrected)` et `(age, off_corrected)` pour avoir deux fois
# plus de signal par âge.  Les deux séries sont sur la même échelle
# (corrections rater + PK appliquées).  La courbe ajustée `f(age)` représente
# la progression globale corrigée du patient.
#
# Résidus :
# - `on_time_residual_from_curve` = on_time_corrected − f(age)
# - `off_residual_from_curve`     = off_corrected     − f(age)
#
# ### Anti-overfitting
# * Calcul uniquement depuis les features (pas de target).
# * Patients avec >= 4 points valides (après concat ON_time+OFF) → poly degré 2.
# * Patients avec 2–3 points → droite (degré 1).
# * Patients avec 1 seul point → constante.

# %%
import numpy as np
import pandas as pd
import skore
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.validation import check_is_fitted
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset
from parkinsons.evaluate import get_cv_folds
from parkinsons.hub import load_skore_credentials
from parkinsons.features import build_features, FEATURE_COLS
from parkinsons.postprocess import smooth_by_age_poly

# %%
cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)

DATA_DIR = PROJECT_ROOT / "data"
X_train, y_train = load_dataset(DATA_DIR)
print(f"Train: {X_train.shape}")

folds = get_cv_folds(X_train, y_train)

# %% [markdown]
# ## Courbe ON/OFF combinée par patient

# %%
def _add_onoff_curve(df: pd.DataFrame, min_pts_for_poly2: int = 4) -> pd.DataFrame:
    """Ajoute une courbe de progression ON_time+OFF par patient comme feature.

    Pour chaque patient, ajuste un polynôme sur la concaténation de ses
    points (age, on_time_corrected) et (age, off_corrected).  Les deux
    séries sont sur la même échelle (corrections rater + PK appliquées).
    Évalue la courbe à l'âge de chaque visite → `onoff_curve_at_age`.

    Le calcul n'utilise pas la target et est appliqué directement sur le
    DataFrame courant (train ou val) : GroupKFold garantit que toutes les
    visites d'un patient sont dans le même split, donc aucune fuite.

    Visites sans on_time_corrected valide ne contribuent pas à la courbe
    (dropna) mais reçoivent quand même une valeur évaluée sur la courbe
    ajustée depuis les autres points du patient.

    Patients avec >= min_pts_for_poly2 points valides → degré 2.
    Patients avec 2–3 points → degré 1 (droite).
    Patients avec 1 point   → constante.
    Patients sans points valides → NaN (LightGBM/XGB gèrent nativement).

    Parameters
    ----------
    df :
        DataFrame contenant patient_id, age, on_time_corrected, off_corrected.
    min_pts_for_poly2 :
        Nombre minimal de points valides (après concat ON_time+OFF) pour
        utiliser un polynôme de degré 2.  Default 4.

    Returns
    -------
    pd.DataFrame
        df avec colonnes onoff_curve_at_age, on_time_residual_from_curve,
        off_residual_from_curve ajoutées.
    """
    df = df.copy()
    curve_vals = np.full(len(df), np.nan)

    for pid, grp in df.groupby("patient_id"):
        ages_visit = grp["age"].values
        pos = df.index.get_indexer(grp.index)

        # Concat on_time_corrected et off_corrected (même échelle après corrections)
        on_valid  = grp[["age", "on_time_corrected"]].dropna()
        off_valid = grp[["age", "off_corrected"]].dropna()

        ages_all = np.concatenate([on_valid["age"].values, off_valid["age"].values])
        vals_all = np.concatenate([on_valid["on_time_corrected"].values,
                                   off_valid["off_corrected"].values])
        nv = len(ages_all)

        if nv == 0:
            continue  # garde NaN
        elif nv == 1:
            curve_vals[pos] = vals_all[0]
        else:
            degree = 2 if nv >= min_pts_for_poly2 else 1
            degree = min(degree, nv - 1)  # garde degrés <= nv-1
            coeffs = np.polyfit(ages_all, vals_all, degree)
            curve_vals[pos] = np.polyval(coeffs, ages_visit)

    df["onoff_curve_at_age"]           = curve_vals
    df["on_time_residual_from_curve"]  = df["on_time_corrected"] - df["onoff_curve_at_age"]
    df["off_residual_from_curve"]      = df["off_corrected"]     - df["onoff_curve_at_age"]

    # Vérification explicite : pas de patient entier avec NaN (debug)
    n_nan_patients = (
        df.groupby("patient_id")["onoff_curve_at_age"]
        .apply(lambda x: x.isna().all())
        .sum()
    )
    if n_nan_patients > 0:
        print(f"  WARNING: {n_nan_patients} patients ont onoff_curve_at_age entièrement NaN")

    return df


# %% [markdown]
# ## Feature engineering (exp 10 + courbe ON/OFF corrigée)

# %%
def add_extra_features_v5c(X_fe, X_raw, y_train=None, saved_stats=None):
    """Features exp 06-10 + courbe ON_time+OFF par patient (exp 11c)."""
    df = X_fe.copy()
    df["patient_id"] = X_raw["patient_id"].values
    df["rater_id"]   = X_raw["rater_id"].values
    df["gene"]       = X_raw["gene"].fillna("Unknown").values
    df["cohort"]     = X_raw["cohort"].values

    # -- exp-06 lags --
    ds = df.sort_values(["patient_id", "age"])
    g = ds.groupby("patient_id")
    ds["off_corrected_prev2"]     = g["off_corrected"].shift(2)
    ds["on_corrected_prev2"]      = g["on_corrected"].shift(2)
    ds["off_lag_delta"]           = ds["off_corrected"] - ds["off_corrected_prev"]
    ds["inter_visit_gap"]         = (ds["age"] - g["age"].shift(1)).clip(lower=0)
    ds["off_lag_delta_per_year"]  = ds["off_lag_delta"] / ds["inter_visit_gap"].replace(0, np.nan)
    ds["off_corrected_next2"]     = g["off_corrected"].shift(-2)
    ds = ds.loc[df.index]
    for c in ["off_corrected_prev2", "on_corrected_prev2", "off_lag_delta",
              "inter_visit_gap", "off_lag_delta_per_year", "off_corrected_next2"]:
        df[c] = ds[c].values

    df["patient_on_max"]  = df["patient_id"].map(df.groupby("patient_id")["on_corrected"].max())
    df["patient_on_q75"]  = df["patient_id"].map(df.groupby("patient_id")["on_corrected"].quantile(0.75))
    df["patient_on_std"]  = df["patient_id"].map(df.groupby("patient_id")["on_corrected"].std().fillna(0))

    on_slopes = {}
    for pid, gp in df.groupby("patient_id"):
        v = gp[["on_corrected", "age"]].dropna()
        on_slopes[pid] = float(np.polyfit(v["age"].values, v["on_corrected"].values, 1)[0]) if len(v) >= 2 else np.nan
    med_on_sl = float(np.nanmedian(list(on_slopes.values())))
    for pid in on_slopes:
        if np.isnan(on_slopes[pid]):
            on_slopes[pid] = med_on_sl
    df["patient_on_slope"] = df["patient_id"].map(on_slopes)

    df["ledd_off_ratio"]     = df["ledd"] / df["off_corrected"].replace(0, np.nan)
    df["ledd_per_off_trend"] = df["ledd"] / df["patient_trend_at_age"].replace(0, np.nan)
    df["patient_off_iqr"]    = df["patient_id"].map(
        df.groupby("patient_id")["off_corrected"].apply(
            lambda x: float(np.nanpercentile(x.dropna().values, 75) - np.nanpercentile(x.dropna().values, 25))
            if x.notna().sum() >= 2 else 0.0
        )
    )

    ds2 = df.sort_values(["patient_id", "age"])
    g2  = ds2.groupby("patient_id")
    ds2["off_cummin"]             = g2["off_corrected"].cummin()
    ds2["on_time_corrected_prev"] = g2["on_time_corrected"].shift(1)
    ds2 = ds2.loc[df.index]
    df["off_cummin"]             = ds2["off_cummin"].values
    df["on_time_corrected_prev"] = ds2["on_time_corrected_prev"].values

    # -- exp-07 --
    sl2 = {}
    for pid, gp in df.groupby("patient_id"):
        v = gp[["off_corrected", "age"]].dropna()
        sl2[pid] = float(np.polyfit(v["age"].values, v["off_corrected"].values, 2)[0]) if len(v) >= 4 else np.nan
    med_sl2 = float(np.nanmedian(list(sl2.values())))
    for pid in sl2:
        if np.isnan(sl2[pid]):
            sl2[pid] = med_sl2
    df["patient_slope2"] = df["patient_id"].map(sl2)

    ds3 = df.sort_values(["patient_id", "age"])
    g3  = ds3.groupby("patient_id")
    ds3["off_residual_prev"]      = g3["off_residual_from_trend"].shift(1)
    ds3["on_time_corrected_next"] = g3["on_time_corrected"].shift(-1)
    ds3["ledd_prev"]              = g3["ledd"].shift(1)
    ds3["ledd_next"]              = g3["ledd"].shift(-1)
    ds3["off_lag_accel"]          = (ds3["off_corrected"] - ds3["off_corrected_prev"]) - \
                                    (ds3["off_corrected_prev"] - ds3["off_corrected_prev2"])
    ds3 = ds3.loc[df.index]
    for c in ["off_residual_prev", "on_time_corrected_next", "ledd_prev", "ledd_next", "off_lag_accel"]:
        df[c] = ds3[c].values

    df["patient_ledd_mean"] = df["patient_id"].map(df.groupby("patient_id")["ledd"].mean())
    df["patient_ledd_max"]  = df["patient_id"].map(df.groupby("patient_id")["ledd"].max())
    pl_std = df.groupby("patient_id")["ledd"].std().fillna(1.0)
    df["patient_ledd_std"]  = df["patient_id"].map(pl_std)
    df["ledd_zscore"]       = (df["ledd"] - df["patient_ledd_mean"]) / df["patient_ledd_std"].replace(0, np.nan)

    # -- exp-08 fold-aware stats --
    if y_train is not None:
        dfy = df.copy()
        dfy["_t"] = y_train.values
        rm, rs, rc = {}, {}, {}
        for rid, gr in dfy.groupby("rater_id"):
            v = gr["_t"].dropna()
            rm[rid] = float(v.mean()) if len(v) > 0 else np.nan
            rs[rid] = float(v.std())  if len(v) > 1 else 0.0
            rc[rid] = len(v)
        grm = float(np.nanmean(list(rm.values())))
        grs = float(np.nanmean(list(rs.values())))
        gom = df.groupby("gene")["off_corrected"].mean().to_dict()
        gsm = {g_: float(np.mean([on_slopes[p] for p in gr["patient_id"].unique() if p in on_slopes]) or np.nan)
               for g_, gr in df.groupby("gene")}
        com = df.groupby("cohort")["off_corrected"].mean().to_dict()
        cos = df.groupby("cohort")["off_corrected"].std().to_dict()
        csm = {c_: float(np.mean([on_slopes[p] for p in gr["patient_id"].unique() if p in on_slopes]) or np.nan)
               for c_, gr in df.groupby("cohort")}
        saved_stats_out = {"rm": rm, "rs": rs, "rc": rc, "grm": grm, "grs": grs,
                           "gom": gom, "gsm": gsm, "com": com, "cos": cos, "csm": csm}
    else:
        saved_stats_out = None

    st = saved_stats if saved_stats is not None else saved_stats_out
    if st:
        df["rater_target_mean"]     = df["rater_id"].map(st["rm"]).fillna(st["grm"])
        df["rater_target_std"]      = df["rater_id"].map(st["rs"]).fillna(st["grs"])
        df["rater_target_count"]    = df["rater_id"].map(st["rc"]).fillna(0)
        df["gene_off_mean_feat"]    = df["gene"].map(st["gom"])
        df["gene_slope_mean_feat"]  = df["gene"].map(st["gsm"])
        df["cohort_off_mean_feat"]  = df["cohort"].map(st["com"])
        df["cohort_off_std_feat"]   = df["cohort"].map(st["cos"])
        df["cohort_slope_mean_feat"]= df["cohort"].map(st["csm"])
    else:
        for c in ["rater_target_mean", "rater_target_std", "rater_target_count",
                  "gene_off_mean_feat", "gene_slope_mean_feat",
                  "cohort_off_mean_feat", "cohort_off_std_feat", "cohort_slope_mean_feat"]:
            df[c] = np.nan

    mn = df.groupby("patient_id")["age"].min()
    mx = df.groupby("patient_id")["age"].max()
    ot = (mx - mn).clip(lower=0.1)
    df["patient_observation_time"] = df["patient_id"].map(ot)
    df["visits_per_year"]          = df["patient_n_visits"] / df["patient_observation_time"]
    df["dd_ledd_interaction"]      = df["disease_duration"] * df["ledd"].fillna(0)
    df["dd_ledd_velocity"]         = df["disease_duration"] * df["ledd_velocity"].fillna(0)

    ds4 = df.sort_values(["patient_id", "age"])
    ds4["off_running_std"] = ds4.groupby("patient_id")["off_corrected"].expanding().std().reset_index(level=0, drop=True)
    ds4 = ds4.loc[df.index]
    df["off_running_std"] = ds4["off_running_std"].values

    # -- exp-09 --
    ds5 = df.sort_values(["patient_id", "age"])
    g5  = ds5.groupby("patient_id")
    sh_off = g5["off_corrected"].shift(1)
    sh_on  = g5["on_corrected"].shift(1)
    ds5["off_ewma"] = sh_off.groupby(ds5["patient_id"]).transform(lambda x: x.ewm(alpha=0.5, adjust=False).mean())
    ds5["on_ewma"]  = sh_on.groupby(ds5["patient_id"]).transform(lambda x: x.ewm(alpha=0.5, adjust=False).mean())
    ds5 = ds5.loc[df.index]
    df["off_ewma"] = ds5["off_ewma"].values
    df["on_ewma"]  = ds5["on_ewma"].values

    pk = {}
    for pid, gp in df.groupby("patient_id"):
        v = gp[["on_corrected", "on_time_corrected"]].dropna()
        if len(v) >= 2:
            pk[pid] = float((v["on_corrected"] / v["on_time_corrected"].replace(0, np.nan)).median())
        else:
            pk[pid] = np.nan
    med_pk = float(np.nanmedian(list(pk.values())))
    for pid in pk:
        if np.isnan(pk[pid]):
            pk[pid] = med_pk
    df["patient_pk_ratio"] = df["patient_id"].map(pk)

    if st:
        df["off_vs_gene_mean"]   = df["off_corrected"] - df["gene_off_mean_feat"]
        df["off_vs_cohort_mean"] = df["off_corrected"] - df["cohort_off_mean_feat"]
    else:
        df["off_vs_gene_mean"]   = np.nan
        df["off_vs_cohort_mean"] = np.nan

    df["visit_rev_index"] = df["patient_n_visits"] - df["visit_index"]

    ds6 = df.sort_values(["patient_id", "age"])
    sh_ledd = ds6.groupby("patient_id")["ledd"].shift(1)
    ds6["ledd_ewma"] = sh_ledd.groupby(ds6["patient_id"]).transform(lambda x: x.ewm(alpha=0.5, adjust=False).mean())
    ds6 = ds6.loc[df.index]
    df["ledd_ewma"] = ds6["ledd_ewma"].values

    # -- exp-11b : courbe ON/OFF combinée par patient --
    # Calculée directement sur df courant (train ou val) depuis les visites du
    # patient présentes dans ce split — pas de lookup cross-fold, pas de NaN.
    df = _add_onoff_curve(df)

    df = df.drop(columns=["patient_id", "rater_id", "gene", "cohort", "patient_ledd_std"])
    return df, saved_stats_out


# %%
EXP06_COLS = ["off_corrected_prev2", "on_corrected_prev2", "off_lag_delta", "inter_visit_gap",
              "off_lag_delta_per_year", "off_corrected_next2", "patient_on_max", "patient_on_q75",
              "patient_on_std", "patient_on_slope", "ledd_off_ratio", "ledd_per_off_trend",
              "patient_off_iqr", "off_cummin", "on_time_corrected_prev"]
EXP07_COLS = ["patient_slope2", "off_residual_prev", "on_time_corrected_next", "ledd_prev", "ledd_next",
              "off_lag_accel", "patient_ledd_mean", "patient_ledd_max", "ledd_zscore",
              "rater_target_mean", "rater_target_std", "rater_target_count"]
EXP08_COLS = ["gene_off_mean_feat", "gene_slope_mean_feat", "cohort_off_mean_feat", "cohort_off_std_feat",
              "cohort_slope_mean_feat", "patient_observation_time", "visits_per_year",
              "dd_ledd_interaction", "dd_ledd_velocity", "off_running_std"]
EXP09_COLS = ["off_ewma", "on_ewma", "patient_pk_ratio", "off_vs_gene_mean", "off_vs_cohort_mean",
              "visit_rev_index", "ledd_ewma"]
EXP11C_COLS = ["onoff_curve_at_age", "on_time_residual_from_curve", "off_residual_from_curve"]
ALL_COLS = FEATURE_COLS + EXP06_COLS + EXP07_COLS + EXP08_COLS + EXP09_COLS + EXP11C_COLS
print(f"Total features: {len(ALL_COLS)}")

# %% [markdown]
# ## Model factories (identiques à exp 10)

# %%
def make_lgbm_1():
    return LGBMRegressor(
        n_estimators=3000, learning_rate=0.008, min_child_samples=40,
        reg_lambda=2.0, num_leaves=127, subsample=0.85, colsample_bytree=0.85,
        subsample_freq=1, n_jobs=-1, random_state=42, verbose=-1,
    )

def make_lgbm_2():
    return LGBMRegressor(
        n_estimators=3000, learning_rate=0.008, min_child_samples=50,
        reg_lambda=1.5, num_leaves=95, subsample=0.80, colsample_bytree=0.80,
        subsample_freq=1, n_jobs=-1, random_state=2025, verbose=-1,
    )

def make_xgb():
    return XGBRegressor(
        n_estimators=3000, learning_rate=0.008, min_child_weight=40,
        reg_lambda=2.0, max_depth=6, subsample=0.85, colsample_bytree=0.85,
        tree_method="hist", n_jobs=-1, random_state=99, verbosity=0,
    )

# %% [markdown]
# ## Smoothing utilities (identiques à exp 10)

# %%
def _smooth_isotonic_per_patient(predictions: np.ndarray, X_val: pd.DataFrame) -> np.ndarray:
    """Isotonic regression (increasing) per patient, sorted by age."""
    preds = predictions.copy().astype(float)
    df = X_val[["patient_id", "age"]].copy()
    df["_pred"] = preds
    ir = IsotonicRegression(increasing=True)
    for _pid, grp in df.groupby("patient_id"):
        if len(grp) < 2:
            continue
        grp_sorted = grp.sort_values("age")
        fitted = ir.fit_transform(grp_sorted["age"].values, grp_sorted["_pred"].values)
        df.loc[grp_sorted.index, "_pred"] = fitted
    return df["_pred"].values


def smooth_poly2_then_isotonic(predictions: np.ndarray, X_val: pd.DataFrame) -> np.ndarray:
    """Poly2 lissage par âge puis isotonic regression par patient."""
    p1 = smooth_by_age_poly(predictions, X_val, degree=2)
    return _smooth_isotonic_per_patient(p1, X_val)

# %% [markdown]
# ## Wrapper sklearn avec courbe ON/OFF corrigée et lissage intégré

# %%
class EnsemblePipelineV5cCourbe(BaseEstimator, RegressorMixin):
    """Exp-11c : ensemble exp-10 + courbe ON_time+OFF combinée par patient.

    Identique à exp-11b sauf que on_time_corrected remplace on_corrected dans
    la construction de la courbe, pour que les deux séries soient sur la même
    échelle après correction pharmacocinétique.
    """

    def __init__(self):
        pass

    def fit(self, X, y, sample_weight=None):
        X_fe, _ = build_features(X, y, X_val=None)
        X_ext, saved_stats = add_extra_features_v5c(X_fe, X, y_train=y)
        self._X_fit = X
        self._y_fit = y
        self._saved_stats = saved_stats
        X_ext = X_ext[ALL_COLS]
        cat_cols = [c for c in ALL_COLS if str(X_ext[c].dtype) == "category"]
        self._cat_dtypes = {c: X_ext[c].dtype for c in cat_cols}
        self._cat_codes  = {c: X_ext[c].cat.categories for c in cat_cols}
        y_arr = y.values if hasattr(y, "values") else y
        self.m1_ = make_lgbm_1()
        self.m1_.fit(X_ext, y_arr, categorical_feature=cat_cols)
        self.m2_ = make_lgbm_2()
        self.m2_.fit(X_ext, y_arr, categorical_feature=cat_cols)
        X_xgb = X_ext.copy()
        for c, dtype in self._cat_dtypes.items():
            X_xgb[c] = X_xgb[c].cat.codes.astype("float32")
        self.mxgb_ = make_xgb()
        self.mxgb_.fit(X_xgb.values, y_arr)
        self.feature_names_in_ = ALL_COLS
        self.n_features_in_ = len(ALL_COLS)
        return self

    def predict(self, X):
        check_is_fitted(self, "m1_")
        _, X_fe = build_features(self._X_fit, self._y_fit, X_val=X)
        # saved_stats passé pour les stats rater/gene/cohort uniquement ;
        # la courbe ON/OFF est recalculée sur X_fe (les visites val du patient)
        X_ext, _ = add_extra_features_v5c(X_fe, X, saved_stats=self._saved_stats)
        X_ext = X_ext[ALL_COLS]
        for c, dtype in self._cat_dtypes.items():
            X_ext[c] = X_ext[c].astype(dtype)
        p1 = self.m1_.predict(X_ext)
        p2 = self.m2_.predict(X_ext)
        X_xgb = X_ext.copy()
        for c, cats in self._cat_codes.items():
            cm = {v: i for i, v in enumerate(cats)}
            X_xgb[c] = X_ext[c].map(lambda x, cm=cm: cm.get(x, -1)).astype("float32")
        p_xgb = self.mxgb_.predict(X_xgb.values)
        preds_raw = (p1 + p2 + p_xgb) / 3.0
        # Lissage poly2 → isotonic intégré (identique exp 10)
        return smooth_poly2_then_isotonic(preds_raw, X)

# %% [markdown]
# ## Push vers Skore hub + comparaison fold-par-fold avec exp 10

# %%
print("Running CrossValidationReport courbe ON/OFF corrigée (n_jobs=1, 5 folds)…")
report = skore.CrossValidationReport(
    EnsemblePipelineV5cCourbe(),
    X=X_train, y=y_train, splitter=folds, n_jobs=1,
)
project.put("11c_courbe_on_off", report)
print("Report saved: 11c_courbe_on_off")

rmse_df = report.metrics.rmse()
print(rmse_df)

# %% [markdown]
# ## Comparaison fold-par-fold : exp 10 / 11b / 11c

# %%
# RMSE agrégés des expériences précédentes (lus dans leurs sorties de run)
# Exp 10 : mean=3.3997 std=0.0299  (run exp 10)
# Exp 11b: mean=3.3812 std=0.0389  (run exp 11b)
KNOWN_MEAN = {"10": 3.3997, "11b": 3.3812}
KNOWN_STD  = {"10": 0.0299, "11b": 0.0389}

rmse_11c_agg = report.metrics.rmse()
mean_11c = float(rmse_11c_agg.loc["RMSE", "mean"])
std_11c  = float(rmse_11c_agg.loc["RMSE", "std"])

# RMSE par fold pour 11c (disponible sur le report courant)
try:
    fold_series = report.metrics.rmse(data_source="X_test")
    folds_11c = fold_series.values.flatten().tolist()
    n_folds = len(folds_11c)
except Exception as e:
    folds_11c = None
    n_folds = 5
    print(f"(fold-level non disponible pour 11c : {e})")

print("\n" + "=" * 65)
print(f"  {'':6} {'Exp 10':>10} {'Exp 11b':>10} {'Exp 11c':>10}   Δ11cvs10")
print("-" * 65)

if folds_11c:
    for i, r11c in enumerate(folds_11c):
        print(f"  Fold {i+1}      N/A          N/A   {r11c:>10.4f}")
else:
    for i in range(n_folds):
        print(f"  Fold {i+1}      N/A          N/A          N/A")

print("-" * 65)
d_mean_11b = KNOWN_MEAN["11b"] - KNOWN_MEAN["10"]
d_mean_11c = mean_11c - KNOWN_MEAN["10"]
sign_b = "+" if d_mean_11b > 0 else ""
sign_c = "+" if d_mean_11c > 0 else ""
print(f"  Mean   {KNOWN_MEAN['10']:>10.4f} {KNOWN_MEAN['11b']:>10.4f} {mean_11c:>10.4f}   {sign_c}{d_mean_11c:.4f}")
print(f"  Std    {KNOWN_STD['10']:>10.4f} {KNOWN_STD['11b']:>10.4f} {std_11c:>10.4f}")
print(f"  Δvs10           {sign_b}{d_mean_11b:>9.4f} {sign_c}{d_mean_11c:>9.4f}")
print("=" * 65)
