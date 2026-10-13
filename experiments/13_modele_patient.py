# %% [markdown]
# # Experiment 13 — Modèle patient (correction du biais inter-patient)
#
# **Date:** 2025-01-07
# **Base:** Exp 12 (stacking 2 étages, RMSE 3.31)
#
# ## Motivation
#
# Le diagnostic sur les prédictions OOF de l'exp 12 révèle que 67.6 % du MSE
# total provient d'un biais moyen constant par patient (inter-patient), et
# seulement 32.4 % de la variance intra-patient.  Corriger ce biais en
# prédisant la moyenne-cible de chaque patient est donc la piste prioritaire.
#
# ## Architecture
#
# ### Étage 0 : modèle patient
# Un LightGBM entraîné sur une table agrégée (une ligne par patient) pour
# prédire la moyenne des targets du patient.  Features :
#   - moyennes, max, écarts-types de on_corrected et off_corrected
#   - nombre de visites, gene, cohort, ledd moyen, age min / max
#
# ### Étages 1 & 2 : identiques à l'exp 12
# La prédiction du modèle patient est ajoutée comme feature supplémentaire
# à l'étage 2 (en plus des features de stacking).
#
# ## Anti-overfitting
#
# Dans chaque fold externe :
#   - Sur le **fold train** : les prédictions du modèle patient sont produites
#     par GroupKFold interne (4 folds sur les patients du train), exactement
#     comme les prédictions de l'étage 1.
#   - Sur le **fold val** : le modèle patient est entraîné sur tous les patients
#     du fold train, puis appliqué aux patients du fold val.
#
# ## Comparaison
#
# Le RMSE de chaque fold (étage 2 + lissage) est affiché pendant le run.
# Le rapport skore est poussé sous "13_modele_patient" pour comparaison.

# %%
import numpy as np
import pandas as pd
import skore
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
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
# ## Courbe ON/OFF par patient (identique exp 12)

# %%
def _add_onoff_curve(df: pd.DataFrame, min_pts_for_poly2: int = 4) -> pd.DataFrame:
    """Courbe ON+OFF combinée par patient."""
    df = df.copy()
    curve_vals = np.full(len(df), np.nan)

    for pid, grp in df.groupby("patient_id"):
        ages_visit = grp["age"].values
        pos = df.index.get_indexer(grp.index)

        on_valid  = grp[["age", "on_corrected"]].dropna()
        off_valid = grp[["age", "off_corrected"]].dropna()

        ages_all = np.concatenate([on_valid["age"].values, off_valid["age"].values])
        vals_all = np.concatenate([on_valid["on_corrected"].values,
                                   off_valid["off_corrected"].values])
        nv = len(ages_all)

        if nv == 0:
            continue
        elif nv == 1:
            curve_vals[pos] = vals_all[0]
        else:
            degree = 2 if nv >= min_pts_for_poly2 else 1
            degree = min(degree, nv - 1)
            coeffs = np.polyfit(ages_all, vals_all, degree)
            curve_vals[pos] = np.polyval(coeffs, ages_visit)

    df["onoff_curve_at_age"]      = curve_vals
    df["on_residual_from_curve"]  = df["on_corrected"]  - df["onoff_curve_at_age"]
    df["off_residual_from_curve"] = df["off_corrected"] - df["onoff_curve_at_age"]
    return df


# %% [markdown]
# ## Feature engineering (identique exp 12)

# %%
def add_extra_features_v5b(X_fe, X_raw, y_train=None, saved_stats=None):
    """Features exp 06-11b (identiques à exp 12)."""
    df = X_fe.copy()
    df["patient_id"] = X_raw["patient_id"].values
    df["rater_id"]   = X_raw["rater_id"].values
    df["gene"]       = X_raw["gene"].fillna("Unknown").values
    df["cohort"]     = X_raw["cohort"].values

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
EXP11B_COLS = ["onoff_curve_at_age", "on_residual_from_curve", "off_residual_from_curve"]
# Features étage 1 (stacking) + modèle patient
STACK_COLS = ["s1_pred_curr", "s1_pred_prev", "s1_pred_next"]
PAT_COLS   = ["patient_mean_pred"]

ALL_COLS_BASE = FEATURE_COLS + EXP06_COLS + EXP07_COLS + EXP08_COLS + EXP09_COLS + EXP11B_COLS
ALL_COLS      = ALL_COLS_BASE + STACK_COLS + PAT_COLS
print(f"Base features (étage 1): {len(ALL_COLS_BASE)}")
print(f"Total features (étage 2 + patient): {len(ALL_COLS)}")

# %% [markdown]
# ## Table patient et modèle patient

# %%
def build_patient_table(X_raw: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Construit une table avec une ligne par patient pour le modèle patient.

    Travaille sur les colonnes brutes disponibles dans X_raw (on, off, ledd,
    age, gene, cohort) pour être utilisable à chaque inner fold sans avoir à
    reconstruire les features engineered.

    Parameters
    ----------
    X_raw :
        DataFrame brut contenant patient_id, on, off, gene, cohort, ledd, age.
    y :
        Targets alignées avec X_raw.

    Returns
    -------
    pd.DataFrame
        Table agrégée (index = patient_id) avec la colonne cible
        ``patient_target_mean``.
    """
    df = X_raw.copy()
    df["_target"] = y.values

    agg = df.groupby("patient_id").agg(
        on_mean     = ("on", "mean"),
        on_max      = ("on", "max"),
        on_std      = ("on", "std"),
        off_mean    = ("off", "mean"),
        off_max     = ("off", "max"),
        off_std     = ("off", "std"),
        n_visits    = ("patient_id", "count"),
        ledd_mean   = ("ledd", "mean"),
        age_min     = ("age", "min"),
        age_max     = ("age", "max"),
        patient_target_mean = ("_target", "mean"),
    )

    # Catégoriels : gene et cohort (mode)
    agg["gene"]   = df.groupby("patient_id")["gene"].agg(lambda x: x.fillna("Unknown").mode().iloc[0])
    agg["cohort"] = df.groupby("patient_id")["cohort"].agg(lambda x: x.mode().iloc[0])

    # Remplir NaN dans on/off std (patients à 1 seule visite)
    agg["on_std"]  = agg["on_std"].fillna(0.0)
    agg["off_std"] = agg["off_std"].fillna(0.0)

    # Encoder gene et cohort en entiers (LightGBM n'accepte pas les str)
    agg["gene"]   = pd.Categorical(agg["gene"]).codes.astype("int32")
    agg["cohort"] = pd.Categorical(agg["cohort"]).codes.astype("int32")

    return agg


def make_patient_model() -> LGBMRegressor:
    """LightGBM léger pour prédire la moyenne-cible par patient."""
    return LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        subsample_freq=1,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )


# %% [markdown]
# ## Factories de modèles (étages 1 & 2, identiques à exp 12)

# %%
def make_lgbm_stage1():
    return LGBMRegressor(
        n_estimators=3000, learning_rate=0.008, min_child_samples=40,
        reg_lambda=2.0, num_leaves=127, subsample=0.85, colsample_bytree=0.85,
        subsample_freq=1, n_jobs=-1, random_state=42, verbose=-1,
    )

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
# ## Lissage (identique exp 12)

# %%
def _smooth_isotonic_per_patient(predictions: np.ndarray, X_val: pd.DataFrame) -> np.ndarray:
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
    p1 = smooth_by_age_poly(predictions, X_val, degree=2)
    return _smooth_isotonic_per_patient(p1, X_val)

# %% [markdown]
# ## Utilitaire : lag des prédictions étage 1 par patient

# %%
def _add_stage1_lag_features(X_raw: pd.DataFrame, s1_preds: np.ndarray) -> pd.DataFrame:
    df = X_raw[["patient_id", "age"]].copy()
    df["s1_pred_curr"] = s1_preds

    df_sorted = df.sort_values(["patient_id", "age"])
    grp = df_sorted.groupby("patient_id")
    df_sorted["s1_pred_prev"] = grp["s1_pred_curr"].shift(1)
    df_sorted["s1_pred_next"] = grp["s1_pred_curr"].shift(-1)
    df_sorted = df_sorted.loc[df.index]

    result = df.copy()
    result["s1_pred_prev"] = df_sorted["s1_pred_prev"].values
    result["s1_pred_next"] = df_sorted["s1_pred_next"].values
    return result[["s1_pred_curr", "s1_pred_prev", "s1_pred_next"]]

# %% [markdown]
# ## Pipeline stacking 3 étages (exp 13)

# %%
class StackingPipeline13(BaseEstimator, RegressorMixin):
    """Exp-13 : exp-12 + modèle patient (étage 0) pour corriger le biais inter-patient.

    fit() :
      1. Table patient sur le fold train → entraîne modèle patient via
         GroupKFold interne (4 folds) pour produire des prédictions OOF.
      2. Features 11b sur le fold train.
      3. Prédictions étage 1 OOF via GroupKFold interne (4 folds, mêmes splits).
      4. Ajoute s1_pred_curr/prev/next + patient_mean_pred aux features train.
      5. Entraîne le modèle patient complet et l'étage 1 complet (pour predict).
      6. Entraîne l'ensemble étage 2 (LGBM×2 + XGB) sur les features augmentées.

    predict() :
      1. Construit features 11b sur X_val.
      2. Prédiction modèle patient (entraîné sur tout le fold train).
      3. Prédiction étage 1 (entraîné sur tout le fold train).
      4. Ajoute s1_pred_curr/prev/next + patient_mean_pred.
      5. Prédiction étage 2 + lissage poly2→isotonic.
    """

    def __init__(self, n_inner_folds: int = 4):
        self.n_inner_folds = n_inner_folds

    def _build_features(self, X, y=None, saved_stats=None):
        if y is not None:
            X_fe, _ = build_features(X, y, X_val=None)
            X_ext, stats = add_extra_features_v5b(X_fe, X, y_train=y)
        else:
            _, X_fe = build_features(self._X_fit, self._y_fit, X_val=X)
            X_ext, stats = add_extra_features_v5b(X_fe, X, saved_stats=saved_stats)
        return X_ext, stats

    def _prepare_for_lgbm(self, X_ext, all_cols, cat_dtypes=None):
        X_sel = X_ext[all_cols].copy()
        if cat_dtypes is not None:
            for c, dtype in cat_dtypes.items():
                if c in X_sel.columns:
                    X_sel[c] = X_sel[c].astype(dtype)
        return X_sel

    def _patient_model_oof(
        self,
        X_raw: pd.DataFrame,
        y_arr: np.ndarray,
        patient_groups: np.ndarray,
    ) -> np.ndarray:
        """Prédictions OOF du modèle patient via GroupKFold interne.

        Entraîne le modèle patient (table agrégée par patient) sur les patients
        internes au fold train, et prédit les patients du fold val interne.
        Retourne un array de prédictions alignées avec X_raw (une valeur par
        visite = la prédiction de la moyenne-cible du patient correspondant).
        """
        y_ser = pd.Series(y_arr, index=X_raw.index)
        inner_gkf = GroupKFold(n_splits=self.n_inner_folds)
        pat_oof = np.full(len(X_raw), np.nan)

        # Mapping patient_id → indice de position dans X_raw (pour l'OOF)
        pid_to_positions: dict = {}
        for pos, pid in enumerate(patient_groups):
            pid_to_positions.setdefault(pid, []).append(pos)

        for inner_tr_idx, inner_val_idx in inner_gkf.split(X_raw, y_ser, groups=patient_groups):
            X_inner_tr = X_raw.iloc[inner_tr_idx].reset_index(drop=True)
            y_inner_tr = y_ser.iloc[inner_tr_idx].reset_index(drop=True)
            X_inner_val = X_raw.iloc[inner_val_idx]

            # Table patient sur le sous-ensemble interne train
            pat_table_tr = build_patient_table(X_inner_tr, y_inner_tr)
            pat_feats = [c for c in pat_table_tr.columns if c != "patient_target_mean"]
            y_pat_tr  = pat_table_tr["patient_target_mean"]
            X_pat_tr  = pat_table_tr[pat_feats]

            m_pat = make_patient_model()
            m_pat.fit(X_pat_tr, y_pat_tr)

            # Prédire les patients de inner_val (unseen)
            val_pids = X_inner_val["patient_id"].unique()
            # Table patient pour les patients val (on utilise leurs features X,
            # mais pas leurs targets — on met target=0 et on prédit ensuite)
            pat_table_val = build_patient_table(X_inner_val, pd.Series(np.zeros(len(X_inner_val)), index=X_inner_val.index))
            X_pat_val = pat_table_val[pat_feats]
            pat_preds_val = m_pat.predict(X_pat_val)  # indexé par patient_id

            # Propager la prédiction patient → chaque visite du fold val interne
            pred_series = pd.Series(pat_preds_val, index=val_pids)
            for i, pid in enumerate(X_inner_val["patient_id"].values):
                pat_oof[inner_val_idx[i]] = pred_series.get(pid, np.nan)

        return pat_oof

    def fit(self, X, y, sample_weight=None):
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)

        # ── Étape 1 : features 11b sur le fold train ──────────────────────
        X_ext_tr, saved_stats = self._build_features(X, y=y)
        self._X_fit      = X
        self._y_fit      = y
        self._saved_stats = saved_stats

        X_base = X_ext_tr[ALL_COLS_BASE].copy()
        cat_cols = [c for c in ALL_COLS_BASE if str(X_base[c].dtype) == "category"]
        self._cat_dtypes = {c: X_base[c].dtype for c in cat_cols}
        self._cat_codes  = {c: X_base[c].cat.categories for c in cat_cols}

        patient_groups = X["patient_id"].values
        inner_gkf = GroupKFold(n_splits=self.n_inner_folds)

        # ── Étape 2 : prédictions OOF du modèle patient ───────────────────
        pat_oof = self._patient_model_oof(X, y_arr, patient_groups)

        # ── Étape 3 : prédictions OOF de l'étage 1 ───────────────────────
        s1_oof = np.full(len(X), np.nan)
        for inner_tr_idx, inner_val_idx in inner_gkf.split(X, y, groups=patient_groups):
            m_s1_inner = make_lgbm_stage1()
            m_s1_inner.fit(
                X_base.iloc[inner_tr_idx], y_arr[inner_tr_idx],
                categorical_feature=cat_cols,
            )
            s1_oof[inner_val_idx] = m_s1_inner.predict(X_base.iloc[inner_val_idx])

        # ── Étape 4 : lags des prédictions OOF étage 1 ───────────────────
        lag_tr = _add_stage1_lag_features(X, s1_oof)

        # ── Étape 5 : entraîner le modèle patient complet ─────────────────
        pat_table_full = build_patient_table(X, pd.Series(y_arr, index=X.index))
        pat_feats_full = [c for c in pat_table_full.columns if c != "patient_target_mean"]
        self._pat_feats  = pat_feats_full
        self.m_pat_      = make_patient_model()
        self.m_pat_.fit(pat_table_full[pat_feats_full], pat_table_full["patient_target_mean"])
        # Moyenne globale des targets pour les nouveaux patients (fallback)
        self._pat_global_mean = float(pat_table_full["patient_target_mean"].mean())

        # ── Étape 6 : entraîner l'étage 1 complet ────────────────────────
        self.m_s1_ = make_lgbm_stage1()
        self.m_s1_.fit(X_base, y_arr, categorical_feature=cat_cols)

        # ── Étape 7 : features augmentées pour l'étage 2 ─────────────────
        X_aug = X_base.copy()
        for col in STACK_COLS:
            X_aug[col] = lag_tr[col].values
        X_aug["patient_mean_pred"] = pat_oof

        X_xgb = X_aug.copy()
        for c, dtype in self._cat_dtypes.items():
            X_xgb[c] = X_xgb[c].cat.codes.astype("float32")

        # ── Étape 8 : entraîner l'étage 2 ────────────────────────────────
        self.m1_ = make_lgbm_1()
        self.m1_.fit(X_aug, y_arr, categorical_feature=cat_cols)
        self.m2_ = make_lgbm_2()
        self.m2_.fit(X_aug, y_arr, categorical_feature=cat_cols)
        self.mxgb_ = make_xgb()
        self.mxgb_.fit(X_xgb.values, y_arr)

        self.feature_names_in_ = ALL_COLS
        self.n_features_in_    = len(ALL_COLS)
        return self

    def predict(self, X):
        check_is_fitted(self, "m1_")

        # ── Features 11b sur le split de validation ───────────────────────
        X_ext_val, _ = self._build_features(X, saved_stats=self._saved_stats)
        X_base_val = self._prepare_for_lgbm(X_ext_val, ALL_COLS_BASE, self._cat_dtypes)

        # ── Prédiction modèle patient ─────────────────────────────────────
        # Table patient pour les patients du fold val
        pat_table_val = build_patient_table(X, pd.Series(np.zeros(len(X)), index=X.index))
        X_pat_val = pat_table_val[self._pat_feats]
        pat_pred_by_pid = pd.Series(
            self.m_pat_.predict(X_pat_val),
            index=pat_table_val.index,
        )
        pat_pred_visits = X["patient_id"].map(pat_pred_by_pid).fillna(self._pat_global_mean).values

        # ── Prédictions étage 1 ───────────────────────────────────────────
        s1_val = self.m_s1_.predict(X_base_val)
        lag_val = _add_stage1_lag_features(X, s1_val)

        # ── Features augmentées ───────────────────────────────────────────
        X_aug_val = X_base_val.copy()
        for col in STACK_COLS:
            X_aug_val[col] = lag_val[col].values
        X_aug_val["patient_mean_pred"] = pat_pred_visits

        # ── Prédictions étage 2 ───────────────────────────────────────────
        p1 = self.m1_.predict(X_aug_val)
        p2 = self.m2_.predict(X_aug_val)

        X_xgb_val = X_aug_val.copy()
        for c, cats in self._cat_codes.items():
            cm = {v: i for i, v in enumerate(cats)}
            X_xgb_val[c] = X_aug_val[c].map(lambda x, cm=cm: cm.get(x, -1)).astype("float32")
        p_xgb = self.mxgb_.predict(X_xgb_val.values)

        preds_raw = (p1 + p2 + p_xgb) / 3.0
        return smooth_poly2_then_isotonic(preds_raw, X)

# %% [markdown]
# ## Boucle manuelle pour afficher le RMSE fold par fold

# %%
print("=" * 60)
print("Boucle manuelle fold-par-fold (étage 2 + lissage)")
print("=" * 60)

fold_rmses = []
for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    X_tr  = X_train.iloc[tr_idx].reset_index(drop=True)
    y_tr  = y_train.iloc[tr_idx].reset_index(drop=True)
    X_val = X_train.iloc[val_idx].reset_index(drop=True)
    y_val = y_train.iloc[val_idx].reset_index(drop=True)

    model = StackingPipeline13()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)

    rmse_fold = root_mean_squared_error(y_val, preds)
    fold_rmses.append(rmse_fold)
    print(f"  Fold {fold_idx + 1}  RMSE = {rmse_fold:.4f}", flush=True)

mean_rmse = float(np.mean(fold_rmses))
std_rmse  = float(np.std(fold_rmses))
print("-" * 60)
print(f"  Mean RMSE = {mean_rmse:.4f}  std = {std_rmse:.4f}")
print(f"  Exp 12    = 3.3090  std = ?")
delta = mean_rmse - 3.3090
sign = "+" if delta > 0 else ""
print(f"  Delta     = {sign}{delta:.4f}  ({'worse' if delta > 0 else 'better'})")
print("=" * 60)

# %% [markdown]
# ## Push vers Skore hub

# %%
print("\nRunning CrossValidationReport (n_jobs=1)…")
report = skore.CrossValidationReport(
    StackingPipeline13(),
    X=X_train, y=y_train, splitter=folds, n_jobs=1,
)
project.put("13_modele_patient", report)
print("Report saved: 13_modele_patient")
print(report.metrics.rmse())
