# %% [markdown]
# # Experiment 12 — Stacking 2 étages
#
# **Date:** 2025-01-07
# **Base:** Exp 11b (RMSE 3.381, ensemble3 + courbe ON/OFF + lissage poly2→isotonic)
#
# ## Architecture
#
# ### Étage 1 : LightGBM unique
# Un seul `LGBMRegressor` entraîné sur les features 11b (`ALL_COLS`) pour
# produire une prédiction de la target.
#
# ### Étage 2 : ensemble 11b augmenté
# Le même ensemble (LGBM×2 + XGB) de l'exp 11b, avec en plus trois nouvelles
# features issues des prédictions de l'étage 1 :
#
# - `s1_pred_curr`  : prédiction étage 1 pour la visite actuelle
# - `s1_pred_prev`  : prédiction étage 1 pour la visite précédente du même
#                     patient (triées par âge), NaN pour la première visite
# - `s1_pred_next`  : prédiction étage 1 pour la visite suivante, NaN pour
#                     la dernière visite
#
# ## Anti-overfitting
#
# Les prédictions de l'étage 1 sur le **fold d'entraînement** sont produites
# par validation croisée interne (GroupKFold à 4 folds sur les patients du
# fold train) : chaque patient est prédit par un modèle étage 1 qui ne l'a
# jamais vu.  L'étage 2 reçoit donc des prédictions hors-échantillon pour
# toutes les lignes d'entraînement, ce qui évite le sur-apprentissage de
# l'étage 1.
#
# Pour les données de **validation** (ou test), l'étage 1 est entraîné sur
# la totalité du fold d'entraînement.
#
# ## Comparaison en cours de run
#
# Le RMSE de chaque fold (étage 2 avec lissage) est affiché pendant le run
# via une boucle manuelle.  Le rapport skore est ensuite poussé pour
# comparaison officielle.

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
# ## Courbe ON/OFF par patient (identique exp 11b)

# %%
def _add_onoff_curve(df: pd.DataFrame, min_pts_for_poly2: int = 4) -> pd.DataFrame:
    """Courbe ON+OFF combinée par patient — identique à exp 11b."""
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
# ## Feature engineering (identique exp 11b)

# %%
def add_extra_features_v5b(X_fe, X_raw, y_train=None, saved_stats=None):
    """Features exp 06-11b (identiques à exp 11b)."""
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

    # -- exp-11b : courbe ON/OFF combinée --
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
# Features étage 1 (prédictions stacking)
STACK_COLS = ["s1_pred_curr", "s1_pred_prev", "s1_pred_next"]

ALL_COLS_BASE = FEATURE_COLS + EXP06_COLS + EXP07_COLS + EXP08_COLS + EXP09_COLS + EXP11B_COLS
ALL_COLS      = ALL_COLS_BASE + STACK_COLS
print(f"Base features (étage 1): {len(ALL_COLS_BASE)}")
print(f"Total features (étage 2): {len(ALL_COLS)}")

# %% [markdown]
# ## Factories de modèles

# %%
def make_lgbm_stage1():
    """Étage 1 : LightGBM unique, identique à make_lgbm_1 de l'exp 11b."""
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
# ## Lissage (identique exp 10/11b)

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
def _add_stage1_lag_features(
    X_raw: pd.DataFrame,
    s1_preds: np.ndarray,
) -> pd.DataFrame:
    """Ajoute s1_pred_curr/prev/next en triant par âge au sein de chaque patient.

    Parameters
    ----------
    X_raw :
        DataFrame raw (doit contenir patient_id et age), aligné avec s1_preds.
    s1_preds :
        Prédictions de l'étage 1, alignées avec X_raw.

    Returns
    -------
    pd.DataFrame avec colonnes s1_pred_curr, s1_pred_prev, s1_pred_next.
    """
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
# ## Wrapper sklearn stacking 2 étages

# %%
class StackingPipeline12(BaseEstimator, RegressorMixin):
    """Exp-12 : stacking 2 étages sur base exp-11b.

    fit() :
      1. Construit les features 11b sur le fold train (build_features +
         add_extra_features_v5b).
      2. Produit les prédictions étage 1 hors-échantillon sur le train via
         GroupKFold interne (4 folds sur les patients du train).
      3. Ajoute s1_pred_curr/prev/next aux features train.
      4. Entraîne l'étage 1 complet sur tout le train (pour predict).
      5. Entraîne l'ensemble étage 2 (LGBM×2 + XGB) sur les features
         augmentées.

    predict() :
      1. Construit les features 11b sur X_val.
      2. Prédit avec l'étage 1 complet (entraîné sur tout le train).
      3. Ajoute s1_pred_curr/prev/next.
      4. Prédit avec l'étage 2 et applique le lissage poly2→isotonic.
    """

    def __init__(self, n_inner_folds: int = 4):
        self.n_inner_folds = n_inner_folds

    def _build_features(self, X, y=None, saved_stats=None):
        """Construit features 11b + retourne saved_stats."""
        if y is not None:
            X_fe, _ = build_features(X, y, X_val=None)
            X_ext, stats = add_extra_features_v5b(X_fe, X, y_train=y)
        else:
            _, X_fe = build_features(self._X_fit, self._y_fit, X_val=X)
            X_ext, stats = add_extra_features_v5b(X_fe, X, saved_stats=saved_stats)
        return X_ext, stats

    def _prepare_for_lgbm(self, X_ext, all_cols, cat_dtypes=None):
        """Sélectionne les colonnes et aligne les dtypes catégoriels."""
        X_sel = X_ext[all_cols].copy()
        if cat_dtypes is not None:
            for c, dtype in cat_dtypes.items():
                if c in X_sel.columns:
                    X_sel[c] = X_sel[c].astype(dtype)
        return X_sel

    def fit(self, X, y, sample_weight=None):
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)

        # ---- Étape 1 : features 11b sur le fold train ----
        X_ext_tr, saved_stats = self._build_features(X, y=y)
        self._X_fit      = X
        self._y_fit      = y
        self._saved_stats = saved_stats

        X_base = X_ext_tr[ALL_COLS_BASE].copy()

        # Dtypes catégoriels pour l'étage 1 et l'étage 2
        cat_cols = [c for c in ALL_COLS_BASE if str(X_base[c].dtype) == "category"]
        self._cat_dtypes = {c: X_base[c].dtype for c in cat_cols}
        self._cat_codes  = {c: X_base[c].cat.categories for c in cat_cols}

        # ---- Étape 2 : prédictions étage 1 hors-échantillon sur le train ----
        # GroupKFold interne sur les patients du fold train pour éviter la fuite.
        patient_groups = X["patient_id"].values
        inner_gkf = GroupKFold(n_splits=self.n_inner_folds)
        s1_oof = np.full(len(X), np.nan)

        for inner_tr_idx, inner_val_idx in inner_gkf.split(X, y, groups=patient_groups):
            X_inner_tr = X_base.iloc[inner_tr_idx]
            y_inner_tr = y_arr[inner_tr_idx]
            X_inner_val = X_base.iloc[inner_val_idx]

            m_s1_inner = make_lgbm_stage1()
            m_s1_inner.fit(X_inner_tr, y_inner_tr, categorical_feature=cat_cols)
            s1_oof[inner_val_idx] = m_s1_inner.predict(X_inner_val)

        # ---- Étape 3 : lags des prédictions OOF ----
        lag_tr = _add_stage1_lag_features(X, s1_oof)

        # ---- Étape 4 : entraîner l'étage 1 complet sur tout le fold train ----
        self.m_s1_ = make_lgbm_stage1()
        self.m_s1_.fit(X_base, y_arr, categorical_feature=cat_cols)

        # ---- Étape 5 : features augmentées pour l'étage 2 ----
        X_aug = X_base.copy()
        for col in STACK_COLS:
            X_aug[col] = lag_tr[col].values

        # Préparer XGB (codes entiers pour les catégoriels)
        X_xgb = X_aug.copy()
        for c, dtype in self._cat_dtypes.items():
            X_xgb[c] = X_xgb[c].cat.codes.astype("float32")

        # ---- Étape 6 : entraîner l'étage 2 ----
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

        # ---- Features 11b sur le split de validation ----
        X_ext_val, _ = self._build_features(X, saved_stats=self._saved_stats)
        X_base_val = self._prepare_for_lgbm(X_ext_val, ALL_COLS_BASE, self._cat_dtypes)

        # ---- Prédictions étage 1 (modèle entraîné sur tout le fold train) ----
        s1_val = self.m_s1_.predict(X_base_val)

        # ---- Lags des prédictions étage 1 ----
        lag_val = _add_stage1_lag_features(X, s1_val)

        # ---- Features augmentées ----
        X_aug_val = X_base_val.copy()
        for col in STACK_COLS:
            X_aug_val[col] = lag_val[col].values

        # Prédictions étage 2
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
print("=" * 55)
print("Boucle manuelle fold-par-fold (étage 2 + lissage)")
print("=" * 55)

fold_rmses = []
for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    X_tr = X_train.iloc[tr_idx].reset_index(drop=True)
    y_tr = y_train.iloc[tr_idx].reset_index(drop=True)
    X_val = X_train.iloc[val_idx].reset_index(drop=True)
    y_val = y_train.iloc[val_idx].reset_index(drop=True)

    model = StackingPipeline12()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)

    rmse_fold = root_mean_squared_error(y_val, preds)
    fold_rmses.append(rmse_fold)
    print(f"  Fold {fold_idx + 1}  RMSE = {rmse_fold:.4f}")

mean_rmse = float(np.mean(fold_rmses))
std_rmse  = float(np.std(fold_rmses))
print("-" * 55)
print(f"  Mean RMSE = {mean_rmse:.4f}  std = {std_rmse:.4f}")
print(f"  Exp 11b   = 3.3812  std = 0.0389")
delta = mean_rmse - 3.3812
sign = "+" if delta > 0 else ""
print(f"  Delta     = {sign}{delta:.4f}  ({'worse' if delta > 0 else 'better'})")
print("=" * 55)

# %% [markdown]
# ## Push vers Skore hub

# %%
print("\nRunning CrossValidationReport stacking (n_jobs=1)…")
report = skore.CrossValidationReport(
    StackingPipeline12(),
    X=X_train, y=y_train, splitter=folds, n_jobs=1,
)
project.put("12_stacking", report)
print("Report saved: 12_stacking")
print(report.metrics.rmse())
