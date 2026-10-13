# %% [markdown]
# # Experiment 18 — Heterogeneous Voting Ensemble (LightGBM + XGBoost + HistGBR) + Poly2+Isotonic
#
# **Date:** 2025-01-10
# **Base:** Exp 16 feature set (76 cols after correlation pruning) + NoisyRaterEncoder
#
# ## Objective
#
# Combine the algorithmic diversity of our best single models (Exp 08/11 gradient boosting
# ensemble and Exp 16 tuned HistGBR) into a heterogeneous voting regressor to push RMSE
# below the current record of 3.45.
#
# ## Architecture
#
# 1. Feature engineering: full pipeline from Exp 16 — lags, Bateman PK, longitudinal
#    aggregates per patient, NoisyRaterEncoder for rater_id (leak-safe).
# 2. Correlation pruning: drop one column from each |r| > 0.98 pair → 76-feature matrix.
# 3. Voting ensemble:
#    - Model A: LGBMRegressor  (lr=0.015, 1000 trees, subsample=0.8, colsample=0.6, λ=5.0)
#    - Model B: XGBRegressor   (lr=0.015, 1000 trees, max_depth=5,  subsample=0.8, λ=5.0)
#    - Model C: HistGradientBoostingRegressor (lr=0.03, 900 iters, max_leaf=45, λ=20.0)
#    - Raw prediction = (pred_lgb + pred_xgb + pred_hgbr) / 3
# 4. Post-processing: per-patient poly2 smooth → isotonic → clip [0.0, 109.5].
# 5. Validation: GroupKFold(n_splits=5) grouped on patient_id.
# 6. Skore evaluation and push, then full-train refit and test-set inference.

# %%
import numpy as np
import pandas as pd
import skore
import skrub
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.evaluate import get_cv_folds
from parkinsons.features import build_features, FEATURE_COLS
from parkinsons.hub import load_skore_credentials

# ── Skore login ────────────────────────────────────────────────────────────────
cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)

DATA_DIR = PROJECT_ROOT / "data"
EXP_KEY  = "18_ensemble_heterogene_poly2iso"

# ── Pharmacokinetic parameters (Bateman biexponential) ─────────────────────────
_KA = 1.5
_KE = 0.46


# ============================================================
# § 1 — Feature column groups (same as Exp 16)
# ============================================================

EXP06_COLS = [
    "off_corrected_prev2", "on_corrected_prev2", "off_lag_delta",
    "inter_visit_gap", "off_lag_delta_per_year", "off_corrected_next2",
    "patient_on_max", "patient_on_q75", "patient_on_std", "patient_on_slope",
    "ledd_off_ratio", "ledd_per_off_trend", "patient_off_iqr",
    "off_cummin", "on_time_corrected_prev",
]
EXP07_COLS = [
    "patient_slope2",
    "off_residual_prev", "on_time_corrected_next",
    "ledd_prev", "ledd_next", "off_lag_accel",
    "patient_ledd_mean", "patient_ledd_max", "ledd_zscore",
    "rater_target_mean", "rater_target_std", "rater_target_count",
]
EXP08_COLS = [
    "gene_off_mean_feat", "gene_slope_mean_feat",
    "cohort_off_mean_feat", "cohort_off_std_feat", "cohort_slope_mean_feat",
    "patient_observation_time", "visits_per_year",
    "dd_ledd_interaction", "dd_ledd_velocity",
    "off_running_std",
]
EXP11_COLS = [
    "off_ewma", "on_ewma",
    "pk_concentration_on", "pk_concentration_off",
    "off_vs_gene_mean", "off_vs_cohort_mean",
    "visit_rev_index",
    "ledd_ewma",
]

ALL_EXTRA_COLS = EXP06_COLS + EXP07_COLS + EXP08_COLS + EXP11_COLS


# ============================================================
# § 2 — Bateman PK helper
# ============================================================

def compute_bateman_concentration(
    ledd: pd.Series,
    time_since_intake: pd.Series,
    ka: float = _KA,
    ke: float = _KE,
) -> pd.Series:
    """Plasma levodopa concentration via the Bateman biexponential equation."""
    safe_ledd = ledd.fillna(0.0)
    safe_time = time_since_intake.fillna(0.0)
    c = (safe_ledd * ka / (ka - ke)) * (
        np.exp(-ke * safe_time) - np.exp(-ka * safe_time)
    )
    mask = ledd.isna() | time_since_intake.isna()
    c[mask] = np.nan
    return c


# ============================================================
# § 3 — Full extra feature engineering (Exp 16 version)
# ============================================================

def add_extra_features_v4(X_fe, X_raw, y_train=None, saved_stats=None):
    """Compute the full set of extra features used in experiments 6–11 and 14."""
    df = X_fe.copy()
    df["patient_id"] = X_raw["patient_id"].values
    df["rater_id"]   = X_raw["rater_id"].values
    df["gene"]       = X_raw["gene"].fillna("Unknown").values
    df["cohort"]     = X_raw["cohort"].values

    df_sorted = df.sort_values(["patient_id", "age"])
    grp = df_sorted.groupby("patient_id")
    df_sorted["off_corrected_prev2"]   = grp["off_corrected"].shift(2)
    df_sorted["on_corrected_prev2"]    = grp["on_corrected"].shift(2)
    df_sorted["off_lag_delta"]         = (
        df_sorted["off_corrected"] - df_sorted["off_corrected_prev"]
    )
    df_sorted["inter_visit_gap"]       = (
        df_sorted["age"] - grp["age"].shift(1)
    ).clip(lower=0)
    gap_safe = df_sorted["inter_visit_gap"].replace(0, np.nan)
    df_sorted["off_lag_delta_per_year"] = df_sorted["off_lag_delta"] / gap_safe
    df_sorted["off_corrected_next2"]   = grp["off_corrected"].shift(-2)
    df_sorted = df_sorted.loc[df.index]
    for col in [
        "off_corrected_prev2", "on_corrected_prev2", "off_lag_delta",
        "inter_visit_gap", "off_lag_delta_per_year", "off_corrected_next2",
    ]:
        df[col] = df_sorted[col].values

    df["patient_on_max"] = df["patient_id"].map(
        df.groupby("patient_id")["on_corrected"].max()
    )
    df["patient_on_q75"] = df["patient_id"].map(
        df.groupby("patient_id")["on_corrected"].quantile(0.75)
    )
    df["patient_on_std"] = df["patient_id"].map(
        df.groupby("patient_id")["on_corrected"].std().fillna(0.0)
    )
    on_slopes = {}
    for pid, grp_p in df.groupby("patient_id"):
        valid = grp_p[["on_corrected", "age"]].dropna()
        if len(valid) >= 2:
            on_slopes[pid] = float(
                np.polyfit(valid["age"].values, valid["on_corrected"].values, 1)[0]
            )
        else:
            on_slopes[pid] = np.nan
    median_on_slope = float(np.nanmedian(list(on_slopes.values())))
    for pid in on_slopes:
        if np.isnan(on_slopes[pid]):
            on_slopes[pid] = median_on_slope
    df["patient_on_slope"] = df["patient_id"].map(on_slopes)

    off_safe = df["off_corrected"].replace(0, np.nan)
    df["ledd_off_ratio"]     = df["ledd"] / off_safe
    trend_safe = df["patient_trend_at_age"].replace(0, np.nan)
    df["ledd_per_off_trend"] = df["ledd"] / trend_safe
    df["patient_off_iqr"]    = df["patient_id"].map(
        df.groupby("patient_id")["off_corrected"].apply(
            lambda x: float(
                np.nanpercentile(x.dropna().values, 75)
                - np.nanpercentile(x.dropna().values, 25)
            )
            if x.notna().sum() >= 2
            else 0.0
        )
    )
    df_sorted2 = df.sort_values(["patient_id", "age"])
    grp2 = df_sorted2.groupby("patient_id")
    df_sorted2["off_cummin"]            = grp2["off_corrected"].cummin()
    df_sorted2["on_time_corrected_prev"] = grp2["on_time_corrected"].shift(1)
    df_sorted2 = df_sorted2.loc[df.index]
    df["off_cummin"]             = df_sorted2["off_cummin"].values
    df["on_time_corrected_prev"] = df_sorted2["on_time_corrected_prev"].values

    slope2_map = {}
    for pid, grp_p in df.groupby("patient_id"):
        valid = grp_p[["off_corrected", "age"]].dropna()
        if len(valid) >= 4:
            coeffs = np.polyfit(
                valid["age"].values, valid["off_corrected"].values, 2
            )
            slope2_map[pid] = float(coeffs[0])
        else:
            slope2_map[pid] = np.nan
    median_slope2 = float(np.nanmedian(list(slope2_map.values())))
    for pid in slope2_map:
        if np.isnan(slope2_map[pid]):
            slope2_map[pid] = median_slope2
    df["patient_slope2"] = df["patient_id"].map(slope2_map)

    df_sorted3 = df.sort_values(["patient_id", "age"])
    grp3 = df_sorted3.groupby("patient_id")
    df_sorted3["off_residual_prev"]       = grp3["off_residual_from_trend"].shift(1)
    df_sorted3["on_time_corrected_next"]  = grp3["on_time_corrected"].shift(-1)
    df_sorted3["ledd_prev"]               = grp3["ledd"].shift(1)
    df_sorted3["ledd_next"]               = grp3["ledd"].shift(-1)
    prev_delta = (
        df_sorted3["off_corrected_prev"] - df_sorted3["off_corrected_prev2"]
    )
    curr_delta = df_sorted3["off_corrected"] - df_sorted3["off_corrected_prev"]
    df_sorted3["off_lag_accel"] = curr_delta - prev_delta
    df_sorted3 = df_sorted3.loc[df.index]
    for col in [
        "off_residual_prev", "on_time_corrected_next",
        "ledd_prev", "ledd_next", "off_lag_accel",
    ]:
        df[col] = df_sorted3[col].values

    df["patient_ledd_mean"] = df["patient_id"].map(
        df.groupby("patient_id")["ledd"].mean()
    )
    df["patient_ledd_max"]  = df["patient_id"].map(
        df.groupby("patient_id")["ledd"].max()
    )
    patient_ledd_std        = df.groupby("patient_id")["ledd"].std().fillna(1.0)
    df["patient_ledd_std"]  = df["patient_id"].map(patient_ledd_std)
    ledd_std_safe           = df["patient_ledd_std"].replace(0, np.nan)
    df["ledd_zscore"]       = (df["ledd"] - df["patient_ledd_mean"]) / ledd_std_safe

    if y_train is not None:
        df_y = df.copy()
        df_y["_target"] = y_train.values
        rater_mean_map, rater_std_map, rater_count_map = {}, {}, {}
        for rid, grp_r in df_y.groupby("rater_id"):
            vals = grp_r["_target"].dropna()
            rater_mean_map[rid]  = float(vals.mean()) if len(vals) > 0 else np.nan
            rater_std_map[rid]   = float(vals.std())  if len(vals) > 1 else 0.0
            rater_count_map[rid] = len(vals)
        _g_rater_mean = float(np.nanmean(list(rater_mean_map.values())))
        _g_rater_std  = float(np.nanmean(list(rater_std_map.values())))
        gene_off_mean_map  = df.groupby("gene")["off_corrected"].mean().to_dict()
        gene_slope_mean_map = {}
        for gene, grp_g in df.groupby("gene"):
            slopes = [
                on_slopes[pid]
                for pid in grp_g["patient_id"].unique()
                if pid in on_slopes
            ]
            gene_slope_mean_map[gene] = float(np.mean(slopes)) if slopes else np.nan
        cohort_off_mean_map   = df.groupby("cohort")["off_corrected"].mean().to_dict()
        cohort_off_std_map    = df.groupby("cohort")["off_corrected"].std().to_dict()
        cohort_slope_mean_map = {}
        for coh, grp_c in df.groupby("cohort"):
            slopes_c = [
                on_slopes[pid]
                for pid in grp_c["patient_id"].unique()
                if pid in on_slopes
            ]
            cohort_slope_mean_map[coh] = (
                float(np.mean(slopes_c)) if slopes_c else np.nan
            )
        saved_stats_out = {
            "rater_mean":         rater_mean_map,
            "rater_std":          rater_std_map,
            "rater_count":        rater_count_map,
            "g_rater_mean":       _g_rater_mean,
            "g_rater_std":        _g_rater_std,
            "gene_off_mean":      gene_off_mean_map,
            "gene_slope_mean":    gene_slope_mean_map,
            "cohort_off_mean":    cohort_off_mean_map,
            "cohort_off_std":     cohort_off_std_map,
            "cohort_slope_mean":  cohort_slope_mean_map,
        }
    else:
        saved_stats_out = None

    stats = saved_stats if saved_stats is not None else saved_stats_out

    if stats is not None:
        df["rater_target_mean"]   = df["rater_id"].map(stats["rater_mean"]).fillna(stats["g_rater_mean"])
        df["rater_target_std"]    = df["rater_id"].map(stats["rater_std"]).fillna(stats["g_rater_std"])
        df["rater_target_count"]  = df["rater_id"].map(stats["rater_count"]).fillna(0)
        df["gene_off_mean_feat"]  = df["gene"].map(stats["gene_off_mean"])
        df["gene_slope_mean_feat"]= df["gene"].map(stats["gene_slope_mean"])
        df["cohort_off_mean_feat"]= df["cohort"].map(stats["cohort_off_mean"])
        df["cohort_off_std_feat"] = df["cohort"].map(stats["cohort_off_std"])
        df["cohort_slope_mean_feat"] = df["cohort"].map(stats["cohort_slope_mean"])
    else:
        for col in [
            "rater_target_mean", "rater_target_std", "rater_target_count",
            "gene_off_mean_feat", "gene_slope_mean_feat",
            "cohort_off_mean_feat", "cohort_off_std_feat",
            "cohort_slope_mean_feat",
        ]:
            df[col] = np.nan

    pat_min_age = df.groupby("patient_id")["age"].min()
    pat_max_age = df.groupby("patient_id")["age"].max()
    obs_time    = (pat_max_age - pat_min_age).clip(lower=0.1)
    df["patient_observation_time"] = df["patient_id"].map(obs_time)
    df["visits_per_year"]          = df["patient_n_visits"] / df["patient_observation_time"]
    df["dd_ledd_interaction"]      = df["disease_duration"] * df["ledd"].fillna(0)
    df["dd_ledd_velocity"]         = df["disease_duration"] * df["ledd_velocity"].fillna(0)
    df_sorted4 = df.sort_values(["patient_id", "age"])
    grp4 = df_sorted4.groupby("patient_id")
    df_sorted4["off_running_std"] = (
        grp4["off_corrected"].expanding().std().reset_index(level=0, drop=True)
    )
    df_sorted4 = df_sorted4.loc[df.index]
    df["off_running_std"] = df_sorted4["off_running_std"].values

    df_sorted5 = df.sort_values(["patient_id", "age"])
    grp5 = df_sorted5.groupby("patient_id")
    off_shifted = grp5["off_corrected"].shift(1)
    on_shifted  = grp5["on_corrected"].shift(1)
    df_sorted5["off_ewma"] = off_shifted.groupby(
        df_sorted5["patient_id"]
    ).transform(lambda x: x.ewm(alpha=0.5, adjust=False).mean())
    df_sorted5["on_ewma"] = on_shifted.groupby(
        df_sorted5["patient_id"]
    ).transform(lambda x: x.ewm(alpha=0.5, adjust=False).mean())
    df_sorted5 = df_sorted5.loc[df.index]
    df["off_ewma"] = df_sorted5["off_ewma"].values
    df["on_ewma"]  = df_sorted5["on_ewma"].values

    df["pk_concentration_on"]  = compute_bateman_concentration(
        df["ledd"], df["time_since_intake_on"]
    ).values
    df["pk_concentration_off"] = compute_bateman_concentration(
        df["ledd"], df["time_since_intake_off"]
    ).values

    if stats is not None:
        df["off_vs_gene_mean"]   = df["off_corrected"] - df["gene_off_mean_feat"]
        df["off_vs_cohort_mean"] = df["off_corrected"] - df["cohort_off_mean_feat"]
    else:
        df["off_vs_gene_mean"]   = np.nan
        df["off_vs_cohort_mean"] = np.nan

    df["visit_rev_index"] = df["patient_n_visits"] - df["visit_index"]

    df_sorted6 = df.sort_values(["patient_id", "age"])
    grp6 = df_sorted6.groupby("patient_id")
    ledd_shifted = grp6["ledd"].shift(1)
    df_sorted6["ledd_ewma"] = ledd_shifted.groupby(
        df_sorted6["patient_id"]
    ).transform(lambda x: x.ewm(alpha=0.5, adjust=False).mean())
    df_sorted6 = df_sorted6.loc[df.index]
    df["ledd_ewma"] = df_sorted6["ledd_ewma"].values

    df = df.drop(columns=["patient_id", "gene", "cohort", "patient_ledd_std"])
    return df, saved_stats_out


# ============================================================
# § 4 — Correlation pruning
# ============================================================

def drop_highly_correlated(df: pd.DataFrame, threshold: float = 0.98):
    """Remove one column from each |Pearson r| > threshold pair."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return df, []
    filled  = df[num_cols].fillna(df[num_cols].median())
    corr    = filled.corr().abs()
    upper   = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    cols_to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    dropped = [c for c in cols_to_drop if c in df.columns]
    print(f"Auto-drop {len(dropped)} highly-correlated cols: {dropped}")
    return df.drop(columns=dropped, errors="ignore"), dropped


# ============================================================
# § 5 — NoisyRaterEncoder
# ============================================================

class NoisyRaterEncoder(BaseEstimator, TransformerMixin):
    """Replace rater_id with the per-rater target mean (+ Gaussian noise at transform)."""

    def __init__(self, noise_std: float = 0.1, random_state: int = 0):
        self.noise_std    = noise_std
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):
        if y is None:
            raise ValueError("NoisyRaterEncoder requires y in fit().")
        y_arr = np.asarray(y, dtype=float)
        df = pd.DataFrame({"rater_id": X["rater_id"].values, "_y": y_arr})
        self.rater_mean_  = df.groupby("rater_id")["_y"].mean().to_dict()
        self.global_mean_ = float(np.nanmean(y_arr))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out     = X.copy()
        encoded = (
            out["rater_id"]
            .map(self.rater_mean_)
            .fillna(self.global_mean_)
            .astype(float)
        )
        rng     = np.random.default_rng(self.random_state)
        encoded = encoded + rng.normal(loc=0.0, scale=self.noise_std, size=len(out))
        out["rater_id"] = encoded
        return out


# ============================================================
# § 6 — Heterogeneous voting ensemble wrapper
# ============================================================

class HeterogeneousVotingRegressor(BaseEstimator, RegressorMixin):
    """Equally-weighted average of LGBMRegressor, XGBRegressor, and HistGBR.

    XGBoost requires numeric features, so a numeric mapping of any remaining
    object/category columns is applied before fitting and predicting with XGB.
    """

    def __init__(
        self,
        lgbm_params: dict | None = None,
        xgb_params:  dict | None = None,
        hgbr_params: dict | None = None,
    ):
        self.lgbm_params = lgbm_params or {}
        self.xgb_params  = xgb_params  or {}
        self.hgbr_params = hgbr_params or {}

    # ------------------------------------------------------------------
    def _encode_for_xgb(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Map object/category columns to integer codes for XGBoost."""
        df = X.copy()
        if fit:
            self._cat_encoders: dict = {}
        for col in df.select_dtypes(include=["object", "category"]).columns:
            if fit:
                uniq = {v: i for i, v in enumerate(sorted(df[col].dropna().unique()))}
                self._cat_encoders[col] = uniq
            enc = self._cat_encoders.get(col, {})
            df[col] = df[col].map(enc).fillna(-1).astype(float)
        return df.values.astype(float)

    # ------------------------------------------------------------------
    def fit(self, X, y):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        self.lgbm_ = LGBMRegressor(**self.lgbm_params)
        self.lgbm_.fit(X_df, y)

        X_xgb = self._encode_for_xgb(X_df, fit=True)
        self.xgb_ = XGBRegressor(**self.xgb_params)
        self.xgb_.fit(X_xgb, y)

        self.hgbr_ = HistGradientBoostingRegressor(**self.hgbr_params)
        self.hgbr_.fit(X_df, y)

        return self

    # ------------------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        X_df  = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        X_xgb = self._encode_for_xgb(X_df, fit=False)

        pred_lgb  = self.lgbm_.predict(X_df)
        pred_xgb  = self.xgb_.predict(X_xgb)
        pred_hgbr = self.hgbr_.predict(X_df)

        return (pred_lgb + pred_xgb + pred_hgbr) / 3.0


# ============================================================
# § 7 — Post-processing: poly2 smooth → isotonic → clip
# ============================================================

def smooth_poly2_then_isotonic(
    predictions: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """Per-patient poly2 smooth then isotonic regression then clip [0, 109.5]."""
    preds = predictions.copy().astype(float)
    df    = X_val[["patient_id", "age"]].copy()
    df["_pred"] = preds
    iso   = IsotonicRegression(increasing=True, out_of_bounds="clip")

    for pid, grp in df.groupby("patient_id"):
        n          = len(grp)
        grp_sorted = grp.sort_values("age")
        ages       = grp_sorted["age"].values
        p          = grp_sorted["_pred"].values

        if n >= 3:
            degree = 2 if n >= 4 else 1
            coeffs = np.polyfit(ages, p, degree)
            p      = np.polyval(coeffs, ages)

        if n >= 2:
            p = iso.fit_transform(ages, p)

        df.loc[grp_sorted.index, "_pred"] = p

    return np.clip(df["_pred"].values, 0.0, 109.5)


# ============================================================
# § 8 — Full-fold feature builder
# ============================================================

def build_all_features(X_tr_raw, y_tr, X_va_raw=None, saved_stats=None):
    """Build the Exp-16 feature matrix (76 cols) for one fold or the full train set."""
    X_tr_fe, X_va_fe = build_features(X_tr_raw, y_tr, X_val=X_va_raw)

    X_tr_ext, stats_out = add_extra_features_v4(X_tr_fe, X_tr_raw, y_train=y_tr)
    if X_va_raw is not None:
        X_va_ext, _ = add_extra_features_v4(X_va_fe, X_va_raw, saved_stats=stats_out)

    ALL_COLS = FEATURE_COLS + ALL_EXTRA_COLS
    avail_tr = [c for c in ALL_COLS if c in X_tr_ext.columns]
    # rater_id must be present for NoisyRaterEncoder
    if "rater_id" not in avail_tr and "rater_id" in X_tr_ext.columns:
        avail_tr = ["rater_id"] + avail_tr

    X_tr_raw_feats = X_tr_ext[avail_tr].copy()
    X_tr_pruned, corr_drop = drop_highly_correlated(X_tr_raw_feats, threshold=0.98)

    X_va_pruned = None
    if X_va_raw is not None:
        avail_va   = [c for c in avail_tr if c in X_va_ext.columns]
        X_va_raw_f = X_va_ext[avail_va].copy()
        X_va_pruned = X_va_raw_f.drop(columns=corr_drop, errors="ignore")

    return X_tr_pruned, X_va_pruned, corr_drop, stats_out, avail_tr


# ============================================================
# § 9 — Ensemble hyper-parameters
# ============================================================

LGBM_PARAMS = dict(
    learning_rate=0.015,
    n_estimators=1000,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_lambda=5.0,
    random_state=42,
    verbose=-1,
)

XGB_PARAMS = dict(
    learning_rate=0.015,
    n_estimators=1000,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_lambda=5.0,
    random_state=42,
    verbosity=0,
)

HGBR_PARAMS = dict(
    learning_rate=0.03,
    max_iter=900,
    max_leaf_nodes=45,
    min_samples_leaf=60,
    l2_regularization=20.0,
    max_features=0.4,
    random_state=42,
)


# ============================================================
# § 10 — Load data and fixed folds
# ============================================================

X_train_raw, y_train = load_dataset(DATA_DIR)
X_test_raw           = load_test_dataset(DATA_DIR)
cv_folds             = get_cv_folds(X_train_raw, y_train)

print(f"Train: {X_train_raw.shape}   Test: {X_test_raw.shape}")
print(f"Folds: {len(cv_folds)}")


# ============================================================
# § 11 — Cross-validation loop
# ============================================================

print("\n=== Cross-validation (5 folds, GroupKFold on patient_id) ===")

fold_rmse_raw: list[float] = []
fold_rmse_pp:  list[float] = []

for fold_idx, (tr_idx, val_idx) in enumerate(cv_folds):
    X_tr_raw = X_train_raw.iloc[tr_idx].reset_index(drop=True)
    X_va_raw = X_train_raw.iloc[val_idx].reset_index(drop=True)
    y_tr     = y_train.iloc[tr_idx].reset_index(drop=True)
    y_va     = y_train.iloc[val_idx].reset_index(drop=True)

    X_tr_f, X_va_f, _, _, _ = build_all_features(X_tr_raw, y_tr, X_va_raw)

    pipeline = make_pipeline(
        NoisyRaterEncoder(noise_std=0.1, random_state=0),
        skrub.TableVectorizer(),
        HeterogeneousVotingRegressor(
            lgbm_params=LGBM_PARAMS,
            xgb_params=XGB_PARAMS,
            hgbr_params=HGBR_PARAMS,
        ),
    )
    pipeline.fit(X_tr_f, y_tr)

    raw_preds = pipeline.predict(X_va_f)
    pp_preds  = smooth_poly2_then_isotonic(raw_preds, X_va_raw)

    rmse_raw = root_mean_squared_error(y_va.values, raw_preds)
    rmse_pp  = root_mean_squared_error(y_va.values, pp_preds)
    fold_rmse_raw.append(rmse_raw)
    fold_rmse_pp.append(rmse_pp)
    print(f"  Fold {fold_idx + 1}: RMSE_raw={rmse_raw:.4f}  RMSE_pp={rmse_pp:.4f}")

mean_rmse_raw = float(np.mean(fold_rmse_raw))
mean_rmse_pp  = float(np.mean(fold_rmse_pp))
print(f"\nMean RMSE (raw)           : {mean_rmse_raw:.4f}")
print(f"Mean RMSE (poly2+isotonic): {mean_rmse_pp:.4f}")


# ============================================================
# § 12 — Full-train feature engineering and skore evaluation
# ============================================================

print("\nBuilding full-train feature set …")

X_full_f, _, corr_cols_to_drop, stats_full, avail_cols_full = \
    build_all_features(X_train_raw, y_train)

# skrub DataOps graph: attach patient_id and target so GroupKFold can group correctly
visits_df = X_full_f.copy()
visits_df["target"]     = y_train.values
visits_df["patient_id"] = X_train_raw["patient_id"].values

data_node   = skrub.var("visits", visits_df)
groups_node = data_node["patient_id"]

X_op = data_node.drop(["target", "patient_id"], axis=1).skb.mark_as_X(
    cv=GroupKFold(n_splits=5),
    split_kwargs={"groups": groups_node},
)
y_op = data_node["target"].skb.mark_as_y()

pipeline_for_skore = make_pipeline(
    NoisyRaterEncoder(noise_std=0.1, random_state=0),
    skrub.TableVectorizer(),
    HeterogeneousVotingRegressor(
        lgbm_params=LGBM_PARAMS,
        xgb_params=XGB_PARAMS,
        hgbr_params=HGBR_PARAMS,
    ),
)

pred_node = X_op.skb.apply(pipeline_for_skore, y=y_op)

print("Running skore.evaluate …")
report = skore.evaluate(pred_node, data={"visits": visits_df})
project.put(EXP_KEY, report)
print(f"Skore report pushed: {EXP_KEY}")
try:
    print(report.metrics.rmse())
except Exception:
    pass


# ============================================================
# § 13 — Full-train refit and test-set inference
# ============================================================

print("\nRefitting on full training set …")

pipeline_full = make_pipeline(
    NoisyRaterEncoder(noise_std=0.1, random_state=0),
    skrub.TableVectorizer(),
    HeterogeneousVotingRegressor(
        lgbm_params=LGBM_PARAMS,
        xgb_params=XGB_PARAMS,
        hgbr_params=HGBR_PARAMS,
    ),
)
pipeline_full.fit(X_full_f, y_train)

# Build test features using full-train calibration stats
X_te_fe, _ = build_features(X_train_raw, y_train, X_val=X_test_raw)
X_te_ext, _ = add_extra_features_v4(X_te_fe, X_test_raw, saved_stats=stats_full)

avail_te    = [c for c in avail_cols_full if c in X_te_ext.columns]
X_te_raw_f  = X_te_ext[avail_te].copy()
X_te_pruned = X_te_raw_f.drop(columns=corr_cols_to_drop, errors="ignore")

raw_preds_test = pipeline_full.predict(X_te_pruned)
pp_preds_test  = smooth_poly2_then_isotonic(raw_preds_test, X_test_raw)

# Hard clip to eliminate any residual negative or out-of-range values
pp_preds_test = np.clip(pp_preds_test, 0.0, 109.5)

print(f"Test predictions: {len(pp_preds_test)} rows  "
      f"min={pp_preds_test.min():.2f}  max={pp_preds_test.max():.2f}")


# ============================================================
# § 14 — Save submission
# ============================================================

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
sub_path = submissions_dir / f"{EXP_KEY}.csv"

sub           = X_test_raw[["Index"]].copy()
sub["target"] = pp_preds_test
sub.to_csv(sub_path, index=False)
print(f"Submission saved: {sub_path}  ({len(sub)} rows)")
print(sub.head())
print("\nDone.")
