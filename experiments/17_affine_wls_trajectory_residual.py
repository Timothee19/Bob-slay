# %% [markdown]
# # Experiment 17 — Affine WLS rater calibration + linear patient trend + HistGBR
#
# **Date:** 2025-01-10
# **Base:** Exp 16 feature set (76 cols, full engineering pipeline)
#
# ## What this experiment adds over Exp 16
#
# 1. **Affine rater calibration** (fold-aware, Ridge α=10):
#    For each rater_id fit `target ~ off` and `target ~ on_bateman` on the
#    training fold.  Produces `off_affine_corrected`, `on_affine_corrected`, and
#    the raw calibration residual `off_affine_delta` as extra features.
#    Fallback to global affine for raters with < 10 observations.
#
# 2. **Per-patient linear trend** (degree-1 only, clipped [0, 110]):
#    Fit `off_affine_corrected ~ age` per patient (median-regularised for
#    patients with < 3 valid visits).  Adds `patient_linear_trend` (the
#    personalised prediction at each visit age) and
#    `patient_trend_residual` (deviation from that trend).
#
# 3. All 76 original features are retained; the ~4 new features are appended.
#
# 4. Post-processing: per-patient poly2 smooth → isotonic → clip [0, 109.5].

# %%
import numpy as np
import pandas as pd
import skore
import skrub
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline

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
EXP_KEY  = "17_affine_wls_trajectory_residual"

# ── Pharmacokinetic parameters (Bateman biexponential) ─────────────────────────
_KA = 1.5
_KE = 0.46


# ============================================================
# § 1 — Feature column groups (same as Exp 14 / 16)
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

# New features added in Exp 17
EXP17_COLS = [
    "off_affine_corrected",
    "on_affine_corrected",
    "off_affine_delta",
    "patient_linear_trend",
    "patient_trend_residual",
]

ALL_EXTRA_COLS = EXP06_COLS + EXP07_COLS + EXP08_COLS + EXP11_COLS


# ============================================================
# § 2 — Helpers (Bateman, extra features v4 from Exp 16)
# ============================================================

def compute_bateman_concentration(
    ledd: pd.Series,
    time_since_intake: pd.Series,
    ka: float = _KA,
    ke: float = _KE,
) -> pd.Series:
    """Plasma levodopa via the Bateman biexponential equation."""
    safe_ledd = ledd.fillna(0.0)
    safe_time = time_since_intake.fillna(0.0)
    c = (safe_ledd * ka / (ka - ke)) * (
        np.exp(-ke * safe_time) - np.exp(-ka * safe_time)
    )
    mask = ledd.isna() | time_since_intake.isna()
    c[mask] = np.nan
    return c


def add_extra_features_v4(X_fe, X_raw, y_train=None, saved_stats=None):
    """Full extra feature set from experiments 6–11 and 14 (Exp 16 version)."""
    df = X_fe.copy()
    df["patient_id"] = X_raw["patient_id"].values
    df["rater_id"] = X_raw["rater_id"].values
    df["gene"] = X_raw["gene"].fillna("Unknown").values
    df["cohort"] = X_raw["cohort"].values

    df_sorted = df.sort_values(["patient_id", "age"])
    grp = df_sorted.groupby("patient_id")
    df_sorted["off_corrected_prev2"] = grp["off_corrected"].shift(2)
    df_sorted["on_corrected_prev2"] = grp["on_corrected"].shift(2)
    df_sorted["off_lag_delta"] = (
        df_sorted["off_corrected"] - df_sorted["off_corrected_prev"]
    )
    df_sorted["inter_visit_gap"] = (
        df_sorted["age"] - grp["age"].shift(1)
    ).clip(lower=0)
    gap_safe = df_sorted["inter_visit_gap"].replace(0, np.nan)
    df_sorted["off_lag_delta_per_year"] = df_sorted["off_lag_delta"] / gap_safe
    df_sorted["off_corrected_next2"] = grp["off_corrected"].shift(-2)
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
    df["ledd_off_ratio"] = df["ledd"] / off_safe
    trend_safe = df["patient_trend_at_age"].replace(0, np.nan)
    df["ledd_per_off_trend"] = df["ledd"] / trend_safe
    df["patient_off_iqr"] = df["patient_id"].map(
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
    df_sorted2["off_cummin"] = grp2["off_corrected"].cummin()
    df_sorted2["on_time_corrected_prev"] = grp2["on_time_corrected"].shift(1)
    df_sorted2 = df_sorted2.loc[df.index]
    df["off_cummin"] = df_sorted2["off_cummin"].values
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
    df_sorted3["off_residual_prev"] = grp3["off_residual_from_trend"].shift(1)
    df_sorted3["on_time_corrected_next"] = grp3["on_time_corrected"].shift(-1)
    df_sorted3["ledd_prev"] = grp3["ledd"].shift(1)
    df_sorted3["ledd_next"] = grp3["ledd"].shift(-1)
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
    df["patient_ledd_max"] = df["patient_id"].map(
        df.groupby("patient_id")["ledd"].max()
    )
    patient_ledd_std = df.groupby("patient_id")["ledd"].std().fillna(1.0)
    df["patient_ledd_std"] = df["patient_id"].map(patient_ledd_std)
    ledd_std_safe = df["patient_ledd_std"].replace(0, np.nan)
    df["ledd_zscore"] = (df["ledd"] - df["patient_ledd_mean"]) / ledd_std_safe

    if y_train is not None:
        df_y = df.copy()
        df_y["_target"] = y_train.values
        rater_mean_map, rater_std_map, rater_count_map = {}, {}, {}
        for rid, grp_r in df_y.groupby("rater_id"):
            vals = grp_r["_target"].dropna()
            rater_mean_map[rid] = float(vals.mean()) if len(vals) > 0 else np.nan
            rater_std_map[rid] = float(vals.std()) if len(vals) > 1 else 0.0
            rater_count_map[rid] = len(vals)
        _g_rater_mean = float(np.nanmean(list(rater_mean_map.values())))
        _g_rater_std = float(np.nanmean(list(rater_std_map.values())))
        gene_off_mean_map = df.groupby("gene")["off_corrected"].mean().to_dict()
        gene_slope_mean_map = {}
        for gene, grp_g in df.groupby("gene"):
            slopes = [
                on_slopes[pid]
                for pid in grp_g["patient_id"].unique()
                if pid in on_slopes
            ]
            gene_slope_mean_map[gene] = (
                float(np.mean(slopes)) if slopes else np.nan
            )
        cohort_off_mean_map = (
            df.groupby("cohort")["off_corrected"].mean().to_dict()
        )
        cohort_off_std_map = (
            df.groupby("cohort")["off_corrected"].std().to_dict()
        )
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
            "rater_mean": rater_mean_map,
            "rater_std": rater_std_map,
            "rater_count": rater_count_map,
            "g_rater_mean": _g_rater_mean,
            "g_rater_std": _g_rater_std,
            "gene_off_mean": gene_off_mean_map,
            "gene_slope_mean": gene_slope_mean_map,
            "cohort_off_mean": cohort_off_mean_map,
            "cohort_off_std": cohort_off_std_map,
            "cohort_slope_mean": cohort_slope_mean_map,
        }
    else:
        saved_stats_out = None

    stats = saved_stats if saved_stats is not None else saved_stats_out

    if stats is not None:
        df["rater_target_mean"] = df["rater_id"].map(
            stats["rater_mean"]
        ).fillna(stats["g_rater_mean"])
        df["rater_target_std"] = df["rater_id"].map(
            stats["rater_std"]
        ).fillna(stats["g_rater_std"])
        df["rater_target_count"] = df["rater_id"].map(
            stats["rater_count"]
        ).fillna(0)
        df["gene_off_mean_feat"] = df["gene"].map(stats["gene_off_mean"])
        df["gene_slope_mean_feat"] = df["gene"].map(stats["gene_slope_mean"])
        df["cohort_off_mean_feat"] = df["cohort"].map(stats["cohort_off_mean"])
        df["cohort_off_std_feat"] = df["cohort"].map(stats["cohort_off_std"])
        df["cohort_slope_mean_feat"] = df["cohort"].map(
            stats["cohort_slope_mean"]
        )
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
    obs_time = (pat_max_age - pat_min_age).clip(lower=0.1)
    df["patient_observation_time"] = df["patient_id"].map(obs_time)
    df["visits_per_year"] = df["patient_n_visits"] / df["patient_observation_time"]
    df["dd_ledd_interaction"] = df["disease_duration"] * df["ledd"].fillna(0)
    df["dd_ledd_velocity"] = df["disease_duration"] * df["ledd_velocity"].fillna(0)
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
    on_shifted = grp5["on_corrected"].shift(1)
    df_sorted5["off_ewma"] = off_shifted.groupby(
        df_sorted5["patient_id"]
    ).transform(lambda x: x.ewm(alpha=0.5, adjust=False).mean())
    df_sorted5["on_ewma"] = on_shifted.groupby(
        df_sorted5["patient_id"]
    ).transform(lambda x: x.ewm(alpha=0.5, adjust=False).mean())
    df_sorted5 = df_sorted5.loc[df.index]
    df["off_ewma"] = df_sorted5["off_ewma"].values
    df["on_ewma"] = df_sorted5["on_ewma"].values

    df["pk_concentration_on"] = compute_bateman_concentration(
        df["ledd"], df["time_since_intake_on"]
    ).values
    df["pk_concentration_off"] = compute_bateman_concentration(
        df["ledd"], df["time_since_intake_off"]
    ).values

    if stats is not None:
        df["off_vs_gene_mean"] = df["off_corrected"] - df["gene_off_mean_feat"]
        df["off_vs_cohort_mean"] = (
            df["off_corrected"] - df["cohort_off_mean_feat"]
        )
    else:
        df["off_vs_gene_mean"] = np.nan
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

    df = df.drop(
        columns=["patient_id", "gene", "cohort", "patient_ledd_std"]
    )
    return df, saved_stats_out


def drop_highly_correlated(df: pd.DataFrame, threshold: float = 0.98):
    """Remove one column from each |Pearson r| > threshold pair."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return df, []
    filled = df[num_cols].fillna(df[num_cols].median())
    corr = filled.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    cols_to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    dropped = [c for c in cols_to_drop if c in df.columns]
    print(f"Auto-drop {len(dropped)} highly-correlated cols: {dropped}")
    return df.drop(columns=dropped, errors="ignore"), dropped


# ============================================================
# § 3 — Affine rater calibration (Exp 17 new feature block)
# ============================================================

def fit_affine_rater_calibration(
    X_fe: pd.DataFrame,
    X_raw: pd.DataFrame,
    y_train: pd.Series,
    on_bateman: pd.Series,
    min_samples: int = 10,
    alpha: float = 10.0,
) -> dict:
    """Fit per-rater Ridge affine models: target ~ off and target ~ on_bateman.

    Returns a stats dict suitable for ``apply_affine_rater_features``.
    Only use on the training fold.
    """
    rater_ids = X_raw["patient_id"].values  # NOT used here; use X_raw["rater_id"]
    rater_col = X_raw["rater_id"].values
    off_col   = X_fe["off_corrected"].values   # already rater-bias-divided
    y_arr     = y_train.values

    # Global fallback
    def _ridge_fit(x, y):
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() < 2:
            return (0.0, 1.0)
        r = Ridge(alpha=alpha, fit_intercept=True)
        r.fit(x[ok].reshape(-1, 1), y[ok])
        return float(r.intercept_), float(r.coef_[0])

    global_off   = _ridge_fit(off_col, y_arr)
    on_pk_arr    = on_bateman.values
    global_on    = _ridge_fit(on_pk_arr, y_arr)

    rater_off: dict = {}
    rater_on:  dict = {}
    rater_counts: dict = {}

    for rater in np.unique(rater_col):
        mask = rater_col == rater
        n_ok_off = int(np.sum(~np.isnan(off_col[mask])))
        rater_counts[rater] = n_ok_off
        if n_ok_off >= min_samples:
            rater_off[rater] = _ridge_fit(off_col[mask], y_arr[mask])
        n_ok_on = int(np.sum(~np.isnan(on_pk_arr[mask])))
        if n_ok_on >= min_samples:
            rater_on[rater]  = _ridge_fit(on_pk_arr[mask], y_arr[mask])

    return {
        "global_off": global_off,
        "global_on":  global_on,
        "rater_off":  rater_off,
        "rater_on":   rater_on,
        "rater_counts": rater_counts,
    }


def apply_affine_rater_features(
    X_fe: pd.DataFrame,
    X_raw: pd.DataFrame,
    on_bateman: pd.Series,
    calib_stats: dict,
) -> pd.DataFrame:
    """Compute off_affine_corrected, on_affine_corrected, off_affine_delta.

    Also computes per-patient linear trend on off_affine_corrected and adds
    patient_linear_trend + patient_trend_residual.
    """
    df = X_fe.copy()
    raters   = X_raw["rater_id"].values
    off_arr  = df["off_corrected"].values
    on_arr   = on_bateman.values

    n = len(df)
    off_affine = np.full(n, np.nan)
    on_affine  = np.full(n, np.nan)

    g_off = calib_stats["global_off"]
    g_on  = calib_stats["global_on"]

    for i in range(n):
        r = raters[i]
        # off affine correction
        a_off, b_off = calib_stats["rater_off"].get(r, g_off)
        if not np.isnan(off_arr[i]):
            off_affine[i] = a_off + b_off * off_arr[i]

        # on_bateman affine correction
        a_on, b_on = calib_stats["rater_on"].get(r, g_on)
        if not np.isnan(on_arr[i]):
            on_affine[i] = a_on + b_on * on_arr[i]

    df["off_affine_corrected"] = off_affine
    df["on_affine_corrected"]  = on_affine
    df["off_affine_delta"]     = off_affine - off_arr   # calibration correction magnitude

    # Per-patient linear trend on off_affine_corrected (degree 1, clipped)
    patient_ids = X_raw["patient_id"].values
    ages        = df["age"].values

    # Build a temporary DF grouped by patient for the linear fit
    tmp = pd.DataFrame({
        "patient_id": patient_ids,
        "age":        ages,
        "off_a":      off_affine,
    })

    slope_map: dict = {}
    intercept_map: dict = {}
    for pid, grp in tmp.groupby("patient_id"):
        ok = ~np.isnan(grp["off_a"].values)
        n_ok = ok.sum()
        if n_ok >= 2:
            c = np.polyfit(grp["age"].values[ok], grp["off_a"].values[ok], 1)
            slope_map[pid]     = float(c[0])
            intercept_map[pid] = float(c[1])
        else:
            slope_map[pid]     = np.nan
            intercept_map[pid] = np.nan

    med_slope     = float(np.nanmedian(list(slope_map.values())))
    med_intercept = float(np.nanmedian(list(intercept_map.values())))
    for pid in slope_map:
        if np.isnan(slope_map[pid]):
            slope_map[pid]     = med_slope
            intercept_map[pid] = med_intercept

    # Force < 3 valid visits to use median slope (regularise)
    visit_counts = tmp.groupby("patient_id")["off_a"].apply(
        lambda x: int((~np.isnan(x.values)).sum())
    ).to_dict()
    for pid in slope_map:
        if visit_counts.get(pid, 0) < 3:
            slope_map[pid] = med_slope

    trend = np.array([
        intercept_map[pid] + slope_map[pid] * age
        for pid, age in zip(patient_ids, ages)
    ], dtype=float)
    trend = np.clip(trend, 0.0, 110.0)

    df["patient_linear_trend"]   = trend
    df["patient_trend_residual"] = off_affine - trend   # NaN where off_affine is NaN

    return df


# ============================================================
# § 4 — NoisyRaterEncoder (same as Exp 16)
# ============================================================

class NoisyRaterEncoder(BaseEstimator, TransformerMixin):
    """Replace rater_id with the per-rater target mean (+ Gaussian noise)."""

    def __init__(self, noise_std: float = 0.1, random_state: int = 0):
        self.noise_std    = noise_std
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):
        if y is None:
            raise ValueError("NoisyRaterEncoder requires y in fit().")
        y_arr = np.asarray(y, dtype=float)
        df = pd.DataFrame({"rater_id": X["rater_id"].values, "_y": y_arr})
        self.rater_mean_ = df.groupby("rater_id")["_y"].mean().to_dict()
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
        encoded = encoded + rng.normal(0.0, self.noise_std, size=len(out))
        out["rater_id"] = encoded
        return out


# ============================================================
# § 5 — Post-processing: poly2 → isotonic → clip
# ============================================================

def smooth_poly2_isotonic(
    predictions: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """Per-patient poly2 smooth then isotonic regression then clip [0, 109.5]."""
    preds = predictions.copy().astype(float)
    df = X_val[["patient_id", "age"]].copy()
    df["_pred"] = preds
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")

    for pid, grp in df.groupby("patient_id"):
        n = len(grp)
        grp_sorted = grp.sort_values("age")
        ages = grp_sorted["age"].values
        p    = grp_sorted["_pred"].values

        if n >= 3:
            degree = 2 if n >= 4 else 1
            coeffs = np.polyfit(ages, p, degree)
            p = np.polyval(coeffs, ages)

        if n >= 2:
            p = iso.fit_transform(ages, p)

        df.loc[grp_sorted.index, "_pred"] = p

    return np.clip(df["_pred"].values, 0.0, 109.5)


# ============================================================
# § 6 — Full-fold feature builder
# ============================================================

def build_all_features(
    X_tr_raw, y_tr,
    X_va_raw=None,
    saved_stats=None,
):
    """Build the full feature matrix (Exp 16 + Exp 17 extras) for one fold.

    When ``X_va_raw`` is None, only training features are returned.
    Returns ``(X_tr_final, X_va_final, corr_cols_to_drop, stats_full)``.
    ``X_va_final`` is None when ``X_va_raw`` is None.
    """
    # Base FE from parkinsons.features (rater bias + PK + lags + trend)
    X_tr_fe, X_va_fe = build_features(X_tr_raw, y_tr, X_val=X_va_raw)

    # Add extra features v4 (Exp 06–11)
    X_tr_ext, stats_out = add_extra_features_v4(
        X_tr_fe, X_tr_raw, y_train=y_tr
    )
    if X_va_raw is not None:
        X_va_ext, _ = add_extra_features_v4(
            X_va_fe, X_va_raw, saved_stats=stats_out
        )

    # ── Exp 17: affine rater calibration features ─────────────────────────
    on_bateman_tr = compute_bateman_concentration(
        X_tr_raw["ledd"], X_tr_raw["time_since_intake_on"]
    ).values
    on_bateman_tr = pd.Series(on_bateman_tr, index=X_tr_raw.index)

    calib_stats = fit_affine_rater_calibration(
        X_tr_fe, X_tr_raw, y_tr, on_bateman_tr, min_samples=10, alpha=10.0
    )

    X_tr_ext = apply_affine_rater_features(
        X_tr_ext, X_tr_raw, on_bateman_tr, calib_stats
    )

    if X_va_raw is not None:
        on_bateman_va = compute_bateman_concentration(
            X_va_raw["ledd"], X_va_raw["time_since_intake_on"]
        ).values
        on_bateman_va = pd.Series(on_bateman_va, index=X_va_raw.index)
        X_va_ext = apply_affine_rater_features(
            X_va_ext, X_va_raw, on_bateman_va, calib_stats
        )

    # Select feature columns
    ALL_COLS = FEATURE_COLS + ALL_EXTRA_COLS + EXP17_COLS
    avail_tr = [c for c in ALL_COLS if c in X_tr_ext.columns]
    # Add rater_id for NoisyRaterEncoder
    if "rater_id" not in avail_tr and "rater_id" in X_tr_ext.columns:
        avail_tr = ["rater_id"] + avail_tr

    X_tr_raw_feats = X_tr_ext[avail_tr].copy()

    # Correlation pruning (computed from training fold only)
    X_tr_pruned, corr_drop = drop_highly_correlated(X_tr_raw_feats, threshold=0.98)

    X_va_pruned = None
    if X_va_raw is not None:
        avail_va = [c for c in avail_tr if c in X_va_ext.columns]
        X_va_raw_feats = X_va_ext[avail_va].copy()
        X_va_pruned = X_va_raw_feats.drop(columns=corr_drop, errors="ignore")

    return X_tr_pruned, X_va_pruned, corr_drop, stats_out, calib_stats, avail_tr


# ============================================================
# § 7 — Load data and fixed folds
# ============================================================

X_train_raw, y_train = load_dataset(DATA_DIR)
X_test_raw           = load_test_dataset(DATA_DIR)
cv_folds             = get_cv_folds(X_train_raw, y_train)

print(f"Train: {X_train_raw.shape}   Test: {X_test_raw.shape}")
print(f"Folds: {len(cv_folds)}")

# ── HistGBR hyper-parameters (best from Exp 14/16 search) ────────────────────
HGBR_PARAMS = dict(
    learning_rate=0.02,
    max_iter=2700,
    max_leaf_nodes=93,
    min_samples_leaf=67,
    l2_regularization=1.45,
    max_features=0.57,
    random_state=42,
)


# ============================================================
# § 8 — Cross-validation loop
# ============================================================

print("\n=== Cross-validation (5 folds) ===")

fold_rmse_raw: list[float] = []
fold_rmse_pp:  list[float] = []

for fold_idx, (tr_idx, val_idx) in enumerate(cv_folds):
    X_tr_raw = X_train_raw.iloc[tr_idx].reset_index(drop=True)
    X_va_raw = X_train_raw.iloc[val_idx].reset_index(drop=True)
    y_tr     = y_train.iloc[tr_idx].reset_index(drop=True)
    y_va     = y_train.iloc[val_idx].reset_index(drop=True)

    X_tr_f, X_va_f, _, _, _, _ = build_all_features(X_tr_raw, y_tr, X_va_raw)

    pipeline = make_pipeline(
        NoisyRaterEncoder(noise_std=0.1, random_state=0),
        skrub.TableVectorizer(),
        HistGradientBoostingRegressor(**HGBR_PARAMS),
    )
    pipeline.fit(X_tr_f, y_tr)

    raw_preds = pipeline.predict(X_va_f)
    pp_preds  = smooth_poly2_isotonic(raw_preds, X_va_raw)

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
# § 9 — Full-train feature engineering and skore evaluation
# ============================================================

print("\nBuilding full-train feature set …")

X_full_f, _, corr_cols_to_drop, stats_full, calib_stats_full, avail_cols_full = \
    build_all_features(X_train_raw, y_train)

# skrub DataOps graph: NoisyRaterEncoder expects rater_id as a column in X
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
    HistGradientBoostingRegressor(**HGBR_PARAMS),
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
# § 10 — Full-train refit and test-set inference
# ============================================================

print("\nRefitting on full training set …")
pipeline_full = make_pipeline(
    NoisyRaterEncoder(noise_std=0.1, random_state=0),
    skrub.TableVectorizer(),
    HistGradientBoostingRegressor(**HGBR_PARAMS),
)
pipeline_full.fit(X_full_f, y_train)

# Build test features using full-train calibration stats
X_te_fe, _ = build_features(X_train_raw, y_train, X_val=X_test_raw)
X_te_ext, _ = add_extra_features_v4(X_te_fe, X_test_raw, saved_stats=stats_full)

on_bateman_te = compute_bateman_concentration(
    X_test_raw["ledd"], X_test_raw["time_since_intake_on"]
).values
on_bateman_te = pd.Series(on_bateman_te, index=X_test_raw.index)
X_te_ext = apply_affine_rater_features(
    X_te_ext, X_test_raw, on_bateman_te, calib_stats_full
)

avail_te   = [c for c in avail_cols_full if c in X_te_ext.columns]
X_te_raw_f = X_te_ext[avail_te].copy()
X_te_pruned = X_te_raw_f.drop(columns=corr_cols_to_drop, errors="ignore")

raw_preds_test = pipeline_full.predict(X_te_pruned)
pp_preds_test  = smooth_poly2_isotonic(raw_preds_test, X_test_raw)

print(f"Test predictions: {len(pp_preds_test)} rows  "
      f"min={pp_preds_test.min():.2f}  max={pp_preds_test.max():.2f}")


# ============================================================
# § 11 — Save submission
# ============================================================

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
sub_path        = submissions_dir / f"{EXP_KEY}.csv"

sub           = X_test_raw[["Index"]].copy()
sub["target"] = pp_preds_test
sub.to_csv(sub_path, index=False)
print(f"Submission saved: {sub_path}  ({len(sub)} rows)")
print(sub.head())
print("\nDone.")
