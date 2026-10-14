# %% [markdown]
# # Experiment 20 — bench ratchet
#
# **Date:** 2025-01-08
# **Independent pipeline** — uses only `load_dataset`, `load_test_dataset`,
# `get_cv_folds`, `load_skore_credentials` from the `parkinsons` package.
# Everything else (feature engineering, post-processing) is defined here.
#
# ## Features (all target-dependent stats computed on fold train only)
#
# 1. **Rater bias** — exp(q75(log(off/target))) per rater_id, renormalised to
#    preserve the global median(off/target) level.
#    off_c = off / bias ; on_c = on / (bias * f(t)) where f(t) is the median
#    on/target curve by time_since_intake_on bins, fitted on train.
#
# 2. **Patient drug-response k** — from visits that have both on and off:
#    k = exp(mean(log(on_c / off_c))).
#    z = log(off_c) when off exists, else log(on_c / k).
#
# 3. **Per-patient MLE curve** — log T ~ a + b*(age - age_mean), residual
#    density learned by KDE on train, prior on slope.
#    Outputs: mle, mle_slope, z_resid.
#
# 4. **Detrended z** — zd = z - mle_slope*(age - age_mean):
#    quantiles 50/75/90, max, top2 mean, std, count, sliding windows
#    (max/mean/q75 on 3/5/7 visits), cummax fwd/bwd, lags +-1 +-2.
#
# 5. **Rater reliability** — median, std, q10, q90 of (z - log(target))
#    per rater, fitted on train.
#
# 6. **KDE mode of zd** per patient (h=0.04).
#
# No raw rater_id in the feature matrix.
#
# ## Models
# LGBM (1500 trees, lr=0.02, 63 leaves, min_child_samples=40)
# + XGB (1500 trees, lr=0.02, depth=7) — averaged.
#
# ## Post-processing
# 1. Poly2 smoothing per patient on age.
# 2. Ratchet: sort by age per patient; prediction never drops; when it would
#    stagnate/drop, pred_i = 1.02 * pred_{i-1}.  Logic verified on y_train.
# 3. Clip to 0.
#
# ## Parallelism
# CrossValidationReport n_jobs = min(5, n_cores);
# LGBM / XGB n_jobs = max(1, n_cores // 5).

# %%
import os
import multiprocessing

import numpy as np
import pandas as pd
import skore
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.evaluate import get_cv_folds
from parkinsons.hub import load_skore_credentials

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

N_CORES = multiprocessing.cpu_count()
CV_JOBS = min(5, N_CORES)
MODEL_JOBS = max(1, N_CORES // 5)
print(f"n_cores={N_CORES}  cv_jobs={CV_JOBS}  model_jobs={MODEL_JOBS}")


# %% [markdown]
# ## Feature engineering helpers — all target-aware code guarded on train only

# %%
def _rater_bias_q75(X_raw: pd.DataFrame, y: pd.Series) -> dict:
    """Compute exp(q75(log(off/target))) per rater on the train fold.

    Returns dict with keys 'b_r' (rater_id -> float) and 'global_fallback'.
    Normalised so that the weighted global level equals median(off/target).
    """
    off = X_raw["off"].values
    rater = X_raw["rater_id"].values
    target = y.values

    mask = pd.notna(off) & (off > 0) & (target > 5)
    off_m = off[mask]
    tgt_m = target[mask]
    rid_m = rater[mask]

    if len(off_m) == 0:
        return {"b_r": {}, "global_fallback": 0.85}

    log_ratio = np.log(off_m / tgt_m)
    global_med_ratio = float(np.median(off_m / tgt_m))
    global_q75_log = float(np.percentile(log_ratio, 75))
    global_exp_q75 = float(np.exp(global_q75_log))
    norm = global_med_ratio / global_exp_q75 if global_exp_q75 > 0 else 1.0

    b_r = {}
    for rid in np.unique(rid_m):
        sel = rid_m == rid
        b_r[rid] = float(np.exp(np.percentile(log_ratio[sel], 75))) * norm

    return {"b_r": b_r, "global_fallback": global_exp_q75 * norm}


def _on_target_curve(X_raw: pd.DataFrame, y: pd.Series, n_bins: int = 10) -> dict:
    """Median on/target ratio per time_since_intake_on bin, fitted on train.

    Returns dict with 'edges', 'medians', 'global_median'.
    Apply: on_c = on / (bias * f(t)).
    """
    t = X_raw["time_since_intake_on"].values
    on = X_raw["on"].values
    target = y.values

    mask = pd.notna(t) & pd.notna(on) & (on > 0) & (target > 5)
    t_m = t[mask]
    ratio_m = on[mask] / target[mask]

    global_median = float(np.median(ratio_m)) if len(ratio_m) > 0 else 1.0

    if len(t_m) < 2 * n_bins:
        return {"edges": np.array([-np.inf, np.inf]),
                "medians": np.array([global_median]),
                "global_median": global_median}

    edges = np.percentile(t_m, np.linspace(0, 100, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    medians = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (t_m >= lo) & (t_m < hi)
        medians.append(float(np.median(ratio_m[sel])) if sel.sum() > 0 else global_median)
    return {"edges": edges, "medians": np.array(medians), "global_median": global_median}


def _apply_ft(t_vals: np.ndarray, curve: dict) -> np.ndarray:
    """Evaluate f(t) = median on/target ratio for each time_since_intake_on."""
    edges = curve["edges"]
    medians = curve["medians"]
    result = np.full(len(t_vals), curve["global_median"])
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = pd.notna(t_vals) & (t_vals >= lo) & (t_vals < hi)
        result[sel] = medians[i]
    result = np.where(pd.notna(t_vals), result, curve["global_median"])
    return np.clip(result, 0.05, 20.0)


def _correct_observations(X_raw: pd.DataFrame, bias: dict, curve: dict) -> pd.DataFrame:
    """Compute off_c and on_c for every row using train-fitted bias and curve."""
    df = X_raw[["patient_id", "rater_id", "age", "on", "off",
                "time_since_intake_on", "time_since_intake_off",
                "ledd", "gene", "cohort", "sexM",
                "age_at_diagnosis"]].copy()

    b_r = df["rater_id"].map(bias["b_r"]).fillna(bias["global_fallback"]).clip(lower=0.1).values
    ft = _apply_ft(df["time_since_intake_on"].values, curve)

    off_c = np.where(
        pd.notna(df["off"].values) & (df["off"].values > 0),
        df["off"].values / b_r,
        np.nan,
    )
    on_c = np.where(
        pd.notna(df["on"].values) & (df["on"].values > 0),
        df["on"].values / (b_r * ft),
        np.nan,
    )
    df["off_c"] = off_c
    df["on_c"] = on_c
    df["rater_bias"] = b_r
    df["ft"] = ft
    df["disease_duration"] = (df["age"] - df["age_at_diagnosis"]).clip(lower=0.0)
    return df


def _patient_k(df: pd.DataFrame) -> pd.Series:
    """Patient drug-response k = exp(mean(log(on_c / off_c))).

    Computed per patient from rows that have both on_c > 1 and off_c > 1.
    Returns a Series aligned to df.index.
    """
    k_map = {}
    for pid, grp in df.groupby("patient_id"):
        mask = grp["on_c"].notna() & grp["off_c"].notna() & (grp["off_c"] > 1) & (grp["on_c"] > 1)
        valid = grp.loc[mask]
        if len(valid) == 0:
            k_map[pid] = 1.0
        else:
            ratio = (valid["on_c"] / valid["off_c"]).clip(lower=1e-6)
            k_map[pid] = float(np.exp(np.log(ratio).mean()))
    return df["patient_id"].map(k_map).clip(lower=1e-6)


def _compute_z(df: pd.DataFrame, k: pd.Series) -> pd.Series:
    """z = log(off_c) when available, else log(on_c / k)."""
    log_off = np.where(
        df["off_c"].notna() & (df["off_c"] > 0),
        np.log(df["off_c"].clip(lower=1e-9)),
        np.nan,
    )
    log_on_k = np.where(
        df["on_c"].notna() & (df["on_c"] > 0) & (k.values > 0),
        np.log((df["on_c"] / k).clip(lower=1e-9)),
        np.nan,
    )
    z = np.where(~np.isnan(log_off), log_off, log_on_k)
    return pd.Series(z, index=df.index)


# %%
class _TabDensity:
    """Tabulated density (picklable, used in MLE)."""

    def __init__(self, grid, values):
        self.grid = np.asarray(grid, dtype=float)
        self.values = np.asarray(values, dtype=float)

    def __call__(self, u):
        u_a = np.asarray(u, dtype=float)
        return np.maximum(np.interp(u_a, self.grid, self.values), 1e-4)


def _build_density(df_train: pd.DataFrame, z_train: pd.Series, y_train: pd.Series) -> "_TabDensity":
    """KDE of z - log(target) on training rows with target > 3."""
    log_tgt = np.log(y_train.clip(lower=1e-9))
    mask = (y_train > 3) & z_train.notna()
    u = (z_train[mask] - log_tgt[mask]).values
    grid = np.linspace(-4.0, 1.5, 1000)
    if len(u) < 5:
        return _TabDensity(grid, np.full_like(grid, 1e-4))
    std_u = u.std(ddof=1) if u.std(ddof=1) > 0 else 1.0
    kde = gaussian_kde(u, bw_method=0.04 / std_u)
    return _TabDensity(grid, np.maximum(kde(grid), 1e-4))


def _slope_prior(df_train: pd.DataFrame, z_train: pd.Series, y_train: pd.Series):
    """Median and std of log(target)~age slopes per patient (>=4 visits)."""
    slopes = []
    tmp = df_train[["patient_id", "age"]].copy()
    tmp["_logy"] = np.log(y_train.clip(lower=1e-9)).values
    for _, grp in tmp.groupby("patient_id"):
        v = grp[["age", "_logy"]].dropna()
        if len(v) >= 4:
            slopes.append(float(np.polyfit(v["age"].values, v["_logy"].values, 1)[0]))
    if len(slopes) < 2:
        return 0.0, 1.0
    s = np.array(slopes)
    return float(np.median(s)), float(np.std(s, ddof=1))


def _fit_mle(z: np.ndarray, ages: np.ndarray, dens, slope_med: float, slope_std: float):
    """Fit log T = a + b*(age - age_mean) per patient by MLE with slope prior."""
    valid = ~np.isnan(z)
    z_v = z[valid]
    if len(z_v) == 0:
        return 0.0, slope_med
    x_v = ages[valid] - ages[valid].mean()

    def neg_ll(params):
        a, b = params
        u = z_v - (a + b * x_v)
        ll = np.sum(np.log(dens(u)))
        pen = 0.5 * ((b - slope_med) / max(slope_std, 1e-6)) ** 2
        return -(ll - pen)

    z_med = float(np.median(z_v))
    best_val, best_a, best_b = np.inf, z_med, slope_med
    for offset in [0.0, 0.1, -0.1, 0.2]:
        res = minimize(neg_ll, [z_med + offset, slope_med], method="Nelder-Mead",
                       options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 2000})
        if res.fun < best_val:
            best_val = res.fun
            best_a, best_b = float(res.x[0]), float(res.x[1])
    return best_a, best_b


def _rater_stats(df_train: pd.DataFrame, z_train: pd.Series, y_train: pd.Series) -> dict:
    """Rater reliability: median, std, q10, q90 of (z - log(target)) per rater."""
    log_tgt = np.log(y_train.clip(lower=1e-9))
    mask = (y_train > 3) & z_train.notna()
    u_all = (z_train[mask] - log_tgt[mask]).values
    rids = df_train.loc[mask, "rater_id"].values

    g_med = float(np.median(u_all)) if len(u_all) > 0 else 0.0
    g_std = float(np.std(u_all, ddof=1)) if len(u_all) > 1 else 0.0
    g_q10 = float(np.percentile(u_all, 10)) if len(u_all) > 0 else 0.0
    g_q90 = float(np.percentile(u_all, 90)) if len(u_all) > 0 else 0.0

    r_med, r_std, r_q10, r_q90 = {}, {}, {}, {}
    for rid in np.unique(rids):
        uv = u_all[rids == rid]
        r_med[rid] = float(np.median(uv))
        r_std[rid] = float(np.std(uv, ddof=1)) if len(uv) > 1 else 0.0
        r_q10[rid] = float(np.percentile(uv, 10))
        r_q90[rid] = float(np.percentile(uv, 90))

    return {"r_med": r_med, "r_std": r_std, "r_q10": r_q10, "r_q90": r_q90,
            "g_med": g_med, "g_std": g_std, "g_q10": g_q10, "g_q90": g_q90}


def _window_max_mean(arr: np.ndarray, i: int, half: int):
    lo, hi = max(0, i - half), min(len(arr), i + half + 1)
    v = arr[lo:hi]
    valid = v[~np.isnan(v)]
    if len(valid) == 0:
        return np.nan, np.nan
    return float(np.max(valid)), float(np.mean(valid))


def _window_q75(arr: np.ndarray, i: int, half: int):
    lo, hi = max(0, i - half), min(len(arr), i + half + 1)
    v = arr[lo:hi]
    valid = v[~np.isnan(v)]
    return float(np.percentile(valid, 75)) if len(valid) > 0 else np.nan


def _kde_mode(vals: np.ndarray, h: float):
    """Score each value as Σ_j exp(-0.5*((v_i - v_j)/h)^2); return (best_val, best_score)."""
    if len(vals) == 0:
        return np.nan, np.nan
    diff = (vals[:, None] - vals[None, :]) / h
    scores = np.sum(np.exp(-0.5 * diff ** 2), axis=1)
    best_i = int(np.argmax(scores + vals * 1e-12))
    return float(vals[best_i]), float(scores[best_i])


def _patient_features(
    pid,
    ages: np.ndarray,
    z: np.ndarray,
    mle_a: float,
    mle_b: float,
    rater_ids: np.ndarray,
    r_stats: dict,
) -> dict:
    """All per-patient features for one patient (sorted by age, returned in original order)."""
    n = len(ages)
    sort_ord = np.argsort(ages, kind="stable")
    inv_ord = np.argsort(sort_ord, kind="stable")

    ages_s = ages[sort_ord]
    z_s = z[sort_ord]
    rids_s = rater_ids[sort_ord]

    age_mean = float(np.mean(ages_s))
    trend_s = mle_b * (ages_s - age_mean)
    zd_s = z_s - trend_s

    # --- Global zd stats ---
    valid_zd = zd_s[~np.isnan(zd_s)]
    nv = len(valid_zd)

    zd_med = float(np.median(valid_zd)) if nv >= 1 else 0.0
    zd_q75 = float(np.percentile(valid_zd, 75)) if nv >= 1 else 0.0
    zd_q90 = float(np.percentile(valid_zd, 90)) if nv >= 1 else 0.0
    zd_max = float(np.max(valid_zd)) if nv >= 1 else 0.0
    zd_top2 = float(np.mean(np.sort(valid_zd)[-2:])) if nv >= 2 else zd_max
    zd_std = float(np.std(valid_zd, ddof=1)) if nv >= 2 else 0.0
    zd_count = float(nv)

    env_med = np.exp(zd_med + trend_s)
    env_q75 = np.exp(zd_q75 + trend_s)
    env_q90 = np.exp(zd_q90 + trend_s)
    env_max = np.exp(zd_max + trend_s)
    env_top2 = np.exp(zd_top2 + trend_s)

    # --- Sliding windows (max / mean / q75 on 3/5/7 visits) ---
    win3_max = np.full(n, np.nan); win3_avg = np.full(n, np.nan); win3_q75 = np.full(n, np.nan)
    win5_max = np.full(n, np.nan); win5_avg = np.full(n, np.nan); win5_q75 = np.full(n, np.nan)
    win7_max = np.full(n, np.nan); win7_avg = np.full(n, np.nan); win7_q75 = np.full(n, np.nan)

    for i in range(n):
        t = trend_s[i]
        m3, a3 = _window_max_mean(zd_s, i, 1)
        m5, a5 = _window_max_mean(zd_s, i, 2)
        m7, a7 = _window_max_mean(zd_s, i, 3)
        win3_max[i] = np.exp(m3 + t) if not np.isnan(m3) else np.nan
        win3_avg[i] = np.exp(a3 + t) if not np.isnan(a3) else np.nan
        win3_q75[i] = np.exp(_window_q75(zd_s, i, 1) + t) if not np.isnan(_window_q75(zd_s, i, 1)) else np.nan
        win5_max[i] = np.exp(m5 + t) if not np.isnan(m5) else np.nan
        win5_avg[i] = np.exp(a5 + t) if not np.isnan(a5) else np.nan
        win5_q75[i] = np.exp(_window_q75(zd_s, i, 2) + t) if not np.isnan(_window_q75(zd_s, i, 2)) else np.nan
        win7_max[i] = np.exp(m7 + t) if not np.isnan(m7) else np.nan
        win7_avg[i] = np.exp(a7 + t) if not np.isnan(a7) else np.nan
        win7_q75[i] = np.exp(_window_q75(zd_s, i, 3) + t) if not np.isnan(_window_q75(zd_s, i, 3)) else np.nan

    # --- Cummax fwd / bwd ---
    cummax_fwd = np.full(n, np.nan)
    cummax_bwd = np.full(n, np.nan)
    rm = -np.inf
    for i in range(n):
        if not np.isnan(zd_s[i]):
            rm = max(rm, zd_s[i])
        if rm > -np.inf:
            cummax_fwd[i] = np.exp(rm + trend_s[i])
    rm = -np.inf
    for i in range(n - 1, -1, -1):
        if not np.isnan(zd_s[i]):
            rm = max(rm, zd_s[i])
        if rm > -np.inf:
            cummax_bwd[i] = np.exp(rm + trend_s[i])

    # --- Lags +-1 +-2 ---
    lag_m2 = np.full(n, np.nan); lag_m1 = np.full(n, np.nan)
    lag_p1 = np.full(n, np.nan); lag_p2 = np.full(n, np.nan)
    for i in range(n):
        t = trend_s[i]
        if i >= 2 and not np.isnan(zd_s[i - 2]):
            lag_m2[i] = np.exp(zd_s[i - 2] + t)
        if i >= 1 and not np.isnan(zd_s[i - 1]):
            lag_m1[i] = np.exp(zd_s[i - 1] + t)
        if i + 1 < n and not np.isnan(zd_s[i + 1]):
            lag_p1[i] = np.exp(zd_s[i + 1] + t)
        if i + 2 < n and not np.isnan(zd_s[i + 2]):
            lag_p2[i] = np.exp(zd_s[i + 2] + t)

    # --- Rater reliability features per row ---
    def _r(d, fallback_k):
        return np.array([d.get(r, r_stats[fallback_k]) for r in rids_s], dtype=float)

    r_med_a = _r(r_stats["r_med"], "g_med")
    r_std_a = _r(r_stats["r_std"], "g_std")
    r_q10_a = _r(r_stats["r_q10"], "g_q10")
    r_q90_a = _r(r_stats["r_q90"], "g_q90")

    # --- KDE mode of zd ---
    mode_c, mode_sc = _kde_mode(valid_zd, h=0.04)
    mode_val = np.exp(mode_c + trend_s) if not np.isnan(mode_c) else np.full(n, np.nan)
    mode_score = np.full(n, mode_sc if not np.isnan(mode_sc) else np.nan)

    # Re-order to original
    def _ro(a):
        return a[inv_ord]

    return {
        "env_med":    _ro(env_med),
        "env_q75":    _ro(env_q75),
        "env_q90":    _ro(env_q90),
        "env_max":    _ro(env_max),
        "env_top2":   _ro(env_top2),
        "zd_std":     _ro(np.full(n, zd_std)),
        "zd_count":   _ro(np.full(n, zd_count)),
        "win3_max":   _ro(win3_max), "win3_avg": _ro(win3_avg), "win3_q75": _ro(win3_q75),
        "win5_max":   _ro(win5_max), "win5_avg": _ro(win5_avg), "win5_q75": _ro(win5_q75),
        "win7_max":   _ro(win7_max), "win7_avg": _ro(win7_avg), "win7_q75": _ro(win7_q75),
        "cummax_fwd": _ro(cummax_fwd),
        "cummax_bwd": _ro(cummax_bwd),
        "lag_m2":     _ro(lag_m2), "lag_m1": _ro(lag_m1),
        "lag_p1":     _ro(lag_p1), "lag_p2": _ro(lag_p2),
        "r_med":      _ro(r_med_a), "r_std": _ro(r_std_a),
        "r_q10":      _ro(r_q10_a), "r_q90": _ro(r_q90_a),
        "mode_val":   _ro(mode_val),
        "mode_score": _ro(mode_score),
    }


# %%
def build_features_v20(
    X_raw: pd.DataFrame,
    y: pd.Series | None = None,
    train_stats: dict | None = None,
) -> tuple[pd.DataFrame, dict | None]:
    """Build feature matrix.

    When y is given (train mode), compute all target-dependent statistics on
    the supplied rows and return them in `train_stats`.
    When y is None (predict mode), train_stats must be provided.
    """
    is_train = y is not None

    # ---- Rater bias + on/target curve ----
    if is_train:
        bias = _rater_bias_q75(X_raw, y)
        curve = _on_target_curve(X_raw, y)
    else:
        bias = train_stats["bias"]
        curve = train_stats["curve"]

    df = _correct_observations(X_raw, bias, curve)

    # ---- Patient k and z ----
    k = _patient_k(df)
    df["k"] = k
    z = _compute_z(df, k)
    df["z"] = z

    # ---- MLE density + slope prior (train only) ----
    if is_train:
        y_ser = pd.Series(y.values if hasattr(y, "values") else np.asarray(y), index=df.index)
        dens = _build_density(df, z, y_ser)
        slope_med, slope_std = _slope_prior(df, z, y_ser)
        r_stats = _rater_stats(df, z, y_ser)
    else:
        dens = train_stats["dens"]
        slope_med = train_stats["slope_med"]
        slope_std = train_stats["slope_std"]
        r_stats = train_stats["r_stats"]

    # ---- Per-patient MLE fit ----
    mle_a_map: dict = {}
    mle_b_map: dict = {}
    for pid, grp in df.groupby("patient_id"):
        z_pat = z.loc[grp.index].values
        ages_pat = grp["age"].values
        a, b = _fit_mle(z_pat, ages_pat, dens, slope_med, slope_std)
        mle_a_map[pid] = a
        mle_b_map[pid] = b

    age_mean_map: dict = {}
    for pid, grp in df.groupby("patient_id"):
        age_mean_map[pid] = float(grp["age"].mean())

    df["mle_a"] = df["patient_id"].map(mle_a_map)
    df["mle_b"] = df["patient_id"].map(mle_b_map)
    df["age_mean_pat"] = df["patient_id"].map(age_mean_map)
    df["x_centered"] = df["age"] - df["age_mean_pat"]
    df["mle"] = np.exp(df["mle_a"] + df["mle_b"] * df["x_centered"])
    df["z_resid"] = z - (df["mle_a"] + df["mle_b"] * df["x_centered"])

    # ---- Per-patient envelope + window + lag + rater + mode features ----
    pat_feat_arrays: dict[str, np.ndarray] = {}
    for pid, grp in df.groupby("patient_id"):
        pos = df.index.get_indexer(grp.index)
        res = _patient_features(
            pid,
            grp["age"].values,
            z.loc[grp.index].values,
            mle_a_map[pid],
            mle_b_map[pid],
            grp["rater_id"].values,
            r_stats,
        )
        for col, arr in res.items():
            if col not in pat_feat_arrays:
                pat_feat_arrays[col] = np.full(len(df), np.nan)
            pat_feat_arrays[col][pos] = arr

    for col, arr in pat_feat_arrays.items():
        df[col] = arr

    # ---- Scalar raw features (no target dependency) ----
    df["visit_n"] = df.groupby("patient_id").cumcount()
    df["visit_n_total"] = df["patient_id"].map(df.groupby("patient_id")["age"].transform("count"))
    df["visit_rev"] = df["visit_n_total"] - df["visit_n"]
    df["ledd_off_c"] = df["ledd"] / df["off_c"].replace(0, np.nan)
    df["ledd_on_c"] = df["ledd"] / df["on_c"].replace(0, np.nan)
    df["off_c_on_c_ratio"] = df["off_c"] / df["on_c"].replace(0, np.nan)

    # ---- Encode categoricals numerically (no raw rater_id) ----
    gene_map = {v: i for i, v in enumerate(sorted(df["gene"].fillna("Unknown").unique()))}
    cohort_map = {v: i for i, v in enumerate(sorted(df["cohort"].unique()))}
    df["gene_enc"] = df["gene"].fillna("Unknown").map(gene_map).fillna(-1).astype("int32")
    df["cohort_enc"] = df["cohort"].map(cohort_map).fillna(-1).astype("int32")

    feature_cols = [
        # corrected observations
        "off_c", "on_c", "rater_bias", "ft", "k",
        # clinical
        "age", "disease_duration", "ledd", "sexM",
        "time_since_intake_on", "time_since_intake_off",
        # MLE curve
        "mle", "mle_b", "z_resid",
        # envelope patient
        "env_med", "env_q75", "env_q90", "env_max", "env_top2",
        "zd_std", "zd_count",
        # sliding windows
        "win3_max", "win3_avg", "win3_q75",
        "win5_max", "win5_avg", "win5_q75",
        "win7_max", "win7_avg", "win7_q75",
        # cummax
        "cummax_fwd", "cummax_bwd",
        # lags
        "lag_m2", "lag_m1", "lag_p1", "lag_p2",
        # rater reliability
        "r_med", "r_std", "r_q10", "r_q90",
        # KDE mode
        "mode_val", "mode_score",
        # derived scalars
        "visit_n", "visit_n_total", "visit_rev",
        "ledd_off_c", "ledd_on_c", "off_c_on_c_ratio",
        # categoricals (encoded)
        "gene_enc", "cohort_enc",
    ]

    X_out = df[feature_cols].copy()

    new_stats = None
    if is_train:
        new_stats = {
            "bias": bias,
            "curve": curve,
            "dens": dens,
            "slope_med": slope_med,
            "slope_std": slope_std,
            "r_stats": r_stats,
        }

    return X_out, new_stats


# %% [markdown]
# ## Post-processing : poly2 smooth + ratchet + clip

# %%
def _smooth_poly2(preds: np.ndarray, X_raw: pd.DataFrame) -> np.ndarray:
    """Fit quadratic pred ~ age per patient; fall back to linear if < 4 visits."""
    p = preds.copy().astype(float)
    df = X_raw[["patient_id", "age"]].copy()
    df["_p"] = p
    for _, grp in df.groupby("patient_id"):
        n = len(grp)
        if n < 3:
            continue
        ages = grp["age"].values
        d = 2 if n >= 4 else 1
        coeffs = np.polyfit(ages, grp["_p"].values, d)
        df.loc[grp.index, "_p"] = np.polyval(coeffs, ages)
    return df["_p"].values


def _ratchet(preds: np.ndarray, X_raw: pd.DataFrame, rate: float = 1.02) -> np.ndarray:
    """Monotone ratchet per patient sorted by age.

    The target in y_train never drops; when it stagnates it increases ~2% per
    visit.  Mirror that: if pred_i <= pred_{i-1}, set pred_i = rate * pred_{i-1}.
    """
    p = preds.copy().astype(float)
    df = X_raw[["patient_id", "age"]].copy().reset_index(drop=True)
    df["_p"] = p
    df["_orig"] = df.index
    df_s = df.sort_values(["patient_id", "age"])

    for _, grp in df_s.groupby("patient_id"):
        idxs = grp["_orig"].values
        vals = grp["_p"].values.copy()
        for i in range(1, len(vals)):
            if vals[i] <= vals[i - 1]:
                vals[i] = rate * vals[i - 1]
        df_s.loc[grp.index, "_p"] = vals

    df_s = df_s.sort_values("_orig")
    return df_s["_p"].values


def postprocess(preds: np.ndarray, X_raw: pd.DataFrame) -> np.ndarray:
    """Poly2 smooth → ratchet → clip(0)."""
    p = _smooth_poly2(preds, X_raw)
    p = _ratchet(p, X_raw)
    return np.clip(p, 0, None)


# %% [markdown]
# ## sklearn wrapper

# %%
class RatchetPipeline20(BaseEstimator, RegressorMixin):
    """LGBM + XGB averaged, with poly2 + ratchet post-processing."""

    def __init__(self, model_jobs: int = 1):
        self.model_jobs = model_jobs

    def fit(self, X, y):
        y_arr = y.values if hasattr(y, "values") else np.asarray(y, dtype=float)
        y_ser = pd.Series(y_arr, index=X.index if hasattr(X, "index") else np.arange(len(y_arr)))
        self._X_raw_fit = X.copy() if hasattr(X, "copy") else pd.DataFrame(X)
        self._y_fit = y_ser

        X_feat, self._train_stats = build_features_v20(
            self._X_raw_fit, y=y_ser
        )

        n_jobs = self.model_jobs
        self.lgbm_ = LGBMRegressor(
            n_estimators=1500, learning_rate=0.02, num_leaves=63,
            min_child_samples=40, subsample=0.85, colsample_bytree=0.85,
            subsample_freq=1, reg_lambda=1.0, n_jobs=n_jobs,
            random_state=42, verbose=-1,
        )
        self.xgb_ = XGBRegressor(
            n_estimators=1500, learning_rate=0.02, max_depth=7,
            subsample=0.85, colsample_bytree=0.85,
            reg_lambda=1.0, tree_method="hist",
            n_jobs=n_jobs, random_state=42, verbosity=0,
        )

        X_np = X_feat.values
        self.lgbm_.fit(X_np, y_arr)
        self.xgb_.fit(X_np, y_arr)
        self.feature_names_in_ = np.array(X_feat.columns.tolist())
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def predict(self, X):
        check_is_fitted(self, "lgbm_")
        X_raw = X.copy() if hasattr(X, "copy") else pd.DataFrame(X)
        X_feat, _ = build_features_v20(X_raw, train_stats=self._train_stats)
        X_np = X_feat.values
        p_lgbm = self.lgbm_.predict(X_np)
        p_xgb = self.xgb_.predict(X_np)
        preds_avg = (p_lgbm + p_xgb) / 2.0
        return postprocess(preds_avg, X_raw)


# %% [markdown]
# ## Quick 1-fold test

# %%
print("=" * 60)
print("Exp 20 — bench ratchet : 1-fold smoke test")
print("=" * 60)

tr_idx, val_idx = folds[0]
X_tr = X_train.iloc[tr_idx].reset_index(drop=True)
y_tr = y_train.iloc[tr_idx].reset_index(drop=True)
X_val = X_train.iloc[val_idx].reset_index(drop=True)
y_val = y_train.iloc[val_idx].reset_index(drop=True)

model_1fold = RatchetPipeline20(model_jobs=MODEL_JOBS)
model_1fold.fit(X_tr, y_tr)
preds_1fold = model_1fold.predict(X_val)

from sklearn.metrics import root_mean_squared_error
rmse_1fold = root_mean_squared_error(y_val, preds_1fold)
print(f"Fold 1 RMSE (smoke test) = {rmse_1fold:.4f}")
print("Smoke test passed — proceeding to full CV.")

# %% [markdown]
# ## Full CrossValidationReport

# %%
print(f"\nRunning CrossValidationReport exp 20 (n_jobs={CV_JOBS})…")
report = skore.CrossValidationReport(
    RatchetPipeline20(model_jobs=MODEL_JOBS),
    X=X_train,
    y=y_train,
    splitter=folds,
    n_jobs=CV_JOBS,
)
project.put("20_bench_ratchet", report)
print("Report saved: 20_bench_ratchet")

# RMSE mean + std + per fold
rmse_df = report.metrics.rmse(aggregate=None)
print(rmse_df)

try:
    if "RMSE" in rmse_df.columns:
        fold_vals = rmse_df["RMSE"].values
    else:
        fold_vals = rmse_df.iloc[:, 0].values
    for i, v in enumerate(fold_vals):
        print(f"  Fold {i + 1}  RMSE = {v:.4f}")
    print(f"  Mean RMSE = {float(np.mean(fold_vals)):.4f}  std = {float(np.std(fold_vals)):.4f}")
except Exception as exc:
    print(f"  (Could not parse fold values: {exc})")

# %% [markdown]
# ## Refit on full train + submission

# %%
print("\nRefitting on full train…")
final_model = RatchetPipeline20(model_jobs=MODEL_JOBS)
final_model.fit(X_train, y_train)

X_test = load_test_dataset(DATA_DIR)
print(f"Test: {X_test.shape}")

test_preds = final_model.predict(X_test)

import pathlib
pathlib.Path("submissions").mkdir(exist_ok=True)

submission = pd.DataFrame({
    "Index": X_test["Index"] if "Index" in X_test.columns else X_test.index,
    "target": test_preds,
})
submission.to_csv("submissions/20_bench_ratchet.csv", index=False)
print(f"Submission saved: submissions/20_bench_ratchet.csv  ({len(submission)} rows)")
