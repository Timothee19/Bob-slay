# %% [markdown]
# # Experiment 11 — 3-model ensemble (2×LightGBM + XGBoost) + Bateman PK model
#
# **Date:** 2025-01-07
# **Base:** Exp 09 (RMSE 3.456 poly2+isotonic / 3.589 raw ensemble)
#
# ## New ideas this iteration
#
# ### A. Bateman pharmacokinetic model replaces the PK proxy
# Experiment 09 used a per-patient heuristic (median on_corrected / on_time_corrected)
# as a crude PK proxy.  Here we replace it with the Bateman biexponential equation,
# which models plasma drug concentration as a function of LEDD (dose) and time since
# last intake:
#
#   C(t) = (ledd * ka / (ka - ke)) * (exp(-ke * t) - exp(-ka * t))
#
# where ka=1.5 (absorption rate constant, h⁻¹) and ke=0.46 (elimination rate
# constant, h⁻¹) are population-level levodopa PK parameters.
#
# Two derived features are created:
# - ``pk_concentration_on``  — concentration at time_since_intake_on
# - ``pk_concentration_off`` — concentration at time_since_intake_off
#
# Missing ledd or time values propagate as NaN (no imputation); the tree models
# handle NaN natively.

# %%
import numpy as np
import pandas as pd
import skore
from sklearn.metrics import root_mean_squared_error
from sklearn.isotonic import IsotonicRegression

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
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
# ## Post-processing utilities

# %%
def smooth_isotonic(predictions: np.ndarray, X_val: pd.DataFrame) -> np.ndarray:
    preds = predictions.copy().astype(float)
    df = X_val[["patient_id", "age"]].copy()
    df["_pred"] = preds
    ir = IsotonicRegression(increasing=True)
    for pid, grp in df.groupby("patient_id"):
        if len(grp) < 2:
            continue
        grp_sorted = grp.sort_values("age")
        ages = grp_sorted["age"].values
        fitted = ir.fit_transform(ages, grp_sorted["_pred"].values)
        df.loc[grp_sorted.index, "_pred"] = fitted
    return df["_pred"].values


def smooth_poly2_then_isotonic(predictions, X_val):
    p1 = smooth_by_age_poly(predictions, X_val, degree=2)
    return smooth_isotonic(p1, X_val)


# %% [markdown]
# ## Bateman pharmacokinetic model

# %%
def compute_bateman_concentration(
    ledd: pd.Series,
    time_since_intake: pd.Series,
    ka: float = 1.5,
    ke: float = 0.46,
) -> pd.Series:
    """Compute plasma drug concentration using the Bateman biexponential equation.

    C(t) = (ledd * ka / (ka - ke)) * (exp(-ke * t) - exp(-ka * t))

    Parameters
    ----------
    ledd : pd.Series
        Levodopa equivalent daily dose (mg).
    time_since_intake : pd.Series
        Time since last levodopa intake (hours).
    ka : float
        Absorption rate constant (h⁻¹), default 1.5.
    ke : float
        Elimination rate constant (h⁻¹), default 0.46.

    Returns
    -------
    pd.Series
        Estimated drug concentration.  NaN where ledd or time_since_intake is NaN.
    """
    concentration = (ledd * ka / (ka - ke)) * (
        np.exp(-ke * time_since_intake) - np.exp(-ka * time_since_intake)
    )
    # Preserve NaN: if either input is NaN the arithmetic already yields NaN,
    # but be explicit to guard against any future dtype changes.
    nan_mask = ledd.isna() | time_since_intake.isna()
    concentration[nan_mask] = np.nan
    return concentration


# %% [markdown]
# ## Feature engineering (v4 — full accumulation + Bateman PK)

# %%
def add_extra_features_v4(
    X_fe: pd.DataFrame,
    X_raw: pd.DataFrame,
    y_train: pd.Series | None = None,
    saved_stats: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """All features from exp 06-09, with patient_pk_ratio replaced by Bateman PK features."""
    df = X_fe.copy()
    df["patient_id"] = X_raw["patient_id"].values
    df["rater_id"] = X_raw["rater_id"].values
    df["gene"] = X_raw["gene"].fillna("Unknown").values
    df["cohort"] = X_raw["cohort"].values

    # ------------------------------------------------------------------ #
    # Exp-06 features
    # ------------------------------------------------------------------ #
    df_sorted = df.sort_values(["patient_id", "age"])
    grp = df_sorted.groupby("patient_id")
    df_sorted["off_corrected_prev2"] = grp["off_corrected"].shift(2)
    df_sorted["on_corrected_prev2"]  = grp["on_corrected"].shift(2)
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
    for col in ["off_corrected_prev2", "on_corrected_prev2", "off_lag_delta",
                "inter_visit_gap", "off_lag_delta_per_year", "off_corrected_next2"]:
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
            on_slopes[pid] = float(np.polyfit(valid["age"].values,
                                              valid["on_corrected"].values, 1)[0])
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
            lambda x: float(np.nanpercentile(x.dropna().values, 75) -
                            np.nanpercentile(x.dropna().values, 25))
            if x.notna().sum() >= 2 else 0.0
        )
    )

    df_sorted2 = df.sort_values(["patient_id", "age"])
    grp2 = df_sorted2.groupby("patient_id")
    df_sorted2["off_cummin"] = grp2["off_corrected"].cummin()
    df_sorted2["on_time_corrected_prev"] = grp2["on_time_corrected"].shift(1)
    df_sorted2 = df_sorted2.loc[df.index]
    df["off_cummin"] = df_sorted2["off_cummin"].values
    df["on_time_corrected_prev"] = df_sorted2["on_time_corrected_prev"].values

    # ------------------------------------------------------------------ #
    # Exp-07 features
    # ------------------------------------------------------------------ #
    slope2_map = {}
    for pid, grp_p in df.groupby("patient_id"):
        valid = grp_p[["off_corrected", "age"]].dropna()
        if len(valid) >= 4:
            coeffs = np.polyfit(valid["age"].values, valid["off_corrected"].values, 2)
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
    prev_delta = df_sorted3["off_corrected_prev"] - df_sorted3["off_corrected_prev2"]
    curr_delta = df_sorted3["off_corrected"] - df_sorted3["off_corrected_prev"]
    df_sorted3["off_lag_accel"] = curr_delta - prev_delta
    df_sorted3 = df_sorted3.loc[df.index]
    for col in ["off_residual_prev", "on_time_corrected_next",
                "ledd_prev", "ledd_next", "off_lag_accel"]:
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

    # ------------------------------------------------------------------ #
    # Exp-08 fold-aware stats + features
    # ------------------------------------------------------------------ #
    if y_train is not None:
        df_y = df.copy()
        df_y["_target"] = y_train.values
        rater_mean_map, rater_std_map, rater_count_map = {}, {}, {}
        for rid, grp_r in df_y.groupby("rater_id"):
            vals = grp_r["_target"].dropna()
            rater_mean_map[rid] = float(vals.mean()) if len(vals) > 0 else np.nan
            rater_std_map[rid]  = float(vals.std())  if len(vals) > 1 else 0.0
            rater_count_map[rid] = len(vals)
        _g_rater_mean = float(np.nanmean(list(rater_mean_map.values())))
        _g_rater_std  = float(np.nanmean(list(rater_std_map.values())))

        gene_off_mean_map = df.groupby("gene")["off_corrected"].mean().to_dict()
        gene_slope_mean_map = {}
        for gene, grp_g in df.groupby("gene"):
            slopes = [on_slopes[pid] for pid in grp_g["patient_id"].unique()
                      if pid in on_slopes]
            gene_slope_mean_map[gene] = float(np.mean(slopes)) if slopes else np.nan

        cohort_off_mean_map = df.groupby("cohort")["off_corrected"].mean().to_dict()
        cohort_off_std_map  = df.groupby("cohort")["off_corrected"].std().to_dict()
        cohort_slope_mean_map = {}
        for coh, grp_c in df.groupby("cohort"):
            slopes_c = [on_slopes[pid] for pid in grp_c["patient_id"].unique()
                        if pid in on_slopes]
            cohort_slope_mean_map[coh] = float(np.mean(slopes_c)) if slopes_c else np.nan

        saved_stats_out = {
            "rater_mean": rater_mean_map, "rater_std": rater_std_map,
            "rater_count": rater_count_map,
            "g_rater_mean": _g_rater_mean, "g_rater_std": _g_rater_std,
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
        df["rater_target_mean"]  = df["rater_id"].map(stats["rater_mean"]).fillna(stats["g_rater_mean"])
        df["rater_target_std"]   = df["rater_id"].map(stats["rater_std"]).fillna(stats["g_rater_std"])
        df["rater_target_count"] = df["rater_id"].map(stats["rater_count"]).fillna(0)
        df["gene_off_mean_feat"] = df["gene"].map(stats["gene_off_mean"])
        df["gene_slope_mean_feat"] = df["gene"].map(stats["gene_slope_mean"])
        df["cohort_off_mean_feat"] = df["cohort"].map(stats["cohort_off_mean"])
        df["cohort_off_std_feat"]  = df["cohort"].map(stats["cohort_off_std"])
        df["cohort_slope_mean_feat"] = df["cohort"].map(stats["cohort_slope_mean"])
    else:
        for col in ["rater_target_mean", "rater_target_std", "rater_target_count",
                    "gene_off_mean_feat", "gene_slope_mean_feat",
                    "cohort_off_mean_feat", "cohort_off_std_feat", "cohort_slope_mean_feat"]:
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
    df_sorted4["off_running_std"] = grp4["off_corrected"].expanding().std().reset_index(level=0, drop=True)
    df_sorted4 = df_sorted4.loc[df.index]
    df["off_running_std"] = df_sorted4["off_running_std"].values

    # ------------------------------------------------------------------ #
    # Exp-11 new features (replaces Exp-09 PK proxy)
    # ------------------------------------------------------------------ #

    # B. EWMA of off_corrected and on_corrected (shifted by 1 = prev-only)
    df_sorted5 = df.sort_values(["patient_id", "age"])
    grp5 = df_sorted5.groupby("patient_id")
    # Shift first so current visit is NOT included in EWMA (no leakage)
    off_shifted = grp5["off_corrected"].shift(1)
    on_shifted  = grp5["on_corrected"].shift(1)
    df_sorted5["off_ewma"] = off_shifted.groupby(df_sorted5["patient_id"]).transform(
        lambda x: x.ewm(alpha=0.5, adjust=False).mean()
    )
    df_sorted5["on_ewma"] = on_shifted.groupby(df_sorted5["patient_id"]).transform(
        lambda x: x.ewm(alpha=0.5, adjust=False).mean()
    )
    df_sorted5 = df_sorted5.loc[df.index]
    df["off_ewma"] = df_sorted5["off_ewma"].values
    df["on_ewma"]  = df_sorted5["on_ewma"].values

    # C. Bateman PK model: replace patient_pk_ratio with true pharmacokinetic features
    df["pk_concentration_on"] = compute_bateman_concentration(
        df["ledd"], df["time_since_intake_on"]
    ).values
    df["pk_concentration_off"] = compute_bateman_concentration(
        df["ledd"], df["time_since_intake_off"]
    ).values

    # D. Relative severity vs gene/cohort mean
    if stats is not None:
        df["off_vs_gene_mean"] = df["off_corrected"] - df["gene_off_mean_feat"]
        df["off_vs_cohort_mean"] = df["off_corrected"] - df["cohort_off_mean_feat"]
    else:
        df["off_vs_gene_mean"] = np.nan
        df["off_vs_cohort_mean"] = np.nan

    # E. Visit reverse index (distance from last visit)
    df["visit_rev_index"] = df["patient_n_visits"] - df["visit_index"]

    # F. Exponentially-weighted of ledd (same shifted logic)
    df_sorted6 = df.sort_values(["patient_id", "age"])
    grp6 = df_sorted6.groupby("patient_id")
    ledd_shifted = grp6["ledd"].shift(1)
    df_sorted6["ledd_ewma"] = ledd_shifted.groupby(df_sorted6["patient_id"]).transform(
        lambda x: x.ewm(alpha=0.5, adjust=False).mean()
    )
    df_sorted6 = df_sorted6.loc[df.index]
    df["ledd_ewma"] = df_sorted6["ledd_ewma"].values

    df = df.drop(columns=["patient_id", "rater_id", "gene", "cohort", "patient_ledd_std"])
    return df, saved_stats_out


# %%
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
ALL_COLS = FEATURE_COLS + EXP06_COLS + EXP07_COLS + EXP08_COLS + EXP11_COLS
print(f"Feature count: {len(ALL_COLS)} ({len(FEATURE_COLS)} base + "
      f"{len(EXP06_COLS)}+{len(EXP07_COLS)}+{len(EXP08_COLS)}+{len(EXP11_COLS)} new)")

# %% [markdown]
# ## Model factories

# %%
try:
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError(f"Missing package: {e}") from e


def make_lgbm_1():
    return LGBMRegressor(
        n_estimators=3000, learning_rate=0.008,
        min_child_samples=40, reg_lambda=2.0, num_leaves=127,
        subsample=0.85, colsample_bytree=0.85, subsample_freq=1,
        n_jobs=-1, random_state=42, verbose=-1,
    )

def make_lgbm_2():
    return LGBMRegressor(
        n_estimators=3000, learning_rate=0.008,
        min_child_samples=50, reg_lambda=1.5, num_leaves=95,
        subsample=0.80, colsample_bytree=0.80, subsample_freq=1,
        n_jobs=-1, random_state=2025, verbose=-1,
    )

def make_xgb():
    return XGBRegressor(
        n_estimators=3000, learning_rate=0.008,
        min_child_weight=40, reg_lambda=2.0, max_depth=6,
        subsample=0.85, colsample_bytree=0.85,
        tree_method="hist",
        n_jobs=-1, random_state=99, verbosity=0,
    )

# %% [markdown]
# ## Cross-validation

# %%
all_y_true = []
all_y_pred_raw = []
fold_rmses_raw = []
fold_data = []

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    X_tr_raw = X_train.iloc[tr_idx].reset_index(drop=True)
    y_tr     = y_train.iloc[tr_idx].reset_index(drop=True)
    X_va_raw = X_train.iloc[va_idx].reset_index(drop=True)
    y_va     = y_train.iloc[va_idx].reset_index(drop=True)

    X_tr_fe, X_va_fe = build_features(X_tr_raw, y_tr, X_val=X_va_raw)
    X_tr_ext, saved_stats = add_extra_features_v4(X_tr_fe, X_tr_raw, y_train=y_tr)
    X_va_ext, _           = add_extra_features_v4(X_va_fe, X_va_raw, saved_stats=saved_stats)

    X_tr_ext = X_tr_ext[ALL_COLS]
    X_va_ext = X_va_ext[ALL_COLS]
    cat_cols = [c for c in ALL_COLS if str(X_tr_ext[c].dtype) == "category"]
    cat_indices = [i for i, c in enumerate(ALL_COLS) if c in set(cat_cols)]

    # LightGBM models
    m1 = make_lgbm_1()
    m1.fit(X_tr_ext, y_tr.values, categorical_feature=cat_cols)
    p1 = m1.predict(X_va_ext)

    m2 = make_lgbm_2()
    m2.fit(X_tr_ext, y_tr.values, categorical_feature=cat_cols)
    p2 = m2.predict(X_va_ext)

    # XGBoost — encode categoricals as integers (no native category support)
    X_tr_xgb = X_tr_ext.copy()
    X_va_xgb = X_va_ext.copy()
    for col in cat_cols:
        X_tr_xgb[col] = X_tr_xgb[col].cat.codes.astype("float32")
        X_va_xgb[col] = X_va_xgb[col].cat.codes.astype("float32")
    # Align val categories (codes may differ if new categories appear)
    for col in cat_cols:
        # Remap val codes to match train
        train_cats = X_tr_ext[col].cat.categories
        val_series = X_va_ext[col]
        # Map val values to train integer codes
        code_map = {v: i for i, v in enumerate(train_cats)}
        X_va_xgb[col] = val_series.map(lambda x, cm=code_map: cm.get(x, -1)).astype("float32")

    mxgb = make_xgb()
    mxgb.fit(X_tr_xgb.values, y_tr.values)
    p_xgb = mxgb.predict(X_va_xgb.values)

    # Ensemble: average of 3
    preds = (p1 + p2 + p_xgb) / 3.0

    rmse = root_mean_squared_error(y_va.values, preds)
    r1 = root_mean_squared_error(y_va.values, p1)
    r2 = root_mean_squared_error(y_va.values, p2)
    rxgb = root_mean_squared_error(y_va.values, p_xgb)
    fold_rmses_raw.append(rmse)
    all_y_true.append(y_va.values)
    all_y_pred_raw.append(preds)
    fold_data.append((X_va_raw, y_va, preds))
    print(f"  fold {fold_idx}: RMSE lgbm1={r1:.4f}  lgbm2={r2:.4f}  xgb={rxgb:.4f}  ens={rmse:.4f}")

overall_rmse_raw = root_mean_squared_error(
    np.concatenate(all_y_true), np.concatenate(all_y_pred_raw)
)
print(f"\nEnsemble raw — Overall RMSE: {overall_rmse_raw:.4f}  "
      f"| mean: {np.mean(fold_rmses_raw):.4f} +/- {np.std(fold_rmses_raw):.4f}")

# %% [markdown]
# ## Post-processing comparison

# %%
smoothed_poly2  = [smooth_by_age_poly(p, X_va, degree=2) for X_va, _, p in fold_data]
smoothed_p2iso  = [smooth_poly2_then_isotonic(p, X_va) for X_va, _, p in fold_data]

rmse_poly2 = root_mean_squared_error(np.concatenate(all_y_true), np.concatenate(smoothed_poly2))
rmse_p2iso = root_mean_squared_error(np.concatenate(all_y_true), np.concatenate(smoothed_p2iso))

print(f"Poly2:        {rmse_poly2:.4f}  (delta vs exp09: {rmse_poly2 - 3.456:+.4f})")
print(f"Poly2+Iso:    {rmse_p2iso:.4f}  (delta vs exp09: {rmse_p2iso - 3.456:+.4f})")
best_label = "poly2" if rmse_poly2 <= rmse_p2iso else "poly2+isotonic"
best_rmse = min(rmse_poly2, rmse_p2iso)
print(f"Best: {best_label} ({best_rmse:.4f})")

# %% [markdown]
# ## Refit on full train + submission

# %%
X_tr_full_fe, _ = build_features(X_train, y_train)
X_tr_full_ext, stats_full = add_extra_features_v4(X_tr_full_fe, X_train, y_train=y_train)
X_tr_full_ext = X_tr_full_ext[ALL_COLS]
cat_full = [c for c in ALL_COLS if str(X_tr_full_ext[c].dtype) == "category"]

X_test = load_test_dataset(DATA_DIR)
_, X_te_fe = build_features(X_train, y_train, X_val=X_test)
X_te_ext, _ = add_extra_features_v4(X_te_fe, X_test, saved_stats=stats_full)
X_te_ext = X_te_ext[ALL_COLS]
for col in cat_full:
    X_te_ext[col] = X_te_ext[col].astype(X_tr_full_ext[col].dtype)

# XGBoost copies
X_tr_xgb_full = X_tr_full_ext.copy()
X_te_xgb = X_te_ext.copy()
for col in cat_full:
    X_tr_xgb_full[col] = X_tr_xgb_full[col].cat.codes.astype("float32")
    train_cats_full = X_tr_full_ext[col].cat.categories
    code_map_full = {v: i for i, v in enumerate(train_cats_full)}
    X_te_xgb[col] = X_te_ext[col].map(lambda x, cm=code_map_full: cm.get(x, -1)).astype("float32")

fm1 = make_lgbm_1()
fm1.fit(X_tr_full_ext, y_train.values, categorical_feature=cat_full)
fm2 = make_lgbm_2()
fm2.fit(X_tr_full_ext, y_train.values, categorical_feature=cat_full)
fmxgb = make_xgb()
fmxgb.fit(X_tr_xgb_full.values, y_train.values)

test_preds_raw = (fm1.predict(X_te_ext) + fm2.predict(X_te_ext) + fmxgb.predict(X_te_xgb.values)) / 3.0
test_preds_poly2 = smooth_by_age_poly(test_preds_raw, X_test, degree=2)
test_preds_p2iso = smooth_poly2_then_isotonic(test_preds_raw, X_test)

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

sub = X_test[["Index"]].copy()
sub["target"] = test_preds_p2iso  # best smoother
sub.to_csv(submissions_dir / "10_ensemble3_bateman.csv", index=False)
print(f"Submission: 10_ensemble3_bateman.csv  ({len(sub)} rows)")
sub_p2 = X_test[["Index"]].copy()
sub_p2["target"] = test_preds_poly2
sub_p2.to_csv(submissions_dir / "10_ensemble3_bateman_poly2.csv", index=False)
print(f"Submission: 10_ensemble3_bateman_poly2.csv")
print(sub.head())

# %% [markdown]
# ## Push to skore hub

# %%
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class EnsemblePipelineV4Bateman(BaseEstimator, RegressorMixin):
    """Exp-11: 3-model ensemble (2×LightGBM + XGBoost) with Bateman PK features."""

    def __init__(self):
        pass

    def _encode_cat_for_xgb(self, X_ext, cat_dtypes):
        X_xgb = X_ext.copy()
        for col, dtype in cat_dtypes.items():
            X_xgb[col] = X_xgb[col].cat.codes.astype("float32")
        return X_xgb

    def fit(self, X, y, sample_weight=None):
        X_fe, _ = build_features(X, y, X_val=None)
        X_ext, saved_stats = add_extra_features_v4(X_fe, X, y_train=y)
        self._X_fit = X
        self._y_fit = y
        self._saved_stats = saved_stats
        X_ext = X_ext[ALL_COLS]
        cat_cols = [c for c in ALL_COLS if str(X_ext[c].dtype) == "category"]
        self._cat_dtypes = {c: X_ext[c].dtype for c in cat_cols}
        self._cat_codes = {c: X_ext[c].cat.categories for c in cat_cols}
        y_arr = y.values if hasattr(y, "values") else y
        self.m1_ = make_lgbm_1()
        self.m1_.fit(X_ext, y_arr, categorical_feature=cat_cols)
        self.m2_ = make_lgbm_2()
        self.m2_.fit(X_ext, y_arr, categorical_feature=cat_cols)
        X_xgb = self._encode_cat_for_xgb(X_ext, self._cat_dtypes)
        self.mxgb_ = make_xgb()
        self.mxgb_.fit(X_xgb.values, y_arr)
        self.feature_names_in_ = ALL_COLS
        self.n_features_in_ = len(ALL_COLS)
        return self

    def predict(self, X):
        check_is_fitted(self, "m1_")
        _, X_fe = build_features(self._X_fit, self._y_fit, X_val=X)
        X_ext, _ = add_extra_features_v4(X_fe, X, saved_stats=self._saved_stats)
        X_ext = X_ext[ALL_COLS]
        for col, dtype in self._cat_dtypes.items():
            X_ext[col] = X_ext[col].astype(dtype)
        p1 = self.m1_.predict(X_ext)
        p2 = self.m2_.predict(X_ext)
        X_xgb = X_ext.copy()
        for col, cats in self._cat_codes.items():
            cm = {v: i for i, v in enumerate(cats)}
            X_xgb[col] = X_ext[col].map(lambda x, cm=cm: cm.get(x, -1)).astype("float32")
        p_xgb = self.mxgb_.predict(X_xgb.values)
        return (p1 + p2 + p_xgb) / 3.0


report = skore.CrossValidationReport(
    EnsemblePipelineV4Bateman(),
    X=X_train, y=y_train, splitter=folds, n_jobs=1,
)
project.put("10_ensemble3_bateman", report)
print("Report saved: 10_ensemble3_bateman")
print(report.metrics.rmse())
