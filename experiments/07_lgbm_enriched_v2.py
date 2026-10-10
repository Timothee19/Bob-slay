# %% [markdown]
# # Experiment 07 — Monotone post-processing + rater aggregates + quadratic patient fit
#
# **Date:** 2025-01-07
# **Base:** Exp 06 (RMSE 3.532 poly2 / 3.706 raw)
#
# ## New ideas this iteration
#
# ### A. Isotonic / monotone post-processing
# Parkinson's disease only progresses — the true OFF score cannot decrease over
# time for a patient. After poly2 smoothing, we apply isotonic regression
# (monotone non-decreasing) to the per-patient prediction sequence sorted by age.
# This replaces the quadratic constraint with a piecewise-constant monotone one.
#
# We compare three post-processing strategies:
# 1. poly2 (best from exp 05/06)
# 2. isotonic (new: PAVA on sorted predictions)
# 3. poly2 then isotonic (two-stage)
#
# ### B. Per-rater target statistics (fold-aware, no leak)
# Beyond the rater_bias_feature (median ratio), add per-rater aggregates of the
# target from the training fold: mean target per rater, std, and count.
# These capture absolute severity differences across clinical cohorts.
#
# ### C. Quadratic patient trajectory features
# The linear `patient_slope` / `patient_intercept` captures only the first-order
# rate of progression. Add:
# - `patient_slope2`: coefficient of age^2 in a quadratic fit to off_corrected.
#   Positive = accelerating; negative = decelerating (plateau). For patients
#   with < 4 visits, fallback to 0.
#
# ### D. More lag features
# - `off_residual_prev`: off_residual_from_trend at previous visit. Captures
#   whether the patient was above/below their own trend last time.
# - `on_time_corrected_next`: PK-corrected ON at the next visit.
# - `ledd_prev`, `ledd_next`: LEDD at adjacent visits — captures dose changes.
# - `off_lag_delta2`: off_corrected_prev2 − off_corrected_prev (acceleration).
#
# ### E. Per-patient LEDD aggregates
# - `patient_ledd_mean`, `patient_ledd_max`: dose level aggregates.
# - `ledd_zscore`: (ledd − patient_ledd_mean) / patient_ledd_std. Encodes
#   whether the current visit is a high-dose or low-dose visit relative to
#   the patient's own history.
#
# ### F. Same LightGBM + subsampling for regularisation
# - `subsample=0.85`, `colsample_bytree=0.85`: reduce variance through
#   random feature/row subsampling per tree.
# - Keep 2000 trees, lr=0.01.

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

# %% [markdown]
# ## Project connection

# %%
cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)

# %% [markdown]
# ## Data

# %%
DATA_DIR = PROJECT_ROOT / "data"
X_train, y_train = load_dataset(DATA_DIR)
print(f"Train: {X_train.shape}")

folds = get_cv_folds(X_train, y_train)

# %% [markdown]
# ## Extra feature engineering (exp 06 features + new ones)

# %%
def add_extra_features_v2(
    X_fe: pd.DataFrame,
    X_raw: pd.DataFrame,
    y_train: pd.Series | None = None,
    rater_stats: dict | None = None,
) -> pd.DataFrame:
    """Add enriched features on top of the 36-feature set.

    Parameters
    ----------
    X_fe : pd.DataFrame
        Output of build_features.
    X_raw : pd.DataFrame
        Raw X DataFrame for patient_id and rater_id.
    y_train : pd.Series or None
        Training-fold target. If provided, rater_stats is computed from it.
        If None, rater_stats must be provided for validation/test.
    rater_stats : dict or None
        Pre-computed per-rater target statistics dict. Required when y_train
        is None (validation/test split).

    Returns
    -------
    pd.DataFrame
        Extended feature matrix (no patient_id column).
    """
    df = X_fe.copy()
    df["patient_id"] = X_raw["patient_id"].values
    df["rater_id"] = X_raw["rater_id"].values

    # ------------------------------------------------------------------ #
    # All exp-06 features
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
    # New exp-07 features
    # ------------------------------------------------------------------ #

    # C. Quadratic patient trajectory
    slope2_map = {}
    for pid, grp_p in df.groupby("patient_id"):
        valid = grp_p[["off_corrected", "age"]].dropna()
        if len(valid) >= 4:
            coeffs = np.polyfit(valid["age"].values, valid["off_corrected"].values, 2)
            slope2_map[pid] = float(coeffs[0])  # quadratic term
        else:
            slope2_map[pid] = np.nan
    median_slope2 = float(np.nanmedian(list(slope2_map.values())))
    for pid in slope2_map:
        if np.isnan(slope2_map[pid]):
            slope2_map[pid] = median_slope2
    df["patient_slope2"] = df["patient_id"].map(slope2_map)

    # D. More lag features
    df_sorted3 = df.sort_values(["patient_id", "age"])
    grp3 = df_sorted3.groupby("patient_id")

    df_sorted3["off_residual_prev"] = grp3["off_residual_from_trend"].shift(1)
    df_sorted3["on_time_corrected_next"] = grp3["on_time_corrected"].shift(-1)
    df_sorted3["ledd_prev"] = grp3["ledd"].shift(1)
    df_sorted3["ledd_next"] = grp3["ledd"].shift(-1)
    # acceleration: prev2 → prev → current slope
    prev_delta = df_sorted3["off_corrected_prev"] - df_sorted3["off_corrected_prev2"]
    curr_delta = df_sorted3["off_corrected"] - df_sorted3["off_corrected_prev"]
    df_sorted3["off_lag_accel"] = curr_delta - prev_delta

    df_sorted3 = df_sorted3.loc[df.index]
    for col in ["off_residual_prev", "on_time_corrected_next",
                "ledd_prev", "ledd_next", "off_lag_accel"]:
        df[col] = df_sorted3[col].values

    # E. Per-patient LEDD aggregates
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

    # B. Per-rater target statistics (fold-aware)
    if y_train is not None:
        # Compute from training fold
        rater_mean_map = {}
        rater_std_map = {}
        rater_count_map = {}
        df_with_y = df.copy()
        df_with_y["_target"] = y_train.values
        for rid, grp_r in df_with_y.groupby("rater_id"):
            vals = grp_r["_target"].dropna()
            rater_mean_map[rid] = float(vals.mean()) if len(vals) > 0 else np.nan
            rater_std_map[rid] = float(vals.std()) if len(vals) > 1 else 0.0
            rater_count_map[rid] = len(vals)
        _global_mean = float(np.nanmean(list(rater_mean_map.values())))
        _global_std = float(np.nanmean(list(rater_std_map.values())))
        # Store for validation use
        rater_stats_out = {
            "mean": rater_mean_map,
            "std": rater_std_map,
            "count": rater_count_map,
            "global_mean": _global_mean,
            "global_std": _global_std,
        }
    else:
        rater_stats_out = None

    if rater_stats is not None:
        df["rater_target_mean"] = df["rater_id"].map(rater_stats["mean"]).fillna(
            rater_stats["global_mean"]
        )
        df["rater_target_std"] = df["rater_id"].map(rater_stats["std"]).fillna(
            rater_stats["global_std"]
        )
        df["rater_target_count"] = df["rater_id"].map(rater_stats["count"]).fillna(0)
    else:
        # Training fold: compute self-statistics (not using target as feature,
        # using them as metadata about the rater's patient cohort)
        df["rater_target_mean"] = np.nan
        df["rater_target_std"] = np.nan
        df["rater_target_count"] = np.nan

    df = df.drop(columns=["patient_id", "rater_id"])
    return df, rater_stats_out


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

ALL_COLS = FEATURE_COLS + EXP06_COLS + EXP07_COLS
print(f"Feature count: {len(ALL_COLS)} ({len(FEATURE_COLS)} base + "
      f"{len(EXP06_COLS)} exp06 + {len(EXP07_COLS)} exp07)")

# %% [markdown]
# ## Isotonic post-processing

# %%
def smooth_isotonic(predictions: np.ndarray, X_val: pd.DataFrame) -> np.ndarray:
    """Enforce per-patient monotone non-decreasing predictions (by age).

    Since Parkinson's disease only progresses, the true OFF score should be
    non-decreasing over time. IsotonicRegression (PAVA) finds the closest
    monotone non-decreasing sequence in L2 sense.

    Parameters
    ----------
    predictions : np.ndarray
        Raw predictions aligned with X_val.
    X_val : pd.DataFrame
        Must contain patient_id and age.

    Returns
    -------
    np.ndarray
        Monotone predictions in original row order.
    """
    preds = predictions.copy().astype(float)
    df = X_val[["patient_id", "age"]].copy()
    df["_pred"] = preds
    df["_orig_idx"] = np.arange(len(df))
    ir = IsotonicRegression(increasing=True)

    for pid, grp in df.groupby("patient_id"):
        if len(grp) < 2:
            continue
        grp_sorted = grp.sort_values("age")
        ages = grp_sorted["age"].values
        p = grp_sorted["_pred"].values
        fitted = ir.fit_transform(ages, p)
        df.loc[grp_sorted.index, "_pred"] = fitted

    return df["_pred"].values


def smooth_poly2_then_isotonic(
    predictions: np.ndarray, X_val: pd.DataFrame
) -> np.ndarray:
    """Apply poly2 smoothing then isotonic constraint."""
    p1 = smooth_by_age_poly(predictions, X_val, degree=2)
    return smooth_isotonic(p1, X_val)

# %% [markdown]
# ## Cross-validation loop

# %%
try:
    from lightgbm import LGBMRegressor
except ImportError as e:
    raise ImportError("lightgbm not installed. Run: pip install lightgbm") from e

all_y_true = []
all_y_pred_raw = []
fold_rmses_raw = []
fold_data = []   # (X_va_raw, y_va, preds_raw)

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    X_tr_raw = X_train.iloc[tr_idx].reset_index(drop=True)
    y_tr     = y_train.iloc[tr_idx].reset_index(drop=True)
    X_va_raw = X_train.iloc[va_idx].reset_index(drop=True)
    y_va     = y_train.iloc[va_idx].reset_index(drop=True)

    X_tr_fe, X_va_fe = build_features(X_tr_raw, y_tr, X_val=X_va_raw)

    # Enrich: pass y_tr for rater stats on train, then apply to val
    X_tr_fe_ext, rater_stats = add_extra_features_v2(X_tr_fe, X_tr_raw, y_train=y_tr)
    # Validation: apply train rater stats
    X_va_fe_ext, _ = add_extra_features_v2(X_va_fe, X_va_raw,
                                            y_train=None, rater_stats=rater_stats)

    X_tr_fe_ext = X_tr_fe_ext[ALL_COLS]
    X_va_fe_ext = X_va_fe_ext[ALL_COLS]

    cat_cols = [c for c in ALL_COLS if str(X_tr_fe_ext[c].dtype) == "category"]

    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        min_child_samples=40,
        reg_lambda=2.0,
        num_leaves=127,
        subsample=0.85,
        colsample_bytree=0.85,
        subsample_freq=1,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_tr_fe_ext, y_tr.values, categorical_feature=cat_cols)
    preds = model.predict(X_va_fe_ext)

    rmse = root_mean_squared_error(y_va.values, preds)
    fold_rmses_raw.append(rmse)
    all_y_true.append(y_va.values)
    all_y_pred_raw.append(preds)
    fold_data.append((X_va_raw, y_va, preds))
    print(f"  fold {fold_idx}: RMSE (raw) = {rmse:.4f}")

overall_rmse_raw = root_mean_squared_error(
    np.concatenate(all_y_true), np.concatenate(all_y_pred_raw)
)
print(f"\nRaw — Overall RMSE: {overall_rmse_raw:.4f}  "
      f"| mean: {np.mean(fold_rmses_raw):.4f} +/- {np.std(fold_rmses_raw):.4f}")

# %% [markdown]
# ## Smoothing comparison

# %%
def eval_smooth(fold_data, all_y_true, fn, label):
    smoothed = [fn(p, X_va_raw) for X_va_raw, _, p in fold_data]
    overall = root_mean_squared_error(np.concatenate(all_y_true), np.concatenate(smoothed))
    per_fold = [root_mean_squared_error(y_va.values, s) for (_, y_va, _), s in zip(fold_data, smoothed)]
    print(f"  {label:35s}: {overall:.4f}  | {[f'{r:.4f}' for r in per_fold]}")
    return smoothed, overall

print("\nSmoothing comparison:")
_, rmse_poly2 = eval_smooth(fold_data, all_y_true,
    lambda p, X: smooth_by_age_poly(p, X, degree=2), "Poly2(age)")
_, rmse_iso = eval_smooth(fold_data, all_y_true,
    lambda p, X: smooth_isotonic(p, X), "Isotonic")
smoothed_best, rmse_poly2_iso = eval_smooth(fold_data, all_y_true,
    lambda p, X: smooth_poly2_then_isotonic(p, X), "Poly2 + Isotonic")

print(f"\nDelta vs exp06 poly2 (3.532):")
print(f"  Poly2:          {rmse_poly2 - 3.532:+.4f}")
print(f"  Isotonic:       {rmse_iso - 3.532:+.4f}")
print(f"  Poly2+Isotonic: {rmse_poly2_iso - 3.532:+.4f}")

# Pick best
best_rmse = min(rmse_poly2, rmse_iso, rmse_poly2_iso)
if best_rmse == rmse_poly2:
    best_label = "poly2"
    best_smooth_fn = lambda p, X: smooth_by_age_poly(p, X, degree=2)
elif best_rmse == rmse_iso:
    best_label = "isotonic"
    best_smooth_fn = lambda p, X: smooth_isotonic(p, X)
else:
    best_label = "poly2+isotonic"
    best_smooth_fn = lambda p, X: smooth_poly2_then_isotonic(p, X)
print(f"\nBest smoothing: {best_label} (RMSE={best_rmse:.4f})")

# %% [markdown]
# ## Feature importance (top 25)

# %%
X_tr_raw_f0 = X_train.iloc[folds[0][0]].reset_index(drop=True)
y_tr_f0     = y_train.iloc[folds[0][0]].reset_index(drop=True)
X_tr_fe_f0, X_va_fe_f0 = build_features(X_tr_raw_f0, y_tr_f0)
X_tr_fe_f0_ext, _ = add_extra_features_v2(X_tr_fe_f0, X_tr_raw_f0, y_train=y_tr_f0)
X_tr_fe_f0_ext = X_tr_fe_f0_ext[ALL_COLS]
cat_cols_f0 = [c for c in ALL_COLS if str(X_tr_fe_f0_ext[c].dtype) == "category"]

model_f0 = LGBMRegressor(
    n_estimators=2000, learning_rate=0.01, min_child_samples=40,
    reg_lambda=2.0, num_leaves=127, subsample=0.85, colsample_bytree=0.85,
    subsample_freq=1, n_jobs=-1, random_state=42, verbose=-1,
)
model_f0.fit(X_tr_fe_f0_ext, y_tr_f0.values, categorical_feature=cat_cols_f0)
importance = pd.Series(model_f0.feature_importances_, index=ALL_COLS)
print("\nTop 25 features by importance:")
print(importance.sort_values(ascending=False).head(25).to_string())

# %% [markdown]
# ## Refit on full train + generate submission

# %%
X_tr_full_fe, _ = build_features(X_train, y_train)
X_tr_full_ext, rater_stats_full = add_extra_features_v2(
    X_tr_full_fe, X_train, y_train=y_train
)
X_tr_full_ext = X_tr_full_ext[ALL_COLS]
cat_cols_full = [c for c in ALL_COLS if str(X_tr_full_ext[c].dtype) == "category"]

X_test = load_test_dataset(DATA_DIR)
_, X_te_fe = build_features(X_train, y_train, X_val=X_test)
X_te_ext, _ = add_extra_features_v2(
    X_te_fe, X_test, y_train=None, rater_stats=rater_stats_full
)
X_te_ext = X_te_ext[ALL_COLS]
# Align categoricals
for col in cat_cols_full:
    X_te_ext[col] = X_te_ext[col].astype(X_tr_full_ext[col].dtype)

final_model = LGBMRegressor(
    n_estimators=2000, learning_rate=0.01, min_child_samples=40,
    reg_lambda=2.0, num_leaves=127, subsample=0.85, colsample_bytree=0.85,
    subsample_freq=1, n_jobs=-1, random_state=42, verbose=-1,
)
final_model.fit(X_tr_full_ext, y_train.values, categorical_feature=cat_cols_full)

test_preds_raw = final_model.predict(X_te_ext)
test_preds_best = best_smooth_fn(test_preds_raw, X_test)

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
sub = X_test[["Index"]].copy()
sub["target"] = test_preds_best
fname = f"07_lgbm_enrichedv2_{best_label}.csv"
sub.to_csv(submissions_dir / fname, index=False)
print(f"Submission written: {len(sub)} rows -> {fname}")

# Also save poly2 version for comparison
test_preds_poly2 = smooth_by_age_poly(test_preds_raw, X_test, degree=2)
sub_p2 = X_test[["Index"]].copy()
sub_p2["target"] = test_preds_poly2
sub_p2.to_csv(submissions_dir / "07_lgbm_enrichedv2_poly2.csv", index=False)
print("Also saved: 07_lgbm_enrichedv2_poly2.csv")
print(sub.head())

# %% [markdown]
# ## Push to skore hub

# %%
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

class EnrichedPipelineV2(BaseEstimator, RegressorMixin):
    """Experiment-07 pipeline with rater stats and extended features."""

    def __init__(self, n_estimators=2000, learning_rate=0.01,
                 min_child_samples=40, reg_lambda=2.0, num_leaves=127,
                 subsample=0.85, colsample_bytree=0.85):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.reg_lambda = reg_lambda
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

    def fit(self, X, y, sample_weight=None):
        X_fe, _ = build_features(X, y, X_val=None)
        X_fe_ext, rater_stats = add_extra_features_v2(X_fe, X, y_train=y)
        self._X_fit = X
        self._y_fit = y
        self._rater_stats = rater_stats
        X_fe_ext = X_fe_ext[ALL_COLS]
        cat_cols = [c for c in ALL_COLS if str(X_fe_ext[c].dtype) == "category"]
        self._cat_dtypes = {c: X_fe_ext[c].dtype for c in cat_cols}
        self.model_ = LGBMRegressor(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples, reg_lambda=self.reg_lambda,
            num_leaves=self.num_leaves, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree, subsample_freq=1,
            n_jobs=-1, random_state=42, verbose=-1,
        )
        self.model_.fit(X_fe_ext, y.values if hasattr(y, "values") else y,
                        categorical_feature=cat_cols)
        self.feature_names_in_ = ALL_COLS
        self.n_features_in_ = len(ALL_COLS)
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        _, X_fe = build_features(self._X_fit, self._y_fit, X_val=X)
        X_fe_ext, _ = add_extra_features_v2(X_fe, X, y_train=None,
                                             rater_stats=self._rater_stats)
        X_fe_ext = X_fe_ext[ALL_COLS]
        for col, dtype in self._cat_dtypes.items():
            X_fe_ext[col] = X_fe_ext[col].astype(dtype)
        return self.model_.predict(X_fe_ext)


report = skore.CrossValidationReport(
    EnrichedPipelineV2(),
    X=X_train, y=y_train, splitter=folds, n_jobs=1,
)
project.put("07_lgbm_enriched_v2", report)
print("Report saved: 07_lgbm_enriched_v2")
print(report.metrics.rmse())
