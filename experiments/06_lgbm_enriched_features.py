# %% [markdown]
# # Experiment 06 — Enriched features: deeper lags, inter-visit gap, on aggregates,
# #                   LEDD–OFF ratio, and a stronger LightGBM (lr=0.01, 2000 trees)
#
# **Date:** 2025-01-07
# **Base:** Exp 05 (RMSE 3.739 with poly2 smoothing)
#
# ## New ideas this iteration
#
# ### A. Deeper lag features
# - ``off_corrected_prev2``: off_corrected two visits back. Provides a second
#   anchor point for the trajectory; helps when the immediately preceding visit
#   off is missing.
# - ``off_lag_delta``: off_corrected − off_corrected_prev (signed velocity of
#   symptom change). Captures acceleration in disease progression.
# - ``off_lag_delta_per_year``: delta normalised by the inter-visit time gap.
#   Removes confounding by visit frequency.
#
# ### B. Inter-visit time gap
# - ``inter_visit_gap``: time in years between the current and previous visit
#   (from age difference). NaN for first visit. Useful alongside lag features.
#
# ### C. Richer ON aggregates
# - ``patient_on_max``: per-patient max of on_corrected. Upper bound on
#   medication benefit, correlates with overall disease severity.
# - ``patient_on_q75``: 75th percentile of on_corrected.
# - ``patient_on_std``: variability in ON response.
# - ``patient_on_slope``: slope of on_corrected vs age.
#
# ### D. LEDD–OFF ratio
# - ``ledd_off_ratio``: ledd / off_corrected. When ledd increases but OFF
#   worsens anyway, this ratio captures treatment resistance. Correlated with
#   disease stage.
# - ``ledd_per_off_trend``: ledd / patient_trend_at_age (normalised by expected).
#
# ### E. Patient-level off_corrected variability
# - ``patient_off_iqr``: IQR (q75 - q25) of off_corrected per patient. Measures
#   intra-patient measurement variability / fluctuation severity.
#
# ### F. Stronger LightGBM
# - Lower learning rate (0.01) + more trees (2000) for better convergence.
# - Keep same regularisation (min_child_samples=40, reg_lambda=2).
#
# ## Expected improvements
# Deeper lags and inter-visit gaps give the model more temporal context.
# ON aggregates help for visits where OFF is missing (model relies on ON).
# The LEDD ratio encodes treatment resistance, a strong prognostic signal.

# %%
import numpy as np
import pandas as pd
import skore
from sklearn.metrics import root_mean_squared_error

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
print(f"Target: mean={y_train.mean():.2f}, std={y_train.std():.2f}")

# %% [markdown]
# ## Extra feature engineering on top of the existing pipeline
#
# We inject additional features AFTER build_features() runs.
# This keeps the existing fold-aware code intact and adds stateless
# (or same-fold-aware) features safely.

# %%
def add_extra_features(X_fe: pd.DataFrame, X_raw: pd.DataFrame) -> pd.DataFrame:
    """Add enriched features on top of the existing 36-feature set.

    All features here are either:
    - Fully stateless (computed from X_fe columns, no target access).
    - Within-patient aggregates that respect GroupKFold (same fold = same patient).

    Parameters
    ----------
    X_fe : pd.DataFrame
        Output of build_features (already has FEATURE_COLS + patient_id).
    X_raw : pd.DataFrame
        Raw X DataFrame aligned with X_fe (for patient_id lookup if needed).

    Returns
    -------
    pd.DataFrame
        Extended feature matrix.
    """
    df = X_fe.copy()
    # patient_id is needed for group aggregates — bring it in from raw
    df["patient_id"] = X_raw["patient_id"].values

    # ------------------------------------------------------------------ #
    # A. Deeper lag features (sort by patient + age, within-split)
    # ------------------------------------------------------------------ #
    # off_corrected_prev2: two visits back
    df_sorted = df.sort_values(["patient_id", "age"])
    grp = df_sorted.groupby("patient_id")

    df_sorted["off_corrected_prev2"] = grp["off_corrected"].shift(2)
    df_sorted["on_corrected_prev2"]  = grp["on_corrected"].shift(2)
    # off_lag_delta: current minus prev (can be NaN when prev is missing)
    df_sorted["off_lag_delta"] = (
        df_sorted["off_corrected"] - df_sorted["off_corrected_prev"]
    )
    # inter-visit gap: age difference to previous visit
    df_sorted["inter_visit_gap"] = (
        df_sorted["age"] - grp["age"].shift(1)
    ).clip(lower=0)
    # delta per year of gap (velocity of change)
    gap_safe = df_sorted["inter_visit_gap"].replace(0, np.nan)
    df_sorted["off_lag_delta_per_year"] = df_sorted["off_lag_delta"] / gap_safe

    # off_corrected_next2 (two visits ahead)
    df_sorted["off_corrected_next2"] = grp["off_corrected"].shift(-2)

    # Restore original order
    df_sorted = df_sorted.loc[df.index]
    for col in [
        "off_corrected_prev2", "on_corrected_prev2",
        "off_lag_delta", "inter_visit_gap",
        "off_lag_delta_per_year", "off_corrected_next2",
    ]:
        df[col] = df_sorted[col].values

    # ------------------------------------------------------------------ #
    # B. Richer ON aggregates
    # ------------------------------------------------------------------ #
    df["patient_on_max"] = df["patient_id"].map(
        df.groupby("patient_id")["on_corrected"].max()
    )
    df["patient_on_q75"] = df["patient_id"].map(
        df.groupby("patient_id")["on_corrected"].quantile(0.75)
    )
    df["patient_on_std"] = df["patient_id"].map(
        df.groupby("patient_id")["on_corrected"].std().fillna(0.0)
    )

    # ON slope vs age (captures how ON response deteriorates)
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

    # ------------------------------------------------------------------ #
    # C. LEDD–OFF ratio and LEDD-per-trend
    # ------------------------------------------------------------------ #
    off_safe = df["off_corrected"].replace(0, np.nan)
    df["ledd_off_ratio"] = df["ledd"] / off_safe

    trend_safe = df["patient_trend_at_age"].replace(0, np.nan)
    df["ledd_per_off_trend"] = df["ledd"] / trend_safe

    # ------------------------------------------------------------------ #
    # D. Patient-level off_corrected IQR
    # ------------------------------------------------------------------ #
    df["patient_off_iqr"] = df["patient_id"].map(
        df.groupby("patient_id")["off_corrected"].apply(
            lambda x: float(np.nanpercentile(x.dropna().values, 75) -
                            np.nanpercentile(x.dropna().values, 25))
            if x.notna().sum() >= 2 else 0.0
        )
    )

    # ------------------------------------------------------------------ #
    # E. Running statistics: off_cummin (disease best state seen so far)
    # ------------------------------------------------------------------ #
    df_sorted2 = df.sort_values(["patient_id", "age"])
    grp2 = df_sorted2.groupby("patient_id")
    df_sorted2["off_cummin"] = grp2["off_corrected"].cummin()
    df_sorted2 = df_sorted2.loc[df.index]
    df["off_cummin"] = df_sorted2["off_cummin"].values

    # ------------------------------------------------------------------ #
    # F. on_time_corrected lags
    # ------------------------------------------------------------------ #
    df_sorted3 = df.sort_values(["patient_id", "age"])
    grp3 = df_sorted3.groupby("patient_id")
    df_sorted3["on_time_corrected_prev"] = grp3["on_time_corrected"].shift(1)
    df_sorted3 = df_sorted3.loc[df.index]
    df["on_time_corrected_prev"] = df_sorted3["on_time_corrected_prev"].values

    # Drop patient_id (was injected locally)
    df = df.drop(columns=["patient_id"])

    return df


# %%
# Extra feature column names (in addition to FEATURE_COLS)
EXTRA_COLS = [
    "off_corrected_prev2",
    "on_corrected_prev2",
    "off_lag_delta",
    "inter_visit_gap",
    "off_lag_delta_per_year",
    "off_corrected_next2",
    "patient_on_max",
    "patient_on_q75",
    "patient_on_std",
    "patient_on_slope",
    "ledd_off_ratio",
    "ledd_per_off_trend",
    "patient_off_iqr",
    "off_cummin",
    "on_time_corrected_prev",
]

ALL_COLS = FEATURE_COLS + EXTRA_COLS

print(f"Feature count: {len(ALL_COLS)} ({len(FEATURE_COLS)} base + {len(EXTRA_COLS)} new)")

# %% [markdown]
# ## Fixed folds

# %%
folds = get_cv_folds(X_train, y_train)

# %% [markdown]
# ## Cross-validation loop with enriched features

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

    # Standard fold-aware feature engineering
    X_tr_fe, X_va_fe = build_features(X_tr_raw, y_tr, X_val=X_va_raw)

    # Enrich with extra features (fold-aware within same split)
    X_tr_fe = add_extra_features(X_tr_fe, X_tr_raw)
    X_va_fe = add_extra_features(X_va_fe, X_va_raw)

    X_tr_fe = X_tr_fe[ALL_COLS]
    X_va_fe = X_va_fe[ALL_COLS]

    # Align categorical dtypes for extra non-categorical features
    cat_cols = [c for c in ALL_COLS if str(X_tr_fe[c].dtype) == "category"]

    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        min_child_samples=40,
        reg_lambda=2.0,
        num_leaves=127,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_tr_fe, y_tr.values, categorical_feature=cat_cols)
    preds = model.predict(X_va_fe)

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
# ## Poly2 smoothing

# %%
smoothed_preds = [smooth_by_age_poly(p, X_va_raw, degree=2)
                  for X_va_raw, _, p in fold_data]
overall_rmse_poly2 = root_mean_squared_error(
    np.concatenate(all_y_true), np.concatenate(smoothed_preds)
)
fold_rmses_poly2 = [
    root_mean_squared_error(y_va.values, s)
    for (_, y_va, _), s in zip(fold_data, smoothed_preds)
]
print(f"Poly2 — Overall RMSE: {overall_rmse_poly2:.4f}  "
      f"| mean: {np.mean(fold_rmses_poly2):.4f} +/- {np.std(fold_rmses_poly2):.4f}")
print(f"Delta vs exp05 raw: {overall_rmse_raw - 3.939:+.4f}")
print(f"Delta vs exp05 poly2: {overall_rmse_poly2 - 3.739:+.4f}")

# %% [markdown]
# ## Feature importance (top 20)

# %%
# Refit on fold 0 for quick importance check
X_tr_raw_f0 = X_train.iloc[folds[0][0]].reset_index(drop=True)
y_tr_f0     = y_train.iloc[folds[0][0]].reset_index(drop=True)
X_tr_fe_f0, _ = build_features(X_tr_raw_f0, y_tr_f0)
X_tr_fe_f0 = add_extra_features(X_tr_fe_f0, X_tr_raw_f0)[ALL_COLS]
cat_cols_f0 = [c for c in ALL_COLS if str(X_tr_fe_f0[c].dtype) == "category"]

model_f0 = LGBMRegressor(
    n_estimators=2000, learning_rate=0.01, min_child_samples=40,
    reg_lambda=2.0, num_leaves=127, n_jobs=-1, random_state=42, verbose=-1,
)
model_f0.fit(X_tr_fe_f0, y_tr_f0.values, categorical_feature=cat_cols_f0)

importance = pd.Series(model_f0.feature_importances_, index=ALL_COLS)
print("\nTop 20 features by importance:")
print(importance.sort_values(ascending=False).head(20).to_string())

# %% [markdown]
# ## Refit on full train + generate submission

# %%
X_tr_full_fe, _ = build_features(X_train, y_train)
X_tr_full_fe = add_extra_features(X_tr_full_fe, X_train)[ALL_COLS]
cat_cols_full = [c for c in ALL_COLS if str(X_tr_full_fe[c].dtype) == "category"]

X_test = load_test_dataset(DATA_DIR)
_, X_te_fe = build_features(X_train, y_train, X_val=X_test)
X_te_fe = add_extra_features(X_te_fe, X_test)[ALL_COLS]
# Align categorical dtypes
for col in cat_cols_full:
    X_te_fe[col] = X_te_fe[col].astype(X_tr_full_fe[col].dtype)

final_model = LGBMRegressor(
    n_estimators=2000, learning_rate=0.01, min_child_samples=40,
    reg_lambda=2.0, num_leaves=127, n_jobs=-1, random_state=42, verbose=-1,
)
final_model.fit(X_tr_full_fe, y_train.values, categorical_feature=cat_cols_full)

test_preds_raw   = final_model.predict(X_te_fe)
test_preds_poly2 = smooth_by_age_poly(test_preds_raw, X_test, degree=2)

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
sub = X_test[["Index"]].copy()
sub["target"] = test_preds_poly2
sub.to_csv(submissions_dir / "06_lgbm_enriched_poly2.csv", index=False)
print(f"Submission written: {len(sub)} rows -> 06_lgbm_enriched_poly2.csv")
print(sub.head())

# %% [markdown]
# ## Push to skore hub
#
# Note: skore CrossValidationReport expects a sklearn-compatible estimator.
# We wrap everything in a custom pipeline.

# %%
from sklearn.base import BaseEstimator, RegressorMixin
from parkinsons.pipeline import ParkinsonsPipeline

class EnrichedPipeline(BaseEstimator, RegressorMixin):
    """ParkinsonsPipeline extended with extra features for skore reporting."""

    def __init__(self, n_estimators=2000, learning_rate=0.01,
                 min_child_samples=40, reg_lambda=2.0, num_leaves=127):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.reg_lambda = reg_lambda
        self.num_leaves = num_leaves

    def fit(self, X, y, sample_weight=None):
        X_fe, _ = build_features(X, y, X_val=None)
        X_fe = add_extra_features(X_fe, X)[ALL_COLS]
        self._X_fit = X
        self._y_fit = y
        cat_cols = [c for c in ALL_COLS if str(X_fe[c].dtype) == "category"]
        self._cat_dtypes = {c: X_fe[c].dtype for c in cat_cols}
        self.model_ = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            reg_lambda=self.reg_lambda,
            num_leaves=self.num_leaves,
            n_jobs=-1, random_state=42, verbose=-1,
        )
        self.model_.fit(X_fe, y.values if hasattr(y, "values") else y,
                        categorical_feature=cat_cols)
        self.feature_names_in_ = ALL_COLS
        self.n_features_in_ = len(ALL_COLS)
        return self

    def predict(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "model_")
        _, X_fe = build_features(self._X_fit, self._y_fit, X_val=X)
        X_fe = add_extra_features(X_fe, X)[ALL_COLS]
        # Align categorical dtypes to match training
        for col, dtype in self._cat_dtypes.items():
            X_fe[col] = X_fe[col].astype(dtype)
        return self.model_.predict(X_fe)


report = skore.CrossValidationReport(
    EnrichedPipeline(),
    X=X_train, y=y_train, splitter=folds, n_jobs=1,
)
project.put("06_lgbm_enriched_features", report)
print("Report saved: 06_lgbm_enriched_features")
print(report.metrics.rmse())
