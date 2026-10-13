# %% [markdown]
# # Experiment 16 — SOTA + HalvingSearch anti-overfitting + Poly2+Isotonic post-processing
#
# **Date:** 2025-01-10
# **Base:** Exp 15 (full feature set, 76 cols after correlation pruning)
#
# ## What this experiment does
#
# 1. Full feature engineering (lags, Bateman PK, longitudinal aggregates).
# 2. NoisyRaterEncoder: target-encodes rater_id with Gaussian noise for leak-safe smoothing.
# 3. Pipeline: NoisyRaterEncoder → TableVectorizer → HistGradientBoostingRegressor.
# 4. HalvingRandomSearchCV with anti-overfitting grid (60 candidates, resource=max_iter).
# 5. skore.evaluate with GroupKFold(5) + project.put.
# 6. Full-train refit → raw test predictions → per-patient poly2 + isotonic post-processing.

# %%
import numpy as np
import pandas as pd
import skore
import skrub
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, HalvingRandomSearchCV
from sklearn.pipeline import make_pipeline

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.features import build_features, FEATURE_COLS
from parkinsons.hub import load_skore_credentials

# ── Login ──────────────────────────────────────────────────────────────────────
cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)

DATA_DIR = PROJECT_ROOT / "data"

# %%
# ── Feature column groups (identical to exp 14 / 15) ─────────────────────────
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


# %%
# ── Bateman PK biexponential helper ───────────────────────────────────────────
def compute_bateman_concentration(
    ledd: pd.Series,
    time_since_intake: pd.Series,
    ka: float = 1.5,
    ke: float = 0.46,
) -> pd.Series:
    """Plasma levodopa concentration via the Bateman biexponential equation."""
    concentration = (ledd * ka / (ka - ke)) * (
        np.exp(-ke * time_since_intake) - np.exp(-ka * time_since_intake)
    )
    nan_mask = ledd.isna() | time_since_intake.isna()
    concentration[nan_mask] = np.nan
    return concentration


# %%
# ── Full feature engineering (experiments 6–11 + 14) ─────────────────────────
def add_extra_features_v4(X_fe, X_raw, y_train=None, saved_stats=None):
    """Compute the full set of extra features used in experiments 6–11 and 14."""
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

    # Retain rater_id as a plain column for NoisyRaterEncoder downstream;
    # drop other categorical origin columns used only to build aggregates.
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


# %%
# ── NoisyRaterEncoder: leak-safe target encoding for rater_id ─────────────────
class NoisyRaterEncoder(BaseEstimator, TransformerMixin):
    """Replace rater_id with the per-rater target mean (+ Gaussian noise).

    Fitted on the training fold: learns the mean UPDRS target per rater.
    On transform, the rater_id column is replaced by the learned mean,
    with a small Gaussian jitter (σ=0.1) to prevent exact memorisation.
    Unseen raters fall back to the global mean.
    """

    def __init__(self, noise_std: float = 0.1, random_state: int = 0):
        self.noise_std = noise_std
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
        out = X.copy()
        encoded = (
            out["rater_id"]
            .map(self.rater_mean_)
            .fillna(self.global_mean_)
            .astype(float)
        )
        rng = np.random.default_rng(self.random_state)
        encoded = encoded + rng.normal(0.0, self.noise_std, size=len(out))
        out["rater_id"] = encoded
        return out


# %%
# ── Load and engineer features ────────────────────────────────────────────────
X_train, y_train = load_dataset(DATA_DIR)
print(f"Train raw: {X_train.shape}")

X_fe_full, _ = build_features(X_train, y_train)
X_ext_full, stats_full = add_extra_features_v4(X_fe_full, X_train, y_train=y_train)

# Assemble declared feature columns (includes rater_id for NoisyRaterEncoder)
ALL_COLS = FEATURE_COLS + ALL_EXTRA_COLS
available_cols = [c for c in ALL_COLS if c in X_ext_full.columns]
# Ensure rater_id is present for the encoder
if "rater_id" not in available_cols and "rater_id" in X_ext_full.columns:
    available_cols = ["rater_id"] + available_cols

X_raw = X_ext_full[available_cols].copy()

# Remove highly correlated columns (|r| > 0.98); rater_id is string — excluded
X_pruned, corr_cols_to_drop = drop_highly_correlated(X_raw, threshold=0.98)
print(f"After correlation pruning: {X_pruned.shape}")

groups_train = X_train["patient_id"].values

# %%
# ── Pipeline: NoisyRaterEncoder → TableVectorizer → HistGBR ──────────────────
pipeline = make_pipeline(
    NoisyRaterEncoder(noise_std=0.1, random_state=0),
    skrub.TableVectorizer(),
    HistGradientBoostingRegressor(random_state=0),
)

param_distributions = {
    "histgradientboostingregressor__learning_rate": [0.015, 0.02, 0.03],
    "histgradientboostingregressor__max_leaf_nodes": [15, 23, 31, 45],
    "histgradientboostingregressor__min_samples_leaf": [60, 90, 120, 150],
    "histgradientboostingregressor__l2_regularization": [2.0, 5.0, 10.0, 20.0],
    "histgradientboostingregressor__max_features": [0.4, 0.55, 0.7],
}

cv = GroupKFold(n_splits=5)

search = HalvingRandomSearchCV(
    pipeline,
    param_distributions=param_distributions,
    resource="histgradientboostingregressor__max_iter",
    min_resources=300,
    max_resources=1500,
    factor=3,
    n_candidates=60,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    refit=True,
    n_jobs=-1,
    random_state=0,
    verbose=2,
)

print("Starting HalvingRandomSearchCV …")
search.fit(X_pruned, y_train, groups=groups_train)
print(f"Best params: {search.best_params_}")
print(f"Best CV RMSE: {-search.best_score_:.4f}")

# %%
# ── Evaluate best estimator with skore ────────────────────────────────────────
best_pipeline = search.best_estimator_

# Build a skrub DataOps graph around the already-engineered data so skore can
# consume it with the same GroupKFold splitter.
visits_df = X_pruned.copy()
visits_df["target"] = y_train.values
visits_df["patient_id"] = groups_train

data_node = skrub.var("visits", visits_df)
groups_node = data_node["patient_id"]

X_op = data_node.drop(["target", "patient_id"], axis=1).skb.mark_as_X(
    cv=GroupKFold(n_splits=5),
    split_kwargs={"groups": groups_node},
)
y_op = data_node["target"].skb.mark_as_y()

pred_node = X_op.skb.apply(best_pipeline, y=y_op)

report = skore.evaluate(pred_node, data={"visits": visits_df})
project.put("16_sota_halving_poly2iso", report)
print("Skore report pushed: 16_sota_halving_poly2iso")
print(report.metrics.rmse())

# %%
# ── Full-train refit and test-set inference ───────────────────────────────────
print("Refitting best pipeline on full training set …")
best_pipeline.fit(X_pruned, y_train)

X_test = load_test_dataset(DATA_DIR)
_, X_te_fe = build_features(X_train, y_train, X_val=X_test)
X_te_ext, _ = add_extra_features_v4(X_te_fe, X_test, saved_stats=stats_full)

available_te_cols = [c for c in available_cols if c in X_te_ext.columns]
X_te_raw = X_te_ext[available_te_cols].copy()
X_te_pruned = X_te_raw.drop(columns=corr_cols_to_drop, errors="ignore")

raw_preds = best_pipeline.predict(X_te_pruned)
print(f"Raw test predictions: {len(raw_preds)} rows")

# %%
# ── Post-processing: per-patient poly2 + isotonic regression ─────────────────
def smooth_poly2_isotonic(
    predictions: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """Fit a degree-2 polynomial then constrain with isotonic regression per patient.

    For each patient's visits (sorted by age):
    1. Fit  pred ~ age + age²  to obtain a smooth parabola.
    2. Constrain the parabola values with IsotonicRegression(increasing=True)
       to enforce physiologically plausible non-decreasing UPDRS trajectories.

    Patients with fewer than 3 visits are left unchanged (insufficient data
    for a meaningful polynomial fit).
    """
    preds = predictions.copy().astype(float)
    df = X_val[["patient_id", "age"]].copy()
    df["_pred"] = preds

    for pid, grp in df.groupby("patient_id"):
        n = len(grp)
        if n < 3:
            continue
        grp_sorted = grp.sort_values("age")
        ages = grp_sorted["age"].values
        p = grp_sorted["_pred"].values

        # Degree-2 polynomial fit (falls back to degree 1 when n < 4)
        degree = 2 if n >= 4 else 1
        coeffs = np.polyfit(ages, p, degree)
        poly_fitted = np.polyval(coeffs, ages)

        # Isotonic constraint: force non-decreasing UPDRS along age axis
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso_fitted = iso.fit_transform(ages, poly_fitted)

        df.loc[grp_sorted.index, "_pred"] = iso_fitted

    return df["_pred"].values


# X_test carries patient_id and age for the post-processing pass
post_preds = smooth_poly2_isotonic(raw_preds, X_test)
print(f"Post-processed predictions: min={post_preds.min():.2f}  max={post_preds.max():.2f}")

# %%
# ── Save submission ───────────────────────────────────────────────────────────
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
sub = X_test[["Index"]].copy()
sub["target"] = post_preds
sub.to_csv(submissions_dir / "16_sota_halving_poly2iso.csv", index=False)
print(f"Submission saved: submissions/16_sota_halving_poly2iso.csv  ({len(sub)} rows)")
print(sub.head())
