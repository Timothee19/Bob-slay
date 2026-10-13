# %% [markdown]
# # Experiment 14 — Drop correlated / redundant columns before skrub pipeline
#
# **Date:** 2025-01-09
# **Base:** Exp 13 (skrub tabular_pipeline + HalvingRandomSearchCV tuned)
#
# ## Changes vs Exp 13
#
# - **Manual categorical-origin drop**: after `add_extra_features_v4`, the raw
#   categorical source columns (`rater_id`, `gene`, `cohort`) are already removed
#   inside that function — but skore Cramer's V alert fires because the
#   *origin column* sometimes leaks through `FEATURE_COLS`.  We explicitly drop
#   any such column from `X_search` before it enters the pipeline.
# - **Auto high-correlation pruning**: `drop_highly_correlated` scans the
#   numeric column pairs and removes one from each pair whose absolute Pearson
#   correlation exceeds 0.98 on the training set.  This eliminates near-duplicate
#   lags / EWMAs flagged by skore.
# - `n_candidates` reduced to 100 (faster wall-clock, same search quality per
#   budget as halving does most of the work via resource doubling).
# - Everything else preserved: GroupKFold(5), min_resources=300,
#   max_resources=3000, resource=max_iter.

# %%
import numpy as np
import pandas as pd
import skore
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import GroupKFold, HalvingRandomSearchCV
from sklearn.metrics import root_mean_squared_error
from sklearn.utils.validation import check_is_fitted

import skrub

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
# ## Bateman PK model (from Exp 11)

# %%
def compute_bateman_concentration(
    ledd: pd.Series,
    time_since_intake: pd.Series,
    ka: float = 1.5,
    ke: float = 0.46,
) -> pd.Series:
    """Plasma drug concentration via the Bateman biexponential equation.

    C(t) = (ledd * ka / (ka - ke)) * (exp(-ke * t) - exp(-ka * t))
    NaN is propagated where either input is NaN.
    """
    concentration = (ledd * ka / (ka - ke)) * (
        np.exp(-ke * time_since_intake) - np.exp(-ka * time_since_intake)
    )
    nan_mask = ledd.isna() | time_since_intake.isna()
    concentration[nan_mask] = np.nan
    return concentration


# %% [markdown]
# ## Feature engineering (v4 — identical to Exp 13)

# %%
def add_extra_features_v4(
    X_fe: pd.DataFrame,
    X_raw: pd.DataFrame,
    y_train: pd.Series | None = None,
    saved_stats: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """All features from exp 06-11 (Bateman PK replaces patient_pk_ratio)."""
    df = X_fe.copy()
    df["patient_id"] = X_raw["patient_id"].values
    df["rater_id"] = X_raw["rater_id"].values
    df["gene"] = X_raw["gene"].fillna("Unknown").values
    df["cohort"] = X_raw["cohort"].values

    # Exp-06 features
    df_sorted = df.sort_values(["patient_id", "age"])
    grp = df_sorted.groupby("patient_id")
    df_sorted["off_corrected_prev2"] = grp["off_corrected"].shift(2)
    df_sorted["on_corrected_prev2"]  = grp["on_corrected"].shift(2)
    df_sorted["off_lag_delta"] = df_sorted["off_corrected"] - df_sorted["off_corrected_prev"]
    df_sorted["inter_visit_gap"] = (df_sorted["age"] - grp["age"].shift(1)).clip(lower=0)
    gap_safe = df_sorted["inter_visit_gap"].replace(0, np.nan)
    df_sorted["off_lag_delta_per_year"] = df_sorted["off_lag_delta"] / gap_safe
    df_sorted["off_corrected_next2"] = grp["off_corrected"].shift(-2)
    df_sorted = df_sorted.loc[df.index]
    for col in ["off_corrected_prev2", "on_corrected_prev2", "off_lag_delta",
                "inter_visit_gap", "off_lag_delta_per_year", "off_corrected_next2"]:
        df[col] = df_sorted[col].values

    df["patient_on_max"] = df["patient_id"].map(df.groupby("patient_id")["on_corrected"].max())
    df["patient_on_q75"] = df["patient_id"].map(df.groupby("patient_id")["on_corrected"].quantile(0.75))
    df["patient_on_std"] = df["patient_id"].map(df.groupby("patient_id")["on_corrected"].std().fillna(0.0))
    on_slopes = {}
    for pid, grp_p in df.groupby("patient_id"):
        valid = grp_p[["on_corrected", "age"]].dropna()
        if len(valid) >= 2:
            on_slopes[pid] = float(np.polyfit(valid["age"].values, valid["on_corrected"].values, 1)[0])
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

    # Exp-07 features
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
    for col in ["off_residual_prev", "on_time_corrected_next", "ledd_prev", "ledd_next", "off_lag_accel"]:
        df[col] = df_sorted3[col].values

    df["patient_ledd_mean"] = df["patient_id"].map(df.groupby("patient_id")["ledd"].mean())
    df["patient_ledd_max"]  = df["patient_id"].map(df.groupby("patient_id")["ledd"].max())
    patient_ledd_std = df.groupby("patient_id")["ledd"].std().fillna(1.0)
    df["patient_ledd_std"] = df["patient_id"].map(patient_ledd_std)
    ledd_std_safe = df["patient_ledd_std"].replace(0, np.nan)
    df["ledd_zscore"] = (df["ledd"] - df["patient_ledd_mean"]) / ledd_std_safe

    # Exp-08 fold-aware stats
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
            slopes = [on_slopes[pid] for pid in grp_g["patient_id"].unique() if pid in on_slopes]
            gene_slope_mean_map[gene] = float(np.mean(slopes)) if slopes else np.nan
        cohort_off_mean_map = df.groupby("cohort")["off_corrected"].mean().to_dict()
        cohort_off_std_map  = df.groupby("cohort")["off_corrected"].std().to_dict()
        cohort_slope_mean_map = {}
        for coh, grp_c in df.groupby("cohort"):
            slopes_c = [on_slopes[pid] for pid in grp_c["patient_id"].unique() if pid in on_slopes]
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

    # Exp-11 features (EWMA + Bateman PK + relative severity + LEDD EWMA)
    df_sorted5 = df.sort_values(["patient_id", "age"])
    grp5 = df_sorted5.groupby("patient_id")
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

    df["pk_concentration_on"] = compute_bateman_concentration(
        df["ledd"], df["time_since_intake_on"]
    ).values
    df["pk_concentration_off"] = compute_bateman_concentration(
        df["ledd"], df["time_since_intake_off"]
    ).values

    if stats is not None:
        df["off_vs_gene_mean"] = df["off_corrected"] - df["gene_off_mean_feat"]
        df["off_vs_cohort_mean"] = df["off_corrected"] - df["cohort_off_mean_feat"]
    else:
        df["off_vs_gene_mean"] = np.nan
        df["off_vs_cohort_mean"] = np.nan

    df["visit_rev_index"] = df["patient_n_visits"] - df["visit_index"]

    df_sorted6 = df.sort_values(["patient_id", "age"])
    grp6 = df_sorted6.groupby("patient_id")
    ledd_shifted = grp6["ledd"].shift(1)
    df_sorted6["ledd_ewma"] = ledd_shifted.groupby(df_sorted6["patient_id"]).transform(
        lambda x: x.ewm(alpha=0.5, adjust=False).mean()
    )
    df_sorted6 = df_sorted6.loc[df.index]
    df["ledd_ewma"] = df_sorted6["ledd_ewma"].values

    # Drop raw categorical columns and the intermediate std helper.
    # rater_id, gene, cohort have been replaced by target-encoded / mean features.
    df = df.drop(columns=["patient_id", "rater_id", "gene", "cohort", "patient_ledd_std"])
    return df, saved_stats_out


# %% [markdown]
# ## Auto high-correlation pruning (Pearson |r| > 0.98)

# %%
def drop_highly_correlated(df: pd.DataFrame, threshold: float = 0.98) -> tuple[pd.DataFrame, list[str]]:
    """Remove one column from each numeric pair with |Pearson r| > threshold.

    Keeps the *first* column encountered in each correlated pair (alphabetical
    order within the correlation matrix), ensuring the same set of columns is
    dropped for train and test when called with the same `cols_to_drop` list.

    Returns the pruned DataFrame and the list of dropped column names so the
    test matrix can apply the same drop via ``df.drop(columns=cols_to_drop)``.
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return df, []

    # Fill NaN with column median for correlation computation only
    filled = df[num_cols].fillna(df[num_cols].median())
    corr = filled.corr().abs()

    # Upper triangle only — avoid double-counting
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    cols_to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    dropped = [c for c in cols_to_drop if c in df.columns]
    print(f"Auto-drop {len(dropped)} highly correlated columns (|r| > {threshold}): {dropped}")
    return df.drop(columns=dropped, errors="ignore"), dropped


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
print(f"Feature count before pruning: {len(ALL_COLS)} ({len(FEATURE_COLS)} base + "
      f"{len(EXP06_COLS)}+{len(EXP07_COLS)}+{len(EXP08_COLS)}+{len(EXP11_COLS)} new)")

# %% [markdown]
# ## FeatureSelector transformer — toggles feature groups on/off

# %%
class FeatureGroupSelector(BaseEstimator, TransformerMixin):
    """Drop feature groups flagged False.

    Parameters
    ----------
    use_exp06, use_exp07, use_exp08, use_exp11 : bool
        Whether to include each feature group.
    """

    def __init__(
        self,
        use_exp06: bool = True,
        use_exp07: bool = True,
        use_exp08: bool = True,
        use_exp11: bool = True,
    ):
        self.use_exp06 = use_exp06
        self.use_exp07 = use_exp07
        self.use_exp08 = use_exp08
        self.use_exp11 = use_exp11

    def fit(self, X, y=None):
        self._cols_to_drop = []
        if not self.use_exp06:
            self._cols_to_drop.extend(c for c in EXP06_COLS if c in X.columns)
        if not self.use_exp07:
            self._cols_to_drop.extend(c for c in EXP07_COLS if c in X.columns)
        if not self.use_exp08:
            self._cols_to_drop.extend(c for c in EXP08_COLS if c in X.columns)
        if not self.use_exp11:
            self._cols_to_drop.extend(c for c in EXP11_COLS if c in X.columns)
        return self

    def transform(self, X):
        return X.drop(columns=self._cols_to_drop, errors="ignore")


# %% [markdown]
# ## Build full feature matrix — then prune redundant columns

# %%
X_fe_full, _ = build_features(X_train, y_train)
X_ext_full, stats_full = add_extra_features_v4(X_fe_full, X_train, y_train=y_train)

# Keep only the declared feature columns — this already excludes rater_id / gene / cohort
# (they were dropped inside add_extra_features_v4; their encoded replacements are in EXP07/EXP08).
available_cols = [c for c in ALL_COLS if c in X_ext_full.columns]
X_search_raw = X_ext_full[available_cols].copy()
print(f"Before correlation pruning: {X_search_raw.shape}")

# Auto-prune highly correlated numeric columns (|r| > 0.98).
# Save the list so the same columns can be dropped from the test matrix.
X_search, corr_cols_to_drop = drop_highly_correlated(X_search_raw, threshold=0.98)
print(f"After correlation pruning:  {X_search.shape}")

# %% [markdown]
# ## Build the skrub pipeline (HistGBR, NaN-native)

# %%
from sklearn.pipeline import make_pipeline

base_pipeline = make_pipeline(
    FeatureGroupSelector(),
    skrub.tabular_pipeline("regressor"),
)

# %% [markdown]
# ## HalvingRandomSearchCV — 100 candidates, GroupKFold(5)
# Same resource schedule as Exp 13 (min=300, max=3000, factor=3).
# n_candidates reduced to 100 for faster wall-clock.

# %%
param_distributions = {
    # Feature group toggles
    "featuregroupselector__use_exp06": [True, False],
    "featuregroupselector__use_exp07": [True, False],
    "featuregroupselector__use_exp08": [True, False],
    "featuregroupselector__use_exp11": [True, False],
    # HistGBR hyperparameters — low learning rates only
    "pipeline__histgradientboostingregressor__learning_rate": [0.005, 0.01, 0.02, 0.05],
    "pipeline__histgradientboostingregressor__max_leaf_nodes": randint(20, 512),
    "pipeline__histgradientboostingregressor__min_samples_leaf": randint(10, 100),
    "pipeline__histgradientboostingregressor__l2_regularization": loguniform(1e-3, 30.0),
    "pipeline__histgradientboostingregressor__max_features": uniform(0.5, 0.5),
}

cv = GroupKFold(n_splits=5)

search = HalvingRandomSearchCV(
    estimator=base_pipeline,
    param_distributions=param_distributions,
    n_candidates=100,
    resource="pipeline__histgradientboostingregressor__max_iter",
    max_resources=3000,
    min_resources=300,
    factor=3,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    refit=True,
    random_state=42,
    n_jobs=-1,
    verbose=2,
)

print("Starting HalvingRandomSearchCV (100 candidates, resource=max_iter, max=3000, min=300)...")
search.fit(X_search, y_train.values, groups=X_train["patient_id"].values)

print(f"\nBest RMSE (CV): {-search.best_score_:.4f}")
print(f"Best params: {search.best_params_}")

# %% [markdown]
# ## Wrap best params in a serialisable sklearn estimator for skore

# %%
class SkrubHalvingBestModel(BaseEstimator):
    """Wrapper holding the best HalvingRandomSearch params; rebuilds the pipeline on fit.

    Needed because skore hub serialises a clone of the unfitted estimator, which
    fails for nested sklearn Pipelines containing TableVectorizer.  Wrapping the
    params in a plain BaseEstimator makes the object serialisable.
    """

    def __init__(
        self,
        use_exp06: bool = True,
        use_exp07: bool = True,
        use_exp08: bool = True,
        use_exp11: bool = True,
        learning_rate: float = 0.01,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        max_features: float = 1.0,
        max_iter: int = 100,
    ):
        self.use_exp06 = use_exp06
        self.use_exp07 = use_exp07
        self.use_exp08 = use_exp08
        self.use_exp11 = use_exp11
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_features = max_features
        self.max_iter = max_iter

    def _build_pipeline(self):
        from sklearn.pipeline import make_pipeline
        import skrub as _skrub

        pipe = make_pipeline(
            FeatureGroupSelector(
                use_exp06=self.use_exp06,
                use_exp07=self.use_exp07,
                use_exp08=self.use_exp08,
                use_exp11=self.use_exp11,
            ),
            _skrub.tabular_pipeline("regressor"),
        )
        pipe.set_params(**{
            "pipeline__histgradientboostingregressor__learning_rate": self.learning_rate,
            "pipeline__histgradientboostingregressor__max_leaf_nodes": self.max_leaf_nodes,
            "pipeline__histgradientboostingregressor__min_samples_leaf": self.min_samples_leaf,
            "pipeline__histgradientboostingregressor__l2_regularization": self.l2_regularization,
            "pipeline__histgradientboostingregressor__max_features": self.max_features,
            "pipeline__histgradientboostingregressor__max_iter": self.max_iter,
        })
        return pipe

    def fit(self, X, y, sample_weight=None):
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "pipeline_")
        return self.pipeline_.predict(X)


# Extract best params from search
bp = search.best_params_
best_model = SkrubHalvingBestModel(
    use_exp06=bp["featuregroupselector__use_exp06"],
    use_exp07=bp["featuregroupselector__use_exp07"],
    use_exp08=bp["featuregroupselector__use_exp08"],
    use_exp11=bp["featuregroupselector__use_exp11"],
    learning_rate=float(bp["pipeline__histgradientboostingregressor__learning_rate"]),
    max_leaf_nodes=int(bp["pipeline__histgradientboostingregressor__max_leaf_nodes"]),
    min_samples_leaf=int(bp["pipeline__histgradientboostingregressor__min_samples_leaf"]),
    l2_regularization=float(bp["pipeline__histgradientboostingregressor__l2_regularization"]),
    max_features=float(bp["pipeline__histgradientboostingregressor__max_features"]),
    max_iter=int(bp["pipeline__histgradientboostingregressor__max_iter"]),
)

# %% [markdown]
# ## Evaluate best model with skore.CrossValidationReport

# %%
report = skore.CrossValidationReport(
    best_model,
    X=X_search,
    y=y_train,
    splitter=folds,
    n_jobs=1,
)
project.put("14_skrub_drop_correlated", report)
print("Report saved: 14_skrub_drop_correlated")
print(report.metrics.rmse())

# %% [markdown]
# ## Submission — refit on full train, predict test

# %%
X_test = load_test_dataset(DATA_DIR)
_, X_te_fe = build_features(X_train, y_train, X_val=X_test)
X_te_ext, _ = add_extra_features_v4(X_te_fe, X_test, saved_stats=stats_full)

# Apply the same column selection and correlation pruning as for train
available_te_cols = [c for c in ALL_COLS if c in X_te_ext.columns]
X_te_search_raw = X_te_ext[available_te_cols].copy()
X_te_search = X_te_search_raw.drop(columns=corr_cols_to_drop, errors="ignore")
print(f"Test matrix after pruning: {X_te_search.shape}")

test_preds = best_model.predict(X_te_search)

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
sub = X_test[["Index"]].copy()
sub["target"] = test_preds
sub.to_csv(submissions_dir / "14_skrub_drop_correlated.csv", index=False)
print(f"Submission: 14_skrub_drop_correlated.csv  ({len(sub)} rows)")
print(sub.head())
