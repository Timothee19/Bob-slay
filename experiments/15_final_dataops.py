# %% [markdown]
# # Experiment 15 — Clean DataOps graph (DAY2 §6 pattern)
#
# **Date:** 2025-01-10
# **Base:** Exp 14 (HistGBR, best params, 76 features after correlation pruning)
#
# ## What this experiment does
#
# Instead of passing a pre-split NumPy / DataFrame to `CrossValidationReport`,
# we freeze the full data-prep + model as a **skrub DataOps graph** following
# the DAY2.md §6 recipe.  The `visits` DataFrame already contains the cleaned
# feature columns plus `target` and `patient_id` — `skrub.var` names it, the
# DataOps marks split info (GroupKFold groups), and `skore.evaluate(pred)` reads
# everything from the graph without a separate `splitter=` argument.
#
# ## Best hyperparameters (frozen from Exp 14 search)
#
# lr=0.02, max_iter=2700, max_leaf_nodes=93, min_samples_leaf=67,
# l2=1.4457, max_features=0.5715 — all feature groups active.
# Correlation-pruning removes 5 columns (|r| > 0.98).

# %%
import numpy as np
import pandas as pd
import skore
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

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
# ── Feature column groups (same as exp 14) ────────────────────────────────────
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
# ── Bateman PK model helper ────────────────────────────────────────────────────
def compute_bateman_concentration(
    ledd: pd.Series,
    time_since_intake: pd.Series,
    ka: float = 1.5,
    ke: float = 0.46,
) -> pd.Series:
    """Plasma drug concentration via the Bateman biexponential equation."""
    concentration = (ledd * ka / (ka - ke)) * (
        np.exp(-ke * time_since_intake) - np.exp(-ka * time_since_intake)
    )
    nan_mask = ledd.isna() | time_since_intake.isna()
    concentration[nan_mask] = np.nan
    return concentration


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

    # Drop categorical origin columns that were only used to compute aggregates
    df = df.drop(
        columns=["patient_id", "rater_id", "gene", "cohort", "patient_ledd_std"]
    )
    return df, saved_stats_out


def drop_highly_correlated(df, threshold=0.98):
    """Remove one column from each pair with |Pearson r| > threshold."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return df, []
    filled = df[num_cols].fillna(df[num_cols].median())
    corr = filled.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    cols_to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    dropped = [c for c in cols_to_drop if c in df.columns]
    print(f"Auto-drop {len(dropped)} cols: {dropped}")
    return df.drop(columns=dropped, errors="ignore"), dropped


# %%
# ── Build the cleaned visits DataFrame ────────────────────────────────────────
X_train, y_train = load_dataset(DATA_DIR)
print(f"Train raw: {X_train.shape}")

X_fe_full, _ = build_features(X_train, y_train)
X_ext_full, stats_full = add_extra_features_v4(X_fe_full, X_train, y_train=y_train)

# Keep only declared feature columns (excludes categorical origin cols)
ALL_COLS = FEATURE_COLS + ALL_EXTRA_COLS
available_cols = [c for c in ALL_COLS if c in X_ext_full.columns]
X_raw = X_ext_full[available_cols].copy()

# Remove columns with correlation > 0.98 (same pruning as exp14)
X_pruned, corr_cols_to_drop = drop_highly_correlated(X_raw, threshold=0.98)
print(f"After pruning: {X_pruned.shape}")

# Attach target and patient_id for the DataOps graph
visits = X_pruned.copy()
visits["target"] = y_train.values
visits["patient_id"] = X_train["patient_id"].values

# %%
# ── skrub DataOps graph (DAY2 §6 pattern) ────────────────────────────────────
data = skrub.var("visits", visits)
groups = data["patient_id"]

X_op = data.drop(["target", "patient_id"], axis=1).skb.mark_as_X(
    cv=GroupKFold(n_splits=5),
    split_kwargs={"groups": groups},
)
y_op = data["target"].skb.mark_as_y()

# Best hyperparameters from Exp 14 HalvingRandomSearchCV
hgbr = HistGradientBoostingRegressor(
    learning_rate=0.02,
    max_iter=2700,
    max_leaf_nodes=93,
    min_samples_leaf=67,
    l2_regularization=1.4457497927654140,
    max_features=0.5714958410264179,
    random_state=0,
)

pred = X_op.skb.apply(skrub.TableVectorizer()).skb.apply(hgbr, y=y_op)

# %%
# ── Evaluate and publish ───────────────────────────────────────────────────────
report = skore.evaluate(pred)
project.put("15_final_dataops", report)
print("Report saved: 15_final_dataops")
print(report.metrics.rmse())

# %%
# ── Submission: fit SkrubLearner on full train, predict test ──────────────────
X_test = load_test_dataset(DATA_DIR)
_, X_te_fe = build_features(X_train, y_train, X_val=X_test)
X_te_ext, _ = add_extra_features_v4(X_te_fe, X_test, saved_stats=stats_full)

available_te_cols = [c for c in ALL_COLS if c in X_te_ext.columns]
X_te_raw = X_te_ext[available_te_cols].copy()
X_te_pruned = X_te_raw.drop(columns=corr_cols_to_drop, errors="ignore")

# Attach placeholder target=0 and patient_id so the learner can consume the
# same "visits" dict key (target column is dropped inside X_op).
X_te_visits = X_te_pruned.copy()
X_te_visits["target"] = 0.0
X_te_visits["patient_id"] = X_test["patient_id"].values

learner = pred.skb.make_learner()
learner.fit({"visits": visits})
test_preds = learner.predict({"visits": X_te_visits})

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
sub = X_test[["Index"]].copy()
sub["target"] = test_preds
sub.to_csv(submissions_dir / "15_final_dataops.csv", index=False)
print(f"Submission saved: submissions/15_final_dataops.csv  ({len(sub)} rows)")
print(sub.head())
