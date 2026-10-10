"""Feature engineering for the Parkinson's UPDRS regression.

Three tiers of features:

1. **Stateless** — computed from a single row without any fold information:
   ``disease_duration``, missingness flags, categorical encoding (including
   native ``category`` dtype columns for HGBR/LightGBM).
   These are safe to apply before any CV split.

2. **Rater-bias correction** (fold-aware) — each rater applies a
   multiplicative bias b_r to their scores.  ``b_r`` is estimated as
   ``median(off / target)`` per rater on the *training fold only*, then
   applied to both train and validation folds.  On the test set the full
   training set is used for estimation.

3. **Patient aggregates** (intra-split, full-patient) — for each row,
   slope/intercept/mean/std computed over *all* visits of the same patient
   within the same split.  No leave-one-out: on the test set all visits of
   a patient are available, so using all visits is consistent and gives a
   more stable estimate.  GroupKFold guarantees no cross-fold patient
   overlap, so this is still leak-free.

Public API
----------
build_features(X, y, X_val=None)
    Fit on ``X``/``y`` (training fold), transform both ``X`` and ``X_val``
    (validation / test fold).  Returns ``(X_train_fe, X_val_fe)`` when
    ``X_val`` is provided, otherwise ``(X_train_fe, None)``.

FEATURE_COLS
    Ordered list of final numeric feature column names, excluding
    ``patient_id`` and ``Index``.

GENE_CATEGORIES / COHORT_CATEGORIES
    Ordered label lists used for ordinal encoding.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GENE_CATEGORIES: list[str] = ["No Mutation", "GBA+", "LRRK2+", "Other", "Unknown"]
COHORT_CATEGORIES: list[str] = ["A", "B"]

# Pharmacokinetic binning: half-hour bins from 0 to 6 h
_T_BIN_WIDTH = 0.5
_T_MAX = 6.0
_T_CLAMP_MIN = 0.3   # floor on f̂(t) to avoid explosive division
_T_CLAMP_MAX = 1.0   # ceiling

# Fallback rater-bias factor when a rater has no (off, target) pairs in fold
_GLOBAL_OFF_RATIO = 0.85


# ---------------------------------------------------------------------------
# Stateless helpers
# ---------------------------------------------------------------------------

def _stateless_features(X: pd.DataFrame) -> pd.DataFrame:
    """Compute features that do not require fold-level target information.

    Includes disease duration, missingness flags, and categorical columns
    (using native ``category`` dtype so HGBR/LightGBM can find optimal
    splits without forcing an arbitrary ordinal order).

    Parameters
    ----------
    X :
        Raw feature DataFrame (any split).

    Returns
    -------
    pd.DataFrame
        DataFrame with new columns added; original columns preserved for
        downstream fold-aware steps.
    """
    df = X.copy()

    # --- disease duration -----------------------------------------------
    age_diag = df["age_at_diagnosis"].copy()
    age_diag_filled = age_diag.fillna(age_diag.median())
    df["disease_duration"] = (df["age"] - age_diag_filled).clip(lower=0)

    # --- missingness flags -----------------------------------------------
    df["ledd_missing_flag"] = df["ledd"].isna().astype("float64")
    df["on_missing_flag"] = df["on"].isna().astype("float64")
    df["off_missing_flag"] = df["off"].isna().astype("float64")
    df["time_on_missing_flag"] = df["time_since_intake_on"].isna().astype("float64")

    # --- native categorical columns (HGBR / LightGBM native support) ------
    # gene: fill NaN with "Unknown" sentinel then cast to category
    df["gene_cat"] = df["gene"].fillna("Unknown").astype("category")
    # cohort: already clean (no NaN)
    df["cohort_cat"] = df["cohort"].astype("category")
    # rater_id: high cardinality (50 raters) — category lets the model
    # find optimal partitions; complements the numeric rater_bias_feature
    df["rater_id_cat"] = df["rater_id"].astype("category")

    # --- legacy numeric encodings (kept for backward compat + HGBR bins) --
    df["sexM"] = df["sexM"].astype("float64")

    return df


# ---------------------------------------------------------------------------
# Rater-bias estimation and correction
# ---------------------------------------------------------------------------

def _estimate_rater_bias(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, float]:
    """Estimate per-rater multiplicative bias from training-fold data.

    For rows where both ``off`` and the target are available, compute
    ``ratio = off / target``.  The per-rater bias ``b_r`` is the median
    of those ratios.  Raters absent from the training fold receive the
    global fallback ``_GLOBAL_OFF_RATIO``.

    Parameters
    ----------
    X_train :
        Training-fold feature DataFrame (must include ``rater_id`` and ``off``).
    y_train :
        Training-fold target Series aligned with ``X_train``.

    Returns
    -------
    dict[str, float]
        Mapping from ``rater_id`` to estimated bias factor.
    """
    mask = X_train["off"].notna() & (y_train > 0)
    if mask.sum() == 0:
        return {}
    ratios = X_train.loc[mask, "off"] / y_train.loc[mask]
    rater_bias: dict[str, float] = (
        ratios.groupby(X_train.loc[mask, "rater_id"])
        .median()
        .to_dict()
    )
    return rater_bias


def _apply_rater_correction(
    df: pd.DataFrame,
    rater_bias: dict[str, float],
) -> pd.DataFrame:
    """Divide ``on`` and ``off`` by the per-rater bias factor.

    Parameters
    ----------
    df :
        Feature DataFrame with ``rater_id``, ``on``, ``off`` columns.
    rater_bias :
        Mapping from ``rater_id`` to bias factor ``b_r``.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``on_corrected``, ``off_corrected``, and
        ``rater_bias_feature`` columns added.
    """
    b = df["rater_id"].map(rater_bias).fillna(_GLOBAL_OFF_RATIO)
    b = b.clip(lower=0.1)
    df = df.copy()
    df["rater_bias_feature"] = b
    df["on_corrected"] = df["on"] / b
    df["off_corrected"] = df["off"] / b
    return df


# ---------------------------------------------------------------------------
# Pharmacokinetic correction of ``on``
# ---------------------------------------------------------------------------

def _estimate_pk_curve(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    on_col: str = "on_corrected",
) -> interp1d | None:
    """Estimate the ratio f(t) = on_corrected / target as a function of time.

    Uses the rater-corrected ``on`` score so the bias is already removed
    before fitting the time curve.

    Parameters
    ----------
    X_train :
        Training-fold DataFrame; must include ``time_since_intake_on`` and
        ``on_corrected``.
    y_train :
        Training-fold target Series.
    on_col :
        Column to use as the numerator (default: ``on_corrected``).

    Returns
    -------
    interp1d | None
        Scipy interpolation function f̂(t), clamped to [0.3, 1.0], or
        ``None`` if there are fewer than 3 valid bins.
    """
    mask = (
        X_train[on_col].notna()
        & X_train["time_since_intake_on"].notna()
        & (y_train > 0)
    )
    if mask.sum() < 10:
        return None

    t = X_train.loc[mask, "time_since_intake_on"].values
    ratios = (X_train.loc[mask, on_col] / y_train.loc[mask]).values

    edges = np.arange(0, _T_MAX + _T_BIN_WIDTH, _T_BIN_WIDTH)
    bin_centers = edges[:-1] + _T_BIN_WIDTH / 2

    bin_medians: list[float] = []
    valid_centers: list[float] = []

    for lo, hi, center in zip(edges[:-1], edges[1:], bin_centers):
        in_bin = (t >= lo) & (t < hi)
        if in_bin.sum() >= 3:
            bin_medians.append(float(np.median(ratios[in_bin])))
            valid_centers.append(center)

    if len(valid_centers) < 3:
        return None

    bin_medians_arr = np.clip(bin_medians, _T_CLAMP_MIN, _T_CLAMP_MAX)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_hat = interp1d(
            valid_centers,
            bin_medians_arr,
            kind="linear",
            bounds_error=False,
            fill_value=(bin_medians_arr[0], bin_medians_arr[-1]),
        )
    return f_hat


def _apply_pk_correction(
    df: pd.DataFrame,
    f_hat: interp1d | None,
) -> pd.DataFrame:
    """Compute ``on_time_corrected = on_corrected / f_hat(t)``.

    Parameters
    ----------
    df :
        DataFrame with ``on_corrected`` and ``time_since_intake_on``.
    f_hat :
        Estimated PK curve, or ``None`` (in which case the column is
        filled with NaN).

    Returns
    -------
    pd.DataFrame
        DataFrame with ``on_time_corrected`` column added.
    """
    df = df.copy()
    if f_hat is None:
        df["on_time_corrected"] = np.nan
        return df

    t = df["time_since_intake_on"]
    mask = t.notna() & df["on_corrected"].notna()

    df["on_time_corrected"] = np.nan
    if mask.any():
        t_vals = t[mask].values
        f_vals = np.clip(f_hat(t_vals), _T_CLAMP_MIN, _T_CLAMP_MAX)
        df.loc[mask, "on_time_corrected"] = df.loc[mask, "on_corrected"] / f_vals

    return df


# ---------------------------------------------------------------------------
# Patient full-group aggregates (no leave-one-out)
# ---------------------------------------------------------------------------

def _group_mean(df: pd.DataFrame, col: str) -> pd.Series:
    """Per-patient mean of ``col`` over all visits (including current row).

    Using all visits (no LOO) is consistent with the test-set regime where
    all visits of a patient are available at prediction time.

    Parameters
    ----------
    df :
        DataFrame with ``patient_id`` and ``col``.
    col :
        Column to aggregate.

    Returns
    -------
    pd.Series
        Group mean values aligned with ``df.index``.
    """
    return df["patient_id"].map(df.groupby("patient_id")[col].mean())


def _group_std(df: pd.DataFrame, col: str) -> pd.Series:
    """Per-patient std of ``col`` over all visits (including current row).

    Parameters
    ----------
    df :
        DataFrame with ``patient_id`` and ``col``.
    col :
        Column to aggregate.

    Returns
    -------
    pd.Series
        Group std values aligned with ``df.index``.  NaN when patient has < 2
        non-NaN observations; filled with 0 for single-visit patients.
    """
    result = df["patient_id"].map(df.groupby("patient_id")[col].std().fillna(0.0))
    return result


def _patient_slope_intercept(
    df: pd.DataFrame, col: str
) -> tuple[pd.Series, pd.Series]:
    """Per-patient linear regression slope and intercept of ``col`` vs ``age``.

    Uses all visits of the patient (no LOO).

    Parameters
    ----------
    df :
        DataFrame with ``patient_id``, ``age``, and ``col``.
    col :
        Column to regress (typically ``off_corrected``).

    Returns
    -------
    tuple[pd.Series, pd.Series]
        ``(patient_slope, patient_intercept)`` aligned with ``df.index``.
        Patients with fewer than 2 valid rows fall back to the global
        median slope/intercept.
    """
    slopes: dict[str, float] = {}
    intercepts: dict[str, float] = {}

    for pid, grp in df.groupby("patient_id"):
        valid = grp[[col, "age"]].dropna()
        if len(valid) >= 2:
            coeffs = np.polyfit(valid["age"].values, valid[col].values, 1)
            slopes[pid] = float(coeffs[0])
            intercepts[pid] = float(coeffs[1])
        else:
            slopes[pid] = np.nan
            intercepts[pid] = np.nan

    median_slope = float(np.nanmedian(list(slopes.values())))
    median_intercept = float(np.nanmedian(list(intercepts.values())))
    for pid in slopes:
        if np.isnan(slopes[pid]):
            slopes[pid] = median_slope
            intercepts[pid] = median_intercept

    slope_series = df["patient_id"].map(slopes)
    intercept_series = df["patient_id"].map(intercepts)
    return slope_series, intercept_series


def _ledd_velocity(df: pd.DataFrame) -> pd.Series:
    """Per-patient LEDD velocity: slope of ledd vs age for each visit.

    For patients with >= 2 non-NaN LEDD values, fits a linear regression
    ``ledd ~ age`` and returns the slope (dose increase rate, mg/year).
    Patients with < 2 valid LEDD rows get NaN — HGBR/LightGBM handle it.

    Parameters
    ----------
    df :
        DataFrame with ``patient_id``, ``age``, ``ledd``.

    Returns
    -------
    pd.Series
        LEDD velocity (slope) aligned with ``df.index``.
    """
    vel: dict[str, float] = {}
    for pid, grp in df.groupby("patient_id"):
        valid = grp[["ledd", "age"]].dropna()
        if len(valid) >= 2:
            coeffs = np.polyfit(valid["age"].values, valid["ledd"].values, 1)
            vel[pid] = float(coeffs[0])
        else:
            vel[pid] = np.nan
    return df["patient_id"].map(vel)


def _add_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-patient aggregate features to ``df``.

    All aggregates use only feature columns (no target), so they are
    leak-free.  For each row the aggregates summarise *all* visits of the
    same patient within the same DataFrame (no LOO exclusion).

    Parameters
    ----------
    df :
        DataFrame after rater-bias correction with ``patient_id``,
        ``off_corrected``, ``on_corrected``, and ``age`` columns.

    Returns
    -------
    pd.DataFrame
        ``df`` with ``patient_off_mean``, ``patient_off_std``,
        ``patient_on_mean``, ``patient_slope``, ``patient_intercept``,
        ``patient_n_visits`` added.
    """
    df = df.copy()
    df["patient_off_mean"] = _group_mean(df, "off_corrected")
    df["patient_off_std"] = _group_std(df, "off_corrected")
    df["patient_on_mean"] = _group_mean(df, "on_corrected")
    df["patient_slope"], df["patient_intercept"] = _patient_slope_intercept(
        df, "off_corrected"
    )
    df["patient_n_visits"] = df["patient_id"].map(
        df.groupby("patient_id")["age"].count().astype(float)
    )
    # LEDD velocity: rate of dose increase per year of age
    df["ledd_velocity"] = _ledd_velocity(df)
    # Upper-tail summary: max and 75th percentile of off_corrected per patient.
    # Because scores are always under-reported (off <= true OFF), the upper
    # quantiles are less biased than the mean and correlate better with target.
    df["patient_off_max"] = df["patient_id"].map(
        df.groupby("patient_id")["off_corrected"].max()
    )
    df["patient_off_q75"] = df["patient_id"].map(
        df.groupby("patient_id")["off_corrected"].quantile(0.75)
    )
    return df


def _lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add visit-lag features: off/on values at the previous and next visit.

    All lags are computed *within* the DataFrame (no cross-split look-up),
    after sorting by ``patient_id`` and ``age``.  GroupKFold guarantees all
    visits of a patient land in the same fold, so this is leak-free.

    The cumulative maximum of ``off_corrected`` sorted by age is the single
    strongest aggregate feature (corr=0.95 with target on held-out data):
    since the disease only progresses, the highest ``off`` score seen so far
    is a robust lower bound for the true current OFF.

    Features added
    --------------
    off_corrected_prev   : off_corrected at the immediately preceding visit
                           (NaN for first visit).  Corr~0.88 with target —
                           extremely useful when the current visit has no ``off``.
    off_corrected_next   : off_corrected at the next visit (NaN for last visit).
                           Almost as informative as prev; available when off
                           courant is missing and the next visit has off.
    on_corrected_prev    : on_corrected at the preceding visit.
    off_cummax           : running max of off_corrected up to and including the
                           current visit (sorted by age).  Corr~0.95 — better
                           than any single-visit feature.  Uses only the
                           off_corrected column (no target), so no leakage.

    Parameters
    ----------
    df :
        DataFrame with ``patient_id``, ``age``, ``off_corrected``,
        ``on_corrected``.  Must already have ``visit_index`` so that
        subsequent steps can use the sorted order.

    Returns
    -------
    pd.DataFrame
        ``df`` with lag columns added; row order preserved (original index).
    """
    df = df.copy()
    # Sort by (patient_id, age) to compute lags in chronological order
    original_index = df.index
    df_sorted = df.sort_values(["patient_id", "age"])

    grp = df_sorted.groupby("patient_id")
    df_sorted["off_corrected_prev"] = grp["off_corrected"].shift(1)
    df_sorted["off_corrected_next"] = grp["off_corrected"].shift(-1)
    df_sorted["on_corrected_prev"] = grp["on_corrected"].shift(1)
    # cummax: max off_corrected seen so far (including current visit)
    df_sorted["off_cummax"] = grp["off_corrected"].cummax()

    # Restore original row order
    df_sorted = df_sorted.loc[original_index]
    df["off_corrected_prev"] = df_sorted["off_corrected_prev"].values
    df["off_corrected_next"] = df_sorted["off_corrected_next"].values
    df["on_corrected_prev"] = df_sorted["on_corrected_prev"].values
    df["off_cummax"] = df_sorted["off_cummax"].values
    return df


def _visit_index(df: pd.DataFrame) -> pd.Series:
    """Rank each row within its patient by ascending age (visit order).

    Returns a float Series aligned with ``df.index``.
    """
    return (
        df.groupby("patient_id")["age"]
        .rank(method="first", ascending=True)
        .astype("float64")
    )


# ---------------------------------------------------------------------------
# Derived interaction features
# ---------------------------------------------------------------------------

def _derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute interaction features from corrected ``on`` and ``off``.

    ``on_off_mean`` is intentionally omitted: if ``off`` is biased low at a
    visit, ``on`` is also biased low (same patient, same day, same conditions).
    Averaging them does not reduce this intra-visit bias.

    ``on_off_ratio = off_corrected / on_corrected`` captures motor fluctuation
    independently of the absolute score level: a systematic under-rating by
    the evaluator leaves the ratio unchanged.

    Parameters
    ----------
    df :
        DataFrame with ``on_corrected`` and ``off_corrected``.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``on_off_diff``, ``on_off_ratio``,
        ``patient_trend_at_age``, ``off_residual_from_trend``, and
        ``visit_rel_index`` added.
    """
    df = df.copy()
    df["on_off_diff"] = df["off_corrected"] - df["on_corrected"]

    # Ratio: guard against near-zero on_corrected
    on_safe = df["on_corrected"].replace(0, np.nan)
    df["on_off_ratio"] = df["off_corrected"] / on_safe

    return df


def _trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add personalised trend features using the per-patient linear model.

    These features require ``patient_slope``, ``patient_intercept``, and
    ``patient_n_visits`` to already be present in ``df``.

    Parameters
    ----------
    df :
        DataFrame after ``_add_patient_features``.

    Returns
    -------
    pd.DataFrame
        ``df`` with:
        - ``patient_trend_at_age``: predicted score from the patient's own
          linear trend at the current visit age — a personalised prior.
        - ``off_residual_from_trend``: how far ``off_corrected`` deviates
          from that trend — captures intra-patient fluctuation beyond
          normal progression.
        - ``visit_rel_index``: visit position normalised by total visits
          (0 = first visit, 1 = last); more informative than raw rank.
    """
    df = df.copy()

    # Personalised linear forecast at this visit's age
    df["patient_trend_at_age"] = (
        df["patient_intercept"] + df["patient_slope"] * df["age"]
    )

    # Residual: how far the measured off_corrected is from the trend
    off_safe = df["off_corrected"]  # NaN where off is missing — kept as NaN
    df["off_residual_from_trend"] = off_safe - df["patient_trend_at_age"]

    # Normalised visit position within the patient's sequence
    n = df["patient_n_visits"].replace(0, 1)  # guard division by zero
    df["visit_rel_index"] = df["visit_index"] / n

    return df


# ---------------------------------------------------------------------------
# Final feature column list
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    # Raw numeric features (NaN handled natively by HGBR / LightGBM)
    "age",
    "disease_duration",
    "ledd",
    # Rater-corrected UPDRS scores
    "on_corrected",
    "off_corrected",
    "on_time_corrected",
    # Interaction features
    "on_off_diff",
    "on_off_ratio",
    # Patient-level longitudinal aggregates (full-group, no LOO)
    "patient_off_mean",
    "patient_off_std",
    "patient_off_max",
    "patient_off_q75",
    "patient_on_mean",
    "patient_slope",
    "patient_intercept",
    "patient_n_visits",
    "visit_index",
    "visit_rel_index",
    "ledd_velocity",
    # Personalised trend features
    "patient_trend_at_age",
    "off_residual_from_trend",
    # Lag features (chronological off/on at adjacent visits)
    "off_corrected_prev",
    "off_corrected_next",
    "on_corrected_prev",
    "off_cummax",
    # Time since intake (kept raw; model uses it)
    "time_since_intake_on",
    "time_since_intake_off",
    # Rater bias factor (numeric representation of rater identity)
    "rater_bias_feature",
    # Native categorical columns (HGBR / LightGBM find optimal splits)
    "gene_cat",
    "cohort_cat",
    "rater_id_cat",
    # Sex (binary)
    "sexM",
    # Missingness flags (informative MNAR signal)
    "ledd_missing_flag",
    "on_missing_flag",
    "off_missing_flag",
    "time_on_missing_flag",
]


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def build_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Fit feature engineering on the training fold and transform both splits.

    All fold-aware estimations (rater bias, PK curve, patient aggregates)
    are learned exclusively from ``X_train`` / ``y_train`` and applied to
    both ``X_train`` and ``X_val``.  This guarantees no target leakage from
    the validation fold into the training computation.

    Patient aggregates (slope, mean, std) are computed over *all* visits of
    each patient within the respective split — no leave-one-out.  This is
    consistent with the test-set regime (all visits available) and gives a
    more stable per-patient signal.

    Parameters
    ----------
    X_train :
        Training-fold raw features (includes ``patient_id``, ``rater_id``,
        ``Index``, etc.).
    y_train :
        Training-fold target Series, aligned index with ``X_train``.
    X_val :
        Validation or test raw features.  When ``None``, only the
        training-fold transformed DataFrame is returned.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame | None]
        ``(X_train_fe, X_val_fe)`` — both containing only ``FEATURE_COLS``
        columns.  ``X_val_fe`` is ``None`` when ``X_val`` is ``None``.
    """
    # ---- Step 1: stateless features on both splits ----------------------
    X_tr = _stateless_features(X_train)
    X_v = _stateless_features(X_val) if X_val is not None else None

    # ---- Step 2: rater-bias estimation (train only) ---------------------
    rater_bias = _estimate_rater_bias(X_tr, y_train)

    X_tr = _apply_rater_correction(X_tr, rater_bias)
    if X_v is not None:
        X_v = _apply_rater_correction(X_v, rater_bias)

    # ---- Step 3: PK curve estimation (train only) -----------------------
    f_hat = _estimate_pk_curve(X_tr, y_train)

    X_tr = _apply_pk_correction(X_tr, f_hat)
    if X_v is not None:
        X_v = _apply_pk_correction(X_v, f_hat)

    # ---- Step 4: derived interaction features ---------------------------
    X_tr = _derived_features(X_tr)
    if X_v is not None:
        X_v = _derived_features(X_v)

    # ---- Step 5: patient full-group aggregates (no LOO) -----------------
    # Each split computed independently using only feature columns (no target).
    X_tr = _add_patient_features(X_tr)
    if X_v is not None:
        X_v = _add_patient_features(X_v)

    # ---- Step 6: visit index and relative position ----------------------
    X_tr["visit_index"] = _visit_index(X_tr)
    if X_v is not None:
        X_v["visit_index"] = _visit_index(X_v)

    # ---- Step 7: personalised trend features (requires slope/intercept) -
    X_tr = _trend_features(X_tr)
    if X_v is not None:
        X_v = _trend_features(X_v)

    # ---- Step 7b: lag features (chronological, within-split) ------------
    # Applied after visit_index so the sort order is stable.
    X_tr = _lag_features(X_tr)
    if X_v is not None:
        X_v = _lag_features(X_v)

    # ---- Step 8: align categorical dtypes across train / val ------------
    # Category columns must share the same dtype object (same categories)
    # so the model sees consistent integer codes.
    if X_v is not None:
        for col in ("gene_cat", "cohort_cat", "rater_id_cat"):
            X_v[col] = X_v[col].astype(X_tr[col].dtype)

    # ---- Step 9: select final feature columns ---------------------------
    X_tr_fe = X_tr[FEATURE_COLS].copy()
    X_v_fe = X_v[FEATURE_COLS].copy() if X_v is not None else None

    return X_tr_fe, X_v_fe
