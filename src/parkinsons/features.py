"""Feature engineering for the Parkinson's UPDRS regression.

Three tiers of features:

1. **Stateless** — computed from a single row without any fold information:
   ``disease_duration``, missingness flags, categorical encoding.
   These are safe to apply before any CV split.

2. **Rater-bias correction** (fold-aware) — each rater applies a
   multiplicative bias b_r to their scores.  ``b_r`` is estimated as
   ``median(off / target)`` per rater on the *training fold only*, then
   applied to both train and validation folds.  On the test set the full
   training set is used for estimation.

3. **Patient LOO aggregates** (intra-split, leave-one-out) — for each
   row, the mean/std/slope of the *other* visits of the same patient
   within the same split.  This is leak-free because only features X
   are used (no target).  GroupKFold puts all visits of a patient in
   one fold so there is no cross-fold patient overlap; intra-split LOO
   is the only way to get per-patient context on cold-start patients.

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

def _encode_gene(series: pd.Series) -> pd.Series:
    """Ordinal-encode ``gene``, mapping NaN to the 'Unknown' sentinel."""
    filled = series.fillna("Unknown")
    mapping = {cat: i for i, cat in enumerate(GENE_CATEGORIES)}
    return filled.map(mapping).astype("float64")


def _encode_cohort(series: pd.Series) -> pd.Series:
    """Ordinal-encode ``cohort`` (A → 0, B → 1)."""
    mapping = {cat: i for i, cat in enumerate(COHORT_CATEGORIES)}
    return series.map(mapping).astype("float64")


def _stateless_features(X: pd.DataFrame) -> pd.DataFrame:
    """Compute features that do not require fold-level target information.

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
    # Fill the ~5 % missing with the global median before subtraction only
    age_diag_filled = age_diag.fillna(age_diag.median())
    df["disease_duration"] = (df["age"] - age_diag_filled).clip(lower=0)

    # --- missingness flags -----------------------------------------------
    df["ledd_missing_flag"] = df["ledd"].isna().astype("float64")
    df["on_missing_flag"] = df["on"].isna().astype("float64")
    df["off_missing_flag"] = df["off"].isna().astype("float64")
    df["time_on_missing_flag"] = df["time_since_intake_on"].isna().astype("float64")

    # --- categorical encoding --------------------------------------------
    df["gene_enc"] = _encode_gene(df["gene"])
    df["cohort_enc"] = _encode_cohort(df["cohort"])
    # sexM is already 0/1 integer — keep as float for HGBR
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
    # Guard against zero/near-zero divisors
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

    # Bin edges: 0, 0.5, 1.0, ..., 6.0
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

    # Clamp extreme values before fitting
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
# Patient LOO (leave-one-out) intra-split aggregates
# ---------------------------------------------------------------------------

def _loo_mean(df: pd.DataFrame, col: str) -> pd.Series:
    """Leave-one-out mean: for each row, mean of *other* visits of same patient.

    Uses only feature values (no target), so this is leak-free even when
    applied within the validation fold.  GroupKFold ensures all visits of
    a patient land in the same fold, so each patient's LOO context comes
    exclusively from their own visits in the same split.

    Parameters
    ----------
    df :
        DataFrame with ``patient_id`` and the target column ``col``.
    col :
        Name of the column to aggregate (e.g. ``off_corrected``).

    Returns
    -------
    pd.Series
        LOO mean values aligned with ``df.index``.  NaN when a patient has
        only one non-NaN observation (nothing left to average after LOO).
    """
    notna_mask = df[col].notna().astype(float)
    filled = df[col].fillna(0.0)

    pid = df["patient_id"]
    group_sum = pid.map(df.groupby("patient_id")[col].sum())
    group_count = pid.map(df.groupby("patient_id")[col].count())

    loo_sum = group_sum - filled * notna_mask
    loo_count = group_count - notna_mask

    result = np.where(loo_count > 0, loo_sum / loo_count, np.nan)
    return pd.Series(result, index=df.index)


def _loo_std(df: pd.DataFrame, col: str) -> pd.Series:
    """Leave-one-out std: std of *other* visits of same patient.

    Approximates using the full-patient std when the group has 3+ members.

    Parameters
    ----------
    df :
        DataFrame with ``patient_id`` and the target column ``col``.
    col :
        Column to aggregate.

    Returns
    -------
    pd.Series
        LOO std approximation aligned with ``df.index``.
    """
    pid = df["patient_id"]
    group_std = pid.map(df.groupby("patient_id")[col].std().fillna(0.0))
    group_count = pid.map(df.groupby("patient_id")[col].count())
    result = np.where(group_count >= 3, group_std, 0.0)
    return pd.Series(result, index=df.index)


def _patient_slope_intercept(
    df: pd.DataFrame, col: str
) -> tuple[pd.Series, pd.Series]:
    """Per-patient linear regression slope and intercept of ``col`` vs ``age``.

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
        NaN-patients fall back to the global median slope/intercept.
    """
    slopes: dict[str, float] = {}
    intercepts: dict[str, float] = {}

    for pid, grp in df.groupby("patient_id"):
        valid = grp[[col, "age"]].dropna()
        if len(valid) >= 3:
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


def _add_patient_loo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add LOO intra-split patient aggregate features to ``df``.

    All aggregates use only feature columns (no target), so they are
    leak-free.  For each row the aggregates summarise the *other* visits
    of the same patient within the same DataFrame.

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
    df["patient_off_mean"] = _loo_mean(df, "off_corrected")
    df["patient_off_std"] = _loo_std(df, "off_corrected")
    df["patient_on_mean"] = _loo_mean(df, "on_corrected")
    df["patient_slope"], df["patient_intercept"] = _patient_slope_intercept(
        df, "off_corrected"
    )
    df["patient_n_visits"] = df["patient_id"].map(
        df.groupby("patient_id")["age"].count().astype(float)
    )
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

    Parameters
    ----------
    df :
        DataFrame with ``on_corrected`` and ``off_corrected``.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``on_off_diff`` and ``on_off_mean`` added.
    """
    df = df.copy()
    df["on_off_diff"] = df["off_corrected"] - df["on_corrected"]
    df["on_off_mean"] = (df["off_corrected"] + df["on_corrected"]) / 2
    return df


# ---------------------------------------------------------------------------
# Final feature column list
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    # Raw numeric features (NaN handled natively by HGBR)
    "age",
    "disease_duration",
    "ledd",
    # Rater-corrected UPDRS scores
    "on_corrected",
    "off_corrected",
    "on_time_corrected",
    # Interaction features
    "on_off_diff",
    "on_off_mean",
    # Patient-level longitudinal aggregates
    "patient_off_mean",
    "patient_off_std",
    "patient_on_mean",
    "patient_slope",
    "patient_intercept",
    "patient_n_visits",
    "visit_index",
    # Time since intake (kept raw; HGBR will use it)
    "time_since_intake_on",
    "time_since_intake_off",
    # Rater bias factor (numeric representation of rater identity)
    "rater_bias_feature",
    # Encoded categoricals
    "gene_enc",
    "cohort_enc",
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

    # ---- Step 5: patient LOO aggregates (intra-split, leak-free) --------
    # Computed independently per split using only feature columns (no target).
    X_tr = _add_patient_loo_features(X_tr)
    if X_v is not None:
        X_v = _add_patient_loo_features(X_v)

    # ---- Step 6: visit index (can be done post-merge) -------------------
    X_tr["visit_index"] = _visit_index(X_tr)
    if X_v is not None:
        # For the validation/test set: rank within patient from their own rows
        X_v["visit_index"] = _visit_index(X_v)

    # ---- Step 7: select final feature columns ---------------------------
    X_tr_fe = X_tr[FEATURE_COLS].copy()
    X_v_fe = X_v[FEATURE_COLS].copy() if X_v is not None else None

    return X_tr_fe, X_v_fe
