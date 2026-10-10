"""Post-processing utilities for Parkinson's UPDRS predictions.

Two smoothing strategies are available:

1. **Savitzky-Golay** (``smooth_patient_predictions``): applies a sliding
   polynomial filter over the visit sequence sorted by age.  Fast and
   simple, but treats visits as equally spaced — not ideal for irregular
   visit schedules (mean inter-visit gap ~1 year, std ~0.5).

2. **Age-based polynomial fit** (``smooth_by_age_poly``): fits a low-degree
   polynomial ``pred ~ age`` per patient and uses the fitted values as
   smoothed predictions.  Respects the irregular visit spacing because age
   is the x-axis.  Degree 2 (quadratic) captures both the linear trend and
   a gentle acceleration that characterises PD progression without
   overfitting sparse patient sequences.

Usage
-----
    preds_sg   = smooth_patient_predictions(preds_raw, X_val, window=5)
    preds_poly = smooth_by_age_poly(preds_raw, X_val, degree=2)
    preds_both = smooth_combined(preds_raw, X_val)          # SG then poly2
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def smooth_patient_predictions(
    predictions: np.ndarray,
    X_val: pd.DataFrame,
    window: int = 5,
    polyorder: int = 1,
) -> np.ndarray:
    """Apply per-patient Savitzky-Golay smoothing to predictions.

    Visits are sorted by ``age`` within each patient before smoothing and
    the result is mapped back to the original row order.  Patients with
    fewer rows than ``window`` are left unchanged (not enough signal to
    smooth meaningfully).

    Parameters
    ----------
    predictions :
        Raw model predictions aligned with ``X_val`` (same row order).
    X_val :
        Validation feature DataFrame containing ``patient_id`` and ``age``.
        Must be in the same row order as ``predictions``.
    window :
        Savitzky-Golay window length (must be odd and >= polyorder + 2).
        Defaults to 5.
    polyorder :
        Polynomial order for the filter.  ``polyorder=1`` is a weighted
        moving average; higher values preserve more curvature.

    Returns
    -------
    np.ndarray
        Smoothed predictions in the same row order as ``X_val``.
    """
    if window % 2 == 0:
        window += 1

    preds = predictions.copy()
    df = X_val[["patient_id", "age"]].copy()
    df["_pred"] = preds
    df["_orig_idx"] = np.arange(len(df))

    for pid, grp in df.groupby("patient_id"):
        if len(grp) < window:
            continue
        grp_sorted = grp.sort_values("age")
        smoothed = savgol_filter(
            grp_sorted["_pred"].values,
            window_length=window,
            polyorder=polyorder,
        )
        df.loc[grp_sorted.index, "_pred"] = smoothed

    return df["_pred"].values


def smooth_by_age_poly(
    predictions: np.ndarray,
    X_val: pd.DataFrame,
    degree: int = 2,
    min_visits_for_degree: int = 4,
) -> np.ndarray:
    """Fit a low-degree polynomial ``pred ~ age`` per patient and return fitted values.

    Unlike Savitzky-Golay (which operates on the visit sequence index),
    this smoother uses the actual age as the x-axis so it correctly handles
    irregular visit spacing.

    A degree-2 polynomial is recommended: it captures the known mild
    acceleration of PD motor progression while remaining well-conditioned
    for patients with only 4 visits.

    Patients with fewer than ``min_visits_for_degree`` visits fall back to
    a linear fit (degree 1) to avoid extrapolation artefacts.

    Parameters
    ----------
    predictions :
        Raw model predictions aligned with ``X_val`` (same row order).
    X_val :
        Validation feature DataFrame containing ``patient_id`` and ``age``.
    degree :
        Polynomial degree.  2 (default) works well for PD trajectories.
    min_visits_for_degree :
        Minimum visits required to use ``degree``; fewer rows use degree 1.

    Returns
    -------
    np.ndarray
        Smoothed predictions in the same row order as ``X_val``.
    """
    preds = predictions.copy().astype(float)
    df = X_val[["patient_id", "age"]].copy()
    df["_pred"] = preds

    for pid, grp in df.groupby("patient_id"):
        n = len(grp)
        if n < 3:
            continue
        ages = grp["age"].values
        p = grp["_pred"].values
        d = degree if n >= min_visits_for_degree else 1
        coeffs = np.polyfit(ages, p, d)
        fitted = np.polyval(coeffs, ages)
        df.loc[grp.index, "_pred"] = fitted

    return df["_pred"].values


def smooth_combined(
    predictions: np.ndarray,
    X_val: pd.DataFrame,
    sg_window: int = 5,
    poly_degree: int = 2,
) -> np.ndarray:
    """Apply SG smoothing then age-polynomial fit per patient.

    The two-stage approach:
    1. Savitzky-Golay removes local visit-to-visit noise.
    2. Age-polynomial fit enforces the long-range trajectory shape.

    The SG pass first reduces the influence of individual outlier visits
    on the polynomial coefficients, giving a more stable quadratic fit.

    Parameters
    ----------
    predictions :
        Raw model predictions aligned with ``X_val``.
    X_val :
        Feature DataFrame with ``patient_id`` and ``age``.
    sg_window :
        Savitzky-Golay window (odd integer, >= 3).  Default 5.
    poly_degree :
        Polynomial degree for the age-fit step.  Default 2.

    Returns
    -------
    np.ndarray
        Doubly-smoothed predictions in the same row order as ``X_val``.
    """
    preds_sg = smooth_patient_predictions(predictions, X_val, window=sg_window)
    return smooth_by_age_poly(preds_sg, X_val, degree=poly_degree)
