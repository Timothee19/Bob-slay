"""Post-processing utilities for Parkinson's UPDRS predictions.

Smoothing per-patient prediction sequences reduces the visit-to-visit noise
introduced by predicting each row independently.  The true motor trajectory
for a given patient is a smooth curve; applying a light filter along the
age-sorted visit sequence can reduce RMSE without touching the model.

Usage
-----
    preds_smooth = smooth_patient_predictions(preds_raw, X_val, window=3)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def smooth_patient_predictions(
    predictions: np.ndarray,
    X_val: pd.DataFrame,
    window: int = 3,
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
        Savitzky-Golay window length (must be odd and ≥ polyorder + 2).
        Defaults to 3.
    polyorder :
        Polynomial order for the filter.  ``polyorder=1`` is a weighted
        moving average; higher values preserve more curvature.

    Returns
    -------
    np.ndarray
        Smoothed predictions in the same row order as ``X_val``.
    """
    # Ensure window is odd and >= polyorder + 2
    if window % 2 == 0:
        window += 1

    preds = predictions.copy()
    df = X_val[["patient_id", "age"]].copy()
    df["_pred"] = preds
    df["_orig_idx"] = np.arange(len(df))

    for pid, grp in df.groupby("patient_id"):
        if len(grp) < window:
            # Not enough visits to smooth — skip
            continue
        # Sort by age to get chronological order
        grp_sorted = grp.sort_values("age")
        smoothed = savgol_filter(
            grp_sorted["_pred"].values,
            window_length=window,
            polyorder=polyorder,
        )
        # Write back in original index order
        df.loc[grp_sorted.index, "_pred"] = smoothed

    return df["_pred"].values
