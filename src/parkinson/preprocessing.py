"""Imputation pipeline for the Parkinson true-OFF prediction task.

Handles the structured missingness in X_train / X_test following the
clinical logic agreed for this competition:

  - gene               : fill NaN → 'no_mutation'
  - age_at_diagnosis   : intra-patient propagation first, then age - 5.7
  - ledd               : intra-patient median → cohort median fallback
                         + has_ledd_missing flag when cohort fallback is used
  - time_since_intake_on / off : binary flag first, then intra-patient
                                 median → cohort median fallback
  - on / off           : binary flag first, then intra-patient median
                         → global median fallback

All statistics are computed from the data seen during fit() only,
so the transformer is safe to use in a sklearn Pipeline or cross-
validation loop without leaking test information.

Usage
-----
>>> from parkinson.preprocessing import MissingValueImputer
>>> imp = MissingValueImputer()
>>> X_train_clean = imp.fit_transform(X_train)
>>> X_test_clean  = imp.transform(X_test)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """Impute structured missingness in the Parkinson visit table.

    Parameters
    ----------
    age_diag_offset : float, default=5.7
        Fallback estimate: age_at_diagnosis ≈ age - age_diag_offset,
        used only for patients whose age_at_diagnosis is unknown on
        every visit in the training set.
    """

    def __init__(self, age_diag_offset: float = 5.7) -> None:
        self.age_diag_offset = age_diag_offset

    # ------------------------------------------------------------------
    # fit: learn all statistics from X (training data only)
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None) -> "MissingValueImputer":
        """Compute per-patient and per-cohort statistics from X."""

        # --- age_at_diagnosis: one value per patient (constant) -------
        self.age_at_diag_per_patient_: dict[str, float] = (
            X.dropna(subset=["age_at_diagnosis"])
            .groupby("patient_id")["age_at_diagnosis"]
            .first()
            .to_dict()
        )

        # --- ledd: median per patient, then per cohort ----------------
        self.ledd_per_patient_: dict[str, float] = (
            X.dropna(subset=["ledd"])
            .groupby("patient_id")["ledd"]
            .median()
            .to_dict()
        )
        self.ledd_per_cohort_: dict[str, float] = (
            X.dropna(subset=["ledd"])
            .groupby("cohort")["ledd"]
            .median()
            .to_dict()
        )

        # --- time_since_intake_on: median per patient, then per cohort
        self.tsi_on_per_patient_: dict[str, float] = (
            X.dropna(subset=["time_since_intake_on"])
            .groupby("patient_id")["time_since_intake_on"]
            .median()
            .to_dict()
        )
        self.tsi_on_per_cohort_: dict[str, float] = (
            X.dropna(subset=["time_since_intake_on"])
            .groupby("cohort")["time_since_intake_on"]
            .median()
            .to_dict()
        )

        # --- time_since_intake_off: median per patient, then per cohort
        self.tsi_off_per_patient_: dict[str, float] = (
            X.dropna(subset=["time_since_intake_off"])
            .groupby("patient_id")["time_since_intake_off"]
            .median()
            .to_dict()
        )
        self.tsi_off_per_cohort_: dict[str, float] = (
            X.dropna(subset=["time_since_intake_off"])
            .groupby("cohort")["time_since_intake_off"]
            .median()
            .to_dict()
        )

        # --- on: median per patient, then global median ---------------
        self.on_per_patient_: dict[str, float] = (
            X.dropna(subset=["on"])
            .groupby("patient_id")["on"]
            .median()
            .to_dict()
        )
        self.on_global_: float = float(X["on"].median())

        # --- off: median per patient, then global median --------------
        self.off_per_patient_: dict[str, float] = (
            X.dropna(subset=["off"])
            .groupby("patient_id")["off"]
            .median()
            .to_dict()
        )
        self.off_global_: float = float(X["off"].median())

        return self

    # ------------------------------------------------------------------
    # transform: apply imputation (never modifies the input in place)
    # ------------------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of X with missing values filled."""
        X = X.copy()

        # 1. gene -------------------------------------------------------
        X["gene"] = X["gene"].fillna("no_mutation")

        # 2. age_at_diagnosis -------------------------------------------
        # Intra-patient propagation first.
        mask_diag = X["age_at_diagnosis"].isna()
        if mask_diag.any():
            X.loc[mask_diag, "age_at_diagnosis"] = X.loc[mask_diag, "patient_id"].map(
                self.age_at_diag_per_patient_
            )
        # Remaining NaN → age - offset fallback.
        still_missing = X["age_at_diagnosis"].isna()
        if still_missing.any():
            X.loc[still_missing, "age_at_diagnosis"] = (
                X.loc[still_missing, "age"] - self.age_diag_offset
            )

        # 3. ledd -------------------------------------------------------
        # Flag: True when ledd is unknown for this patient across all visits.
        ledd_missing_mask = X["ledd"].isna()
        if ledd_missing_mask.any():
            # Try intra-patient median.
            X.loc[ledd_missing_mask, "ledd"] = X.loc[
                ledd_missing_mask, "patient_id"
            ].map(self.ledd_per_patient_)

        # Remaining NaN → cohort median.
        still_ledd = X["ledd"].isna()
        X["has_ledd_missing"] = still_ledd.astype(int)
        if still_ledd.any():
            X.loc[still_ledd, "ledd"] = X.loc[still_ledd, "cohort"].map(
                self.ledd_per_cohort_
            )

        # 4. time_since_intake_on ---------------------------------------
        X["has_on_timing"] = X["time_since_intake_on"].notna().astype(int)
        tsi_on_missing = X["time_since_intake_on"].isna()
        if tsi_on_missing.any():
            X.loc[tsi_on_missing, "time_since_intake_on"] = X.loc[
                tsi_on_missing, "patient_id"
            ].map(self.tsi_on_per_patient_)
        still_tsi_on = X["time_since_intake_on"].isna()
        if still_tsi_on.any():
            X.loc[still_tsi_on, "time_since_intake_on"] = X.loc[
                still_tsi_on, "cohort"
            ].map(self.tsi_on_per_cohort_)

        # 5. time_since_intake_off --------------------------------------
        X["has_off_timing"] = X["time_since_intake_off"].notna().astype(int)
        tsi_off_missing = X["time_since_intake_off"].isna()
        if tsi_off_missing.any():
            X.loc[tsi_off_missing, "time_since_intake_off"] = X.loc[
                tsi_off_missing, "patient_id"
            ].map(self.tsi_off_per_patient_)
        still_tsi_off = X["time_since_intake_off"].isna()
        if still_tsi_off.any():
            X.loc[still_tsi_off, "time_since_intake_off"] = X.loc[
                still_tsi_off, "cohort"
            ].map(self.tsi_off_per_cohort_)

        # 6. on ---------------------------------------------------------
        X["on_is_missing"] = X["on"].isna().astype(int)
        on_missing = X["on"].isna()
        if on_missing.any():
            X.loc[on_missing, "on"] = X.loc[on_missing, "patient_id"].map(
                self.on_per_patient_
            )
        still_on = X["on"].isna()
        if still_on.any():
            X.loc[still_on, "on"] = self.on_global_

        # 7. off --------------------------------------------------------
        X["off_is_missing"] = X["off"].isna().astype(int)
        off_missing = X["off"].isna()
        if off_missing.any():
            X.loc[off_missing, "off"] = X.loc[off_missing, "patient_id"].map(
                self.off_per_patient_
            )
        still_off = X["off"].isna()
        if still_off.any():
            X.loc[still_off, "off"] = self.off_global_

        return X
