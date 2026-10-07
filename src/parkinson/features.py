"""Feature engineering transformer for the Parkinson true-OFF prediction task.

All statistics that depend on training data (rater biases, patient-level
aggregates) are learned in fit() and applied in transform(), so the
transformer is safe to use in a sklearn Pipeline or GroupKFold loop.

New columns produced
--------------------
Temporal / visit structure:
  visit_rank          : 0-based visit index within the patient (by Index order)
  n_visits            : total number of visits for this patient in X
  years_since_first   : age - age at first visit in X
  duration            : age - age_at_diagnosis  (NaN kept as NaN when missing)
  age_diag_missing    : 1 if age_at_diagnosis was NaN, else 0

ledd:
  ledd_patient_flag   : 1 if this patient has NO ledd at all in X (cohort fallback)
  ledd_filled         : ledd with intra-patient-then-cohort median fill (NaN → filled)

Patient-level aggregates (mean, median, max, count) of on / off / ledd:
  on_mean, on_median, on_max, on_count
  off_mean, off_median, off_max, off_count
  ledd_mean, ledd_median, ledd_max

Lag / lead features (within patient, sorted by Index):
  on_prev, off_prev   : value at the previous visit
  on_next, off_next   : value at the next visit

Rater bias correction:
  rater_bias          : mean(off - target) per rater, learned from (X, y) in fit()
  off_rater_corrected : off - rater_bias  (NaN if off is NaN)
  on_rater_corrected  : on  - rater_bias  (NaN if on  is NaN)

Pharmacodynamic proxy (levodopa effect at measurement time):
  pd_effect_on   : ledd_filled * exp(-0.5 * time_since_intake_on)
  pd_effect_off  : ledd_filled * exp(-0.5 * time_since_intake_off)

Interaction:
  ledd_x_duration : ledd_filled * duration  (NaN propagated)
  off_minus_on    : off - on

Encoding:
  cohort is mapped A→0, B→1 (in-place)
  gene NaN → 'unknown', then cast to category
  rater_id cast to category
  age_at_diagnosis NaN kept (HGBR handles it), age_diag_missing flag added
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Pharmacodynamic decay constant (hours⁻¹).  Levodopa half-life ≈ 1.5 h
# → k = ln(2)/1.5 ≈ 0.46.  We use 0.5 as a round approximation.
_PD_K = 0.5


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Build all engineered features for experiment 04.

    Parameters
    ----------
    pd_k : float, default=0.5
        Decay constant for the pharmacodynamic proxy ledd*exp(-k*delay).
    """

    def __init__(self, pd_k: float = _PD_K) -> None:
        self.pd_k = pd_k

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self, X: pd.DataFrame, y: pd.Series | pd.DataFrame | None = None
    ) -> "FeatureEngineer":
        """Learn training-set statistics: rater biases, cohort ledd medians."""

        # Rater bias: mean(off - target), learned only when y is available.
        # y can be a Series or a DataFrame with a 'target' column.
        if y is not None:
            y_vals = y["target"] if isinstance(y, pd.DataFrame) else y
            df = X.copy()
            df["_target"] = y_vals.values
            df["_off_bias"] = df["off"] - df["_target"]
            self.rater_bias_: dict[str, float] = (
                df.dropna(subset=["off"])
                .groupby("rater_id")["_off_bias"]
                .mean()
                .to_dict()
            )
            self.rater_bias_global_: float = float(
                df.dropna(subset=["off"])["_off_bias"].mean()
            )
        else:
            self.rater_bias_ = {}
            self.rater_bias_global_ = 0.0

        # Cohort ledd median (fallback when a patient has no ledd at all).
        self.ledd_cohort_median_: dict[str, float] = (
            X.dropna(subset=["ledd"])
            .groupby("cohort")["ledd"]
            .median()
            .to_dict()
        )

        return self

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a fully-engineered copy of X in the SAME ROW ORDER as X.

        The input index is preserved throughout so that y.iloc[i] always
        corresponds to the output row i.  Operations that require chronological
        ordering within a patient (lag/lead) sort locally on a working copy and
        map results back by index before using them.
        """
        df = X.copy()
        # Preserve the original index so every assignment stays aligned with y.
        original_index = df.index

        # ----------------------------------------------------------
        # 1. Encoding — rater_id cast to category LAST (after arithmetic)
        # ----------------------------------------------------------
        df["cohort"] = df["cohort"].map({"A": 0, "B": 1}).astype("category")
        df["gene"] = df["gene"].fillna("unknown").astype("category")
        # rater_id stays as string/object until after rater_bias arithmetic (step 7)

        # ----------------------------------------------------------
        # 2. age_at_diagnosis: keep NaN, add flag
        # ----------------------------------------------------------
        df["age_diag_missing"] = df["age_at_diagnosis"].isna().astype("Int8")
        # duration: NaN propagated naturally when age_at_diagnosis is NaN
        df["duration"] = df["age"] - df["age_at_diagnosis"]

        # ----------------------------------------------------------
        # 3. Visit structure
        # Requires chronological order → sort a working copy, compute,
        # then reindex back to original order.
        # ----------------------------------------------------------
        sorted_df = df.sort_values(["patient_id", "Index"])
        df["visit_rank"] = (
            sorted_df.groupby("patient_id").cumcount().reindex(original_index)
        )
        df["n_visits"] = df.groupby("patient_id")["patient_id"].transform("count")
        df["years_since_first"] = (
            df["age"] - df.groupby("patient_id")["age"].transform("min")
        )

        # ----------------------------------------------------------
        # 4. ledd: intra-patient fill → cohort fill + flag
        # ----------------------------------------------------------
        has_any_ledd = df.groupby("patient_id")["ledd"].transform(
            lambda s: s.notna().any()
        )
        df["ledd_patient_flag"] = (~has_any_ledd).astype("Int8")

        patient_ledd_median = (
            df.dropna(subset=["ledd"])
            .groupby("patient_id")["ledd"]
            .median()
        )
        ledd_filled = df["ledd"].copy()
        mask = ledd_filled.isna()
        if mask.any():
            ledd_filled[mask] = df.loc[mask, "patient_id"].map(patient_ledd_median)
        still_missing = ledd_filled.isna()
        if still_missing.any():
            ledd_filled[still_missing] = df.loc[still_missing, "cohort"].map(
                self.ledd_cohort_median_
            )
        df["ledd_filled"] = ledd_filled

        # ----------------------------------------------------------
        # 5. Patient-level aggregates for on / off / ledd
        # groupby(...).transform() preserves the calling index — safe.
        # ----------------------------------------------------------
        for col in ("on", "off"):
            df[f"{col}_mean"]   = df.groupby("patient_id")[col].transform("mean")
            df[f"{col}_median"] = df.groupby("patient_id")[col].transform("median")
            df[f"{col}_max"]    = df.groupby("patient_id")[col].transform("max")
            df[f"{col}_count"]  = df.groupby("patient_id")[col].transform("count")

        for col in ("ledd",):
            df[f"{col}_mean"]   = df.groupby("patient_id")[col].transform("mean")
            df[f"{col}_median"] = df.groupby("patient_id")[col].transform("median")
            df[f"{col}_max"]    = df.groupby("patient_id")[col].transform("max")

        # ----------------------------------------------------------
        # 6. Lag / lead — must be computed in chronological order.
        # Compute on sorted_df (already sorted above), then reindex.
        # ----------------------------------------------------------
        for col, new_col, n in [
            ("on",  "on_prev",  1), ("off", "off_prev", 1),
            ("on",  "on_next", -1), ("off", "off_next", -1),
        ]:
            shifted = (
                sorted_df.groupby("patient_id")[col]
                .shift(n)
                .reindex(original_index)
            )
            df[new_col] = shifted

        # ----------------------------------------------------------
        # 7. Rater bias correction
        # ----------------------------------------------------------
        df["rater_bias"] = df["rater_id"].map(self.rater_bias_).fillna(
            self.rater_bias_global_
        )
        df["off_rater_corrected"] = df["off"] - df["rater_bias"]
        df["on_rater_corrected"]  = df["on"]  - df["rater_bias"]

        # ----------------------------------------------------------
        # 8. Pharmacodynamic proxies
        # ----------------------------------------------------------
        df["pd_effect_on"] = df["ledd_filled"] * np.exp(
            -self.pd_k * df["time_since_intake_on"]
        )
        df["pd_effect_off"] = df["ledd_filled"] * np.exp(
            -self.pd_k * df["time_since_intake_off"]
        )

        # ----------------------------------------------------------
        # 9. Simple interactions
        # ----------------------------------------------------------
        df["off_minus_on"]    = df["off"] - df["on"]
        df["ledd_x_duration"] = df["ledd_filled"] * df["duration"]

        # ----------------------------------------------------------
        # 10. Cast rater_id to category (now safe — arithmetic is done)
        # ----------------------------------------------------------
        df["rater_id"] = df["rater_id"].astype("category")

        # ----------------------------------------------------------
        # 11. Drop identifiers (not features)
        # ----------------------------------------------------------
        df = df.drop(columns=["Index", "patient_id"])

        return df
