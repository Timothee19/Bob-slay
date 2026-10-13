# %% [markdown]
# # Experiment 17 — Affine WLS trajectory + GBDT residual
#
# **Date:** 2025-01-10
#
# ## Architecture
#
# 1. **RaterAffineCalibrator**: per-rater Ridge affine regression on `off` and
#    `on` (Bateman-PK-corrected).  Fold-aware — fitted only on the training split.
# 2. **WLS trajectory reconstruction**: per-patient quadratic `target ~ Δage`
#    fitted by weighted least squares using the debiased pointwise estimate.
# 3. **GBDT residual model**: `HistGradientBoostingRegressor` on residual
#    `r = target − T̂`, with strict anti-overfitting hyper-parameters.
# 4. **Post-processing**: isotonic regression (PAVA) per patient (non-decreasing).

# %%
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import skore
import skrub
from scipy.interpolate import interp1d  # noqa: F401 (kept for future use)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.evaluate import get_cv_folds
from parkinsons.hub import load_skore_credentials

# ── Skore login ────────────────────────────────────────────────────────────────
cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)

DATA_DIR = PROJECT_ROOT / "data"
EXP_KEY  = "17_affine_wls_trajectory_residual"

# ── Pharmacokinetic parameters (Bateman biexponential) ─────────────────────────
_KA = 1.5
_KE = 0.46
_GLOBAL_OFF_RATIO = 0.85   # fallback when rater is absent from training


# ── Load data & fixed folds ────────────────────────────────────────────────────
X_train_raw, y_train = load_dataset(DATA_DIR)
X_test_raw           = load_test_dataset(DATA_DIR)
cv_folds             = get_cv_folds(X_train_raw, y_train)

print(f"Train: {X_train_raw.shape}   Test: {X_test_raw.shape}")
print(f"Folds loaded: {len(cv_folds)}")


# ============================================================
# § 1 — Stateless feature helpers (no target needed)
# ============================================================

def _bateman_concentration(
    ledd: pd.Series,
    time_since_intake: pd.Series,
    ka: float = _KA,
    ke: float = _KE,
) -> pd.Series:
    """Plasma levodopa via the Bateman biexponential equation."""
    safe_ledd = ledd.fillna(0.0)
    safe_time = time_since_intake.fillna(0.0)
    c = (safe_ledd * ka / (ka - ke)) * (
        np.exp(-ke * safe_time) - np.exp(-ka * safe_time)
    )
    mask = ledd.isna() | time_since_intake.isna()
    c[mask] = np.nan
    return c


def add_stateless_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add disease_duration, missingness flags, and Bateman PK concentration."""
    df = X.copy()
    aad = df["age_at_diagnosis"].fillna(df["age_at_diagnosis"].median())
    df["disease_duration"]     = (df["age"] - aad).clip(lower=0)
    df["ledd_missing_flag"]    = df["ledd"].isna().astype("float64")
    df["on_missing_flag"]      = df["on"].isna().astype("float64")
    df["off_missing_flag"]     = df["off"].isna().astype("float64")
    df["time_on_missing_flag"] = df["time_since_intake_on"].isna().astype("float64")
    df["pk_concentration_on"]  = _bateman_concentration(
        df["ledd"], df["time_since_intake_on"]
    )
    df["pk_concentration_off"] = _bateman_concentration(
        df["ledd"], df["time_since_intake_off"]
    )
    return df


# ============================================================
# § 2 — RaterAffineCalibrator (fold-aware)
# ============================================================

class RaterAffineCalibrator:
    """Per-rater affine debiasing: target ~ a_r + b_r * off (and on_pk).

    Falls back to global affine for unseen raters or raters with fewer than
    ``min_samples`` valid observations.
    """

    def __init__(self, min_samples: int = 5, alpha: float = 1.0):
        self.min_samples = min_samples
        self.alpha       = alpha

    def _fit_one(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Ridge affine fit; returns (intercept, slope)."""
        if len(x) < 2:
            return (0.0, 1.0)
        ridge = Ridge(alpha=self.alpha, fit_intercept=True)
        ridge.fit(x.reshape(-1, 1), y)
        return float(ridge.intercept_), float(ridge.coef_[0])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RaterAffineCalibrator":
        """Fit global fallback then per-rater models on training fold."""
        # Global fallback (using all rows with valid off)
        off_ok = X["off"].notna().values
        if off_ok.sum() >= 2:
            self._global_off = self._fit_one(X.loc[off_ok, "off"].values, y.values[off_ok])
        else:
            self._global_off = (0.0, 1.0 / _GLOBAL_OFF_RATIO)

        on_pk_col = X["pk_concentration_on"].values
        on_ok = ~np.isnan(on_pk_col)
        if on_ok.sum() >= 2:
            self._global_on = self._fit_one(on_pk_col[on_ok], y.values[on_ok])
        else:
            self._global_on = (0.0, 1.0)

        # Per-rater models (vectorised groupby)
        self._rater_off: dict[str, tuple[float, float]] = {}
        self._rater_on:  dict[str, tuple[float, float]] = {}

        for rater, grp in X.groupby("rater_id"):
            idx = grp.index
            y_r = y.loc[idx].values

            off_r  = grp["off"].values
            ok_off = ~np.isnan(off_r)
            if ok_off.sum() >= self.min_samples:
                self._rater_off[rater] = self._fit_one(off_r[ok_off], y_r[ok_off])

            on_pk_r = grp["pk_concentration_on"].values
            ok_on   = ~np.isnan(on_pk_r)
            if ok_on.sum() >= self.min_samples:
                self._rater_on[rater] = self._fit_one(on_pk_r[ok_on], y_r[ok_on])

        return self

    def debiased_estimate(
        self, X: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        """Vectorised computation of (ỹ, w) for every row in X.

        Weights: 1.0 if OFF present, 0.5 if only ON-PK present, 0.0 otherwise.
        """
        n       = len(X)
        y_hat   = np.full(n, np.nan)
        w       = np.zeros(n)
        raters  = X["rater_id"].values
        off_arr = X["off"].values
        on_arr  = X["pk_concentration_on"].values

        for i in range(n):
            r = raters[i]
            o = off_arr[i]
            p = on_arr[i]

            if not np.isnan(o):
                a, b    = self._rater_off.get(r, self._global_off)
                y_hat[i] = a + b * o
                w[i]     = 1.0
            elif not np.isnan(p):
                a, b    = self._rater_on.get(r, self._global_on)
                y_hat[i] = a + b * p
                w[i]     = 0.5

        return (
            pd.Series(y_hat, index=X.index, name="y_debiased"),
            pd.Series(w,     index=X.index, name="w_debiased"),
        )


# ============================================================
# § 3 — Per-patient WLS quadratic trajectory
# ============================================================

def fit_patient_wls_trajectory(
    X: pd.DataFrame,
    y_debiased: pd.Series,
    w: pd.Series,
    global_slope: float | None = None,
    min_valid: int = 3,
) -> tuple[pd.Series, float]:
    """Fit T̂(t) = α + β·Δage + γ·Δage² per patient via weighted least squares.

    Patients with < min_valid valid weighted observations have their linear
    slope regularised toward the cohort-wide median slope.
    """
    T_hat  = pd.Series(np.nan, index=X.index, name="T_hat")
    slopes: list[float] = []
    patient_fits: dict = {}

    # First pass – fit each patient independently
    for pid, grp in X.groupby("patient_id"):
        idx   = grp.index
        w_p   = w.loc[idx].values
        y_p   = y_debiased.loc[idx].values
        age_p = grp["age"].values

        valid = (w_p > 0) & ~np.isnan(y_p)
        n_v   = valid.sum()

        if n_v < 2:
            patient_fits[pid] = None
            continue

        mu  = age_p.mean()
        da  = age_p - mu
        W_d = np.sqrt(w_p[valid])

        if n_v >= min_valid:
            A   = np.column_stack([np.ones(n_v), da[valid], da[valid] ** 2])
            try:
                c, _, _, _ = np.linalg.lstsq(A * W_d[:, None], y_p[valid] * W_d, rcond=None)
                patient_fits[pid] = (mu, c)
                slopes.append(float(c[1]))
            except np.linalg.LinAlgError:
                patient_fits[pid] = None
        else:
            A   = np.column_stack([np.ones(n_v), da[valid]])
            try:
                c, _, _, _ = np.linalg.lstsq(A * W_d[:, None], y_p[valid] * W_d, rcond=None)
                patient_fits[pid] = (mu, np.array([c[0], c[1], 0.0]))
            except np.linalg.LinAlgError:
                patient_fits[pid] = None

    if global_slope is None:
        global_slope = float(np.median(slopes)) if slopes else 1.0

    # Second pass – predict and regularise sparse patients
    for pid, grp in X.groupby("patient_id"):
        idx   = grp.index
        age_p = grp["age"].values
        w_p   = w.loc[idx].values
        y_p   = y_debiased.loc[idx].values
        n_v   = int(((w_p > 0) & ~np.isnan(y_p)).sum())

        fit = patient_fits.get(pid)
        if fit is None:
            # Ultimate fallback: weighted mean of debiased estimates
            ok = (w_p > 0) & ~np.isnan(y_p)
            T_hat.loc[idx] = float(np.average(y_p[ok], weights=w_p[ok])) if ok.any() else y_debiased.mean()
            continue

        mu, c = fit
        if n_v < min_valid:
            # Regularise slope 50/50 toward global median
            c    = c.copy()
            c[1] = 0.5 * c[1] + 0.5 * global_slope

        da = age_p - mu
        T_hat.loc[idx] = c[0] + c[1] * da + c[2] * da ** 2

    return T_hat, global_slope


# ============================================================
# § 4 — GBDT feature matrix
# ============================================================

def build_gbdt_features(
    X: pd.DataFrame,
    y_debiased: pd.Series,
    T_hat: pd.Series,
) -> pd.DataFrame:
    """Build the feature matrix for the GBDT residual learner.

    Columns: disease_duration, missingness flags, cohort_enc, gene_enc,
    ledd_velocity, instant_residual (ỹ − T̂), off_raw, on_pk_raw, ledd_raw, age.
    """
    feats = pd.DataFrame(index=X.index)
    feats["disease_duration"]  = X["disease_duration"].values
    feats["ledd_missing_flag"] = X["ledd_missing_flag"].values
    feats["on_missing_flag"]   = X["on_missing_flag"].values
    feats["off_missing_flag"]  = X["off_missing_flag"].values

    cohort_map = {"A": 0.0, "B": 1.0}
    feats["cohort_enc"] = X["cohort"].map(cohort_map).astype("float64")

    gene_map = {"No Mutation": 0, "GBA+": 1, "LRRK2+": 2, "Other": 3, "Unknown": 4}
    feats["gene_enc"] = X["gene"].fillna("Unknown").map(gene_map).astype("float64")

    # Per-patient ledd velocity (slope ledd ~ age)
    ledd_vel = pd.Series(np.nan, index=X.index)
    for pid, grp in X.groupby("patient_id"):
        l_p = grp["ledd"].values
        a_p = grp["age"].values
        ok  = ~np.isnan(l_p)
        if ok.sum() >= 2:
            ledd_vel.loc[grp.index] = float(np.polyfit(a_p[ok], l_p[ok], 1)[0])
    feats["ledd_velocity"]    = ledd_vel.values
    feats["instant_residual"] = (y_debiased - T_hat).values
    feats["off_raw"]          = X["off"].values
    feats["on_pk_raw"]        = X["pk_concentration_on"].values
    feats["ledd_raw"]         = X["ledd"].values
    feats["age"]              = X["age"].values

    return feats


# ============================================================
# § 5 — Isotonic post-processing (PAVA, per patient, non-decreasing)
# ============================================================

def apply_isotonic_per_patient(
    preds: np.ndarray,
    X: pd.DataFrame,
) -> np.ndarray:
    """Enforce non-decreasing UPDRS trajectories via isotonic regression."""
    result = preds.copy().astype(float)
    df = X[["patient_id", "age"]].copy()
    df["_pred"] = result
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    for _pid, grp in df.groupby("patient_id"):
        if len(grp) < 2:
            continue
        gs = grp.sort_values("age")
        df.loc[gs.index, "_pred"] = iso.fit_transform(gs["age"].values, gs["_pred"].values)
    return df["_pred"].values


# ============================================================
# § 6 — Sklearn-compatible pipeline wrapper (for skore DataOps)
# ============================================================

# NOTE: The skore DataOps graph keeps `patient_id` as a feature column so the
# pipeline can group by patient during both fit and predict.  The pipeline
# never passes `patient_id` to the GBDT (it is used only for groupby).

class WlsGbdtPipeline(BaseEstimator, RegressorMixin):
    """End-to-end WLS trajectory + GBDT residual learner.

    Expects X to contain `patient_id` as a plain column (not the split groups).
    """

    def __init__(
        self,
        rater_min_samples: int = 5,
        rater_alpha: float = 1.0,
        gbdt_max_depth: int = 3,
        gbdt_max_leaf_nodes: int = 15,
        gbdt_min_samples_leaf: int = 60,
        gbdt_learning_rate: float = 0.02,
        gbdt_l2: float = 5.0,
        gbdt_max_iter: int = 500,
    ):
        self.rater_min_samples     = rater_min_samples
        self.rater_alpha           = rater_alpha
        self.gbdt_max_depth        = gbdt_max_depth
        self.gbdt_max_leaf_nodes   = gbdt_max_leaf_nodes
        self.gbdt_min_samples_leaf = gbdt_min_samples_leaf
        self.gbdt_learning_rate    = gbdt_learning_rate
        self.gbdt_l2               = gbdt_l2
        self.gbdt_max_iter         = gbdt_max_iter

    def _add_stateless(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add stateless features if not already present."""
        if "pk_concentration_on" not in X.columns:
            return add_stateless_features(X)
        return X

    def fit(self, X: pd.DataFrame, y) -> "WlsGbdtPipeline":
        if isinstance(y, np.ndarray):
            y = pd.Series(y, index=X.index)
        X = self._add_stateless(X)

        self.calibrator_ = RaterAffineCalibrator(
            min_samples=self.rater_min_samples, alpha=self.rater_alpha
        )
        self.calibrator_.fit(X, y)

        y_deb, w                = self.calibrator_.debiased_estimate(X)
        T_hat, self.global_slope_ = fit_patient_wls_trajectory(X, y_deb, w)

        r     = y.values - T_hat.values
        feats = build_gbdt_features(X, y_deb, T_hat)

        self.gbdt_ = HistGradientBoostingRegressor(
            max_depth=self.gbdt_max_depth,
            max_leaf_nodes=self.gbdt_max_leaf_nodes,
            min_samples_leaf=self.gbdt_min_samples_leaf,
            learning_rate=self.gbdt_learning_rate,
            l2_regularization=self.gbdt_l2,
            max_iter=self.gbdt_max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
        )
        self.gbdt_.fit(feats.values, r)
        self.feature_names_in_ = np.array(X.columns.tolist())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "gbdt_")
        X = self._add_stateless(X)
        y_deb, w = self.calibrator_.debiased_estimate(X)
        T_hat, _ = fit_patient_wls_trajectory(
            X, y_deb, w, global_slope=self.global_slope_
        )
        feats = build_gbdt_features(X, y_deb, T_hat)
        r_hat = self.gbdt_.predict(feats.values)
        y_hat = T_hat.values + r_hat
        return apply_isotonic_per_patient(y_hat, X)


# ============================================================
# § 7 — Manual CV loop (fold-level RMSE reporting)
# ============================================================

print("\n=== Cross-validation (5 folds) ===")

fold_rmse_raw: list[float] = []
fold_rmse_iso: list[float] = []

for fold_idx, (tr_idx, val_idx) in enumerate(cv_folds):
    X_tr_raw = X_train_raw.iloc[tr_idx].reset_index(drop=True)
    X_va_raw = X_train_raw.iloc[val_idx].reset_index(drop=True)
    y_tr     = y_train.iloc[tr_idx].reset_index(drop=True)
    y_va     = y_train.iloc[val_idx].reset_index(drop=True)

    X_tr = add_stateless_features(X_tr_raw)
    X_va = add_stateless_features(X_va_raw)

    calibrator = RaterAffineCalibrator(min_samples=5, alpha=1.0)
    calibrator.fit(X_tr, y_tr)

    y_deb_tr, w_tr = calibrator.debiased_estimate(X_tr)
    T_hat_tr, global_slope = fit_patient_wls_trajectory(X_tr, y_deb_tr, w_tr)

    y_deb_va, w_va = calibrator.debiased_estimate(X_va)
    T_hat_va, _    = fit_patient_wls_trajectory(
        X_va, y_deb_va, w_va, global_slope=global_slope
    )

    r_tr   = y_tr.values - T_hat_tr.values
    feats_tr = build_gbdt_features(X_tr, y_deb_tr, T_hat_tr)
    feats_va = build_gbdt_features(X_va, y_deb_va, T_hat_va)

    gbdt = HistGradientBoostingRegressor(
        max_depth=3, max_leaf_nodes=15, min_samples_leaf=60,
        learning_rate=0.02, l2_regularization=5.0,
        max_iter=500, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=20,
        random_state=42,
    )
    gbdt.fit(feats_tr.values, r_tr)

    r_hat_va  = gbdt.predict(feats_va.values)
    y_hat_raw = T_hat_va.values + r_hat_va
    y_hat_iso = apply_isotonic_per_patient(y_hat_raw, X_va_raw)

    rmse_raw = root_mean_squared_error(y_va.values, y_hat_raw)
    rmse_iso = root_mean_squared_error(y_va.values, y_hat_iso)
    fold_rmse_raw.append(rmse_raw)
    fold_rmse_iso.append(rmse_iso)
    print(f"  Fold {fold_idx + 1}: RMSE_raw={rmse_raw:.4f}  RMSE_iso={rmse_iso:.4f}")

mean_rmse_raw = float(np.mean(fold_rmse_raw))
mean_rmse_iso = float(np.mean(fold_rmse_iso))
print(f"\nMean RMSE (raw)     : {mean_rmse_raw:.4f}")
print(f"Mean RMSE (isotonic): {mean_rmse_iso:.4f}")


# ============================================================
# § 8 — Publish to Skore Hub via skrub DataOps graph
# ============================================================

print("\nBuilding skore evaluation graph …")

# Full stateless features — patient_id is kept as a column in the graph
X_full           = add_stateless_features(X_train_raw)
X_full["target"] = y_train.values

# Build a minimal visits_df: raw + stateless features + target
_KEEP = [
    "patient_id", "age", "off", "on", "ledd",
    "time_since_intake_on", "time_since_intake_off",
    "rater_id", "gene", "cohort",
    "disease_duration", "ledd_missing_flag", "on_missing_flag",
    "off_missing_flag", "time_on_missing_flag",
    "pk_concentration_on", "pk_concentration_off",
    "target",
]
visits_df  = X_full[[c for c in _KEEP if c in X_full.columns]].copy()
# patient_id may have been overwritten; restore from raw
visits_df["patient_id"] = X_train_raw["patient_id"].values

data_node   = skrub.var("visits", visits_df)
groups_node = data_node["patient_id"]

# patient_id is retained in X so WlsGbdtPipeline.predict can group by patient
X_op = data_node.drop(["target"], axis=1).skb.mark_as_X(
    cv=GroupKFold(n_splits=5),
    split_kwargs={"groups": groups_node},
)
y_op = data_node["target"].skb.mark_as_y()

pipeline_for_skore = WlsGbdtPipeline()
pred_node          = X_op.skb.apply(pipeline_for_skore, y=y_op)

print("Running skore.evaluate …")
report = skore.evaluate(pred_node, data={"visits": visits_df})
project.put(EXP_KEY, report)
print(f"Skore report pushed: {EXP_KEY}")
try:
    print(report.metrics.rmse())
except Exception:
    pass


# ============================================================
# § 9 — Full-train refit and test-set inference
# ============================================================

print("\nRefitting on full training set …")
pipeline_full = WlsGbdtPipeline()
pipeline_full.fit(X_full.drop(columns=["target"]), y_train)

X_test    = add_stateless_features(X_test_raw)
raw_preds = pipeline_full.predict(X_test)
print(f"Test predictions: {len(raw_preds)} rows  min={raw_preds.min():.2f}  max={raw_preds.max():.2f}")


# ============================================================
# § 10 — Save submission
# ============================================================

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
sub_path = submissions_dir / f"{EXP_KEY}.csv"

sub          = X_test_raw[["Index"]].copy()
sub["target"] = raw_preds
sub.to_csv(sub_path, index=False)
print(f"Submission saved: {sub_path}  ({len(sub)} rows)")
print(sub.head())
print("\nDone.")
