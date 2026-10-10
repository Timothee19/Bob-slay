"""Push experiment 06 cross-validation report to skore hub.

Run after 06_lgbm_enriched_features.py has already generated the submission.
This script re-runs a 1-job CV to generate the skore report and pushes it.
"""
# %%
import numpy as np
import pandas as pd
import skore
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from lightgbm import LGBMRegressor

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset
from parkinsons.evaluate import get_cv_folds
from parkinsons.hub import load_skore_credentials
from parkinsons.features import build_features, FEATURE_COLS

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
folds = get_cv_folds(X_train, y_train)

# %%
EXTRA_COLS = [
    "off_corrected_prev2",
    "on_corrected_prev2",
    "off_lag_delta",
    "inter_visit_gap",
    "off_lag_delta_per_year",
    "off_corrected_next2",
    "patient_on_max",
    "patient_on_q75",
    "patient_on_std",
    "patient_on_slope",
    "ledd_off_ratio",
    "ledd_per_off_trend",
    "patient_off_iqr",
    "off_cummin",
    "on_time_corrected_prev",
]
ALL_COLS = FEATURE_COLS + EXTRA_COLS


def add_extra_features(X_fe: pd.DataFrame, X_raw: pd.DataFrame) -> pd.DataFrame:
    df = X_fe.copy()
    df["patient_id"] = X_raw["patient_id"].values

    df_sorted = df.sort_values(["patient_id", "age"])
    grp = df_sorted.groupby("patient_id")
    df_sorted["off_corrected_prev2"] = grp["off_corrected"].shift(2)
    df_sorted["on_corrected_prev2"]  = grp["on_corrected"].shift(2)
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
    for col in ["off_corrected_prev2", "on_corrected_prev2", "off_lag_delta",
                "inter_visit_gap", "off_lag_delta_per_year", "off_corrected_next2"]:
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
            on_slopes[pid] = float(np.polyfit(valid["age"].values,
                                              valid["on_corrected"].values, 1)[0])
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
    df_sorted2 = df_sorted2.loc[df.index]
    df["off_cummin"] = df_sorted2["off_cummin"].values

    df_sorted3 = df.sort_values(["patient_id", "age"])
    grp3 = df_sorted3.groupby("patient_id")
    df_sorted3["on_time_corrected_prev"] = grp3["on_time_corrected"].shift(1)
    df_sorted3 = df_sorted3.loc[df.index]
    df["on_time_corrected_prev"] = df_sorted3["on_time_corrected_prev"].values

    df = df.drop(columns=["patient_id"])
    return df


class EnrichedPipeline(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=2000, learning_rate=0.01,
                 min_child_samples=40, reg_lambda=2.0, num_leaves=127):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.reg_lambda = reg_lambda
        self.num_leaves = num_leaves

    def fit(self, X, y, sample_weight=None):
        X_fe, _ = build_features(X, y, X_val=None)
        X_fe = add_extra_features(X_fe, X)[ALL_COLS]
        self._X_fit = X
        self._y_fit = y
        cat_cols = [c for c in ALL_COLS if str(X_fe[c].dtype) == "category"]
        self._cat_dtypes = {c: X_fe[c].dtype for c in cat_cols}
        self.model_ = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            reg_lambda=self.reg_lambda,
            num_leaves=self.num_leaves,
            n_jobs=-1, random_state=42, verbose=-1,
        )
        self.model_.fit(X_fe, y.values if hasattr(y, "values") else y,
                        categorical_feature=cat_cols)
        self.feature_names_in_ = ALL_COLS
        self.n_features_in_ = len(ALL_COLS)
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        _, X_fe = build_features(self._X_fit, self._y_fit, X_val=X)
        X_fe = add_extra_features(X_fe, X)[ALL_COLS]
        for col, dtype in self._cat_dtypes.items():
            X_fe[col] = X_fe[col].astype(dtype)
        return self.model_.predict(X_fe)


# %%
report = skore.CrossValidationReport(
    EnrichedPipeline(),
    X=X_train, y=y_train, splitter=folds, n_jobs=1,
)
project.put("06_lgbm_enriched_features", report)
print("Report saved: 06_lgbm_enriched_features")
print(report.metrics.rmse())
