"""Parkinson's UPDRS regression pipeline.

Wraps the full feature engineering + model into a sklearn-compatible estimator
so it can be used with GroupKFold cross-validation via a manual loop (the
fold-aware feature engineering requires explicit access to the train/validation
split, which sklearn Pipelines do not natively support for grouped data).

Two backends are available:

- ``"hgbr"`` (default): ``HistGradientBoostingRegressor`` — native NaN support,
  categorical features via ``category`` dtype.
- ``"lgbm"``: ``LGBMRegressor`` — native NaN support, categorical features via
  ``category`` dtype, typically faster and slightly more accurate.

The ``ParkinsonsPipeline`` class implements ``fit`` / ``predict`` / ``fit_predict``
and handles the fold-aware feature engineering internally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.utils.validation import check_is_fitted

from parkinsons.features import FEATURE_COLS, build_features

# Indices of categorical columns in FEATURE_COLS for HGBR
_CAT_COLS = {"gene_cat", "cohort_cat", "rater_id_cat"}
_CAT_INDICES = [i for i, c in enumerate(FEATURE_COLS) if c in _CAT_COLS]


class ParkinsonsPipeline(BaseEstimator, RegressorMixin):
    """Full UPDRS regression pipeline: feature engineering + HGBR or LightGBM.

    Feature engineering is fold-aware: rater-bias correction, PK curve,
    and patient-level aggregates are all estimated from the ``X``/``y``
    passed to ``fit`` and applied consistently at ``predict`` time.

    This design means the estimator must always be used inside a
    ``GroupKFold`` loop so that no patient appears in both training and
    validation folds.

    Parameters
    ----------
    backend : str
        ``"hgbr"`` (default) or ``"lgbm"``.
    max_iter : int
        Maximum number of boosting iterations (``n_estimators`` for LightGBM).
    learning_rate : float
        Step size for each boosting iteration.
    min_samples_leaf : int
        Minimum number of samples per leaf — primary overfitting guard.
        Maps to ``min_child_samples`` for LightGBM.
    l2_regularization : float
        L2 regularisation applied to leaf weights (``reg_lambda`` for LightGBM).
    max_leaf_nodes : int
        Maximum number of leaves per tree (``num_leaves`` for LightGBM).
    early_stopping : bool
        Whether to use early stopping (HGBR only; LightGBM always uses it
        when ``early_stopping_rounds`` > 0).
    n_iter_no_change : int
        Patience for early stopping.
    validation_fraction : float
        Fraction of training data used for early stopping (HGBR only).
    random_state : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        backend: str = "hgbr",
        max_iter: int = 500,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.1,
        max_leaf_nodes: int = 63,
        early_stopping: bool = True,
        n_iter_no_change: int = 20,
        validation_fraction: float = 0.1,
        random_state: int | None = 42,
    ) -> None:
        self.backend = backend
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_leaf_nodes = max_leaf_nodes
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def _make_model(self):
        if self.backend == "lgbm":
            return self._make_lgbm()
        return self._make_hgbr()

    def _make_hgbr(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            max_leaf_nodes=self.max_leaf_nodes,
            early_stopping=self.early_stopping,
            n_iter_no_change=self.n_iter_no_change,
            validation_fraction=self.validation_fraction,
            max_bins=255,
            categorical_features=_CAT_INDICES,
            random_state=self.random_state,
        )

    def _make_lgbm(self):
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError(
                "lightgbm is not installed. Run: pip install lightgbm"
            ) from exc
        return LGBMRegressor(
            n_estimators=self.max_iter,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_samples_leaf,
            reg_lambda=self.l2_regularization,
            num_leaves=self.max_leaf_nodes,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=-1,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight=None,
    ) -> "ParkinsonsPipeline":
        """Fit feature engineering and model on the full training data.

        Parameters
        ----------
        X :
            Raw training features (must include ``patient_id``, ``rater_id``,
            and all original columns).
        y :
            Training target (UPDRS score).

        Returns
        -------
        self
        """
        X_fe, _ = build_features(X, y, X_val=None)
        self.model_ = self._make_model()

        if self.backend == "lgbm":
            cat_cols = [c for c in X_fe.columns if str(X_fe[c].dtype) == "category"]
            self.model_.fit(
                X_fe,
                y.values,
                categorical_feature=cat_cols,
            )
        else:
            self.model_.fit(X_fe.values, y.values)

        self._X_train_fit = X
        self._y_train_fit = y
        self.feature_names_in_ = FEATURE_COLS
        self.n_features_in_ = len(FEATURE_COLS)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Transform ``X`` using training-fold statistics and predict.

        Parameters
        ----------
        X :
            Raw features for the split to predict (validation or test).

        Returns
        -------
        np.ndarray
            Predicted UPDRS scores.
        """
        check_is_fitted(self, "model_")
        _, X_fe = build_features(self._X_train_fit, self._y_train_fit, X_val=X)
        if self.backend == "lgbm":
            return self.model_.predict(X_fe)
        return self.model_.predict(X_fe.values)

    # Alias kept for backward compatibility
    @property
    def hgbr_(self):
        return self.model_


def build_learner(
    backend: str = "hgbr",
    max_iter: int = 500,
    learning_rate: float = 0.05,
    min_samples_leaf: int = 20,
    l2_regularization: float = 0.1,
    max_leaf_nodes: int = 63,
) -> ParkinsonsPipeline:
    """Instantiate the Parkinson's UPDRS regression pipeline.

    Parameters
    ----------
    backend :
        ``"hgbr"`` (default) or ``"lgbm"``.
    max_iter, learning_rate, min_samples_leaf, l2_regularization, max_leaf_nodes :
        Forwarded to ``ParkinsonsPipeline``.

    Returns
    -------
    ParkinsonsPipeline
        Unfitted pipeline ready for cross-validation or final fit.
    """
    return ParkinsonsPipeline(
        backend=backend,
        max_iter=max_iter,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        max_leaf_nodes=max_leaf_nodes,
    )


def build_lgbm_learner(
    max_iter: int = 500,
    learning_rate: float = 0.05,
    min_samples_leaf: int = 30,
    l2_regularization: float = 1.0,
    max_leaf_nodes: int = 63,
) -> ParkinsonsPipeline:
    """Instantiate the pipeline with LightGBM backend.

    Conservative defaults (min_samples_leaf=30, l2_regularization=1.0) to
    avoid overfitting on the ~35 000-row training folds.

    Parameters
    ----------
    max_iter, learning_rate, min_samples_leaf, l2_regularization, max_leaf_nodes :
        Forwarded to ``ParkinsonsPipeline`` with LightGBM backend.

    Returns
    -------
    ParkinsonsPipeline
        Unfitted pipeline with LightGBM backend.
    """
    return ParkinsonsPipeline(
        backend="lgbm",
        max_iter=max_iter,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        max_leaf_nodes=max_leaf_nodes,
    )
