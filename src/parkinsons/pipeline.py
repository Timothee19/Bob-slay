"""Parkinson's UPDRS regression pipeline.

Wraps the full feature engineering + HistGradientBoostingRegressor into a
sklearn-compatible estimator so it can be used with GroupKFold cross-validation
via a manual loop (the fold-aware feature engineering requires explicit access
to the train/validation split, which sklearn Pipelines do not natively support
for grouped data).

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


class ParkinsonsPipeline(BaseEstimator, RegressorMixin):
    """Full UPDRS regression pipeline: feature engineering + HGBR.

    Feature engineering is fold-aware: rater-bias correction, PK curve,
    and patient-level aggregates are all estimated from the ``X``/``y``
    passed to ``fit`` and applied consistently at ``predict`` time.

    This design means the estimator must always be used inside a
    ``GroupKFold`` loop (or ``cross_validate``/``skore.evaluate`` with
    a groups array) so that no patient appears in both training and
    validation folds.

    Parameters
    ----------
    max_iter : int
        Maximum number of boosting iterations.
    learning_rate : float
        Step size for each boosting iteration.
    min_samples_leaf : int
        Minimum number of samples per leaf — primary overfitting guard.
    l2_regularization : float
        L2 regularisation applied to leaf weights.
    max_leaf_nodes : int
        Maximum number of leaves per tree.
    early_stopping : bool
        Whether to use early stopping with an inner validation fraction.
    n_iter_no_change : int
        Patience for early stopping.
    validation_fraction : float
        Fraction of training data used for early stopping.
    random_state : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
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
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_leaf_nodes = max_leaf_nodes
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.random_state = random_state

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
            random_state=self.random_state,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight=None,
    ) -> "ParkinsonsPipeline":
        """Fit feature engineering and HGBR on the full training data.

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
        self.hgbr_ = self._make_hgbr()
        self.hgbr_.fit(X_fe.values, y.values)

        # Store fold-level fitting artefacts for predict
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
        check_is_fitted(self, "hgbr_")
        _, X_fe = build_features(self._X_train_fit, self._y_train_fit, X_val=X)
        return self.hgbr_.predict(X_fe.values)


def build_learner(
    max_iter: int = 500,
    learning_rate: float = 0.05,
    min_samples_leaf: int = 20,
    l2_regularization: float = 0.1,
    max_leaf_nodes: int = 63,
) -> ParkinsonsPipeline:
    """Instantiate the default Parkinson's UPDRS regression pipeline.

    Parameters
    ----------
    max_iter, learning_rate, min_samples_leaf, l2_regularization, max_leaf_nodes :
        Forwarded to ``ParkinsonsPipeline``.

    Returns
    -------
    ParkinsonsPipeline
        Unfitted pipeline ready for cross-validation or final fit.
    """
    return ParkinsonsPipeline(
        max_iter=max_iter,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        max_leaf_nodes=max_leaf_nodes,
    )
