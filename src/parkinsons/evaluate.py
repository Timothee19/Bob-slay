"""GroupKFold cross-validation strategy for the Parkinson's UPDRS dataset.

Patients appear ~8 times each in the training set (longitudinal data).
A plain KFold would leak patient-level information into the validation fold,
inflating scores. GroupKFold on ``patient_id`` ensures every row of a given
patient lands in the same fold.

The ``get_cv_folds`` helper materialises the 5 splits once and caches them to
``data/cv_folds.pkl`` so every experiment reuses exactly the same indices.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

from parkinsons import PROJECT_ROOT

# Five folds keeps each fold large enough (~8 900 rows) while giving five
# independent estimates of the RMSE.  The groups vector (patient_id) is
# supplied at evaluation time via the ``groups`` argument of skore.evaluate.
splitter = GroupKFold(n_splits=5)

_DEFAULT_CACHE = PROJECT_ROOT / "data" / "cv_folds.pkl"


def get_cv_folds(
    X: pd.DataFrame,
    y: pd.Series,
    groups: "pd.Series | None" = None,
    cache_path: Path | str | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Return the 5 GroupKFold splits, loading from cache if available.

    The splits are computed once from the full training data and saved to
    ``data/cv_folds.pkl``.  Subsequent calls load the cached file so that
    all experiments are evaluated on exactly the same held-out rows.

    Parameters
    ----------
    X :
        Full training feature DataFrame (must include ``patient_id``).
    y :
        Full training target Series aligned with ``X``.
    groups :
        Groups array for GroupKFold.  Defaults to ``X["patient_id"]``.
    cache_path :
        Path to the pickle cache.  Defaults to ``data/cv_folds.pkl``.

    Returns
    -------
    list[tuple[list[int], list[int]]]
        List of ``(train_indices, val_indices)`` tuples.
    """
    path = Path(cache_path) if cache_path is not None else _DEFAULT_CACHE

    if path.exists():
        with open(path, "rb") as f:
            folds = pickle.load(f)
        print(f"[CV] Loaded {len(folds)} folds from {path}")
        return folds

    if groups is None:
        groups = X["patient_id"]

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(X, y, groups=groups))

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(folds, f)
    print(f"[CV] Created and cached {len(folds)} folds -> {path}")

    # Sanity: print fold sizes
    for i, (tr, va) in enumerate(folds):
        print(f"  fold {i}: train={len(tr)}, val={len(va)}")

    return folds
