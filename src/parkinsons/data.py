"""Data loading for the Parkinson's UPDRS regression dataset.

Loads X_train / y_train / X_test from the project's ``data/`` directory.
The ``patient_id`` column is preserved in the returned DataFrames so it can
be used as the GroupKFold groups array; callers that build feature matrices
should drop it after extracting the groups vector.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from parkinsons import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"


def load_dataset(
    data_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X_train, y_train)`` as pandas objects.

    Parameters
    ----------
    data_dir :
        Directory containing the CSV files. Defaults to ``<project_root>/data``.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``X_train`` retains ``patient_id`` for GroupKFold grouping.
        ``y_train`` is a float Series named ``target``.
    """
    d = Path(data_dir) if data_dir is not None else DATA_DIR
    X = pd.read_csv(d / "X_train.csv")
    y = pd.read_csv(d / "y_train.csv")["target"]
    return X, y


def load_test_dataset(
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Return ``X_test`` as a pandas DataFrame.

    Parameters
    ----------
    data_dir :
        Directory containing the CSV files. Defaults to ``<project_root>/data``.

    Returns
    -------
    pd.DataFrame
        Test features with ``Index`` column preserved for submission.
    """
    d = Path(data_dir) if data_dir is not None else DATA_DIR
    return pd.read_csv(d / "X_test.csv")
