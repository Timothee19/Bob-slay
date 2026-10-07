"""GroupKFold cross-validation strategy for the Parkinson's UPDRS dataset.

Patients appear ~8 times each in the training set (longitudinal data).
A plain KFold would leak patient-level information into the validation fold,
inflating scores. GroupKFold on ``patient_id`` ensures every row of a given
patient lands in the same fold.
"""

from __future__ import annotations

from sklearn.model_selection import GroupKFold

# Five folds keeps each fold large enough (~8 900 rows) while giving five
# independent estimates of the RMSE.  The groups vector (patient_id) is
# supplied at evaluation time via the ``groups`` argument of skore.evaluate.
splitter = GroupKFold(n_splits=5)
