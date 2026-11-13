"""Phase 3 - feature engineering (pandas wrapper around the shared runtime).

All logic lives in ``backend/healthrisk_runtime/features.py`` so training and
serving can never diverge. This module only adapts it to DataFrames and
exposes a scikit-learn compatible, *stateless* transformer.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config  # noqa: F401
from healthrisk_runtime.features import (  # noqa: F401  (re-exported for convenience)
    EXCLUDED_COLUMNS, FEATURE_DESCRIPTIONS, MODEL_CATEGORICAL_FEATURES, MODEL_FEATURES,
    MODEL_NUMERIC_FEATURES, TARGET, engineer_record, engineer_records,
)


def engineer_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raw encounter DataFrame -> engineered model-feature DataFrame (same row order)."""
    records = df.to_dict("records")
    out = pd.DataFrame(engineer_records(records), index=df.index)[MODEL_FEATURES]
    out[MODEL_NUMERIC_FEATURES] = out[MODEL_NUMERIC_FEATURES].astype(float)
    out[MODEL_CATEGORICAL_FEATURES] = out[MODEL_CATEGORICAL_FEATURES].astype(object)
    return out


class EncounterFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless transformer: learns nothing in ``fit`` (cannot leak across splits)."""

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        return engineer_frame(X)
