"""Phase 3 - reusable, leakage-safe preprocessing.

* Numeric: median imputation + standardisation (StandardScaler). Scaling is
  required for the L2-regularised logistic regression and harmless for trees.
* Categorical: constant 'Missing' imputation + one-hot encoding with rare
  categories (< ``min_frequency`` training rows) pooled into an 'infrequent'
  column; unseen categories at inference map to that same column
  (``handle_unknown='infrequent_if_exist'``), so inference never fails on a new
  value.
* Patient-aware splitting: ``StratifiedGroupKFold`` keyed on ``patient_nbr`` so
  no patient appears on both sides of any split.

Everything is fitted inside a Pipeline on training folds only.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_engineering import MODEL_CATEGORICAL_FEATURES, MODEL_NUMERIC_FEATURES


def build_preprocessor(min_frequency: int = 100) -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                                 sparse_output=False, dtype=np.float64)),
    ])
    return ColumnTransformer(
        [("num", numeric, MODEL_NUMERIC_FEATURES), ("cat", categorical, MODEL_CATEGORICAL_FEATURES)],
        remainder="drop", verbose_feature_names_out=True,
    )


def patient_group_split(y: pd.Series, groups: pd.Series, n_folds: int = 5, random_state: int = 42
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Hold out ~1/n_folds of *patients* as the test set, stratified on the label."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(sgkf.split(np.zeros(len(y)), y, groups))
    return train_idx, test_idx


def group_cv(n_splits: int, random_state: int) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def output_feature_sources(pre: ColumnTransformer) -> list[str]:
    """For each transformed column, the engineered feature it came from."""
    sources = []
    for name, trans, cols in pre.transformers_:
        if name == "num":
            sources.extend(cols)
        elif name == "cat":
            enc: OneHotEncoder = trans.named_steps["onehot"]
            for col, n_out in zip(cols, _onehot_widths(enc)):
                sources.extend([col] * n_out)
    return sources


def _onehot_widths(enc: OneHotEncoder) -> list[int]:
    widths = []
    infreq = getattr(enc, "infrequent_categories_", [None] * len(enc.categories_))
    for cats, inf in zip(enc.categories_, infreq):
        n_inf = 0 if inf is None else len(inf)
        widths.append(len(cats) - n_inf + (1 if n_inf else 0))
    return widths
