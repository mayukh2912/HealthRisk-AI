"""Export a fitted (preprocessor + estimator) pipeline to the portable JSON format
evaluated by ``healthrisk_runtime.engine``. Parity is asserted at export time.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import output_feature_sources

UNKNOWN_SENTINEL = "__unseen_category__"


def _f32(v: float) -> float:
    return float(np.float32(v))


def export_preprocessor(pre: ColumnTransformer) -> Dict[str, Any]:
    names = list(pre.get_feature_names_out())
    sources = output_feature_sources(pre)
    numeric, categorical = [], []
    offset = 0
    for name, trans, cols in pre.transformers_:
        if name == "num":
            imp, sc = trans.named_steps["impute"], trans.named_steps["scale"]
            for i, c in enumerate(cols):
                numeric.append({"name": c, "index": offset + i, "median": float(imp.statistics_[i]),
                                "mean": float(sc.mean_[i]), "scale": float(sc.scale_[i])})
            offset += len(cols)
        elif name == "cat":
            enc = trans.named_steps["onehot"]
            width = 0
            for j, c in enumerate(cols):
                cats = list(enc.categories_[j])
                probe = np.array(cats + [UNKNOWN_SENTINEL], dtype=object).reshape(-1, 1)
                # transform each category through the fitted encoder for exact behaviour
                full = np.full((len(probe), len(cols)), cats[0], dtype=object)
                full[:, j] = probe[:, 0]
                onehot = enc.transform(full)
                start = offset + sum(_width(enc, k) for k in range(j))
                block = onehot[:, start - offset:start - offset + _width(enc, j)]
                mapping = {str(cat): (start + int(np.argmax(row)) if row.sum() > 0 else None)
                           for cat, row in zip(cats, block[:-1])}
                unk_row = block[-1]
                categorical.append({"name": c, "mapping": mapping,
                                    "unknown_index": start + int(np.argmax(unk_row)) if unk_row.sum() > 0 else None})
                width += _width(enc, j)
            offset += width
    return {"numeric": numeric, "categorical": categorical, "n_outputs": len(names),
            "output_names": names, "output_source": sources}


def _width(enc, j: int) -> int:
    inf = (getattr(enc, "infrequent_categories_", None) or [None] * len(enc.categories_))[j]
    n_inf = 0 if inf is None else len(inf)
    return len(enc.categories_[j]) - n_inf + (1 if n_inf else 0)


def _sk_tree(tree, scale: float = 1.0, proba: bool = False) -> Dict[str, List]:
    t = tree.tree_
    if proba:
        v = t.value[:, 0, :]
        value = (v[:, 1] / v.sum(axis=1)) * scale
    else:
        value = t.value[:, 0, 0] * scale
    feat = [int(f) if f >= 0 else -1 for f in t.feature]
    return {"feature": feat, "threshold": [float(x) for x in t.threshold],
            "left": [int(x) for x in t.children_left], "right": [int(x) for x in t.children_right],
            "value": [float(x) for x in value], "cover": [float(x) for x in t.weighted_n_node_samples]}


def _xgb_trees(booster) -> List[Dict[str, List]]:
    trees = []
    for dump in booster.get_dump(dump_format="json", with_stats=True):
        root = json.loads(dump)
        nodes: Dict[int, Dict] = {}

        def walk(n):
            nodes[n["nodeid"]] = n
            for ch in n.get("children", []):
                walk(ch)
        walk(root)
        order = sorted(nodes)
        pos = {nid: i for i, nid in enumerate(order)}
        t = {"feature": [], "threshold": [], "left": [], "right": [], "value": [], "cover": []}
        for nid in order:
            n = nodes[nid]
            if "leaf" in n:
                t["feature"].append(-1); t["threshold"].append(0.0)
                t["left"].append(-1); t["right"].append(-1)
                t["value"].append(float(n["leaf"]))
            else:
                t["feature"].append(int(str(n["split"]).lstrip("f")))
                # XGBoost compares float32 x < float32 split; round-trip the printed value to float32
                t["threshold"].append(_f32(float(n["split_condition"])))
                t["left"].append(pos[n["yes"]]); t["right"].append(pos[n["no"]])
                t["value"].append(0.0)
            t["cover"].append(float(n["cover"]))
        trees.append(t)
    return trees


def export_estimator(clf, Xt: np.ndarray) -> Dict[str, Any]:
    if isinstance(clf, LogisticRegression):
        return {"type": "linear", "output_space": "margin", "coef": [float(c) for c in clf.coef_[0]],
                "intercept": float(clf.intercept_[0]), "x_mean": [float(m) for m in Xt.mean(axis=0)]}
    if isinstance(clf, DecisionTreeClassifier):
        return {"type": "tree_ensemble", "output_space": "probability", "compare": "le", "float32": True,
                "base": 0.0, "trees": [_sk_tree(clf, 1.0, proba=True)]}
    if isinstance(clf, RandomForestClassifier):
        k = len(clf.estimators_)
        return {"type": "tree_ensemble", "output_space": "probability", "compare": "le", "float32": True,
                "base": 0.0, "trees": [_sk_tree(e, 1.0 / k, proba=True) for e in clf.estimators_]}
    if isinstance(clf, GradientBoostingClassifier):
        lr = clf.learning_rate
        trees = [_sk_tree(e[0], lr) for e in clf.estimators_]
        x0 = Xt[:1].astype(np.float32)
        base = float(clf.decision_function(x0)[0] - sum(lr * e[0].predict(x0)[0] for e in clf.estimators_))
        return {"type": "tree_ensemble", "output_space": "margin", "compare": "le", "float32": True,
                "base": base, "trees": trees}
    try:
        from xgboost import XGBClassifier
    except ImportError:  # pragma: no cover
        XGBClassifier = None
    if XGBClassifier is not None and isinstance(clf, XGBClassifier):
        import xgboost as xgb
        booster = clf.get_booster()
        trees = _xgb_trees(booster)
        x0 = Xt[:1]
        margin = float(booster.predict(xgb.DMatrix(x0), output_margin=True)[0])
        spec = {"type": "tree_ensemble", "output_space": "margin", "compare": "lt", "float32": True,
                "base": 0.0, "trees": trees}
        from healthrisk_runtime.engine import PortableModel  # local import: avoid cycles
        leaves = PortableModel._tree_sum_for_export(spec, list(map(float, x0[0])))
        spec["base"] = margin - leaves
        return spec
    raise TypeError(f"Unsupported estimator {type(clf).__name__}")


def export_pipeline(pipe: Pipeline, X_train_feat, meta: Dict[str, Any]) -> Dict[str, Any]:
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    Xt = pre.transform(X_train_feat)
    return {"schema_version": 1, **meta, "preprocessor": export_preprocessor(pre), "model": export_estimator(clf, Xt)}
