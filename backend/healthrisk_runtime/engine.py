"""Dependency-free inference engine for the exported HealthRisk AI model.

The training pipeline (``ml/export.py``) serialises the fitted scikit-learn
preprocessor and the selected estimator into a portable JSON document. This
module re-implements the *forward pass only* in pure Python so the serverless
API does not need numpy / scikit-learn / xgboost (keeping the deployment bundle
tiny and cold starts fast). Numerical parity with the original pipeline is
verified by the automated test-suite (``tests/test_export_parity.py``).

Explanations:
* tree ensembles -> exact path-dependent TreeSHAP (Lundberg et al., 2018)
* linear models  -> exact linear SHAP under feature independence
Contributions of one-hot columns are summed back to their source feature
(SHAP values are additive, so grouping preserves the decomposition).
"""
from __future__ import annotations

import json
import math
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .features import FEATURE_LABELS, engineer_record

_F32 = struct.Struct("f")


def _to_f32(x: float) -> float:
    try:
        return _F32.unpack(_F32.pack(x))[0]
    except OverflowError:  # pragma: no cover - defensive
        return math.copysign(float("inf"), x)


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


# ---------------------------------------------------------------------------
# TreeSHAP (Algorithm 2 of Lundberg, Erion & Lee 2018) - pure Python
# ---------------------------------------------------------------------------
def _extend(m: List[List[float]], pz: float, po: float, pi: int) -> List[List[float]]:
    m = [row[:] for row in m]
    l = len(m)
    m.append([pi, pz, po, 1.0 if l == 0 else 0.0])
    for i in range(l - 1, -1, -1):
        m[i + 1][3] += po * m[i][3] * (i + 1) / (l + 1)
        m[i][3] = pz * m[i][3] * (l - i) / (l + 1)
    return m


def _unwind(m: List[List[float]], i: int) -> List[List[float]]:
    m = [row[:] for row in m]
    l = len(m) - 1
    n = m[l][3]
    zi, oi = m[i][1], m[i][2]
    for j in range(l - 1, -1, -1):
        if oi != 0:
            t = m[j][3]
            m[j][3] = n * (l + 1) / ((j + 1) * oi)
            n = t - m[j][3] * zi * (l - j) / (l + 1)
        else:
            m[j][3] = (m[j][3] * (l + 1)) / (zi * (l - j))
    for j in range(i, l):
        m[j][0], m[j][1], m[j][2] = m[j + 1][0], m[j + 1][1], m[j + 1][2]
    return m[:l]


class _Tree:
    __slots__ = ("feature", "threshold", "left", "right", "value", "cover", "lt")

    def __init__(self, spec: Mapping[str, Any], lt: bool):
        self.feature: List[int] = spec["feature"]
        self.threshold: List[float] = spec["threshold"]
        self.left: List[int] = spec["left"]
        self.right: List[int] = spec["right"]
        self.value: List[float] = spec["value"]
        self.cover: List[float] = spec["cover"]
        self.lt = lt

    def _go_left(self, node: int, x: Sequence[float]) -> bool:
        v = x[self.feature[node]]
        return v < self.threshold[node] if self.lt else v <= self.threshold[node]

    def predict(self, x: Sequence[float]) -> float:
        node = 0
        while self.feature[node] >= 0:
            node = self.left[node] if self._go_left(node, x) else self.right[node]
        return self.value[node]

    def expected_value(self) -> float:
        # Cover-weighted mean of leaf values.
        tot, acc = 0.0, 0.0
        for n, f in enumerate(self.feature):
            if f < 0:
                acc += self.value[n] * self.cover[n]
                tot += self.cover[n]
        return acc / tot if tot else 0.0

    def shap(self, x: Sequence[float], phi: List[float]) -> None:
        def recurse(j: int, m: List[List[float]], pz: float, po: float, pi: int) -> None:
            m = _extend(m, pz, po, pi)
            if self.feature[j] < 0:
                for i in range(1, len(m)):
                    w = sum(r[3] for r in _unwind(m, i))
                    phi[int(m[i][0])] += w * (m[i][2] - m[i][1]) * self.value[j]
                return
            d = self.feature[j]
            hot, cold = (self.left[j], self.right[j]) if self._go_left(j, x) else (self.right[j], self.left[j])
            iz, io = 1.0, 1.0
            k = next((idx for idx in range(1, len(m)) if m[idx][0] == d), None)
            if k is not None:
                iz, io = m[k][1], m[k][2]
                m = _unwind(m, k)
            rj = self.cover[j]
            recurse(hot, m, iz * self.cover[hot] / rj, io, d)
            recurse(cold, m, iz * self.cover[cold] / rj, 0.0, d)

        recurse(0, [], 1.0, 1.0, -1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class PortableModel:
    def __init__(self, spec: Mapping[str, Any]):
        self.spec = spec
        self.version: str = spec["model_version"]
        self.name: str = spec["model_name"]
        pre = spec["preprocessor"]
        self.numeric = pre["numeric"]
        self.categorical = pre["categorical"]
        self.n_outputs: int = pre["n_outputs"]
        self.output_source: List[str] = pre["output_source"]
        self.source_features: List[str] = list(dict.fromkeys(self.output_source))
        mdl = spec["model"]
        self.kind: str = mdl["type"]
        self.output_space: str = mdl["output_space"]  # "margin" | "probability"
        if self.kind == "linear":
            self.coef: List[float] = mdl["coef"]
            self.intercept: float = mdl["intercept"]
            self.x_mean: List[float] = mdl["x_mean"]
            self.expected = self.intercept + sum(c * m for c, m in zip(self.coef, self.x_mean))
        elif self.kind == "tree_ensemble":
            lt = mdl["compare"] == "lt"
            self.float32 = bool(mdl.get("float32", True))
            self.base: float = mdl["base"]
            self.trees = [_Tree(t, lt) for t in mdl["trees"]]
            self.expected = self.base + sum(t.expected_value() for t in self.trees)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported model type {self.kind}")
        cal = spec.get("calibration") or {}
        self.cal_a: float = float(cal.get("a", 1.0))
        self.cal_b: float = float(cal.get("b", 0.0))
        self.calibrated: bool = bool(cal)
        self.threshold: float = float(spec["decision_threshold"])
        self.risk_bands: List[Dict[str, Any]] = spec["risk_bands"]

    @staticmethod
    def _tree_sum_for_export(model_spec: Mapping[str, Any], x: Sequence[float]) -> float:
        """Sum of leaf values for one transformed row (used to derive the XGBoost base margin)."""
        lt = model_spec["compare"] == "lt"
        xs = [_to_f32(v) for v in x] if model_spec.get("float32", True) else list(x)
        return sum(_Tree(t, lt).predict(xs) for t in model_spec["trees"])

    # -- preprocessing ------------------------------------------------------
    def transform(self, feats: Mapping[str, Any]) -> List[float]:
        x = [0.0] * self.n_outputs
        for col in self.numeric:
            v = feats.get(col["name"])
            if v is None or (isinstance(v, float) and math.isnan(v)):
                v = col["median"]
            x[col["index"]] = (float(v) - col["mean"]) / col["scale"]
        for col in self.categorical:
            v = feats.get(col["name"])
            v = "Missing" if v is None else str(v)
            idx = col["mapping"].get(v, col["unknown_index"])
            if idx is not None:
                x[idx] = 1.0
        return x

    # -- scoring ------------------------------------------------------------
    def raw_score(self, x: Sequence[float]) -> float:
        if self.kind == "linear":
            return self.intercept + sum(c * v for c, v in zip(self.coef, x))
        xs = [_to_f32(v) for v in x] if self.float32 else x
        return self.base + sum(t.predict(xs) for t in self.trees)

    def _calibrate(self, raw: float) -> float:
        s = raw if self.output_space == "margin" else _logit(raw)
        return _sigmoid(self.cal_a * s + self.cal_b)

    def risk_category(self, p: float) -> str:
        for band in self.risk_bands:
            if p < band["upper"]:
                return band["name"]
        return self.risk_bands[-1]["name"]

    def predict_features(self, feats: Mapping[str, Any]) -> Dict[str, Any]:
        x = self.transform(feats)
        p = self._calibrate(self.raw_score(x))
        return {
            "probability": p,
            "predicted_class": int(p >= self.threshold),
            "risk_category": self.risk_category(p),
        }

    def predict(self, raw_record: Mapping[str, Any]) -> Dict[str, Any]:
        return self.predict_features(engineer_record(raw_record))

    # -- explanations -------------------------------------------------------
    def explain(self, raw_record: Mapping[str, Any], top_k: int = 8) -> Dict[str, Any]:
        feats = engineer_record(raw_record)
        x = self.transform(feats)
        phi = [0.0] * self.n_outputs
        if self.kind == "linear":
            phi = [c * (v - m) for c, v, m in zip(self.coef, x, self.x_mean)]
        else:
            xs = [_to_f32(v) for v in x] if self.float32 else x
            for t in self.trees:
                t.shap(xs, phi)
        scale = self.cal_a if self.output_space == "margin" else 1.0
        grouped: Dict[str, float] = {f: 0.0 for f in self.source_features}
        for src, val in zip(self.output_source, phi):
            grouped[src] += val * scale
        contribs = [
            {
                "feature": f,
                "label": FEATURE_LABELS.get(f, f),
                "value": feats.get(f),
                "contribution": round(c, 6),
                "direction": "increases" if c > 0 else ("decreases" if c < 0 else "neutral"),
            }
            for f, c in grouped.items()
        ]
        contribs.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        raw = self.raw_score(x)
        base = self.expected * scale + (self.cal_b if self.output_space == "margin" else 0.0)
        units = "calibrated log-odds" if self.output_space == "margin" else "probability (uncalibrated)"
        top = contribs[:top_k]
        return {
            "method": "TreeSHAP (exact, path-dependent)" if self.kind == "tree_ensemble" else "Linear SHAP (exact)",
            "units": units,
            "base_value": base,
            "output_value": (self.cal_a * raw + self.cal_b) if self.output_space == "margin" else raw,
            "top_contributors": top,
            "increasing": [c for c in top if c["contribution"] > 0],
            "decreasing": [c for c in top if c["contribution"] < 0],
            "all_contributions": contribs,
        }

    def info(self) -> Dict[str, Any]:
        return {"model_name": self.name, "model_version": self.version}


def load_model(path: str | Path) -> PortableModel:
    with open(path, "r", encoding="utf-8") as fh:
        return PortableModel(json.load(fh))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
