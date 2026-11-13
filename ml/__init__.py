"""HealthRisk AI offline ML pipeline (training, evaluation, explainability, monitoring).

The shared, dependency-free runtime lives in ``backend/healthrisk_runtime`` so the
serving API can be deployed on its own. We put ``backend/`` on the import path so
training uses *exactly* the same feature-engineering code as inference.
"""
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
