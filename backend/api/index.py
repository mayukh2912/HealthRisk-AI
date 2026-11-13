"""Alternative entrypoint (Vercel file-based function / `api/index.py` convention).

When the backend is deployed as its own Vercel project (Root Directory = backend/),
this exposes the same FastAPI app. The default Services deployment uses main:app.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import app  # noqa: E402,F401
