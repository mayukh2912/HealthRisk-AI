# Deployment guide (Vercel)

The repository deploys as **one Vercel project with two services**, configured in `vercel.json`:

| Service | Root | Runtime | Routes |
|---|---|---|---|
| `web` | `frontend/` | Next.js | `/*` |
| `api` | `backend/` | Python 3.12 FastAPI (`main:app`) | `/api/*` (the full path is forwarded) |

> Vercel **Services** is marked *Beta* in Vercel's docs (checked November 2026). If your account cannot use it,
> follow [Option B](#option-b--two-vercel-projects-fallback) below.

## What was verified before packaging

- `vercel build` (CLI 65.15.0) completed successfully for both services with this exact `vercel.json`:
  - The Next.js service built 8 static routes.
  - The Python service resolved Python 3.12 from `backend/.python-version` and installed only `fastapi` and `pydantic`.
  - Its file map contains `artifacts/*.json` and `healthrisk_runtime/*`, about 108 KB of output.
- The production Next.js build and the FastAPI app were smoke-tested together locally with Playwright on all
  7 pages, a single prediction, a batch upload and a CSV download. There were no console errors.
- **Not done:** an actual deployment to Vercel, which needs your account. There is no live URL until you deploy.

## Option A: one project with Services (recommended)

### Via GitHub

1. Push this folder to a new GitHub repository (it already has a `.gitignore`; the raw dataset is excluded).
2. In Vercel, go to **Add New → Project**, import the repository and keep **Root Directory** at the repository root.
   Vercel reads `vercel.json` and detects both services. No build or install command overrides are needed.
3. Environment variables are optional for this layout:
   - `NEXT_PUBLIC_API_BASE_URL`: leave empty (same origin).
   - `APP_ENV=production`.
4. Click **Deploy**.

### Via CLI

```bash
npm i -g vercel        # CLI ≥ 48.1.8
cd HealthRisk-AI
vercel login
vercel                 # preview deployment
vercel --prod          # production deployment
```

## Option B: two Vercel projects (fallback)

1. **API project**:
   - Import the same repository with **Root Directory = `backend`**. FastAPI is auto-detected from `main.py`
     (or `pyproject.toml`'s `[tool.vercel] entrypoint`).
   - Set `ALLOWED_ORIGINS=https://<your-web-project>.vercel.app`.
2. **Web project**:
   - Import again with **Root Directory = `frontend`** and framework Next.js.
   - Set `NEXT_PUBLIC_API_BASE_URL=https://<your-api-project>.vercel.app`.
3. Delete or ignore the root `vercel.json` in this mode. Each project uses its own root.

## Limits and design choices

| Vercel constraint | How the app stays within it |
|---|---|
| Python bundle ≤ 500 MB | About 400 KB of artefacts and code, plus fastapi and pydantic. No numpy, scikit-learn or xgboost. |
| Request body ≤ 4.5 MB (413 above) | Batch CSV capped at 4 MB and 5,000 rows, checked in the browser and on the server |
| Function duration (Hobby 300 s) | `maxDuration: 60`. A single prediction takes milliseconds; a 5,000-row batch took about 0.6 s in local testing. |
| No persistent filesystem or session state | Model loaded read-only at cold start. No writes. Prediction log stored in the browser. |
| No training in functions | Training is offline only (`python -m ml.train`). |
| Large datasets | The raw CSV is never bundled. Only aggregates (`analytics.json`, about 80 KB) are shipped. |

## Local testing

```bash
# API
pip install -r requirements-dev.txt
cd backend && uvicorn main:app --reload --port 8000      # http://localhost:8000/api/docs

# Frontend (second terminal) - /api is proxied to :8000 in dev
cd frontend && npm ci && npm run dev                      # http://localhost:3000

# Production-like local run
cd frontend && API_PROXY_TARGET=http://127.0.0.1:8000 npm run build && API_PROXY_TARGET=http://127.0.0.1:8000 npm start

# Emulate Vercel locally (optional)
vercel build            # needs `vercel link` or a .vercel/project.json
vercel dev --local
```

## Production smoke-test checklist

Replace `$URL` with your deployment URL.

- [ ] `curl $URL/api/health` returns `{"status":"ok","model_loaded":true,"model_version":"1.0.0",...}`
- [ ] `curl $URL/api/model-info` shows `model_name` "Gradient Boosting" and test ROC-AUC ≈ 0.666
- [ ] `$URL/api/docs` opens Swagger
- [ ] `POST $URL/api/predict` with the example in `backend/schemas.py` returns 200 and a probability ≈ 0.218
- [ ] `POST $URL/api/predict` with `{"gender":"X"}` returns 422 with field-level messages
- [ ] Dashboard loads its KPI cards (99,340 encounters; 11.4%)
- [ ] Risk Assessment: "Load illustrative example" → Estimate shows probability, band, version, timestamp and factors
- [ ] Batch: download the template → upload it → 2 rows scored → Download CSV works
- [ ] Batch: a CSV with a `name` column is rejected
- [ ] Performance: the threshold slider updates the confusion matrix
- [ ] Explainability: the SHAP summary image loads (`/figures/shap_summary.png`) and "Explain example prediction" works
- [ ] Monitoring: the registry shows v1.0.0, baseline drift toggles, and a template upload with ≥ 30 rows returns PSI
- [ ] Mobile width (≈ 390 px): no horizontal scrolling and the menu opens

## Updating the model

```bash
python -m ml.train      # writes models/1.0.x and refreshes backend/artifacts + frontend/public/figures
pytest && (cd frontend && npm test)
git add -A && git commit -m "Retrain model" && git push   # Vercel redeploys
```
