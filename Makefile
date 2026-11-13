.PHONY: setup data eda train test api web build
setup:        ## install Python (training+dev) and frontend deps
	pip install -r requirements-dev.txt && cd frontend && npm ci
data:         ## validate dataset + data-quality report
	python -m ml.data_ingestion
eda:
	python -m ml.eda
train:        ## full training run -> models/, reports/, backend/artifacts/
	python -m ml.train
test:
	pytest && cd frontend && npm test
api:
	cd backend && uvicorn main:app --reload --port 8000
web:
	cd frontend && npm run dev
build:
	cd frontend && npm run build
