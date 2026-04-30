# Fraud Thesis Project

Monorepo for a fraud detection thesis project.

## Structure
- backend: FastAPI service for model training and inference
- frontend: React + TypeScript UI
- docs: roadmap, dataset notes, and defense prep

## Quick start (local)
Backend:
1) Create and activate a virtual environment
2) Install dependencies from backend/requirements.txt
3) Run: uvicorn app.main:app --reload --port 8000

Frontend:
1) Install dependencies in frontend
2) Run: npm run dev

Note: Datasets are not checked into this repo. See docs/datasets.md for options and links.
