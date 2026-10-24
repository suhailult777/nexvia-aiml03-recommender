"""Step 8: FastAPI service.

Run:
    C:/Python312/python.exe -m uvicorn src.serve:app --host 127.0.0.1 --port 8011

Endpoints:
    GET /health
    GET /recommend?user_id=1&k=10
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recommender import RecommenderService  # noqa: E402

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load once at startup: 1682 item vectors, a 64 dim embedding matrix and a
    # small MLP, so this is a couple of seconds, not a model server
    STATE["service"] = RecommenderService()
    yield
    STATE.clear()


app = FastAPI(title="AIML-03 Two-Stage Recommender", version="1.0.0",
              lifespan=lifespan)


def get_service() -> RecommenderService:
    svc = STATE.get("service")
    if svc is None:
        raise HTTPException(status_code=503, detail="service not ready")
    return svc


@app.get("/health")
def health() -> dict:
    return get_service().health()


@app.get("/recommend")
def recommend(user_id: int = Query(..., ge=1),
              k: int = Query(10, ge=1, le=50)) -> dict:
    return get_service().recommend(user_id=user_id, k=k)
