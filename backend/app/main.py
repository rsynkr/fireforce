"""FastAPI entry point.

Run locally (from the repository root):
    uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import Settings
from app.state import AppState


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings.default()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.wildfire = AppState(resolved)
        yield

    app = FastAPI(
        title="Wildfire Research API",
        version="0.3.0",
        description=(
            "Local research prototype for comparing M1/M2/M3 next-day fire-mask "
            "predictions. Not an operational forecast."
        ),
        lifespan=lifespan,
    )
    app.include_router(router, prefix="/api")
    # Local development origin only. Production/preview must be served from one
    # of these origins or proxied; no wildcard CORS.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    return app


app = create_app()
