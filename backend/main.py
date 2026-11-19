"""
VIGIL AI – FastAPI Application Entry Point
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from config import settings
from database import init_db
from routers import predict, history, dashboard, analysis, dataset, training
from services.inference import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("VIGIL AI starting up…")
    # Initialize database tables
    await init_db()
    # Pre-load model (non-blocking)
    try:
        loaded = load_model()
        if loaded:
            logger.info("Trained model loaded successfully.")
        else:
            logger.warning("Running in DEMO mode – no trained weights found.")
    except Exception as e:
        logger.error(f"Model load error: {e}")

    # Ensure upload directory exists
    Path("uploads").mkdir(exist_ok=True)

    yield

    logger.info("VIGIL AI shutting down.")


app = FastAPI(
    title="VIGIL AI – Video Violence Detection API",
    version=settings.app_version,
    description=(
        "REST API for VIGIL AI: an intelligent video violence detection and "
        "false positive analysis system built with FastAPI and PyTorch."
    ),
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(predict.router)
app.include_router(history.router)
app.include_router(dashboard.router)
app.include_router(analysis.router)
app.include_router(dataset.router)
app.include_router(training.router)


# ── Serve built React frontend ────────────────────────────────────────────────
from fastapi.responses import FileResponse
from fastapi import Request

_DIST = Path(__file__).parent.parent / "frontend" / "dist"

if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str, request: Request):
        # Serve API routes normally (they're registered above)
        file_path = _DIST / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(_DIST / "index.html"))


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "demo_mode": settings.demo_mode,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
