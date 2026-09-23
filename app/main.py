"""Main application entrypoint configuring and running FastAPI."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app import __version__
from app.api.routes import router as api_router
from app.config import get_settings
from app.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context for startup and shutdown events."""
    settings = get_settings()
    setup_logging(log_level=settings.log_level)
    logger.info("Starting %s (v%s) in [%s] mode", settings.app_name, __version__, settings.app_env)
    yield
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """Application factory for FastAPI instance."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Production-grade Hybrid Search RAG (Dense + BM25 + Cross-Encoder Reranking)",
        lifespan=lifespan,
    )

    # Enable CORS for Streamlit frontend and local testing
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(api_router, prefix="/api/v1")
    # Also mount health directly at root for load balancers
    app.include_router(api_router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
