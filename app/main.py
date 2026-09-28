"""PM-AJAY Virtual Livelihood Assistant — FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.routes.exotel import router as exotel_router
from app.services.bhashini import close_bhashini_client
from app.services.sarvam import close_sarvam_client
from app.services.rag_middleware import close_groq_client, warm_rag_index
from app.services.ingestion import auto_initialize_vector_db

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 1. Auto-initialize ChromaDB vector store with PM-AJAY livelihoods data.
    #    If the collection already exists and is populated, this is a fast no-op.
    await auto_initialize_vector_db()
    # 2. Keep rag_middleware warm (no-op now, kept for compatibility).
    await warm_rag_index()
    try:
        yield
    finally:
        await close_groq_client()
        await close_sarvam_client()
        await close_bhashini_client()


app = FastAPI(
    title="Kaushal Vaani - Virtual Livelihood Assistant",
    description="AI voice assistant for SC beneficiaries and livelihood seekers (కౌశల్ వాణి / कौशल वाणी).",
    version="2.0.0",
    lifespan=lifespan,
)
app.include_router(exotel_router)


@app.get("/health", tags=["operations"])
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "Kaushal Vaani - Virtual Livelihood Assistant"}


@app.exception_handler(Exception)
async def unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_exception")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
