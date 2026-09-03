import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.routes.exotel import router as exotel_router
from app.services.sarvam import close_sarvam_client
from app.services.rag_middleware import warm_rag_index

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Do not accept calls until the local PDF index is ready.  Otherwise the
    # first caller competes with index creation and exceeds the live RAG timeout.
    await warm_rag_index()
    try:
        yield
    finally:
        await close_sarvam_client()


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.include_router(exotel_router)


@app.get("/health", tags=["operations"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.exception_handler(Exception)
async def unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_exception")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
