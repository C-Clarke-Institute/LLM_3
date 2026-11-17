import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import settings


logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.getLogger(__name__).info("Multi-Agent AI application starting.")
    yield
    logging.getLogger(__name__).info("Multi-Agent AI application stopping.")


app = FastAPI(
    title="Multi-Agent AI System",
    description=(
        "A LangGraph-powered multi-agent API with cryptocurrency, automotive, "
        "and general-purpose specialist agents."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health", tags=["Health"])
async def health_check():
    """Basic process health check."""
    return {"status": "healthy", "service": "multi-agent-ai"}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "detail": "Request validation failed.",
            "errors": exc.errors(),
        },
    )
