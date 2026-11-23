"""FastAPI application factory, following the professor's lifecycle architecture."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.schemas.prediction import CustomerData
from app.services.prediction_service import InputDataError, ModelUnavailableError, PredictionService
from src.config import ROOT

logger = logging.getLogger(__name__)


def create_app(artifact_root: Path = ROOT) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        application.state.service = PredictionService(artifact_root)
        application.state.service.load()
        yield
        application.state.service = None

    application = FastAPI(
        title="Credit Default Prediction API",
        description="Original UCI credit-default Random Forest, served through separate routes, schemas and services.",
        version="1.0.0", lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",") if origin.strip()],
        allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
    )

    @application.exception_handler(RequestValidationError)
    async def invalid_input(request: Request, error: RequestValidationError):
        errors = [{"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]} for item in error.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    @application.exception_handler(InputDataError)
    async def invalid_data(request: Request, error: InputDataError):
        return JSONResponse(status_code=422, content={"detail": error.detail})

    @application.exception_handler(ModelUnavailableError)
    async def unavailable(request: Request, error: ModelUnavailableError):
        return JSONResponse(status_code=503, content={"detail": str(error)})

    @application.exception_handler(Exception)
    async def unexpected(request: Request, error: Exception):
        logger.exception("Unexpected backend error processing %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "The backend could not complete this operation. Check the backend log and try again."})

    @application.get("/", tags=["Model"])
    def home():
        return {"message": "Credit Default Prediction API", "version": "1.0.0", "health": "/api/health", "docs": "/docs"}

    # Preserve the two original public routes as small transport aliases.
    @application.get("/health", include_in_schema=False)
    def legacy_health(request: Request):
        result = request.app.state.service.health()
        return JSONResponse(result, status_code=200 if result["model_loaded"] else 503)

    @application.post("/predict", include_in_schema=False)
    def legacy_predict(customer: CustomerData, request: Request):
        result = request.app.state.service.predict(customer.model_dump())
        # Original API used four-decimal default_probability; the canonical /api
        # endpoint retains full precision for frontend scenarios and analytics.
        return {key: result["default_probability_rounded"] if key == "default_probability" else result[key] for key in ("prediction", "prediction_label", "default_probability", "default_probability_percent", "threshold", "model_version")}

    application.include_router(router)
    return application


app = create_app()
