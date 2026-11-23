"""Transport and validation layer; credit predictions are delegated to one service."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from app.schemas.prediction import (
    BatchRequest, BatchResponse, CustomerData, ExplanationRequest, ExplanationResponse,
    HealthResponse, MetricsResponse, ModelInfo, PredictionResponse, ScenarioRequest, ScenarioResponse,
)
from app.services.prediction_service import PredictionService
from src.config import MAX_UPLOAD_BYTES

router = APIRouter(prefix="/api")


def service(request: Request) -> PredictionService:
    value: PredictionService = request.app.state.service
    value.require_model()
    return value


Service = Annotated[PredictionService, Depends(service)]


async def upload_bytes(file: UploadFile) -> bytes:
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "CSV upload exceeds the 20 MiB limit.")
    return content


@router.get("/health", response_model=HealthResponse, tags=["Model"])
def health(request: Request):
    result = request.app.state.service.health()
    return JSONResponse(result, status_code=200 if result["model_loaded"] else 503)


@router.get("/model/info", response_model=ModelInfo, tags=["Model"])
def model_info(value: Service):
    return value.metadata


@router.get("/model/metrics", response_model=MetricsResponse, tags=["Model"])
def model_metrics(value: Service):
    if value.metrics is None:
        raise HTTPException(503, "The original historical evaluation report is unavailable.")
    return value.metrics


@router.get("/model/feature-importance", tags=["Model"])
def feature_importance(value: Service) -> dict:
    return value.feature_importance()


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerData, value: Service):
    return value.predict(customer.model_dump())


@router.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def batch(inputs: BatchRequest, value: Service):
    return value.predict_batch(inputs.customers, inputs.skip_invalid)


@router.post("/predict/batch-csv", response_model=BatchResponse, tags=["Prediction"])
async def batch_csv(value: Service, file: Annotated[UploadFile, File()], skip_invalid: bool = False):
    content = await upload_bytes(file)
    return await run_in_threadpool(value.predict_csv, content, skip_invalid)


@router.post("/scenario", response_model=ScenarioResponse, tags=["Prediction"])
def scenario(inputs: ScenarioRequest, value: Service):
    return value.scenarios(inputs.customer.model_dump(), inputs.factors)


@router.post("/explain", response_model=ExplanationResponse, tags=["Prediction"])
def explain(inputs: ExplanationRequest, value: Service):
    try:
        return value.explain(inputs.customer.model_dump(), inputs.top_n)
    except ImportError as error:
        raise HTTPException(503, "SHAP is unavailable. Install backend/requirements.txt in the backend environment.") from error


@router.post("/analytics/audit", tags=["Analytics"])
async def audit(
    value: Service,
    file: Annotated[UploadFile, File()],
    group_column: Annotated[str, Form()] = "SEX",
    min_group_size: Annotated[int, Form(ge=1, le=50_000)] = 30,
    data_source: Annotated[str, Form(min_length=1, max_length=200)] = "uploaded_csv",
) -> dict:
    content = await upload_bytes(file)
    return await run_in_threadpool(value.audit_csv, content, group_column, min_group_size, data_source)


@router.post("/analytics/monitor", tags=["Analytics"])
async def monitor(value: Service, reference_file: Annotated[UploadFile, File()], current_file: Annotated[UploadFile, File()]) -> dict:
    reference, current = await upload_bytes(reference_file), await upload_bytes(current_file)
    return await run_in_threadpool(value.monitor_csv, reference, current)


@router.get("/data/template", tags=["Data"])
def template(value: Service):
    return Response(value.template_csv(), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="synthetic_customer_template.csv"'})


@router.get("/data/sample", tags=["Data"])
def sample(value: Service, limit: Annotated[int, Query(ge=1, le=1000)] = 100):
    return Response(value.sample_csv(limit), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="uci_credit_sample.csv"'})
