"""Credit-specific API contracts; no Iris fields or fitted preprocessing here."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]
PositiveAmount = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
PaymentAmount = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
RepaymentCode = Annotated[int, Field(strict=True, ge=-2, le=8)]


class CustomerData(BaseModel):
    """The original 23 inputs, including raw UCI codes normalized by original code."""

    model_config = ConfigDict(extra="forbid")
    LIMIT_BAL: PositiveAmount
    SEX: Annotated[int, Field(strict=True, ge=1, le=2)]
    EDUCATION: Annotated[int, Field(strict=True, ge=0, le=6)]
    MARRIAGE: Annotated[int, Field(strict=True, ge=0, le=3)]
    AGE: Annotated[int, Field(strict=True, ge=18, le=100)]
    PAY_0: RepaymentCode
    PAY_2: RepaymentCode
    PAY_3: RepaymentCode
    PAY_4: RepaymentCode
    PAY_5: RepaymentCode
    PAY_6: RepaymentCode
    BILL_AMT1: FiniteNumber
    BILL_AMT2: FiniteNumber
    BILL_AMT3: FiniteNumber
    BILL_AMT4: FiniteNumber
    BILL_AMT5: FiniteNumber
    BILL_AMT6: FiniteNumber
    PAY_AMT1: PaymentAmount
    PAY_AMT2: PaymentAmount
    PAY_AMT3: PaymentAmount
    PAY_AMT4: PaymentAmount
    PAY_AMT5: PaymentAmount
    PAY_AMT6: PaymentAmount


PredictionRequest = CustomerData


class PredictionResponse(BaseModel):
    default_probability: float
    default_probability_rounded: float
    default_probability_percent: float
    non_default_probability: float
    prediction: int
    prediction_label: str
    threshold: float
    model_version: str
    risk_band: Literal["Low", "Moderate", "High"]
    risk_index: float
    risk_level: Literal["Low", "Medium", "High"]
    indicators: dict[str, float | int]
    engineered_features: dict[str, float | int]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    threshold: float | None = None


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    model_name: str
    model_version: str
    algorithm: str
    target: str
    threshold: float
    raw_features: list[str]
    features: list[str]
    processed_features: list[str]
    raw_feature_count: int
    engineered_feature_count: int
    processed_feature_count: int
    classes: list[int]
    model_sha256: str
    sklearn_version: str
    classifier_parameters: dict[str, Any]
    risk_definitions: dict[str, Any]
    artifact_metadata: dict[str, Any]
    metadata_provenance: str


class MetricsResponse(BaseModel):
    source: str
    verified_in_this_integration: bool
    metrics: dict[str, float]
    report: dict[str, Any]


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: Annotated[list[dict[str, Any]], Field(min_length=1, max_length=50_000)]
    skip_invalid: bool = False


class BatchResponse(BaseModel):
    total_rows: int
    scored_rows: int
    skipped_rows: int
    threshold: float
    model_version: str
    results: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    renamed_input_columns: dict[str, str]


class ScenarioRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer: CustomerData
    factors: Annotated[list[Annotated[float, Field(strict=True, ge=0, le=10, allow_inf_nan=False)]], Field(min_length=1, max_length=20)] = [0.0, 0.5, 1.0, 1.5, 2.0]


class ScenarioResponse(BaseModel):
    threshold: float
    model_version: str
    scenarios: list[dict[str, Any]]


class ExplanationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer: CustomerData
    top_n: Annotated[int, Field(strict=True, ge=1, le=41)] = 10


class ExplanationResponse(BaseModel):
    method: str
    positive_class: int
    default_probability: float
    model_version: str
    features: list[dict[str, Any]]
    note: str
