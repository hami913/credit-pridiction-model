"""Application inference owner, adapting the professor's service architecture.

The saved Random Forest and original credit domain functions remain unchanged.
No fitting, threshold optimization, or model serialization occurs in this service.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from pydantic import ValidationError
from sklearn.ensemble import RandomForestClassifier

from app.schemas.prediction import CustomerData
from src.config import (
    INTEGER_FEATURES, MAX_BATCH_ROWS, MAX_UPLOAD_BYTES, MODEL_FILENAME,
    RAW_FEATURES, REQUIRED_SKLEARN_VERSION, ROOT, TARGET,
)
from src.explainability import explain_default_prediction
from src.monitoring import build_monitoring_report
from src.preprocessing import prepare_model_input
from src.reporting import build_audit_report
from src.risk_scoring import assess_risk

logger = logging.getLogger(__name__)


class InputDataError(ValueError):
    def __init__(self, detail: Any):
        super().__init__(str(detail))
        self.detail = detail


class ModelUnavailableError(RuntimeError):
    pass


def safe_errors(error: ValidationError) -> list[dict]:
    """Omit raw input and exception objects, including non-JSON NaN/Infinity."""
    return [{"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]} for item in error.errors()]


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, np.generic):
        return native(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


class PredictionService:
    def __init__(self, root: Path = ROOT):
        self.root = Path(root).resolve()
        self.model = None
        self.pipeline = None
        self.artifact = None
        self.metadata = None
        self.metrics = None
        self.load_error = None

    def load(self) -> None:
        """Load a trusted local artifact once and validate it without modifying it."""
        self.model = self.pipeline = self.artifact = self.metadata = self.metrics = None
        self.load_error = None
        try:
            path = self.root / "models" / MODEL_FILENAME
            metadata = json.loads((self.root / "models/model_metadata.json").read_text(encoding="utf-8"))
            if sklearn.__version__ != REQUIRED_SKLEARN_VERSION:
                raise ValueError(f"This saved model requires scikit-learn {REQUIRED_SKLEARN_VERSION}.")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != metadata["model_sha256"]:
                raise ValueError("Model checksum does not match metadata.")
            artifact = joblib.load(path)
            if not isinstance(artifact, dict):
                raise ValueError("Model artifact must contain the original pipeline and metadata.")
            pipeline = artifact["pipeline"]
            classifier = pipeline.named_steps["classifier"]
            preprocessor = pipeline.named_steps["preprocessor"]
            if not isinstance(classifier, RandomForestClassifier):
                raise ValueError("The deployed credit model must be a RandomForestClassifier.")
            features = list(artifact["features"])
            expected_metadata = {key: native(value) for key, value in artifact.items() if key != "pipeline"}
            if metadata["artifact_metadata"] != expected_metadata:
                raise ValueError("Saved artifact metadata differs from the original model payload.")
            if any(metadata[key] != artifact[key] for key in ("features", "threshold", "target", "model_name", "model_version")):
                raise ValueError("Model metadata does not match the loaded artifact.")
            if (metadata["raw_features"] != RAW_FEATURES or features[:23] != RAW_FEATURES
                    or len(features) != 38 or len(set(features)) != 38
                    or list(preprocessor.feature_names_in_) != features
                    or list(pipeline.classes_) != [0, 1]
                    or metadata["classes"] != [0, 1]
                    or metadata["target"] != TARGET
                    or metadata["sklearn_version"] != REQUIRED_SKLEARN_VERSION
                    or metadata["algorithm"] != "RandomForestClassifier"):
                raise ValueError("Incompatible credit model schema.")
            processed = list(preprocessor.get_feature_names_out())
            if (metadata["processed_features"] != processed or len(processed) != 41
                    or classifier.n_features_in_ != 41
                    or metadata["raw_feature_count"] != 23
                    or metadata["engineered_feature_count"] != 38
                    or metadata["processed_feature_count"] != 41
                    or metadata["classifier_parameters"] != native(classifier.get_params(deep=False))):
                raise ValueError("Preprocessing or classifier metadata is incompatible.")
            cutoff = float(artifact["threshold"])
            if not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
                raise ValueError("Invalid saved decision threshold.")
            rules = metadata["risk_definitions"]
            expected_rules = {
                "binary": {"threshold": cutoff, "comparison": ">=", "positive_class": 1},
                "ui_band": {"low_below": cutoff * 0.6, "moderate_below": cutoff, "high_at_or_above": cutoff},
                "risk_score": {"low_below": 0.30, "medium_below": 0.60, "high_at_or_above": 0.60, "formula": "round(default_probability * 100, 2)"},
            }
            if any(rules[group].get(key) != expected for group, values in expected_rules.items() for key, expected in values.items()):
                raise ValueError("Metadata risk definitions differ from the original prediction rules.")
            self.artifact, self.model, self.pipeline, self.metadata = artifact, pipeline, pipeline, metadata
            self.threshold = cutoff
            self.model_features = features
            self.model_version = str(artifact["model_version"])
            original_report = metadata.get("historical_report")
            report_path = self.root / "reports/credit_risk_regulatory_report.json"
            if original_report is None and report_path.exists():
                original_report = json.loads(report_path.read_text(encoding="utf-8"))
            if original_report:
                names = {"accuracy": "Test_Accuracy", "precision": "Test_Precision", "recall": "Test_Recall", "f1": "Test_F1", "roc_auc": "Test_ROC_AUC", "pr_auc": "Test_PR_AUC"}
                self.metrics = {
                    "source": "Original uploaded reports/credit_risk_regulatory_report.json (historical results)",
                    "verified_in_this_integration": False,
                    "metrics": {name: original_report[key] for name, key in names.items()},
                    "report": original_report,
                }
            logger.info("event=model_loaded model=%s version=%s sha256=%s", artifact["model_name"], self.model_version, digest)
        except Exception as error:
            self.model = self.pipeline = self.artifact = self.metadata = self.metrics = None
            self.load_error = str(error)
            logger.exception("Model unavailable: check the trusted credit artifact, metadata, and pinned dependencies.")

    def require_model(self) -> None:
        if self.model is None:
            raise ModelUnavailableError("Credit model unavailable. Check backend/models and install backend/requirements.txt, then restart the backend.")

    def health(self) -> dict:
        loaded = self.model is not None
        return {
            "status": "healthy" if loaded else "unavailable", "model_loaded": loaded,
            "model_name": self.metadata["model_name"] if loaded else None,
            "model_version": self.model_version if loaded else None,
            "threshold": self.threshold if loaded else None,
        }

    def prepare(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        self.require_model()
        if not isinstance(raw_data, pd.DataFrame) or raw_data.empty:
            raise InputDataError("Supply at least one customer.")
        try:
            features = prepare_model_input(raw_data, self.model_features)
            if features.shape[1] != 38 or not np.isfinite(features.to_numpy(dtype=float)).all():
                raise ValueError("Feature engineering produced nonfinite values.")
            return features
        except (TypeError, ValueError, OverflowError) as error:
            raise InputDataError(str(error)) from error

    def _predict_engineered(self, features: pd.DataFrame) -> np.ndarray:
        self.require_model()
        try:
            probabilities = np.asarray(self.model.predict_proba(features)[:, 1], dtype=float)
        except (ValueError, OverflowError) as error:
            raise InputDataError("The numeric inputs are outside the model's supported range.") from error
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise RuntimeError("Model returned invalid probabilities.")
        return probabilities

    def predict_probabilities(self, raw_data: pd.DataFrame) -> np.ndarray:
        """Shared deployed inference path for API operations and offline validation."""
        self.require_model()
        if not isinstance(raw_data, pd.DataFrame) or raw_data.empty:
            raise InputDataError("Supply a nonempty customer DataFrame.")
        records, _, _ = self._validated_records(raw_data.to_dict(orient="records"), from_csv=True, skip_invalid=False)
        return self._predict_engineered(self.prepare(pd.DataFrame(records, columns=RAW_FEATURES)))

    def _summary(self, probability: float) -> dict:
        probability = float(probability)
        prediction = int(probability >= self.threshold)
        band = "High" if probability >= self.threshold else ("Moderate" if probability >= self.threshold * 0.6 else "Low")
        scored = assess_risk(probability)
        return {
            "default_probability": probability,
            "default_probability_rounded": round(probability, 4),
            "default_probability_percent": round(probability * 100, 2),
            "non_default_probability": 1.0 - probability,
            "prediction": prediction,
            "prediction_label": "DEFAULT / HIGH RISK" if prediction else "NO DEFAULT / LOWER RISK",
            "threshold": self.threshold, "model_version": self.model_version,
            "risk_band": band, "risk_index": scored["risk_index"], "risk_level": scored["risk_level"],
        }

    @staticmethod
    def _indicators(row: pd.Series) -> dict:
        names = {
            "avg_utilization": "AVG_UTILIZATION", "max_utilization": "MAX_UTILIZATION",
            "avg_delay": "AVG_DELAY", "max_delay": "MAX_DELAY", "delayed_months": "DELAY_MONTHS",
            "avg_bill": "AVG_BILL_AMT", "avg_payment": "AVG_PAY_AMT",
            "pay_to_bill": "PAY_TO_BILL_RATIO", "pay_to_limit": "PAY_TO_LIMIT_RATIO",
        }
        return {key: int(row[name]) if key in {"max_delay", "delayed_months"} else float(row[name]) for key, name in names.items()}

    def predict(self, customer: dict) -> dict:
        self.require_model()
        try:
            valid = CustomerData.model_validate(customer).model_dump()
        except ValidationError as error:
            raise InputDataError(safe_errors(error)) from error
        features = self.prepare(pd.DataFrame([valid], columns=RAW_FEATURES))
        probability = self._predict_engineered(features)[0]
        return {**self._summary(probability), "indicators": self._indicators(features.iloc[0]), "engineered_features": native(features.iloc[0].to_dict())}

    def scenarios(self, customer: dict, factors: list[float]) -> dict:
        self.require_model()
        records = []
        labels = {0.0: "No payments", 0.5: "Half", 1.0: "Current", 1.5: "+50%", 2.0: "Double"}
        for factor in factors:
            record = dict(customer)
            for i in range(1, 7):
                record[f"PAY_AMT{i}"] = float(record[f"PAY_AMT{i}"]) * factor
            records.append(record)
        probabilities = self.predict_probabilities(pd.DataFrame(records))
        return {"threshold": self.threshold, "model_version": self.model_version, "scenarios": [
            {"label": labels.get(factor, f"{factor:g}x payments"), "factor": factor, **self._summary(probability)}
            for factor, probability in zip(factors, probabilities)
        ]}

    def explain(self, customer: dict, top_n: int = 10) -> dict:
        features = self.prepare(pd.DataFrame([customer], columns=RAW_FEATURES))
        explanation = explain_default_prediction(self.model, features, top_n=top_n)
        return {
            "method": "SHAP TreeExplainer", "positive_class": 1,
            "default_probability": float(self._predict_engineered(features)[0]),
            "model_version": self.model_version,
            "features": native(explanation.to_dict(orient="records")),
            "note": "Positive SHAP contributions increase the model's default output. These are model associations, not causal effects.",
        }

    def feature_importance(self) -> dict:
        self.require_model()
        names = self.metadata["processed_features"]
        weights = self.model.named_steps["classifier"].feature_importances_
        entries = []
        aggregated = {name: 0.0 for name in self.model_features}
        for feature, importance in zip(names, weights):
            clean = feature.removeprefix("num__").removeprefix("cat__")
            entries.append({"feature": clean, "importance": float(importance)})
            base = next((name for name in ("SEX", "EDUCATION", "MARRIAGE") if feature.startswith(f"cat__{name}_")), clean)
            aggregated[base] += float(importance)
        return {
            "method": "Random Forest mean decrease in impurity (fitted model)",
            "features": sorted(entries, key=lambda row: row["importance"], reverse=True),
            "aggregated_features": sorted([{"feature": key, "importance": value} for key, value in aggregated.items()], key=lambda row: row["importance"], reverse=True),
        }

    @staticmethod
    def read_csv(content: bytes) -> pd.DataFrame:
        if not content:
            raise InputDataError("The CSV file is empty.")
        if len(content) > MAX_UPLOAD_BYTES:
            raise InputDataError("The CSV exceeds the 20 MiB upload limit.")
        try:
            text = content.decode("utf-8-sig")
            rows = csv.reader(io.StringIO(text), strict=True)
            header = next(rows)
            if not header or any(not name.strip() for name in header):
                raise InputDataError("CSV headers must be nonempty.")
            if len(header) != len(set(header)):
                raise InputDataError("Duplicate CSV column names are not allowed.")
            count = 0
            source_row_numbers = []
            previous_line = rows.line_num
            for row in rows:
                row_start = previous_line + 1
                previous_line = rows.line_num
                if not row or (len(row) == 1 and not row[0].strip()):
                    continue
                count += 1
                if len(row) != len(header):
                    raise InputDataError(f"CSV row {row_start} has a different number of fields from its header.")
                if count > MAX_BATCH_ROWS:
                    raise InputDataError(f"A maximum of {MAX_BATCH_ROWS:,} customers is supported per file.")
                source_row_numbers.append(row_start)
            if count == 0:
                raise InputDataError("CSV must contain at least one customer row.")
            # Object/string parsing preserves leading-zero identifiers and passthrough text.
            frame = pd.read_csv(io.StringIO(text), dtype=object, keep_default_na=False)
            if len(frame) != len(source_row_numbers):
                raise InputDataError("CSV row structure is ambiguous; remove empty or malformed records and retry.")
            frame.attrs["source_row_numbers"] = source_row_numbers
        except InputDataError:
            raise
        except (UnicodeDecodeError, csv.Error, StopIteration, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as error:
            raise InputDataError("Provide a valid UTF-8 CSV with a header and consistent columns.") from error
        missing = [name for name in RAW_FEATURES if name not in frame.columns]
        if missing:
            raise InputDataError(f"Missing input features: {missing}")
        return frame

    @staticmethod
    def _normalize_csv_record(record: dict) -> dict:
        normalized = {}
        for name in RAW_FEATURES:
            if name not in record:
                continue
            value = record[name]
            if isinstance(value, bool):
                normalized[name] = value
                continue
            try:
                number = float(value)
                normalized[name] = int(number) if name in INTEGER_FEATURES and math.isfinite(number) and number.is_integer() else number
            except (TypeError, ValueError, OverflowError):
                normalized[name] = value
        return normalized

    def _validated_records(self, records: list[dict], *, from_csv: bool, skip_invalid: bool, row_numbers: list[int] | None = None) -> tuple[list[dict], list[int], list[dict]]:
        if not records:
            raise InputDataError("Supply at least one customer.")
        if len(records) > MAX_BATCH_ROWS:
            raise InputDataError(f"A maximum of {MAX_BATCH_ROWS:,} customers is supported per batch.")
        valid, indices, errors = [], [], []
        for index, record in enumerate(records):
            raw = self._normalize_csv_record(record) if from_csv else {name: record[name] for name in RAW_FEATURES if name in record}
            try:
                valid.append(CustomerData.model_validate(raw).model_dump())
                indices.append(index)
            except ValidationError as error:
                errors.append({"row_number": row_numbers[index] if row_numbers is not None else index + 2, "errors": safe_errors(error)})
        if errors and not skip_invalid:
            raise InputDataError({"message": "Invalid customer rows; no rows were scored. Correct them or explicitly select skip-invalid mode.", "errors": errors})
        if not valid:
            raise InputDataError({"message": "No valid customer rows remain.", "errors": errors})
        return valid, indices, errors

    def predict_batch(self, records: list[dict], skip_invalid: bool = False, *, from_csv: bool = False, row_numbers: list[int] | None = None) -> dict:
        self.require_model()
        valid, indices, errors = self._validated_records(records, from_csv=from_csv, skip_invalid=skip_invalid, row_numbers=row_numbers)
        features = self.prepare(pd.DataFrame(valid, columns=RAW_FEATURES))
        probabilities = self._predict_engineered(features)
        reserved = {"row_number", "default_probability", "default_probability_percent", "prediction", "prediction_label", "risk_band", "risk_index", "risk_level", "delayed_months", "avg_utilization"}
        columns = set().union(*(row.keys() for row in records))
        renamed = {}
        for column in sorted(reserved & columns):
            replacement = "input__" + column
            while replacement in columns:
                replacement = "input__" + replacement
            renamed[column] = replacement
            columns.add(replacement)
        results = []
        for position, (index, probability) in enumerate(zip(indices, probabilities)):
            original = {renamed.get(key, key): native(value) for key, value in records[index].items()}
            summary = self._summary(probability)
            result = {**original, "row_number": row_numbers[index] if row_numbers is not None else index + 2, **{key: summary[key] for key in reserved if key in summary}, "delayed_months": int(features.iloc[position]["DELAY_MONTHS"]), "avg_utilization": float(features.iloc[position]["AVG_UTILIZATION"])}
            results.append(result)
        return {
            "total_rows": len(records), "scored_rows": len(results), "skipped_rows": len(errors),
            "threshold": self.threshold, "model_version": self.model_version,
            "results": results, "errors": errors, "renamed_input_columns": renamed,
        }

    def predict_csv(self, content: bytes, skip_invalid: bool = False) -> dict:
        frame = self.read_csv(content)
        return self.predict_batch(frame.to_dict(orient="records"), skip_invalid, from_csv=True, row_numbers=frame.attrs["source_row_numbers"])

    def _score_analytics_frame(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        records, _, _ = self._validated_records(frame.to_dict(orient="records"), from_csv=True, skip_invalid=False, row_numbers=frame.attrs.get("source_row_numbers"))
        numeric = pd.DataFrame(records, columns=RAW_FEATURES)
        return numeric, self._predict_engineered(self.prepare(numeric))

    @staticmethod
    def _labels(frame: pd.DataFrame, required: bool = True) -> np.ndarray | None:
        available = [name for name in ("actual_default", TARGET) if name in frame]
        if not available:
            if required:
                raise InputDataError(f"Observed outcomes are required: add actual_default or {TARGET} with 0/1 labels.")
            return None
        arrays = []
        for name in available:
            try:
                labels = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float)
            except (ValueError, TypeError) as error:
                raise InputDataError("Observed outcomes must contain only 0 and 1.") from error
            if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
                raise InputDataError("Observed outcomes must contain only 0 and 1.")
            arrays.append(labels.astype(int))
        if len(arrays) == 2 and not np.array_equal(arrays[0], arrays[1]):
            raise InputDataError("actual_default and the original target column disagree.")
        return arrays[0]

    def audit_csv(self, content: bytes, group_column: str = "SEX", min_group_size: int = 30, data_source: str = "uploaded_csv") -> dict:
        self.require_model()
        if group_column not in {"SEX", "EDUCATION", "MARRIAGE", "AGE_GROUP"}:
            raise InputDataError("Audit group must be SEX, EDUCATION, MARRIAGE, or AGE_GROUP.")
        frame = self.read_csv(content)
        labels = self._labels(frame)
        numeric, probabilities = self._score_analytics_frame(frame)
        # Original grouped-category semantics are also used in audit grouping.
        prepared = self.prepare(numeric)
        audit = pd.DataFrame({"actual_default": labels, "default_probability": probabilities})
        if group_column == "AGE_GROUP":
            # Exact right-closed bins from the original notebook's age audit.
            # Its historical "60+" label means strictly over 60, not age 60.
            audit[group_column] = pd.cut(numeric["AGE"], bins=[0, 30, 40, 50, 60, np.inf], labels=["≤30", "31-40", "41-50", "51-60", "60+"])
        else:
            audit[group_column] = prepared[group_column].astype(int)
        report = native(build_audit_report(audit, model_version=self.model_version, group_column=group_column, threshold=self.threshold, min_group_size=min_group_size, data_source=data_source))
        if group_column == "AGE_GROUP":
            report["age_group_definition"] = "Original notebook bins: age ≤30, 31–40, 41–50, 51–60, and >60 (historical label '60+')."
        return report

    def monitor_csv(self, reference: bytes, current: bytes) -> dict:
        self.require_model()
        reference_frame, current_frame = self.read_csv(reference), self.read_csv(current)
        labels = self._labels(current_frame, required=False)
        _, reference_probabilities = self._score_analytics_frame(reference_frame)
        _, current_probabilities = self._score_analytics_frame(current_frame)
        return native(build_monitoring_report(reference_probabilities, current_probabilities, actual_labels=labels, threshold=self.threshold, model_version=self.model_version))

    def template_csv(self) -> bytes:
        customer = {name: 0 for name in RAW_FEATURES}
        customer.update({"LIMIT_BAL": 200000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30})
        customer.update({f"BILL_AMT{i}": 20000 for i in range(1, 7)})
        customer.update({f"PAY_AMT{i}": 5000 for i in range(1, 7)})
        delayed = {**customer, "PAY_0": 2, "PAY_2": 2}
        return pd.DataFrame([{"Customer_ID": "EXAMPLE-001", **customer}, {"Customer_ID": "EXAMPLE-002", **delayed}]).to_csv(index=False).encode("utf-8")

    def sample_csv(self, limit: int = 100) -> bytes:
        path = self.root / "data/processed/UCI_Credit_Card.csv"
        if not path.exists():
            raise ModelUnavailableError("The restored UCI dataset is unavailable. See backend/data/README.md.")
        return pd.read_csv(path, nrows=limit).to_csv(index=False).encode("utf-8")
