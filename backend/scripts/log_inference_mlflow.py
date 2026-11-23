"""Opt-in MLflow logging of an existing read-only RF inference report."""

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Defaults to latest real_data_inference_*.json")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    args = parser.parse_args()
    candidates = list((ROOT / "reports").glob("real_data_inference_*.json"))
    report_path = args.report or (max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None)
    if report_path is None:
        parser.error("No inference report found. Run python -m scripts.verify_real_inference first.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    model_path = ROOT / "models/credit_default_model_v1.joblib"
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if report.get("model_sha256", model_hash) != model_hash:
        raise ValueError("The report belongs to a different model artifact.")

    # MLflow is an optional research dependency; never imported by the application.
    import mlflow

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("Credit_Risk_Inference_Verification")
    with mlflow.start_run(run_name="Random_Forest_v1_Inference") as run:
        mlflow.log_params({
            "model_version": str(report["model_version"]),
            "feature_count": int(report["input_features"]),
            "threshold": float(report["threshold"]),
        })
        mlflow.log_metrics({
            "total_records": report["total_records"],
            "mean_default_probability": report["average_default_probability"],
            "predicted_defaults": report["predicted_default_count"],
            "predicted_nondefaults": report["predicted_nondefault_count"],
        })
        mlflow.set_tags({
            "model_type": "Random Forest",
            "model_sha256": model_hash,
            "run_type": "inference_verification",
            "evaluation_status": "not_independently_evaluated",
        })
        mlflow.log_artifact(str(report_path))
        print(f"MLflow inference report logged. Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
