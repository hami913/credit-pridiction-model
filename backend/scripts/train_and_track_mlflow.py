"""Optional future research: train and log a separate candidate, never deploy it.

This script is not run by application startup, installation, or default tests.
The shipped Random Forest is never fitted, replaced, or registered by it.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.train import train_candidate


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Labeled local dataset for an explicitly requested research experiment")
    parser.add_argument("--output-dir", type=Path, help="A new candidate directory")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"Dataset does not exist: {args.csv}")
    output = args.output_dir or ROOT / "models/candidates" / (
        "research_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
    )
    output = output.resolve()
    if output == (ROOT / "models").resolve():
        parser.error("Use a separate candidate subdirectory, not the deployed models directory.")
    if output.exists() and any(output.iterdir()):
        parser.error("Candidate output directory must be new or empty.")

    import mlflow

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("Credit_Risk_Real_Training_Comparison")
    deployed_path = ROOT / "models/credit_default_model_v1.joblib"
    deployed_hash = hashlib.sha256(deployed_path.read_bytes()).hexdigest()
    dataset_hash = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    result = train_candidate(pd.read_csv(args.csv), output, folds=args.folds, iterations=args.iterations, seed=42)
    if hashlib.sha256(deployed_path.read_bytes()).hexdigest() != deployed_hash:
        raise RuntimeError("The deployed model changed unexpectedly.")
    with mlflow.start_run(run_name="UCI_3_Algorithm_Candidate_Comparison") as parent:
        mlflow.log_params({
            "dataset_rows": result["dataset_rows"],
            "train_rows": result["train_rows"],
            "heldout_test_rows": result["heldout_test_rows"],
            "feature_count": result["feature_count"],
            "cv_folds": args.folds,
            "iterations_per_algorithm": args.iterations,
            "random_seed": 42,
        })
        mlflow.set_tags({
            "run_type": "real_training_comparison",
            "training_executed": "true",
            "selection_metric": result["selection_metric"],
            "selected_algorithm": result["selected_algorithm"],
            "dataset_sha256": dataset_hash,
            "deployment_status": "candidate_only",
        })
        for record in result["comparison"]:
            with mlflow.start_run(run_name=record["algorithm"], nested=True):
                mlflow.log_param("algorithm", record["algorithm"])
                mlflow.log_params(record["best_parameters"])
                mlflow.log_metrics({"cv_average_precision": record["cv_average_precision"], "cv_roc_auc": record["cv_roc_auc"]})
                mlflow.set_tag("run_type", "cross_validation_candidate")
        mlflow.log_metrics({"heldout_" + key.lower(): float(value) for key, value in result["heldout_test"].items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
        mlflow.log_artifact(str(output / "candidate_evaluation.json"), artifact_path="reports")
        mlflow.log_artifact(str(output / "credit_default_candidate.joblib"), artifact_path="candidate_model")
        print(f"Research candidate logged. Run ID: {parent.info.run_id}")
    print(json.dumps(result, indent=2))
    print(f"Candidate saved at: {output}")
    print("The deployed Random Forest remains unchanged.")


if __name__ == "__main__":
    main()
