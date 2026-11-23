"""Opt-in import of the user's historical scores, with original provenance tags."""

import argparse
import os


# Original reported values. These are not new runs or independently verified metrics.
MODELS = [
    {"name": "Logistic_Regression_Baseline", "algorithm": "Logistic Regression",
     "Accuracy": 0.8088, "Precision": 0.6923, "Recall": 0.2442,
     "F1": 0.3610, "ROC_AUC": 0.7101, "PR_AUC": 0.4950},
    {"name": "Regularized_Random_Forest", "algorithm": "Random Forest",
     "Accuracy": 0.8165, "Precision": 0.6619, "Recall": 0.3482,
     "F1": 0.4563, "ROC_AUC": 0.7756, "PR_AUC": 0.5536},
    {"name": "Previously_Reported_Final_RF", "algorithm": "Random Forest",
     "Accuracy": 0.7768, "Precision": 0.4962, "Recall": 0.5863,
     "F1": 0.5375, "ROC_AUC": 0.7777, "PR_AUC": 0.5548},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    args = parser.parse_args()
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(args.tracking_uri)
    experiment_name = "Credit_Risk_Historical_Model_Comparison"
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    for model in MODELS:
        existing = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.historical_import_key = '{model['name']}'",
        )
        if existing:
            print("Already imported:", model["name"])
            continue
        with mlflow.start_run(run_name=model["name"]):
            mlflow.log_params({"algorithm": model["algorithm"], "threshold": 0.5})
            mlflow.log_metrics({key: model[key] for key in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]})
            mlflow.set_tags({
                "run_type": "historical_import",
                "historical_import_key": model["name"],
                "source": "previously_user_reported_colab_results",
                "verification": "not_independently_verified",
                "training_executed_in_this_run": "false",
                "saved_model_linkage": "not_verified",
            })
            print("Imported historical results:", model["name"])


if __name__ == "__main__":
    main()
