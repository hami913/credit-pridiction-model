# Portable diagnostics and optional research

Run commands from `backend/` using the installed backend environment.
Both diagnostic commands use the central `PredictionService`; they do not
reimplement feature engineering or model inference.

```bash
python -m scripts.verify_real_inference
python -m scripts.audit_encoder_categories
```

Both commands default to `backend/data/processed/UCI_Credit_Card.csv` and accept
`--csv /path/to/data.csv`. Inference verification writes a new timestamped JSON
report under `backend/reports/`, or the new path supplied with `--output`.
Existing reports are never overwritten. The report records prediction counts,
not independent held-out model performance.

## Optional MLflow workflows

MLflow is only required for the following explicitly invoked research/logging
commands. It is not required by FastAPI, Streamlit, inference, or default tests.
Install the optional research requirements and start your own MLflow tracking
service before invoking these commands. `--tracking-uri` or
`MLFLOW_TRACKING_URI` configures the destination (default
`http://127.0.0.1:5000`). These commands write to that service when run.

```bash
python -m scripts.log_inference_mlflow --report reports/your_inference_report.json
python -m scripts.import_historical_mlflow
```

The historical import retains the original values and tags marking them
`not_independently_verified`. It skips already-imported named runs.

For a separately authorized future experiment, the original research training
module and portable MLflow wrapper are retained:

```bash
python -m scripts.train_and_track_mlflow --csv data/processed/UCI_Credit_Card.csv
```

This last command **trains new research candidates**. It was not executed during
integration and is not part of the default application or test workflow. It
writes a distinct `credit_default_candidate.joblib` beneath a unique directory
and never replaces the deployed `credit_default_model_v1.joblib`.

The six old HGB-only scripts were excluded from the integrated project because
they require an absent HGB artifact/report chain and contain machine-specific
paths or fixed candidate-run metrics. Their exclusion and exact filenames are
listed in the integration file-mapping/cleanup report. The original uploaded
project remains intact. Deployed explainability, monitoring, fairness, and
reporting are fully available through the original domain modules and API.
