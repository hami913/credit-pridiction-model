# Integrated Credit Default System

**Professor's software architecture + the original credit-default project.**

This is a new, separate project. It uses the professor's FastAPI application
factory, router, Pydantic schema, service layer, configuration, and testing
organization. Its machine learning, data, Streamlit interface, feature engineering,
risk rules, explainability, and analytics belong to the credit-default project.

The deployed model is the **unchanged Credit Default Random Forest v1.0**, with
the saved **0.50 decision threshold**. There is no Iris dataset or classifier,
React application, Node.js, npm, Vite, or JavaScript build requirement.

## Quick start

Use **Python 3.12**. Open a terminal in the extracted
`Integrated_Credit_Default_System` directory.

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install the combined Python dependencies:

```bash
python -m pip install -r requirements.txt
```

**Terminal 1 — FastAPI**, from the project root:

```bash
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

**Terminal 2 — Streamlit**, with the same environment activated and the same
project root as its working directory:

```bash
python -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

- Frontend: http://127.0.0.1:8501
- API documentation and interactive requests: http://127.0.0.1:8000/docs
- Model readiness: http://127.0.0.1:8000/api/health

Use **Load sample history**, then **Run credit risk assessment**, to exercise the
single-customer workflow. The batch page supplies a synthetic CSV template. The
analytics page can supply a small labeled UCI sample for testing the workflow.

The model and complete restored UCI dataset are bundled. Runtime prediction does
not need a network download or retraining. The original theme uses local fallback
fonts and requires no external font download. Start both processes: Streamlit communicates with FastAPI and
does not load a fallback model locally.

### Separate environments

If preferred, install `backend/requirements.txt` in a backend environment and
`frontend/requirements.txt` in a frontend environment. The frontend dependencies
do not include scikit-learn, Joblib, or SHAP. Use each environment to start its
corresponding process. Development tests additionally require
`backend/requirements-dev.txt`.

**Keep `scikit-learn==1.6.1`.** This is the version required by the uploaded
serialized artifact. Loading the original model with the environment's 1.8.0
installation failed; the project therefore retains the original dependency pin
instead of rewriting the artifact or replacing the model.

## Folder architecture

```text
Integrated_Credit_Default_System/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/routes.py
│   │   ├── schemas/prediction.py
│   │   └── services/prediction_service.py
│   ├── config/input_features.json
│   ├── data/
│   │   ├── raw/UCI_Credit_Card.xls
│   │   ├── processed/UCI_Credit_Card.csv
│   │   ├── dataset_provenance.json
│   │   └── README.md
│   ├── models/
│   │   ├── credit_default_model_v1.joblib
│   │   └── model_metadata.json
│   ├── notebooks/
│   ├── reports/
│   ├── scripts/
│   ├── src/
│   ├── tests/
│   │   └── fixtures/
│   ├── .env.example
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── requirements-research.txt
│   └── pytest.ini
├── frontend/
│   ├── app.py
│   ├── pages/
│   │   ├── 1_Batch_Assessment.py
│   │   ├── 2_Model_Information.py
│   │   └── 3_Analytics.py
│   ├── services/api.py
│   ├── components/
│   ├── tests/
│   ├── .env.example
│   ├── requirements.txt
│   └── pytest.ini
├── docs/
│   ├── ARCHITECTURE.md
│   ├── FILE_MAPPING.md
│   ├── FILE_MAPPING.csv
│   ├── REMOVED_FILES.md
│   └── TEST_RESULTS.md
├── .streamlit/config.toml
├── .github/workflows/ci.yml
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```

Empty Python `__init__.py` files, test fixtures, and supporting evidence are not
expanded in this overview. See [the complete file mapping](docs/FILE_MAPPING.md).

## How the layers work

1. Streamlit collects the original 23 raw customer fields and displays charts.
2. `frontend/services/api.py` sends requests and translates HTTP errors.
3. FastAPI's routes validate the request with credit-specific Pydantic schemas.
4. `PredictionService` calls the original preprocessing and feature-engineering
   functions, then invokes the saved pipeline. All serving inference is here.
5. The same service calls the original risk-scoring, SHAP, monitoring, fairness,
   evaluation, and reporting functions. Streamlit presents their results.

The model loads once per backend process at application startup. Its SHA-256,
metadata, class labels, feature order, preprocessing dimensions, estimator type,
hyperparameters, and scikit-learn version are validated. Missing or incompatible
artifacts produce a readable health response and HTTP 503; startup never trains
or manufactures a replacement model. Invalid customer input produces HTTP 422.

See [the architecture explanation](docs/ARCHITECTURE.md) for ownership and design
decisions.

## Preserved ML contract

| Property | Preserved value |
| --- | --- |
| Dataset | UCI Default of Credit Card Clients |
| Target | `default.payment.next.month`, 1 = default |
| Model | RandomForestClassifier, 400 trees |
| Model name/version | Credit Default Random Forest / 1.0 |
| Decision | `probability >= 0.50` |
| Raw predictive fields | 23 |
| Engineered model input | 38 columns: 23 raw + 15 derived |
| Fitted preprocessing output | 41 classifier columns |
| Deployed artifact | Original bytes, not resaved or retrained |

The original categorical normalization is retained: `EDUCATION` 0, 5, and 6 map
to 4; `MARRIAGE` 0 maps to 3. Valid negative repayment codes and negative bill
balances remain supported. `ID` and observed target columns are excluded from
model inputs but preserved as passthrough fields in batch output.

### Two original risk definitions

The active application and `src/risk_scoring.py` use different three-band
definitions. Both are preserved explicitly:

| Output | Low | Middle band | High |
| --- | --- | --- | --- |
| `risk_band` — original dashboard | p < 0.30 | Moderate: 0.30 ≤ p < 0.50 | p ≥ 0.50 |
| `risk_level` — original scoring module | p < 0.30 | Medium: 0.30 ≤ p < 0.60 | p ≥ 0.60 |

The separate `risk_index` is `round(p × 100, 2)`; higher means higher default risk.

For example, probability 0.55 produces default flag **1**, dashboard band
**High**, risk-index band **Medium**, and risk index **55.00**. This is the
original behavior, with distinct names instead of silently combining the rules.
The original source describes the risk-index cutoffs as illustrative.

## Preserved application features

| Feature | Integrated location |
| --- | --- |
| Single customer, full probability, default flag, risk bands/index | Main Streamlit page → `/api/predict` |
| Six-month history editor, demo controls, insights and gauge | Main page and retained UI helpers |
| Monthly bills/payments, status and utilization charts | Main page, payment-trends tab |
| Five payment what-if scenarios | Main page → `/api/scenario` |
| Local positive-class SHAP explanation | Main page → `/api/explain` |
| Batch CSV, validation, IDs, summary, histogram, scored download | Batch page → `/api/predict/batch-csv` |
| Model metadata, historical metrics, actual feature importance | Model Information page |
| Probability drift and optional observed-outcome performance | Analytics page → `/api/analytics/monitor` |
| Group fairness, sample adequacy, evaluation, JSON/CSV reporting | Analytics page → `/api/analytics/audit` |
| Original research and optional experiment tracking | `backend/src`, scripts, and notebooks |

The two old batch implementations are consolidated into one view. Choose strict
validation or explicitly score valid rows and report rejected rows. Customer IDs,
leading zeros, and extra columns survive scoring. Input columns whose names clash
with generated outputs are retained under `input__...` names, with a returned
mapping. Maximum: 50,000 customer rows and 20 MiB per CSV.

Audit uploads require actual labels (`actual_default` or the original target).
If both are supplied, they must agree. Monitoring can run without labels, in
which case it reports drift but does not claim performance metrics. Sex,
education, marriage, and original notebook age groups are available for auditing.
The historical age label `60+` means **over 60**; age 60 belongs to `51-60`.

## Dataset restoration and metric provenance

**The uploaded credit ZIP contained no dataset file.** Its original notebook and
scripts referenced a CSV outside the ZIP. The named official UCI dataset was
therefore downloaded from UCI and bundled, preserving all 30,000 rows and values.
The workbook and a canonical CSV are included; only the target header was renamed
to match the original project. Full source attribution, license, transformations,
and checksums are in [data/README.md](backend/data/README.md) and
`backend/data/dataset_provenance.json`.

The absent original local CSV cannot be compared byte for byte with the restored
copy. This limitation does not prevent checking that the original and integrated
code produce the same predictions on the same restored data.

The supplied `credit_risk_regulatory_report.json` is retained unchanged. Its
accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and historical fairness figures
are labelled **historical supplied results**, not newly verified performance.
The original notebook had no saved outputs or held-out prediction file. No new
training or independent model-performance claim is made by the integration.
Despite its legacy filename, the report is not a regulatory certification.

## API reference

All canonical operations use the `/api` prefix:

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Model readiness, version and threshold |
| GET | `/api/model/info` | Validated model and feature metadata |
| GET | `/api/model/metrics` | Original historical report with provenance |
| GET | `/api/model/feature-importance` | Actual 41-column and aggregated 38-column importance |
| POST | `/api/predict` | One 23-field customer object |
| POST | `/api/predict/batch` | `{customers: [...], skip_invalid: false}` |
| POST | `/api/predict/batch-csv` | Multipart `file`, optional `skip_invalid` query parameter |
| POST | `/api/scenario` | `{customer: {...}, factors: [0, 0.5, 1, 1.5, 2]}` |
| POST | `/api/explain` | `{customer: {...}, top_n: 10}` |
| POST | `/api/analytics/audit` | Labeled CSV and group settings |
| POST | `/api/analytics/monitor` | Reference and current CSV uploads |
| GET | `/api/data/template` | Synthetic example CSV |
| GET | `/api/data/sample?limit=100` | Small restored-UCI example, maximum 1,000 rows |

The original `/health` and `/predict` routes remain as compatibility aliases.
Legacy `/predict` keeps four-decimal `default_probability`. Canonical
`/api/predict` returns full precision plus a separate rounded display value.

JSON input is strict: all 23 fields, finite numbers, integer demographic/status
codes, no unexpected fields, no booleans or numeric strings. CSV numeric strings
are converted and validated on the backend. Swagger at `/docs` exposes the exact
input and output schemas and lets you send a sample request.

## Configuration

Defaults work immediately. Optional backend `.env` values can be copied from
`backend/.env.example`; the backend loads that file relative to itself.

`frontend/.env.example` documents frontend settings; frontend `.env` files are
not automatically loaded. Set or export those variables in the environment
before starting Streamlit, as shown below.

| Variable | Default | Use |
| --- | --- | --- |
| `CREDIT_ARTIFACT_DIR` | Backend directory | Directory containing models, data and reports; relative paths resolve from backend |
| `CORS_ORIGINS` | Localhost/127.0.0.1 port 8501 | Optional browser-origin allowlist |
| `CREDIT_API_BASE_URL` | `http://127.0.0.1:8000` | Backend address used by Streamlit |
| `CREDIT_API_TIMEOUT` | `120` seconds | Backend request timeout |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | Optional offline research scripts only |

For a different backend port, set `CREDIT_API_BASE_URL` before starting Streamlit.
PowerShell: `$env:CREDIT_API_BASE_URL = "http://127.0.0.1:8001"`.
macOS/Linux: `export CREDIT_API_BASE_URL=http://127.0.0.1:8001`.

Streamlit sends Python HTTP requests from its server process. If deployed on a
different machine, this URL must be reachable **from the Streamlit server**.
Deploy FastAPI and Streamlit as separate Python services with the commands above,
using the hosting platform's address/port settings. This package is a local
educational application; it does not introduce an account system or change the
original decision-support scope.

## Automated tests and actual results

Install development dependencies and run:

```bash
python -m pip install -r requirements-dev.txt
python scripts/verify_project.py
```

This cross-platform runner executes both suites, enables real HTTP Streamlit
tests, starts its own temporary backend, and writes XML/text/JSON evidence under
`backend/reports/generated/`. It needs no separately running server.

To run only backend tests: `python -m pytest -c backend/pytest.ini backend/tests -v`.
Frontend unit/UI tests can be run with
`python -m pytest -c frontend/pytest.ini frontend/tests -v`; set
`CREDIT_TEST_LIVE=1` to include its four live-integration tests, or use the runner
above, which sets this automatically. CI also enables these live tests.

The default backend suite excludes the four tests in the optional `research`
candidate-training module. They were not executed during integration. No test
fixture trains the deployed model. The SHAP tests use the shipped Random Forest.

The regression fixture was generated using the **pristine original model and
source code** on all 30,000 restored UCI records. Tests compare all 38 engineered
values, feature order, all probabilities (absolute tolerance `1e-12`), and all
binary predictions with the integrated service. Threaded Random Forest
summation can produce insignificant floating-point differences around `1e-16`.

See [TEST_RESULTS.md](docs/TEST_RESULTS.md) for the actual final counts, executed
commands, live HTTP/Streamlit verification, limitations, and machine-readable
test evidence. The old README's historical test count is not reused.

## Offline research and diagnostics

From `backend/`, with the backend environment active:

```bash
python -m scripts.verify_real_inference
python -m scripts.audit_encoder_categories
```

These diagnostics use the central service and unchanged saved model. The first
reports inference statistics, not new held-out performance. Optional MLflow
scripts are documented in [backend/scripts/README.md](backend/scripts/README.md).
They do not run on startup and require an explicitly configured tracking server.

`backend/notebooks/01_Credit_Default_Inference.ipynb` is a portable, no-training
walkthrough. The original notebook is retained separately as research provenance,
with its original Colab paths and training cells documented in
[backend/notebooks/README.md](backend/notebooks/README.md). It was not executed.
Optional notebook/research dependencies are in `backend/requirements-research.txt`;
they are unnecessary for the application.

## Change records

- [Architecture and design decisions](docs/ARCHITECTURE.md)
- [Every original file mapped or excluded](docs/FILE_MAPPING.md)
- [Machine-readable mapping](docs/FILE_MAPPING.csv)
- [Removed/excluded files and reasons](docs/REMOVED_FILES.md)
- [Executed tests and results](docs/TEST_RESULTS.md)

Neither uploaded source project was overwritten or modified. All cleanup applies
only to the contents selected for this new integrated project.

Original credit project authorship is preserved: **Mian Hamza Ahmad, University
of Lahore**, as stated in the supplied README. The professor's code provides the
architectural reference; no professor ML data, model, or business logic is shipped.
