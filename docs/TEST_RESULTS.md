# Executed integration verification

**Final result: 235 tests passed — 208 backend and 27 frontend — with no failed
tests or skipped selected tests.** Four optional research/candidate-training
tests were deliberately deselected. No model training was performed.

Date: 23 November 2026. Evidence uses UTC timestamps where specified.

## Final clean-environment run

A new Python 3.12.14 virtual environment was created without system site packages.
The supplied combined development requirements installed successfully: 65
resolved packages. The final tests therefore did not depend on undeclared
packages inherited from the initial working environment.

Executed from the integrated project root:

```bash
python scripts/verify_project.py --output-dir backend/reports/verification
```

The runner invoked the two pytest suites in separate processes, enabled
`CREDIT_TEST_LIVE=1`, and set `PYTHONDONTWRITEBYTECODE=1`. The frontend live-test
fixture launched a real Uvicorn process on an available localhost port and
terminated it afterward. No separately running server was assumed.

| Suite | Actual outcome | Pytest-reported time |
| --- | --- | ---: |
| Backend | **208 passed, 4 deselected, 1 warning** | 11.09 seconds |
| Frontend | **27 passed, 0 skipped, 0 warnings** | 7.34 seconds |
| Total executed tests | **235 passed, 0 failures, 0 errors** | |

The runner's process-inclusive timings were 13.612 and 8.276 seconds. These differ
from pytest's own timings because interpreter startup and result handling are
also included. XML, complete text output, commands, exit codes and installed
package versions are saved in:

- `backend/reports/verification/backend_pytest.xml`
- `backend/reports/verification/backend_pytest.txt`
- `backend/reports/verification/frontend_pytest.xml`
- `backend/reports/verification/frontend_pytest.txt`
- `backend/reports/verification/verification_summary.json`

The single backend warning was a Starlette notice that its current TestClient
deprecates the HTTPX transport in favor of `httpx2`. The selected, installed
HTTPX/TestClient combination passed all executed tests; the warning is retained
in the log instead of hidden.

## What the backend tests verified

| Area | Executed checks |
| --- | --- |
| Artifact preservation | Original model SHA-256, original payload metadata, original domain-source hashes, unchanged 0.50 threshold |
| Startup | One model load per process; no estimator fitting; missing/corrupt/incompatible artifacts return 503 |
| Model validation | RF estimator type, class labels, 23/38/41 dimensions, feature ordering, hyperparameters, sklearn version, risk-definition consistency |
| Single prediction | Actual saved-model probability, binary prediction, full precision and original legacy rounding |
| Inputs | Every raw field required; extra fields, strings, booleans, nonfinite values and domain errors rejected; valid negative bills/statuses accepted |
| Feature engineering | Exact original category mappings, all derived values, canonical 38-column order and 41-column preprocessing output |
| Scenarios | All five original payment multipliers scored with the saved model; factor 1 matches the ordinary result |
| Batch | JSON/CSV parity, row order, IDs, leading zeros, passthrough columns, output-name collisions, strict and explicit skip modes |
| CSV errors | Missing/duplicate headers, empty/malformed files, row/upload limits, actual source record-start lines after blanks/multiline fields |
| Explainability | Actual SHAP output from the shipped model, finite contributions and correct feature names |
| Feature importance | Actual fitted RF importances and aggregation from 41 transformed columns to 38 model-input fields |
| Analytics | Observed-label evaluation, fairness sample rules and gaps, monitoring with/without labels, conflicting-label rejection, JSON reports |
| Original age audit | Exact notebook boundary behavior for ages 30/31/40/41/50/51/60/61 |
| Original risk definitions | Boundaries at 0.30, 0.50 and 0.60; binary, dashboard and score bands remain distinct |
| Engineering structure | Required folders/files, arbitrary-working-directory imports, no frontend model inference and no prohibited frontend build assets |

The existing original domain tests were retained where applicable. The SHAP test
fixture was adapted to use the shipped model instead of fitting a toy forest.
The optional research test module was marked and excluded; its four tests remain
in source for future explicitly requested candidate research.

## Independent original-code parity: all 30,000 rows

Before the integrated service was validated, a baseline was generated using the
**pristine extracted original preprocessing/feature-engineering source and the
original uploaded model**, against all 30,000 restored official UCI records.
The baseline records its original archive, code, configuration, model, dataset
and fixture checksums.

The final backend suite compared the integrated service against that fixture:

- All **38 engineered column names and values** match.
- The fitted preprocessing produces **41 classifier columns**.
- All **30,000 default probabilities** match with `rtol=0`, `atol=1e-12`.
- All **30,000 binary predictions** are identical.
- The decision threshold remains **0.50**.
- Model, preserved source, raw-schema and fixture hashes are verified.

The unchanged forest uses `n_jobs=-1`; different threaded accumulation order can
change the final floating-point digits. The tolerance is numerical, not a change
to the scoring rule or an acceptance of materially different predictions.

The fixture is in `backend/tests/fixtures/original_baseline.npz`, with provenance
in the adjacent JSON. This verifies software behavior on a common input set. It
does **not** establish a new held-out accuracy, fairness, calibration or external
validity estimate.

## Frontend verification

| Frontend category | Actual count | Nature of test |
| --- | ---: | --- |
| API client | 16 | Request shapes, multipart payloads, errors and configuration with controlled HTTP responses |
| Isolated Streamlit UI | 7 | Actual Streamlit AppTest execution using explicitly synthetic API fixtures |
| Live integration | 4 | Actual HTTP, Uvicorn, saved model and Streamlit page execution |
| Total | **27** | All passed |

The live tests exercised:

1. Single prediction, batch parity, all five scenarios, real SHAP, feature
   importance, historical metrics, audit and monitoring through the Python client.
2. The Streamlit form and demo history, submitted single result, six rendered
   Plotly charts, scenarios, explanation and Model Information page.
3. Batch upload with leading-zero IDs, strict validation failure followed by
   explicit skip mode, row error display and scoring output.
4. Fairness/reporting and monitoring pages, real UCI sample uploads, rendered
   report-download controls, and clearing stale reports when an input or setting changes.

**UI-test boundary:** Streamlit 1.51 AppTest does not operate a native browser
file-picker dialog. Tests supplied in-memory uploaded-file objects at that
boundary. The CSV bytes, multipart HTTP calls, FastAPI validation, saved-model
inference, analytics, session state and Streamlit rendering were real. No claim
of a manual browser click-through or operating-system file dialog test is made.

## Additional executed checks

### Portable notebook

All **12 executable code cells** in the new portable inference notebook passed in
3.839 seconds. They were executed sequentially using `compile`/`exec` in a shared
namespace, with captured stdout/stderr and four embedded Matplotlib figures.
This was not a live Jupyter kernel or `nbclient` run; those were unavailable in
the notebook verification environment. Its startup cell also worked from
`backend/` and `backend/notebooks/`.

The notebook exercised real-data EDA, service loading, 23/38/41 dimensions,
single/batch/scenario/SHAP operations, feature importance, historical metrics and
unchanged model checksum. It performed no model fitting or artifact writes.
The original historical 204-cell notebook was not executed and remains unchanged.
Evidence: `backend/reports/verification/notebook_execution.json`.

### Portable scripts

All five retained/adapted CLI scripts parsed successfully and their `--help`
entry points exited with code 0. The inference-verification and encoder-audit
scripts also ran on all 30,000 UCI rows and exited with code 0.

The inference diagnostic reported 7,866 default flags and 22,134 nondefault
flags. The encoder audit found no unknown categories after applying the original
normalization, which affected 345 EDUCATION values and 54 MARRIAGE values. These
are functional data/inference diagnostics, not new held-out performance claims.
Evidence: `script_checks.json` and `inference_diagnostics.json` in the verification
directory.

### Original source integrity

Both original ZIPs and all **112 extracted files** were rehashed: 46 credit-project
files and 66 professor-project files. Every hash matched the pre-implementation
inventory. The deployed model, all nine original domain modules, the raw schema,
historical report, and historical notebook retain their original bytes where
marked as retained. See `docs/SOURCE_INTEGRITY.json` and `docs/FILE_MAPPING.csv`.

### Extracted-package verification

The project ZIP was expanded into a separate directory. Its files were checked
against `PACKAGE_MANIFEST.json`, and the original-parity test module was run from
an unrelated `/tmp` working directory against that extracted package. **All three
parity tests passed in 2.44 seconds** (with the same dependency warning). This
checks that the packaged model, data, fixtures and relative paths are usable
without the source-inspection workspace. These repeated parity checks are not
added to the 235-test main-suite total. ZIP CRC and manifest checks also passed.

## Known qualifications and operations not run

- Four optional research tests were deselected. No model retraining, candidate
  selection, threshold optimization, or original research-notebook execution ran.
- Optional live MLflow logging/registration was not performed; no tracking
  server was part of the source upload. The optional CLI interfaces were checked.
- No public/cloud deployment was performed. Local FastAPI HTTP integration and
  Streamlit execution were tested.
- The user ZIP omitted the original local CSV. The official UCI dataset was
  restored with explicit provenance; byte identity to the missing CSV cannot be
  proved.
- Supplied historical model-performance numbers remain historical supplied
  evidence. No accuracy, AUC or fairness improvement is claimed by restructuring.
- The first development-stage architecture check failed while README and
  requirements files were still being authored. After those deliverables were
  completed, the entire final clean-environment suite passed. Earlier partial
  runs are not substituted for the final results above.

## Reproduce

From a new Python 3.12 environment at the project root:

```bash
python -m pip install -r requirements-dev.txt
python scripts/verify_project.py
```

The runner enables live frontend tests automatically and produces fresh timestamped
evidence under `backend/reports/generated/`. It does not overwrite the shipped
model or dataset.
