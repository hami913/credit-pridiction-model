# Credit-default notebooks

## Portable inference and EDA notebook

Open `01_Credit_Default_Inference.ipynb` for a runnable, read-only walkthrough of
the integrated system. It uses `PredictionService` for single prediction,
batch assessment, the original five payment scenarios, one-customer SHAP,
feature importance, and original historical metrics. It does not train a model
or write model/data/report files.

The notebook explicitly verifies the **23 raw → 38 engineered → 41 transformed**
feature contract and explains both preserved risk conventions: UI Low/Moderate/
High bands at 0.30/0.50 and original domain Low/Medium/High levels at 0.30/0.60.
EDA uses a 500-row real UCI sample; batch scoring uses ten rows and SHAP uses
one customer. Sample outputs are not independent test-performance estimates.

From the project root, activate a Python 3.12 virtual environment and run:

```bash
python -m pip install -r backend/requirements-research.txt
python -m jupyterlab backend/notebooks/01_Credit_Default_Inference.ipynb
```

Choose the Python kernel belonging to that same environment. The saved model
requires scikit-learn **1.6.1**, which is pinned in the backend requirements.
The notebook locates the backend when Jupyter starts from the project root,
`backend/`, or `backend/notebooks/`. A running FastAPI server is not needed for
this backend notebook; the separate Streamlit application uses the HTTP API.

### Actual execution verification

All **12 code cells** were executed sequentially under Python 3.12.14, in a
single shared Python namespace, with stdout/stderr and four Matplotlib figures
captured into notebook outputs. This verification used a Python
`compile`/`exec` cell runner because `nbclient` and a Jupyter kernel were not
installed in the verification environment. It was **not** a live Jupyter UI or
`nbclient` run. All 12 cells passed. The setup cell was also checked from both
`backend/` and `backend/notebooks/`; both resolved the correct project.

The model checksum and historical notebook checksum were unchanged after
execution. The shipped notebook contains those execution details in its
metadata and its actual outputs. No training occurred.

## Original preserved research notebook

`Credit_default_pridiction_model_final_project.ipynb` is the original uploaded
204-cell notebook, retained byte for byte as the history of the credit-default
research. It has no saved cell outputs. It was **not executed or retrained**
during integration.

The notebook contains the user's original UCI analysis, category normalization,
feature engineering, model comparisons, SHAP analysis, fairness analysis,
MLflow logging and final Random Forest export. It also includes exploratory
Logistic Regression and XGBoost experiments; these are research history and
do not change the deployed Random Forest.

This historical notebook uses Google Colab-specific `/content/...` paths,
`google.colab.files`, package-install cells, and interactive downloads. It is
not an application startup script and should not be run as part of installing
or testing the serving system. Its model-export cells could overwrite a model
in the notebook's current directory if executed manually.

To inspect the original deployed model safely, use the read-only API endpoints
or, from `backend/`, run:

```bash
python -m scripts.verify_real_inference
python -m scripts.audit_encoder_categories
```

The restored official UCI dataset is available at
`../data/processed/UCI_Credit_Card.csv`. Its provenance is documented in the
data directory. The original upload omitted the external CSV used by the
notebook, so byte identity with that absent local file cannot be checked.

Optional future research uses `src/train.py` to create a separately named
candidate, never the shipped `credit_default_model_v1.joblib`. The optional
candidate-training tests have the `research` marker and are excluded from the
default backend test run. No notebook/research training is required to run
the FastAPI and Streamlit applications.
