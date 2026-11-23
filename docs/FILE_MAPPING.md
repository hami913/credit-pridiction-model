# Complete source file mapping

All 112 source files from both uploaded ZIPs are accounted for. Originals remain unchanged. 'Excluded' means omitted from this new package, never deleted from an uploaded source.

## Credit project

| Original path | Action | Integrated path(s) | Reason |
| --- | --- | --- | --- |
| `.github/workflows/ci.yml` | adapted | `.github/workflows/ci.yml` | Retained CI intent; separate backend and live Streamlit suites use Python 3.12 and new requirements. |
| `.gitignore` | adapted | `.gitignore` | Python/Streamlit caches and local settings excluded; required data/model deliberately retained. |
| `BRANCH_PROTECTION_TEST.md` | excluded | — | Branch-protection probe text has no application, model, deployment, or documentation function. |
| `Credit_default_pridiction_model_final_project.ipynb` | retained byte-for-byte | `backend/notebooks/Credit_default_pridiction_model_final_project.ipynb` | Historical research provenance; all 204 cells preserved; not executed; Colab paths documented. |
| `README.md` | adapted | `README.md`<br>`docs/ARCHITECTURE.md` | Rewritten for the complete integrated architecture, accurate installation, preservation and actual tests; original authorship retained. |
| `api.py` | adapted | `backend/app/main.py`<br>`backend/app/api/routes.py`<br>`backend/app/schemas/prediction.py`<br>`backend/app/services/prediction_service.py` | Original credit API separated into professor-style layers; original /health and rounded /predict aliases preserved. |
| `app.py` | adapted | `frontend/app.py`<br>`frontend/components/ui.py`<br>`frontend/components/common.py`<br>`frontend/components/batch.py`<br>`frontend/services/api.py`<br>`backend/app/services/prediction_service.py` | Original Streamlit UI/charts/input/insights retained; serving inference and scenarios moved to backend; batch consolidated. |
| `app_generator_backup.py` | excluded | — | Obsolete generator containing a duplicate older app; unique model-information/performance display retained in Model Information page. |
| `app_working_backup.py` | excluded | — | Duplicate older application; useful nondefault probability/model card/indicator views integrated into current frontend. |
| `config.toml` | relocated | `.streamlit/config.toml` | Original Streamlit theme moved to its effective configuration directory; content preserved. |
| `config/input_features.json` | retained byte-for-byte | `backend/config/input_features.json` | Canonical original order of all 23 raw features. |
| `models/credit_default_model_v1.joblib` | retained byte-for-byte | `backend/models/credit_default_model_v1.joblib` | Original trained Random Forest artifact; unchanged SHA-256, threshold and fitted pipeline. |
| `pages/1_Batch_Assessment.py` | adapted | `frontend/pages/1_Batch_Assessment.py`<br>`frontend/components/batch.py`<br>`frontend/services/api.py`<br>`backend/app/services/prediction_service.py` | Standalone and embedded batch capabilities combined; original fields/IDs retained; inference centralized; strict/skip modes explicit. |
| `reports/credit_risk_regulatory_report.json` | retained byte-for-byte | `backend/reports/credit_risk_regulatory_report.json` | Original historical values retained with provenance; not treated as freshly verified performance or certification. |
| `requirements.txt` | adapted | `backend/requirements.txt`<br>`backend/requirements-dev.txt`<br>`backend/requirements-research.txt`<br>`frontend/requirements.txt`<br>`requirements.txt`<br>`requirements-dev.txt` | Preserved sklearn 1.6.1; runtime/dev/research/frontend dependencies separated and fresh install verified. |
| `scripts/analyze_business_cost.py` | excluded from integrated project only | — | Requires absent HGB candidate threshold validation report; hypothetical candidate-only cost assumptions are not deployed RF behavior. Core original reporting remains available. |
| `scripts/analyze_oof_thresholds.py` | excluded from integrated project only | — | Requires absent HGB candidate real_20261017_145037_cfaac2 and workstation CSV; performs candidate retraining/threshold exploration that must not change deployed RF threshold. |
| `scripts/audit_encoder_categories.py` | adapted | `backend/scripts/audit_encoder_categories.py` | Replaced Windows/cwd paths with backend-relative defaults and --csv CLI. Uses central service preparation and original fitted encoder; returns unknown-category and normalization counts. |
| `scripts/audit_probability_calibration.py` | excluded from integrated project only | — | Requires absent HGB candidate and hardcoded HGB Brier baseline; not an audit of the uploaded deployed RF. Portable monitoring/evaluation modules remain. |
| `scripts/audit_real_fairness.py` | excluded from integrated project only | — | Requires absent HGB candidate and hardcoded HGB confusion baseline; original portable fairness module now available through API with actual uploaded data. |
| `scripts/import_historical_mlflow.py` | adapted | `backend/scripts/import_historical_mlflow.py` | Preserves all original historical metric values and unverified provenance tags, duplicate-run protection. Optional lazy MLflow import and configurable tracking URI; explicit CLI only. |
| `scripts/log_inference_mlflow.py` | adapted | `backend/scripts/log_inference_mlflow.py` | Preserves original inference tracking metrics/tags. Optional lazy MLflow import, --report and configurable --tracking-uri/MLFLOW_TRACKING_URI; validates model hash when recorded. |
| `scripts/register_hgb_candidate.py` | excluded from integrated project only | — | Requires absent HGB candidate, absent fairness report chain, and external MLflow registry writes; unrelated to serving original RF. No candidate invented or registered. |
| `scripts/train_and_track_mlflow.py` | adapted | `backend/scripts/train_and_track_mlflow.py` | Retains original three-algorithm optional candidate research workflow. CLI dataset/output/URI replaces workstation paths; refuses nonempty outputs/deployed directory and checks deployed artifact hash. Not executed; never called by application/default tests. |
| `scripts/validate_candidate_threshold.py` | excluded from integrated project only | — | Requires absent HGB candidate and hardcoded HGB AUC; experimental threshold 0.25 is not the preserved deployed RF threshold 0.50. |
| `scripts/verify_real_inference.py` | adapted | `backend/scripts/verify_real_inference.py` | Replaced Windows/cwd paths with backend-relative defaults + --csv/--output CLI. Uses central PredictionService and source/model SHA256 checks. Reports inference only; never trains. |
| `src/evaluate.py` | retained byte-for-byte | `backend/src/evaluate.py` | Original credit-default domain logic retained; backend/config/models layout preserves relative paths. |
| `src/explainability.py` | retained byte-for-byte | `backend/src/explainability.py` | Original credit-default domain logic retained; backend/config/models layout preserves relative paths. |
| `src/fairness.py` | retained byte-for-byte | `backend/src/fairness.py` | Original credit-default domain logic retained; backend/config/models layout preserves relative paths. |
| `src/feature_engineering.py` | retained byte-for-byte | `backend/src/feature_engineering.py` | Original credit-default domain logic retained; backend/config/models layout preserves relative paths. |
| `src/monitoring.py` | retained byte-for-byte | `backend/src/monitoring.py` | Original credit-default domain logic retained; backend/config/models layout preserves relative paths. |
| `src/preprocessing.py` | retained byte-for-byte | `backend/src/preprocessing.py` | Original credit-default domain logic retained; backend/config/models layout preserves relative paths. |
| `src/reporting.py` | retained byte-for-byte | `backend/src/reporting.py` | Original credit-default domain logic retained; backend/config/models layout preserves relative paths. |
| `src/risk_scoring.py` | retained byte-for-byte | `backend/src/risk_scoring.py` | Original credit-default domain logic retained; backend/config/models layout preserves relative paths. |
| `src/train.py` | retained byte-for-byte | `backend/src/train.py` | Original credit-default domain logic retained; backend/config/models layout preserves relative paths. |
| `tests/test_api.py` | adapted | `backend/tests/test_api.py`<br>`backend/tests/test_service.py` | Original API checks expanded to saved model, strict input, batch, scenario, analytics and failure-state integration checks. |
| `tests/test_category_mapping.py` | retained byte-for-byte | `backend/tests/test_category_mapping.py` | Original applicable unit test; model paths already resolve relative to test file. |
| `tests/test_evaluate.py` | retained byte-for-byte | `backend/tests/test_evaluate.py` | Original applicable unit test; model paths already resolve relative to test file. |
| `tests/test_explainability.py` | adapted | `backend/tests/test_explainability.py` | Replaced synthetic fitted test model with shipped artifact and real UCI row; now tests actual SHAP without fitting any model. |
| `tests/test_fairness.py` | retained byte-for-byte | `backend/tests/test_fairness.py` | Original applicable unit test; model paths already resolve relative to test file. |
| `tests/test_features.py` | retained byte-for-byte | `backend/tests/test_features.py` | Original applicable unit test; model paths already resolve relative to test file. |
| `tests/test_model.py` | retained byte-for-byte | `backend/tests/test_model.py` | Original applicable unit test; model paths already resolve relative to test file. |
| `tests/test_monitoring.py` | retained byte-for-byte | `backend/tests/test_monitoring.py` | Original applicable unit test; model paths already resolve relative to test file. |
| `tests/test_reporting.py` | retained byte-for-byte | `backend/tests/test_reporting.py` | Original applicable unit test; model paths already resolve relative to test file. |
| `tests/test_risk_scoring.py` | retained byte-for-byte | `backend/tests/test_risk_scoring.py` | Original applicable unit test; model paths already resolve relative to test file. |
| `tests/test_train.py` | adapted | `backend/tests/test_train.py` | Marked entire module research and excluded by default; retained candidate research tests without running training. |

## Professor project: every file excluded as a direct artifact

The professor supplies engineering patterns only. No professor dataset, model, prediction logic, metrics, notebooks, or frontend build files are shipped.

| Professor path | Reference or exclusion reason |
| --- | --- |
| `.gitignore` | Documentation/configuration/dependency organization reference only; credit-specific files written from actual requirements. |
| `README.md` | Documentation/configuration/dependency organization reference only; credit-specific files written from actual requirements. |
| `backend/.env.example` | Documentation/configuration/dependency organization reference only; credit-specific files written from actual requirements. |
| `backend/app/__init__.py` | App factory, router, schemas or service organization reimplemented for credit; no Iris logic copied. |
| `backend/app/api/__init__.py` | App factory, router, schemas or service organization reimplemented for credit; no Iris logic copied. |
| `backend/app/api/routes.py` | App factory, router, schemas or service organization reimplemented for credit; no Iris logic copied. |
| `backend/app/main.py` | App factory, router, schemas or service organization reimplemented for credit; no Iris logic copied. |
| `backend/app/schemas/__init__.py` | App factory, router, schemas or service organization reimplemented for credit; no Iris logic copied. |
| `backend/app/schemas/prediction.py` | App factory, router, schemas or service organization reimplemented for credit; no Iris logic copied. |
| `backend/app/services/__init__.py` | App factory, router, schemas or service organization reimplemented for credit; no Iris logic copied. |
| `backend/app/services/prediction_service.py` | App factory, router, schemas or service organization reimplemented for credit; no Iris logic copied. |
| `backend/data/processed/test.csv` | Iris dataset or Iris split; unrelated to original credit data. |
| `backend/data/processed/train.csv` | Iris dataset or Iris split; unrelated to original credit data. |
| `backend/data/raw/iris.csv` | Iris dataset or Iris split; unrelated to original credit data. |
| `backend/models/iris_decision_tree.joblib` | Iris Decision Tree or Iris metadata; original credit Random Forest retained instead. |
| `backend/models/model_metadata.json` | Iris Decision Tree or Iris metadata; original credit Random Forest retained instead. |
| `backend/notebooks/Decision Tree Notebook.ipynb` | Iris notebook and domain experiments excluded; credit notebook retained. |
| `backend/notebooks/model_prediction.ipynb` | Iris notebook and domain experiments excluded; credit notebook retained. |
| `backend/pytest.ini` | Saved-artifact/validation test organization used as reference; Iris tests and retraining fixture excluded. |
| `backend/reports/eda_summary.json` | Iris-specific EDA, plot or evaluation output; never used as credit evidence. |
| `backend/reports/figures/boxplots.png` | Iris-specific EDA, plot or evaluation output; never used as credit evidence. |
| `backend/reports/figures/class_distribution.png` | Iris-specific EDA, plot or evaluation output; never used as credit evidence. |
| `backend/reports/figures/confusion_matrix.png` | Iris-specific EDA, plot or evaluation output; never used as credit evidence. |
| `backend/reports/figures/correlation.png` | Iris-specific EDA, plot or evaluation output; never used as credit evidence. |
| `backend/reports/figures/decision_tree.png` | Iris-specific EDA, plot or evaluation output; never used as credit evidence. |
| `backend/reports/figures/feature_histograms.png` | Iris-specific EDA, plot or evaluation output; never used as credit evidence. |
| `backend/reports/figures/feature_importance.png` | Iris-specific EDA, plot or evaluation output; never used as credit evidence. |
| `backend/reports/figures/pairplot.png` | Iris-specific EDA, plot or evaluation output; never used as credit evidence. |
| `backend/reports/metrics.json` | Iris-specific EDA, plot or evaluation output; never used as credit evidence. |
| `backend/reports/validation.json` | Iris-specific EDA, plot or evaluation output; never used as credit evidence. |
| `backend/requirements-dev.txt` | Documentation/configuration/dependency organization reference only; credit-specific files written from actual requirements. |
| `backend/requirements-lock.txt` | Documentation/configuration/dependency organization reference only; credit-specific files written from actual requirements. |
| `backend/requirements-training.txt` | Documentation/configuration/dependency organization reference only; credit-specific files written from actual requirements. |
| `backend/requirements.txt` | Documentation/configuration/dependency organization reference only; credit-specific files written from actual requirements. |
| `backend/src/__init__.py` | Offline-source/configuration organization reference; Iris feature/data/training logic excluded. |
| `backend/src/config.py` | Offline-source/configuration organization reference; Iris feature/data/training logic excluded. |
| `backend/src/data_ingestion.py` | Offline-source/configuration organization reference; Iris feature/data/training logic excluded. |
| `backend/src/data_preprocessing.py` | Offline-source/configuration organization reference; Iris feature/data/training logic excluded. |
| `backend/src/data_validation.py` | Offline-source/configuration organization reference; Iris feature/data/training logic excluded. |
| `backend/src/eda.py` | Offline-source/configuration organization reference; Iris feature/data/training logic excluded. |
| `backend/src/feature_engineering.py` | Offline-source/configuration organization reference; Iris feature/data/training logic excluded. |
| `backend/src/model_evaluation.py` | Offline-source/configuration organization reference; Iris feature/data/training logic excluded. |
| `backend/src/model_training.py` | Offline-source/configuration organization reference; Iris feature/data/training logic excluded. |
| `backend/src/training_pipeline.py` | Offline-source/configuration organization reference; Iris feature/data/training logic excluded. |
| `backend/tests/conftest.py` | Saved-artifact/validation test organization used as reference; Iris tests and retraining fixture excluded. |
| `backend/tests/test_api.py` | Saved-artifact/validation test organization used as reference; Iris tests and retraining fixture excluded. |
| `backend/tests/test_prediction.py` | Saved-artifact/validation test organization used as reference; Iris tests and retraining fixture excluded. |
| `backend/tests/test_saved_model.py` | Saved-artifact/validation test organization used as reference; Iris tests and retraining fixture excluded. |
| `frontend/.env.example` | React/JSX/Vite/Node frontend excluded; Streamlit retained. |
| `frontend/index.html` | React/JSX/Vite/Node frontend excluded; Streamlit retained. |
| `frontend/package-lock.json` | React/JSX/Vite/Node frontend excluded; Streamlit retained. |
| `frontend/package.json` | React/JSX/Vite/Node frontend excluded; Streamlit retained. |
| `frontend/src/App.jsx` | React/JSX/Vite/Node frontend excluded; Streamlit retained. |
| `frontend/src/components/Navbar.jsx` | React/JSX/Vite/Node frontend excluded; Streamlit retained. |
| `frontend/src/components/PredictionForm.jsx` | React/JSX/Vite/Node frontend excluded; Streamlit retained. |
| `frontend/src/components/PredictionForm.test.jsx` | React/JSX/Vite/Node frontend excluded; Streamlit retained. |
| `frontend/src/components/PredictionResult.jsx` | React/JSX/Vite/Node frontend excluded; Streamlit retained. |
| `frontend/src/components/SpeciesCard.jsx` | React/JSX/Vite/Node frontend excluded; Streamlit retained. |
| `frontend/src/main.jsx` | React/JSX/Vite/Node frontend excluded; Streamlit retained. |
| `frontend/src/pages/Dashboard.jsx` | React/JSX/Vite/Node frontend excluded; Streamlit retained. |
| `frontend/src/pages/Dashboard.test.jsx` | React/JSX/Vite/Node frontend excluded; Streamlit retained. |
| `frontend/src/pages/ModelPerformance.jsx` | React/JSX/Vite/Node frontend excluded; Streamlit retained. |
| `frontend/src/services/api.js` | Python API-client separation pattern only. |
| `frontend/src/styles.css` | React/JSX/Vite/Node frontend excluded; Streamlit retained. |
| `frontend/src/test-setup.js` | React/JSX/Vite/Node frontend excluded; Streamlit retained. |
| `frontend/vite.config.js` | React/JSX/Vite/Node frontend excluded; Streamlit retained. |

## New integration files

Additional files implement the architecture and missing integration needs: extracted `model_metadata.json`; restored UCI workbook/CSV and provenance; API schemas/service/routes; shared Python HTTP client; Model Information and Analytics pages; independent original-model baseline and regression tests; portable inference notebook; real HTTP Streamlit tests; cross-platform verification runner; and documentation/test evidence. They are new integration artifacts, not copied Iris business logic.

The JSON/CSV mapping includes every original SHA-256 and whether a relocated file is byte-identical. The two source archives and all extracted originals were rehashed after implementation; see `SOURCE_INTEGRITY.json`.
