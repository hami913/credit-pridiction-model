# Removed / excluded files and reasons

No files were removed from either original uploaded project. The following were omitted only from the new integrated package after inspection. Files reorganized into new modules are described in FILE_MAPPING.md and are not counted as discarded functionality.

## Credit-project cleanup

| Original file | Reason for exclusion |
| --- | --- |
| `BRANCH_PROTECTION_TEST.md` | Branch-protection probe text has no application, model, deployment, or documentation function. |
| `app_generator_backup.py` | Obsolete generator containing a duplicate older app; unique model-information/performance display retained in Model Information page. |
| `app_working_backup.py` | Duplicate older application; useful nondefault probability/model card/indicator views integrated into current frontend. |
| `scripts/analyze_business_cost.py` | Requires absent HGB candidate threshold validation report; hypothetical candidate-only cost assumptions are not deployed RF behavior. Core original reporting remains available. |
| `scripts/analyze_oof_thresholds.py` | Requires absent HGB candidate real_20261017_145037_cfaac2 and workstation CSV; performs candidate retraining/threshold exploration that must not change deployed RF threshold. |
| `scripts/audit_probability_calibration.py` | Requires absent HGB candidate and hardcoded HGB Brier baseline; not an audit of the uploaded deployed RF. Portable monitoring/evaluation modules remain. |
| `scripts/audit_real_fairness.py` | Requires absent HGB candidate and hardcoded HGB confusion baseline; original portable fairness module now available through API with actual uploaded data. |
| `scripts/register_hgb_candidate.py` | Requires absent HGB candidate, absent fairness report chain, and external MLflow registry writes; unrelated to serving original RF. No candidate invented or registered. |
| `scripts/validate_candidate_threshold.py` | Requires absent HGB candidate and hardcoded HGB AUC; experimental threshold 0.25 is not the preserved deployed RF threshold 0.50. |

The original portable fairness, evaluation, monitoring and reporting modules remain available through the application. Optional candidate research remains in `src/train.py` and its explicit MLflow wrapper. The missing HGB candidate is not fabricated, substituted for the RF, or registered.

## Professor-project exclusions

All professor project files are excluded as direct deliverables. Relevant architecture is reimplemented using the credit source and Python frontend.

| Original file | Reason |
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

## Generated working files excluded from delivery

Virtual environments, package caches, `__pycache__`, `.pytest_cache`, temporary extraction/analysis files, scratch helpers, and backup copies are excluded. Test-result XML/text/JSON files, fixture baselines, source-integrity evidence, model/data artifacts, CI/configuration and documentation are deliberately retained. No Node/npm build environment was created for the project.
