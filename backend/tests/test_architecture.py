"""Integration boundaries and portable layout are part of the user's contract."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys


def test_required_python_architecture_exists(backend_root):
    project = backend_root.parent
    paths = [
        "backend/app/main.py", "backend/app/api/routes.py",
        "backend/app/schemas/prediction.py", "backend/app/services/prediction_service.py",
        "backend/models/credit_default_model_v1.joblib", "backend/models/model_metadata.json",
        "backend/requirements.txt", "backend/pytest.ini",
        "backend/src/preprocessing.py", "backend/src/feature_engineering.py",
        "backend/src/risk_scoring.py", "backend/src/explainability.py",
        "backend/src/fairness.py", "backend/src/monitoring.py", "backend/src/reporting.py",
        "frontend/app.py", "frontend/pages/1_Batch_Assessment.py",
        "frontend/services/api.py", "frontend/requirements.txt", "README.md", ".gitignore",
    ]
    assert not [path for path in paths if not (project / path).is_file()]
    for directory in ["data/raw", "data/processed", "notebooks", "reports", "tests"]:
        assert (backend_root / directory).is_dir()


def test_frontend_has_no_local_model_or_inference_dependency(backend_root):
    frontend = backend_root.parent / "frontend"
    forbidden_imports = {"joblib", "sklearn", "shap", "src", "backend", "app.services"}
    for path in frontend.rglob("*.py"):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            else:
                imported = []
            assert not [name for name in imported if any(name == bad or name.startswith(bad + ".") for bad in forbidden_imports)], str(path)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"predict_proba", "fit", "fit_transform"}, str(path)


def test_no_iris_react_node_or_vite_artifacts(backend_root):
    project = backend_root.parent
    forbidden_names = {"package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "vite.config.js", "vite.config.ts"}
    wrong = []
    for path in project.rglob("*"):
        if not path.is_file() or any(part in {".venv", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        if path.name in forbidden_names or path.suffix in {".jsx", ".tsx"}:
            wrong.append(path.relative_to(project).as_posix())
        if "iris" in path.name.lower() and path.suffix in {".csv", ".joblib", ".ipynb"}:
            wrong.append(path.relative_to(project).as_posix())
    assert wrong == []


def test_model_paths_and_imports_work_from_an_unrelated_directory(backend_root, tmp_path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(backend_root)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("CREDIT_ARTIFACT_DIR", None)
    code = (
        "import json; from app.main import create_app; from fastapi.testclient import TestClient; "
        "from pathlib import Path; "
        "app = create_app(); "
        "client = TestClient(app); client.__enter__(); "
        "r = client.get('/api/model/info'); "
        "print(json.dumps({'status': r.status_code, 'raw': r.json().get('raw_feature_count'), "
        "'engineered': r.json().get('engineered_feature_count')})); client.__exit__(None,None,None)"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path,
                            env=environment, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip().splitlines()[-1])
    assert output == {"status": 200, "raw": 23, "engineered": 38}


def test_all_python_sources_compile(backend_root):
    for path in backend_root.parent.rglob("*.py"):
        if any(part in {".venv", "__pycache__"} for part in path.parts):
            continue
        compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
