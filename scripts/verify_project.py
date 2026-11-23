"""Run backend tests and real-HTTP Streamlit tests using the active Python env.

Run from any directory: python /path/to/project/scripts/verify_project.py
Reports are written separately from model/data artifacts. No training is run.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def xml_summary(path):
    tree = ET.parse(path).getroot()
    suites = [tree] if tree.tag == "testsuite" else list(tree.iter("testsuite"))
    result = {key: sum(int(suite.get(key, "0")) for suite in suites) for key in ("tests", "failures", "errors", "skipped")}
    result["passed"] = result["tests"] - result["failures"] - result["errors"] - result["skipped"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = (args.output_dir or ROOT / "backend/reports/generated" / ("verification_" + stamp)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CREDIT_TEST_LIVE": "1", "OPENBLAS_NUM_THREADS": "1"}
    results = []
    for name in ("backend", "frontend"):
        xml_path = output / f"{name}_pytest.xml"
        command = [sys.executable, "-m", "pytest", "-c", f"{name}/pytest.ini", f"{name}/tests", "-v", "-p", "no:cacheprovider", f"--junitxml={xml_path}"]
        print(f"Running {name} tests...", flush=True)
        start = time.monotonic()
        process = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
        text = process.stdout + process.stderr
        (output / f"{name}_pytest.txt").write_text(text, encoding="utf-8")
        result = {"suite": name, "command": ["python", *command[1:]], "exit_code": process.returncode, "seconds": round(time.monotonic() - start, 3)}
        if xml_path.exists():
            result.update(xml_summary(xml_path))
        results.append(result)
        print(json.dumps(result), flush=True)
        if process.returncode:
            print(text[-16000:], flush=True)
    packages = {}
    for name in ("fastapi", "pydantic", "uvicorn", "python-multipart", "scikit-learn", "numpy", "pandas", "scipy", "joblib", "shap", "streamlit", "plotly", "httpx", "pytest", "starlette"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "platform": platform.platform(), "packages": packages, "live_frontend_tests_enabled": True, "deployed_model_training_performed": False, "suites": results}
    (output / "verification_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Verification evidence: {output}", flush=True)
    return int(any(result["exit_code"] != 0 for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
