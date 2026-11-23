"""Inspect original fitted encoder coverage without altering data or model."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from app.services.prediction_service import PredictionService


ROOT = Path(__file__).resolve().parents[1]


def audit(csv_path):
    service = PredictionService(ROOT)
    service.load()
    original = pd.read_csv(csv_path)
    prepared = service.prepare(original)
    rows = []
    for name, transformer, columns in service.pipeline.named_steps["preprocessor"].transformers_:
        encoder = transformer if isinstance(transformer, OneHotEncoder) else next(
            (step for _, step in getattr(transformer, "steps", []) if isinstance(step, OneHotEncoder)), None
        )
        if encoder is None:
            continue
        names = [columns] if isinstance(columns, str) else list(columns)
        for column, categories in zip(names, encoder.categories_):
            column = prepared.columns[int(column)] if isinstance(column, (int, np.integer)) else str(column)
            unknown = prepared.loc[~prepared[column].isin(categories), column]
            rows.append({
                "transformer": name,
                "column": column,
                "known_categories": [value.item() if isinstance(value, np.generic) else value for value in categories],
                "unknown_records": int(len(unknown)),
                "normalized_records": int((original[column].to_numpy() != prepared[column].to_numpy()).sum()),
                "unknown_values": {str(k): int(v) for k, v in unknown.value_counts().items()},
            })
    return {"records": len(prepared), "training_performed": False, "columns": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data/processed/UCI_Credit_Card.csv")
    args = parser.parse_args()
    print(json.dumps(audit(args.csv), indent=2))


if __name__ == "__main__":
    main()
