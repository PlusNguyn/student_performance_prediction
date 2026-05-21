from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from django.conf import settings

from preprocessing.pipeline import (
    TARGET_CLASS_COLUMN,
    TARGET_REGRESSION_COLUMN,
    _add_targets,
    build_assessment_features,
    build_modeling_dataset,
    build_vle_features,
    clean_raw_tables,
    load_raw_tables,
    prepare_features,
)


PASS_LABEL = "Pass"
FAIL_LABEL = "Fail / Withdrawn"


class DemoPredictionError(RuntimeError):
    pass


def get_student_prediction(student_id: str) -> dict[str, Any] | None:
    student_id = str(student_id or "").strip()
    if not student_id:
        return None
    for record in get_demo_predictions():
        if str(record["student_id"]) == student_id:
            return record
    return None


def get_lecturer_predictions() -> list[dict[str, Any]]:
    return get_demo_predictions()


def get_demo_summary() -> dict[str, Any]:
    rows = get_demo_predictions()
    total = len(rows)
    high_risk = sum(1 for row in rows if row["risk"] == "High")
    medium_risk = sum(1 for row in rows if row["risk"] == "Medium")
    passed = sum(1 for row in rows if row["prediction_label"] == PASS_LABEL)
    avg_learning = _mean(row["learning_percentage"] for row in rows)
    avg_confidence = _mean(row["confidence"] for row in rows)
    return {
        "total": total,
        "high_risk": high_risk,
        "medium_risk": medium_risk,
        "passed": passed,
        "pass_rate": _percent(passed / total if total else 0),
        "avg_learning": round(avg_learning, 1),
        "avg_confidence": round(avg_confidence, 1),
    }


@lru_cache(maxsize=1)
def get_demo_predictions() -> list[dict[str, Any]]:
    paths = _prediction_paths()
    _ensure_required_files(paths)

    dataset = _build_demo_dataset(paths["demo_dir"])
    encoded_dataset, features, _, _, _, _ = prepare_features(dataset)
    model_features = _read_json(paths["feature_columns"])
    model_medians = _read_json(paths["medians"])
    features = _align_features(features, model_features, model_medians)

    classifier = joblib.load(paths["classification_model"])
    regressor = joblib.load(paths["regression_model"])
    class_predictions = classifier.predict(features)
    confidences = _classification_confidence(classifier, features, class_predictions)
    learning_predictions = np.clip(regressor.predict(features), 0, 100)

    rows: list[dict[str, Any]] = []
    display_dataset = encoded_dataset.copy()
    for index, row in dataset.reset_index(drop=True).iterrows():
        predicted_pass = int(class_predictions[index])
        learning_percentage = float(learning_predictions[index])
        confidence = float(confidences[index])
        rows.append(
            {
                "student_id": str(row["id_student"]),
                "code_module": str(row["code_module"]),
                "code_presentation": str(row["code_presentation"]),
                "prediction_label": PASS_LABEL if predicted_pass else FAIL_LABEL,
                "prediction_value": predicted_pass,
                "learning_percentage": round(learning_percentage, 1),
                "confidence": round(confidence * 100, 1),
                "risk": _risk_label(predicted_pass, learning_percentage),
                "risk_order": _risk_order(predicted_pass, learning_percentage),
                "avg_score": round(float(row.get("avg_score", 0)), 1),
                "total_score": round(float(row.get("total_score", 0)), 1),
                "num_assessments": int(row.get("num_assessments", 0)),
                "total_clicks": int(row.get("total_clicks", 0)),
                "engagement_score": _engagement_score(display_dataset, index),
            }
        )

    return sorted(rows, key=lambda item: (item["risk_order"], item["student_id"]))


def _prediction_paths() -> dict[str, Path]:
    base_dir = Path(settings.BASE_DIR)
    return {
        "demo_dir": base_dir / "data" / "demo",
        "classification_model": base_dir / "models" / "classification_model.joblib",
        "regression_model": base_dir / "models" / "regression_model.joblib",
        "feature_columns": base_dir / "models" / "features.json",
        "medians": base_dir / "models" / "medians.json",
    }


def _ensure_required_files(paths: dict[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise DemoPredictionError("Missing demo prediction artifacts: " + ", ".join(missing))


def _build_demo_dataset(demo_dir: Path) -> pd.DataFrame:
    raw_tables = load_raw_tables(demo_dir)
    cleaned = clean_raw_tables(raw_tables)
    students = _add_targets(cleaned["student_info"])
    assessment_features = build_assessment_features(
        cleaned["student_assess"],
        cleaned["assessments"],
    )
    vle_features = build_vle_features(cleaned["student_vle"])
    return build_modeling_dataset(
        students=students,
        student_reg=cleaned["student_reg"],
        assessment_features=assessment_features,
        vle_features=vle_features,
    )


def _align_features(
    features: pd.DataFrame,
    model_features: list[str],
    medians: dict[str, float],
) -> pd.DataFrame:
    aligned = features.copy()
    for column in model_features:
        if column not in aligned.columns:
            aligned[column] = 0
    aligned = aligned[model_features]
    return aligned.fillna({column: medians.get(column, 0) for column in model_features})


def _classification_confidence(
    classifier: Any,
    features: pd.DataFrame,
    predictions: np.ndarray,
) -> np.ndarray:
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(features)
        return probabilities[np.arange(len(predictions)), predictions.astype(int)]
    return np.ones(len(predictions))


def _risk_label(predicted_pass: int, learning_percentage: float) -> str:
    if not predicted_pass or learning_percentage < 50:
        return "High"
    if learning_percentage < 70:
        return "Medium"
    return "Low"


def _risk_order(predicted_pass: int, learning_percentage: float) -> int:
    return {"High": 0, "Medium": 1, "Low": 2}[_risk_label(predicted_pass, learning_percentage)]


def _engagement_score(dataset: pd.DataFrame, index: int) -> int:
    clicks = dataset["total_clicks"].astype(float)
    max_clicks = clicks.max()
    if not max_clicks:
        return 0
    return int(round(float(clicks.iloc[index]) / float(max_clicks) * 100))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _percent(value: float) -> str:
    return f"{value:.1%}"
