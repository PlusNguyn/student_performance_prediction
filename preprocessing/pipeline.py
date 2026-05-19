from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


SEED = 42

CATEGORICAL_COLUMNS = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
]

VLE_FEATURE_COLUMNS = [
    "total_clicks",
    "active_days",
    "avg_daily_clicks",
    "max_clicks_day",
    "last_activity",
    "first_activity",
    "engagement_span",
]

ASSESSMENT_BEHAVIOR_COLUMNS = [
    "submission_count",
    "late_submissions",
]

ENGINEERED_FEATURE_COLUMNS = [
    "clicks_per_active_day",
    "activity_density",
    "clicks_per_credit",
    "late_submission_rate",
]

TARGET_COLUMN = "learning_percentage"

FEATURE_COLUMNS = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
    "total_clicks",
    "active_days",
    "avg_daily_clicks",
    "max_clicks_day",
    "engagement_span",
    "submission_count",
    "late_submissions",
    "clicks_per_active_day",
    "activity_density",
    "clicks_per_credit",
    "late_submission_rate",
]


@dataclass(frozen=True)
class PreprocessingPaths:
    raw_dir: Path
    preprocessed_dir: Path
    feature_dir: Path

    @classmethod
    def from_defaults(
        cls,
        raw_dir: str | Path | None = None,
        preprocessed_dir: str | Path | None = None,
        feature_dir: str | Path | None = None,
    ) -> "PreprocessingPaths":
        base_dir = Path(settings.BASE_DIR)
        return cls(
            raw_dir=Path(raw_dir) if raw_dir else base_dir / "data" / "raw",
            preprocessed_dir=(
                Path(preprocessed_dir)
                if preprocessed_dir
                else base_dir / "data" / "preprocessed"
            ),
            feature_dir=Path(feature_dir) if feature_dir else base_dir / "data" / "feature",
        )


def run_preprocessing(
    paths: PreprocessingPaths | None = None,
    test_size: float = 0.2,
    random_state: int = SEED,
) -> dict[str, Any]:
    paths = paths or PreprocessingPaths.from_defaults()
    _ensure_directories(paths)

    students = _read_students(paths.raw_dir)
    student_vle = _read_student_vle(paths.raw_dir)
    student_assessment = _read_student_assessment(paths.raw_dir)
    assessments = _read_assessments(paths.raw_dir)

    students = _add_binary_label(students)
    vle_features = build_vle_features(student_vle)
    assessment_features = build_assessment_features(student_assessment, assessments)
    dataset = build_model_dataset(students, vle_features, assessment_features)

    encoded_dataset, encoders = encode_categorical_columns(dataset)
    X = encoded_dataset[FEATURE_COLUMNS]
    medians = X.median(numeric_only=True).to_dict()
    X = X.fillna(medians)
    y = encoded_dataset[TARGET_COLUMN].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    output_paths = save_outputs(
        paths=paths,
        students=students,
        vle_features=vle_features,
        assessment_features=assessment_features,
        encoded_dataset=encoded_dataset,
        X=X,
        y=y,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        encoders=encoders,
        medians=medians,
        test_size=test_size,
        random_state=random_state,
    )
    postgres_sync = _sync_outputs_to_postgres(
        output_paths=output_paths,
        rows=int(encoded_dataset.shape[0]),
        features=len(FEATURE_COLUMNS),
        train_rows=int(X_train.shape[0]),
        test_rows=int(X_test.shape[0]),
        target_mean=float(y.mean()),
        target_min=float(y.min()),
        target_max=float(y.max()),
        test_size=test_size,
        random_state=random_state,
    )

    return {
        "rows": int(encoded_dataset.shape[0]),
        "features": len(FEATURE_COLUMNS),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "target": TARGET_COLUMN,
        "target_mean": float(y.mean()),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "postgres_sync": postgres_sync,
        "paths": output_paths,
    }


def build_vle_features(student_vle: pd.DataFrame) -> pd.DataFrame:
    vle_features = (
        student_vle.groupby("id_student", as_index=False)
        .agg(
            total_clicks=("sum_click", "sum"),
            active_days=("date", "nunique"),
            avg_daily_clicks=("sum_click", "mean"),
            max_clicks_day=("sum_click", "max"),
            last_activity=("date", "max"),
            first_activity=("date", "min"),
        )
    )
    vle_features["engagement_span"] = (
        vle_features["last_activity"] - vle_features["first_activity"]
    )
    return vle_features


def build_assessment_features(
    student_assessment: pd.DataFrame,
    assessments: pd.DataFrame,
) -> pd.DataFrame:
    merged = student_assessment.merge(
        assessments[["id_assessment", "weight"]],
        on="id_assessment",
        how="left",
    )
    merged["weight_filled"] = merged["weight"].fillna(1)
    merged["weighted_score"] = merged["score"] * merged["weight_filled"]

    assessment_features = (
        merged.groupby("id_student", as_index=False)
        .agg(
            avg_score=("score", "mean"),
            min_score=("score", "min"),
            submission_count=("id_assessment", "count"),
            late_submissions=("is_banked", "sum"),
            weighted_score_sum=("weighted_score", "sum"),
            weight_sum=("weight_filled", "sum"),
        )
    )
    assessment_features["weighted_avg"] = (
        assessment_features["weighted_score_sum"] / assessment_features["weight_sum"]
    )
    return assessment_features.drop(columns=["weighted_score_sum", "weight_sum"])


def build_model_dataset(
    students: pd.DataFrame,
    vle_features: pd.DataFrame,
    assessment_features: pd.DataFrame,
) -> pd.DataFrame:
    selected_columns = [
        "id_student",
        "code_module",
        "code_presentation",
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "num_of_prev_attempts",
        "studied_credits",
        "disability",
        "label",
    ]
    dataset = students[selected_columns].copy()
    dataset = dataset.merge(vle_features, on="id_student", how="left")
    dataset = dataset.merge(assessment_features, on="id_student", how="left")
    dataset[VLE_FEATURE_COLUMNS] = dataset[VLE_FEATURE_COLUMNS].fillna(0)
    dataset[ASSESSMENT_BEHAVIOR_COLUMNS] = dataset[ASSESSMENT_BEHAVIOR_COLUMNS].fillna(0)
    dataset = add_engineered_features(dataset)
    dataset[TARGET_COLUMN] = (
        dataset["weighted_avg"]
        .fillna(dataset["avg_score"])
        .fillna(0)
        .clip(lower=0, upper=100)
    )
    return dataset


def add_engineered_features(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.copy()
    dataset["clicks_per_active_day"] = safe_divide(
        dataset["total_clicks"],
        dataset["active_days"],
    )
    dataset["activity_density"] = safe_divide(
        dataset["active_days"],
        dataset["engagement_span"] + 1,
    )
    dataset["clicks_per_credit"] = safe_divide(
        dataset["total_clicks"],
        dataset["studied_credits"],
    )
    dataset["late_submission_rate"] = safe_divide(
        dataset["late_submissions"],
        dataset["submission_count"],
    )
    dataset[ENGINEERED_FEATURE_COLUMNS] = dataset[ENGINEERED_FEATURE_COLUMNS].fillna(0)
    return dataset


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator != 0)).fillna(0)


def encode_categorical_columns(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    encoded_dataset = dataset.copy()
    encoders: dict[str, LabelEncoder] = {}

    for column in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        encoded_dataset[column] = encoder.fit_transform(
            encoded_dataset[column].astype(str)
        )
        encoders[column] = encoder

    return encoded_dataset, encoders


def save_outputs(
    *,
    paths: PreprocessingPaths,
    students: pd.DataFrame,
    vle_features: pd.DataFrame,
    assessment_features: pd.DataFrame,
    encoded_dataset: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    encoders: dict[str, LabelEncoder],
    medians: dict[str, float],
    test_size: float,
    random_state: int,
) -> dict[str, str]:
    preprocessed_files = {
        "students_labeled": paths.preprocessed_dir / "students_labeled.csv",
        "vle_features": paths.preprocessed_dir / "vle_features.csv",
        "assessment_features": paths.preprocessed_dir / "assessment_features.csv",
        "preprocessed_dataset": paths.preprocessed_dir / "preprocessed_dataset.csv",
    }
    students.to_csv(preprocessed_files["students_labeled"], index=False)
    vle_features.to_csv(preprocessed_files["vle_features"], index=False)
    assessment_features.to_csv(preprocessed_files["assessment_features"], index=False)
    encoded_dataset.to_csv(preprocessed_files["preprocessed_dataset"], index=False)

    feature_files = {
        "X": paths.feature_dir / "X.csv",
        "y": paths.feature_dir / "y.csv",
        "X_train": paths.feature_dir / "X_train.csv",
        "X_test": paths.feature_dir / "X_test.csv",
        "y_train": paths.feature_dir / "y_train.csv",
        "y_test": paths.feature_dir / "y_test.csv",
        "feature_columns": paths.feature_dir / "feature_columns.json",
        "medians": paths.feature_dir / "medians.json",
        "encoders": paths.feature_dir / "encoders.joblib",
        "metadata": paths.feature_dir / "metadata.json",
    }
    X.to_csv(feature_files["X"], index=False)
    y.rename(TARGET_COLUMN).to_csv(feature_files["y"], index=False)
    X_train.to_csv(feature_files["X_train"], index=False)
    X_test.to_csv(feature_files["X_test"], index=False)
    y_train.rename(TARGET_COLUMN).to_csv(feature_files["y_train"], index=False)
    y_test.rename(TARGET_COLUMN).to_csv(feature_files["y_test"], index=False)

    _write_json(feature_files["feature_columns"], FEATURE_COLUMNS)
    _write_json(feature_files["medians"], medians)
    joblib.dump(encoders, feature_files["encoders"])
    _write_json(
        feature_files["metadata"],
        {
            "test_size": test_size,
            "random_state": random_state,
            "rows": int(encoded_dataset.shape[0]),
            "features": len(FEATURE_COLUMNS),
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "target_column": TARGET_COLUMN,
            "target_description": (
                "Learning percentage derived from weighted assessment scores. "
                "Students without assessment scores are assigned 0."
            ),
            "target_mean": float(y.mean()),
            "target_min": float(y.min()),
            "target_max": float(y.max()),
            "categorical_columns": CATEGORICAL_COLUMNS,
            "engineered_feature_columns": ENGINEERED_FEATURE_COLUMNS,
            "feature_columns": FEATURE_COLUMNS,
        },
    )

    return {
        name: str(path)
        for name, path in (preprocessed_files | feature_files).items()
    }


def _add_binary_label(students: pd.DataFrame) -> pd.DataFrame:
    students = students.copy()
    students["label"] = students["final_result"].isin(["Pass", "Distinction"]).astype(int)
    return students


def _ensure_directories(paths: PreprocessingPaths) -> None:
    paths.preprocessed_dir.mkdir(parents=True, exist_ok=True)
    paths.feature_dir.mkdir(parents=True, exist_ok=True)


def _read_students(raw_dir: Path) -> pd.DataFrame:
    return pd.read_csv(
        raw_dir / "studentInfo.csv",
        na_values=["?"],
        dtype={
            "id_student": "int32",
            "num_of_prev_attempts": "int16",
            "studied_credits": "int16",
        },
    )


def _read_student_vle(raw_dir: Path) -> pd.DataFrame:
    return pd.read_csv(
        raw_dir / "studentVle.csv",
        usecols=["id_student", "date", "sum_click"],
        na_values=["?"],
        dtype={
            "id_student": "int32",
            "date": "int16",
            "sum_click": "int32",
        },
    )


def _read_student_assessment(raw_dir: Path) -> pd.DataFrame:
    return pd.read_csv(
        raw_dir / "studentAssessment.csv",
        usecols=["id_assessment", "id_student", "is_banked", "score"],
        na_values=["?"],
        dtype={
            "id_assessment": "int32",
            "id_student": "int32",
            "is_banked": "int8",
            "score": "float32",
        },
    )


def _read_assessments(raw_dir: Path) -> pd.DataFrame:
    return pd.read_csv(
        raw_dir / "assessments.csv",
        usecols=["id_assessment", "weight"],
        na_values=["?"],
        dtype={
            "id_assessment": "int32",
            "weight": "float32",
        },
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _sync_outputs_to_postgres(
    *,
    output_paths: dict[str, str],
    rows: int,
    features: int,
    train_rows: int,
    test_rows: int,
    target_mean: float,
    target_min: float,
    target_max: float,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    from preprocessing.storage import sync_preprocessing_outputs_to_postgres

    return sync_preprocessing_outputs_to_postgres(
        output_paths,
        metadata={
            "pipeline": "preprocessing",
            "rows": rows,
            "features": features,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "target_column": TARGET_COLUMN,
            "target_mean": target_mean,
            "target_min": target_min,
            "target_max": target_max,
            "test_size": test_size,
            "random_state": random_state,
        },
    )
