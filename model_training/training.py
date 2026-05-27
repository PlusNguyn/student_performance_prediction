from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)


SEED = 42
TARGET_CLASS_COLUMN = "target_pass"
TARGET_REGRESSION_COLUMN = "learning_percentage"
TARGET_COLUMN = TARGET_REGRESSION_COLUMN
CLASSIFICATION_MODEL_NAME = "random_forest_classifier"
REGRESSION_MODEL_NAME = "random_forest_regressor"


@dataclass(frozen=True)
class TrainingPaths:
    '''
    Chứa các đường dẫn đầu vào và đầu ra được sử dụng khi huấn luyện model.
    '''
    feature_dir: Path
    model_dir: Path
    params_path: Path | None = None

    @classmethod
    def from_defaults(
        cls,
        feature_dir: str | Path | None = None,
        model_dir: str | Path | None = None,
        params_path: str | Path | None = None,
    ) -> "TrainingPaths":
        '''
        Tạo cấu hình đường dẫn từ tham số truyền vào hoặc từ thư mục mặc định.
        '''
        base_dir = Path(settings.BASE_DIR)
        return cls(
            feature_dir=Path(feature_dir) if feature_dir else base_dir / "data" / "feature",
            model_dir=Path(model_dir) if model_dir else base_dir / "models",
            params_path=Path(params_path) if params_path else None,
        )


def train_models(
    paths: TrainingPaths | None = None,
    random_state: int = SEED,
) -> dict[str, Any]:
    '''
    Huấn luyện model phân loại và hồi quy từ các feature split đã tiền xử lý.
    Hàm lưu model, báo cáo đánh giá, đồng bộ tracking và trả về kết quả huấn luyện.
    '''
    paths = paths or TrainingPaths.from_defaults()
    paths.model_dir.mkdir(parents=True, exist_ok=True)

    # Nạp dữ liệu đầu vào và cấu hình tùy chọn trước khi khởi tạo estimator.
    splits = load_feature_splits(paths.feature_dir)
    feature_columns = _read_json(paths.feature_dir / "feature_columns.json")
    medians = _read_json(paths.feature_dir / "medians.json")
    training_config = load_training_config(paths.params_path)

    classifier = RandomForestClassifier(
        **_classification_params(training_config, random_state)
    )
    classifier.fit(splits["X_train_class"], splits["y_train_class"])

    regressor = RandomForestRegressor(
        **_regression_params(training_config, random_state)
    )
    regressor.fit(splits["X_train_reg"], splits["y_train_reg"])

    # Đánh giá hai model trên các tập test tương ứng.
    class_predictions = classifier.predict(splits["X_test_class"])
    reg_predictions = np.clip(
        regressor.predict(splits["X_test_reg"]),
        0,
        100,
    )
    metrics = {
        "classification": evaluate_classifier(
            classifier,
            splits["X_test_class"],
            splits["y_test_class"],
            class_predictions,
        ),
        "regression": evaluate_regressor(
            splits["y_test_reg"],
            reg_predictions,
        ),
    }
    report = classification_report(
        splits["y_test_class"],
        class_predictions,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(splits["y_test_class"], class_predictions).tolist()
    model_selection = {
        "task": "classification_and_regression",
        "classification_model": CLASSIFICATION_MODEL_NAME,
        "regression_model": REGRESSION_MODEL_NAME,
        "classification_target": TARGET_CLASS_COLUMN,
        "regression_target": TARGET_REGRESSION_COLUMN,
        "classification_selection_metric": "f1",
        "regression_selection_metric": "rmse",
        "training_config": training_config,
    }

    output_paths = {
        "classification_model": paths.model_dir / "classification_model.joblib",
        "regression_model": paths.model_dir / "regression_model.joblib",
        "metrics": paths.model_dir / "model_metrics.json",
        "model_selection": paths.model_dir / "model_selection.json",
        "classification_report": paths.model_dir / "classification_report.json",
        "confusion_matrix": paths.model_dir / "confusion_matrix.json",
        "features": paths.model_dir / "features.json",
        "medians": paths.model_dir / "medians.json",
    }
    # Lưu model đã fit và các artifact mô tả kết quả huấn luyện.
    joblib.dump(classifier, output_paths["classification_model"])
    joblib.dump(regressor, output_paths["regression_model"])
    _write_json(output_paths["metrics"], metrics)
    _write_json(output_paths["model_selection"], model_selection)
    _write_json(output_paths["classification_report"], report)
    _write_json(
        output_paths["confusion_matrix"],
        {
            "labels": ["fail_or_withdrawn", "pass_or_distinction"],
            "matrix": confusion,
        },
    )
    _write_json(output_paths["features"], feature_columns)
    _write_json(output_paths["medians"], medians)

    mlflow_tracking = log_training_to_mlflow(
        classifier=classifier,
        regressor=regressor,
        metrics=metrics,
        model_selection=model_selection,
        feature_columns=feature_columns,
        output_paths=output_paths,
        model_dir=paths.model_dir,
        random_state=random_state,
    )
    postgres_sync = _safe_sync_training_outputs_to_postgres(
        output_paths=output_paths,
        metrics=metrics,
        model_selection=model_selection,
        mlflow_tracking=mlflow_tracking,
    )

    return {
        "metrics": metrics,
        "models": {
            "classification": CLASSIFICATION_MODEL_NAME,
            "regression": REGRESSION_MODEL_NAME,
        },
        "mlflow": mlflow_tracking,
        "postgres_sync": postgres_sync,
        "paths": {name: str(path) for name, path in output_paths.items()},
    }


def load_feature_splits(feature_dir: Path) -> dict[str, pd.DataFrame | pd.Series]:
    '''
    Đọc các tập train/test và metadata cần thiết từ đầu ra của bước tiền xử lý.
    Raise lỗi nếu thiếu bất kỳ file bắt buộc nào cho quá trình huấn luyện.
    '''
    required_files = [
        "X_train_class.csv",
        "X_test_class.csv",
        "y_train_class.csv",
        "y_test_class.csv",
        "X_train_reg.csv",
        "X_test_reg.csv",
        "y_train_reg.csv",
        "y_test_reg.csv",
        "feature_columns.json",
        "medians.json",
    ]
    missing = [name for name in required_files if not (feature_dir / name).exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing feature files in {feature_dir}: {joined}. "
            "Run `python manage.py preprocess_data` first."
        )

    return {
        "X_train_class": pd.read_csv(feature_dir / "X_train_class.csv"),
        "X_test_class": pd.read_csv(feature_dir / "X_test_class.csv"),
        "y_train_class": pd.read_csv(feature_dir / "y_train_class.csv")[
            TARGET_CLASS_COLUMN
        ].astype(int),
        "y_test_class": pd.read_csv(feature_dir / "y_test_class.csv")[
            TARGET_CLASS_COLUMN
        ].astype(int),
        "X_train_reg": pd.read_csv(feature_dir / "X_train_reg.csv"),
        "X_test_reg": pd.read_csv(feature_dir / "X_test_reg.csv"),
        "y_train_reg": pd.read_csv(feature_dir / "y_train_reg.csv")[
            TARGET_REGRESSION_COLUMN
        ].astype(float),
        "y_test_reg": pd.read_csv(feature_dir / "y_test_reg.csv")[
            TARGET_REGRESSION_COLUMN
        ].astype(float),
    }


def evaluate_classifier(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
) -> dict[str, float]:
    '''
    Tính các metric đánh giá cho model phân loại.
    ROC AUC chỉ được tính khi tập test chứa đủ hai lớp và model có xác suất dự đoán.
    '''
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }
    if y_test.nunique() > 1 and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
    return metrics


def evaluate_regressor(
    y_test: pd.Series,
    predictions: np.ndarray,
) -> dict[str, float]:
    '''
    Tính các metric đánh giá cho model hồi quy dự đoán phần trăm học tập.
    '''
    errors = np.abs(predictions - y_test)
    return {
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "within_10_percent": float((errors <= 10).mean()),
    }


def log_training_to_mlflow(
    *,
    classifier: RandomForestClassifier,
    regressor: RandomForestRegressor,
    metrics: dict[str, dict[str, float]],
    model_selection: dict[str, Any],
    feature_columns: list[str],
    output_paths: dict[str, Path],
    model_dir: Path,
    random_state: int,
) -> dict[str, Any]:
    '''
    Ghi tham số, metric, artifact và registered models của lần huấn luyện lên MLflow.
    Khi MLflow không khả dụng hoặc logging thất bại, trả về trạng thái và lý do.
    '''
    tracking_uri = getattr(settings, "MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        return {"enabled": False, "reason": "MLFLOW_TRACKING_URI is not configured."}

    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        return {"enabled": False, "reason": f"MLflow is not installed: {exc}"}

    registered_base_name = settings.MLFLOW_REGISTERED_MODEL_NAME
    registered_model_names = {
        "classification": f"{registered_base_name}-classification",
        "regression": f"{registered_base_name}-regression",
    }

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

            with mlflow.start_run(run_name="train-classification-regression") as run:
                mlflow.log_param("task", model_selection["task"])
                mlflow.log_param(
                    "classification_model",
                    model_selection["classification_model"],
                )
                mlflow.log_param("regression_model", model_selection["regression_model"])
                mlflow.log_param("random_state", random_state)
                mlflow.log_param("feature_count", len(feature_columns))
                mlflow.log_param("classification_target", TARGET_CLASS_COLUMN)
                mlflow.log_param("regression_target", TARGET_REGRESSION_COLUMN)
                _log_estimator_params(mlflow, "classification", classifier)
                _log_estimator_params(mlflow, "regression", regressor)

                for task_name, task_metrics in metrics.items():
                    for metric_name, value in task_metrics.items():
                        mlflow.log_metric(f"{task_name}.{metric_name}", value)

                for artifact_name in [
                    "metrics",
                    "model_selection",
                    "classification_report",
                    "confusion_matrix",
                    "features",
                    "medians",
                ]:
                    mlflow.log_artifact(str(output_paths[artifact_name]))

                mlflow.sklearn.log_model(
                    sk_model=classifier,
                    artifact_path="classification_model",
                    registered_model_name=registered_model_names["classification"],
                )
                mlflow.sklearn.log_model(
                    sk_model=regressor,
                    artifact_path="regression_model",
                    registered_model_name=registered_model_names["regression"],
                )

                return {
                    "enabled": True,
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "tracking_uri": tracking_uri,
                    "registered_model_names": registered_model_names,
                    "registered_model_name": ",".join(registered_model_names.values()),
                }
    except Exception as exc:
        return {
            "enabled": True,
            "tracking_uri": tracking_uri,
            "error": str(exc),
        }


def load_training_config(params_path: Path | None) -> dict[str, Any] | None:
    '''
    Đọc cấu hình tham số huấn luyện từ file JSON nếu file được cung cấp và tồn tại.
    '''
    if not params_path or not params_path.exists():
        return None
    return _read_json(params_path)


def _classification_params(
    training_config: dict[str, Any] | None,
    random_state: int,
) -> dict[str, Any]:
    '''
    Tạo bộ tham số cho classifier và ghi đè bằng cấu hình tùy chọn nếu có.
    '''
    params: dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "random_state": random_state,
        "n_jobs": -1,
    }
    params.update((training_config or {}).get("classification_model_params", {}))
    return params


def _regression_params(
    training_config: dict[str, Any] | None,
    random_state: int,
) -> dict[str, Any]:
    '''
    Tạo bộ tham số cho regressor và ghi đè bằng cấu hình tùy chọn nếu có.
    '''
    params: dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 1,
        "random_state": random_state,
        "n_jobs": -1,
    }
    params.update((training_config or {}).get("regression_model_params", {}))
    return params


def _log_estimator_params(mlflow_module: Any, prefix: str, model: Any) -> None:
    '''
    Ghi các tham số đơn giản của estimator vào MLflow với tiền tố theo task.
    '''
    for key, value in model.get_params().items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            mlflow_module.log_param(f"{prefix}.{key}", value)


def _safe_sync_training_outputs_to_postgres(
    *,
    output_paths: dict[str, Path],
    metrics: dict[str, dict[str, float]],
    model_selection: dict[str, Any],
    mlflow_tracking: dict[str, Any],
) -> dict[str, list[str]] | dict[str, Any]:
    '''
    Đồng bộ đầu ra huấn luyện sang PostgreSQL và chuyển lỗi thành kết quả trả về.
    '''
    try:
        return _sync_training_outputs_to_postgres(
            output_paths=output_paths,
            metrics=metrics,
            model_selection=model_selection,
            mlflow_tracking=mlflow_tracking,
        )
    except Exception as exc:
        return {
            "models": [],
            "metadata": [],
            "explainers": [],
            "error": str(exc),
        }


def _sync_training_outputs_to_postgres(
    *,
    output_paths: dict[str, Path],
    metrics: dict[str, dict[str, float]],
    model_selection: dict[str, Any],
    mlflow_tracking: dict[str, Any],
) -> dict[str, list[str]]:
    '''
    Chuyển artifact và metadata của lần huấn luyện sang tầng lưu trữ PostgreSQL.
    '''
    from model_training.storage import sync_training_outputs_to_postgres

    return sync_training_outputs_to_postgres(
        output_paths=output_paths,
        metrics=metrics,
        best_model_name="classification_and_regression",
        model_selection=model_selection,
        mlflow_tracking=mlflow_tracking,
    )


def _read_json(path: Path) -> Any:
    '''
    Đọc và phân tích nội dung của một file JSON UTF-8.
    '''
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    '''
    Ghi payload thành file JSON UTF-8 với định dạng dễ đọc.
    '''
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
