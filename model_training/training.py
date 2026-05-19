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
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor


SEED = 42
TARGET_COLUMN = "learning_percentage"


@dataclass(frozen=True)
class TrainingPaths:
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
    paths = paths or TrainingPaths.from_defaults()
    paths.model_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_feature_splits(paths.feature_dir)
    feature_columns = _read_json(paths.feature_dir / "feature_columns.json")
    medians = _read_json(paths.feature_dir / "medians.json")
    tuning_config = load_tuning_config(paths.params_path)

    xgb = XGBRegressor(**_xgboost_params(tuning_config, random_state))
    xgb.fit(X_train, y_train)

    lgbm = LGBMRegressor(**_lightgbm_params(tuning_config, random_state))
    lgbm.fit(X_train, y_train)

    metrics = {
        "xgboost": evaluate_model(xgb, X_test, y_test),
        "lightgbm": evaluate_model(lgbm, X_test, y_test),
    }
    best_model_name = min(metrics, key=lambda name: metrics[name]["rmse"])
    best_model = {"xgboost": xgb, "lightgbm": lgbm}[best_model_name]
    model_selection = {
        "best_model": best_model_name,
        "selection_metric": "rmse",
        "best_rmse": metrics[best_model_name]["rmse"],
        "target_column": TARGET_COLUMN,
        "tuning_config": tuning_config,
    }

    output_paths = {
        "xgboost_model": paths.model_dir / "xgboost_model.joblib",
        "lightgbm_model": paths.model_dir / "lightgbm_model.joblib",
        "best_model": paths.model_dir / "best_model.joblib",
        "metrics": paths.model_dir / "model_metrics.json",
        "model_selection": paths.model_dir / "model_selection.json",
        "features": paths.model_dir / "features.json",
        "medians": paths.model_dir / "medians.json",
    }
    joblib.dump(xgb, output_paths["xgboost_model"])
    joblib.dump(lgbm, output_paths["lightgbm_model"])
    joblib.dump(best_model, output_paths["best_model"])
    _write_json(output_paths["metrics"], metrics)
    _write_json(output_paths["model_selection"], model_selection)
    _write_json(output_paths["features"], feature_columns)
    _write_json(output_paths["medians"], medians)

    shap_explainer_path = save_shap_explainer_if_available(xgb, paths.model_dir)
    if shap_explainer_path:
        output_paths["xgboost_explainer"] = shap_explainer_path

    mlflow_tracking = log_training_to_mlflow(
        models={"xgboost": xgb, "lightgbm": lgbm},
        best_model_name=best_model_name,
        metrics=metrics,
        tuning_config=tuning_config,
        model_selection=model_selection,
        feature_columns=feature_columns,
        output_paths=output_paths,
        model_dir=paths.model_dir,
        random_state=random_state,
    )
    postgres_sync = _sync_training_outputs_to_postgres(
        output_paths=output_paths,
        metrics=metrics,
        best_model_name=best_model_name,
        model_selection=model_selection,
        mlflow_tracking=mlflow_tracking,
    )

    return {
        "metrics": metrics,
        "best_model": best_model_name,
        "mlflow": mlflow_tracking,
        "postgres_sync": postgres_sync,
        "paths": {name: str(path) for name, path in output_paths.items()},
    }


def load_feature_splits(
    feature_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    required_files = [
        "X_train.csv",
        "X_test.csv",
        "y_train.csv",
        "y_test.csv",
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

    X_train = pd.read_csv(feature_dir / "X_train.csv")
    X_test = pd.read_csv(feature_dir / "X_test.csv")
    target_column = _target_column(feature_dir)
    y_train = pd.read_csv(feature_dir / "y_train.csv")[target_column].astype(float)
    y_test = pd.read_csv(feature_dir / "y_test.csv")[target_column].astype(float)
    return X_train, X_test, y_train, y_test


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = np.clip(model.predict(X_test), 0, 100)
    errors = np.abs(predictions - y_test)
    return {
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "within_10_percent": float((errors <= 10).mean()),
    }


def save_shap_explainer_if_available(model: XGBRegressor, model_dir: Path) -> Path | None:
    try:
        import shap
    except ImportError:
        return None

    explainer = shap.TreeExplainer(model)
    path = model_dir / "xgboost_explainer.joblib"
    joblib.dump(explainer, path)
    return path


def log_training_to_mlflow(
    *,
    models: dict[str, Any],
    best_model_name: str,
    metrics: dict[str, dict[str, float]],
    tuning_config: dict[str, Any] | None,
    model_selection: dict[str, Any],
    feature_columns: list[str],
    output_paths: dict[str, Path],
    model_dir: Path,
    random_state: int,
) -> dict[str, Any]:
    tracking_uri = getattr(settings, "MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        return {"enabled": False, "reason": "MLFLOW_TRACKING_URI is not configured."}

    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        return {"enabled": False, "reason": f"MLflow is not installed: {exc}"}

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

            with mlflow.start_run(run_name=f"train-{best_model_name}") as run:
                mlflow.log_param("best_model", best_model_name)
                mlflow.log_param("selection_metric", model_selection["selection_metric"])
                mlflow.log_param("random_state", random_state)
                mlflow.log_param("feature_count", len(feature_columns))
                mlflow.log_param("target_column", TARGET_COLUMN)

                if tuning_config:
                    mlflow.log_param("tuning.best_model", tuning_config.get("best_model"))
                    mlflow.log_param("tuning.scoring", tuning_config.get("scoring"))
                    mlflow.log_param("tuning.cv_splits", tuning_config.get("cv_splits"))
                    mlflow.log_param(
                        "tuning.n_trials_per_model",
                        tuning_config.get("n_trials_per_model"),
                    )
                    _log_model_params(mlflow, tuning_config)

                _log_estimator_params(mlflow, models)

                for model_name, model_metrics in metrics.items():
                    for metric_name, value in model_metrics.items():
                        mlflow.log_metric(f"{model_name}.{metric_name}", value)
                for metric_name, value in metrics[best_model_name].items():
                    mlflow.log_metric(f"best_model.{metric_name}", value)

                for artifact_name in ["metrics", "model_selection", "features", "medians"]:
                    mlflow.log_artifact(str(output_paths[artifact_name]))

                mlflow.log_artifacts(str(model_dir), artifact_path="local_artifacts")
                mlflow.sklearn.log_model(
                    sk_model=models[best_model_name],
                    artifact_path="best_model",
                    registered_model_name=settings.MLFLOW_REGISTERED_MODEL_NAME,
                )

                return {
                    "enabled": True,
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "tracking_uri": tracking_uri,
                    "registered_model_name": settings.MLFLOW_REGISTERED_MODEL_NAME,
                }
    except Exception as exc:
        return {
            "enabled": True,
            "tracking_uri": tracking_uri,
            "error": str(exc),
        }


def load_tuning_config(params_path: Path | None) -> dict[str, Any] | None:
    if not params_path or not params_path.exists():
        return None

    payload = _read_json(params_path)
    if "best_model" in payload and "best_params" in payload:
        return payload

    return {
        "best_model": "xgboost",
        "best_params": payload,
        "source": "legacy_xgboost_params",
    }


def _xgboost_params(
    tuning_config: dict[str, Any] | None,
    random_state: int,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": random_state,
        "n_jobs": -1,
    }
    params.update(_model_tuned_params(tuning_config, "xgboost"))
    return params


def _lightgbm_params(
    tuning_config: dict[str, Any] | None,
    random_state: int,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "objective": "regression",
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    params.update(_model_tuned_params(tuning_config, "lightgbm"))
    return params


def _model_tuned_params(
    tuning_config: dict[str, Any] | None,
    model_name: str,
) -> dict[str, Any]:
    if not tuning_config:
        return {}

    model_results = tuning_config.get("models", {})
    if model_name in model_results:
        return model_results[model_name]["best_params"]

    if tuning_config["best_model"] == model_name:
        return tuning_config["best_params"]

    return {}


def _log_model_params(mlflow_module: Any, tuning_config: dict[str, Any]) -> None:
    model_results = tuning_config.get("models")
    if model_results:
        for model_name, result in model_results.items():
            for key, value in result.get("best_params", {}).items():
                mlflow_module.log_param(f"{model_name}.{key}", value)
        return

    best_model = tuning_config.get("best_model")
    for key, value in tuning_config.get("best_params", {}).items():
        mlflow_module.log_param(f"{best_model}.{key}", value)


def _log_estimator_params(mlflow_module: Any, models: dict[str, Any]) -> None:
    for model_name, model in models.items():
        for key, value in model.get_params().items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                mlflow_module.log_param(f"trained.{model_name}.{key}", value)


def _target_column(feature_dir: Path) -> str:
    metadata_path = feature_dir / "metadata.json"
    if metadata_path.exists():
        return _read_json(metadata_path).get("target_column", TARGET_COLUMN)
    return TARGET_COLUMN


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sync_training_outputs_to_postgres(
    *,
    output_paths: dict[str, Path],
    metrics: dict[str, dict[str, float]],
    best_model_name: str,
    model_selection: dict[str, Any],
    mlflow_tracking: dict[str, Any],
) -> dict[str, list[str]]:
    from model_training.storage import sync_training_outputs_to_postgres

    return sync_training_outputs_to_postgres(
        output_paths=output_paths,
        metrics=metrics,
        best_model_name=best_model_name,
        model_selection=model_selection,
        mlflow_tracking=mlflow_tracking,
    )
