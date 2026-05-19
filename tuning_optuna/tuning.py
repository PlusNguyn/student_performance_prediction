from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import optuna
import pandas as pd
from django.conf import settings
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor


SEED = 42
SUPPORTED_MODELS = ("xgboost", "lightgbm")
TARGET_COLUMN = "learning_percentage"
DEFAULT_SCORING = "neg_root_mean_squared_error"


@dataclass(frozen=True)
class TuningPaths:
    feature_dir: Path
    tuning_dir: Path

    @classmethod
    def from_defaults(
        cls,
        feature_dir: str | Path | None = None,
        tuning_dir: str | Path | None = None,
    ) -> "TuningPaths":
        base_dir = Path(settings.BASE_DIR)
        return cls(
            feature_dir=Path(feature_dir) if feature_dir else base_dir / "data" / "feature",
            tuning_dir=Path(tuning_dir) if tuning_dir else base_dir / "models" / "tuning_optuna",
        )


def tune_xgboost(
    paths: TuningPaths | None = None,
    n_trials: int = 40,
    cv_splits: int = 5,
    random_state: int = SEED,
    timeout: int | None = None,
    scoring: str = DEFAULT_SCORING,
) -> dict[str, Any]:
    paths = paths or TuningPaths.from_defaults()
    paths.tuning_dir.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train, y_train = load_training_features(paths.feature_dir)
    return tune_model(
        model_name="xgboost",
        X_train=X_train,
        y_train=y_train,
        paths=paths,
        n_trials=n_trials,
        cv_splits=cv_splits,
        random_state=random_state,
        timeout=timeout,
        scoring=scoring,
    )


def tune_best_model(
    paths: TuningPaths | None = None,
    model_names: list[str] | None = None,
    n_trials: int = 40,
    cv_splits: int = 5,
    random_state: int = SEED,
    timeout: int | None = None,
    scoring: str = DEFAULT_SCORING,
) -> dict[str, Any]:
    paths = paths or TuningPaths.from_defaults()
    paths.tuning_dir.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    selected_models = normalize_model_names(model_names)
    X_train, y_train = load_training_features(paths.feature_dir)

    results = {}
    for model_name in selected_models:
        results[model_name] = tune_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            paths=paths,
            n_trials=n_trials,
            cv_splits=cv_splits,
            random_state=random_state,
            timeout=timeout,
            scoring=scoring,
        )

    best_model_name = max(results, key=lambda name: results[name]["best_value"])
    best_result = results[best_model_name]
    best_config = {
        "task": "regression",
        "target_column": TARGET_COLUMN,
        "best_model": best_model_name,
        "best_value": best_result["best_value"],
        "best_params": best_result["best_params"],
        "scoring": scoring,
        "n_trials_per_model": n_trials,
        "cv_splits": cv_splits,
        "random_state": random_state,
        "models": {
            name: {
                "best_value": result["best_value"],
                "best_params": result["best_params"],
            }
            for name, result in results.items()
        },
    }

    output_paths = {
        "best_config": paths.tuning_dir / "best_model_config.json",
        "best_params": paths.tuning_dir / "best_model_params.json",
        "summary": paths.tuning_dir / "best_model_tuning_summary.json",
    }
    _write_json(output_paths["best_config"], best_config)
    _write_json(output_paths["best_params"], best_result["best_params"])
    _write_json(output_paths["summary"], best_config)
    postgres_sync = _sync_tuning_outputs_to_postgres(
        output_paths=output_paths,
        summary=best_config,
        model_name="best_model",
    )

    return {
        **best_config,
        "postgres_sync": postgres_sync,
        "paths": {name: str(path) for name, path in output_paths.items()},
    }


def tune_model(
    *,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    paths: TuningPaths,
    n_trials: int,
    cv_splits: int,
    random_state: int,
    timeout: int | None,
    scoring: str,
) -> dict[str, Any]:
    model_name = normalize_model_names([model_name])[0]

    def objective(trial: optuna.Trial) -> float:
        model = build_model(model_name, suggest_params(model_name, trial, random_state))
        cv = KFold(cv_splits, shuffle=True, random_state=random_state)
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
        )
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    output_paths = {
        "best_params": paths.tuning_dir / f"{model_name}_best_params.json",
        "summary": paths.tuning_dir / f"{model_name}_tuning_summary.json",
        "study": paths.tuning_dir / f"{model_name}_study.joblib",
    }
    summary = {
        "task": "regression",
        "target_column": TARGET_COLUMN,
        "model": model_name,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "scoring": scoring,
        "n_trials": len(study.trials),
        "cv_splits": cv_splits,
        "random_state": random_state,
    }
    _write_json(output_paths["best_params"], study.best_params)
    _write_json(output_paths["summary"], summary)
    joblib.dump(study, output_paths["study"])
    postgres_sync = _sync_tuning_outputs_to_postgres(
        output_paths=output_paths,
        summary=summary,
        model_name=model_name,
    )

    return {
        **summary,
        "postgres_sync": postgres_sync,
        "paths": {name: str(path) for name, path in output_paths.items()},
    }


def build_model(model_name: str, params: dict[str, Any]) -> XGBRegressor | LGBMRegressor:
    if model_name == "xgboost":
        return XGBRegressor(**params)
    if model_name == "lightgbm":
        return LGBMRegressor(**params)
    raise ValueError(f"Unsupported model: {model_name}")


def suggest_params(
    model_name: str,
    trial: optuna.Trial,
    random_state: int,
) -> dict[str, Any]:
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "random_state": random_state,
            "n_jobs": -1,
        }

    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "objective": "regression",
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }

    raise ValueError(f"Unsupported model: {model_name}")


def normalize_model_names(model_names: list[str] | None) -> list[str]:
    if not model_names:
        return list(SUPPORTED_MODELS)

    normalized = []
    for name in model_names:
        model_name = name.strip().lower()
        if model_name not in SUPPORTED_MODELS:
            allowed = ", ".join(SUPPORTED_MODELS)
            raise ValueError(f"Unsupported model `{name}`. Choose from: {allowed}.")
        if model_name not in normalized:
            normalized.append(model_name)

    return normalized


def load_training_features(feature_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    missing = [
        name
        for name in ["X_train.csv", "y_train.csv"]
        if not (feature_dir / name).exists()
    ]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing feature files in {feature_dir}: {joined}. "
            "Run `python manage.py preprocess_data` first."
        )

    X_train = pd.read_csv(feature_dir / "X_train.csv")
    target_column = _target_column(feature_dir)
    y_train = pd.read_csv(feature_dir / "y_train.csv")[target_column].astype(float)
    return X_train, y_train


def _target_column(feature_dir: Path) -> str:
    metadata_path = feature_dir / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8")).get(
            "target_column",
            TARGET_COLUMN,
        )
    return TARGET_COLUMN


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sync_tuning_outputs_to_postgres(
    *,
    output_paths: dict[str, Path],
    summary: dict[str, Any],
    model_name: str,
) -> dict[str, list[str]]:
    from tuning_optuna.storage import sync_tuning_outputs_to_postgres

    return sync_tuning_outputs_to_postgres(
        output_paths=output_paths,
        summary=summary,
        model_name=model_name,
    )
