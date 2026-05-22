from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import optuna
import pandas as pd
from django.conf import settings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score


SEED = 42
CLASSIFICATION_TASK = "classification"
REGRESSION_TASK = "regression"
CLASSIFICATION_MODEL_NAME = "random_forest_classifier"
REGRESSION_MODEL_NAME = "random_forest_regressor"
SUPPORTED_MODELS = (CLASSIFICATION_MODEL_NAME, REGRESSION_MODEL_NAME)
TARGET_CLASS_COLUMN = "target_pass"
TARGET_REGRESSION_COLUMN = "learning_percentage"
DEFAULT_CLASSIFICATION_SCORING = "f1"
DEFAULT_REGRESSION_SCORING = "neg_root_mean_squared_error"
MODEL_ALIASES = {
    "classification": CLASSIFICATION_MODEL_NAME,
    "classifier": CLASSIFICATION_MODEL_NAME,
    "random_forest_classifier": CLASSIFICATION_MODEL_NAME,
    "randomforestclassifier": CLASSIFICATION_MODEL_NAME,
    "rfc": CLASSIFICATION_MODEL_NAME,
    "regression": REGRESSION_MODEL_NAME,
    "regressor": REGRESSION_MODEL_NAME,
    "random_forest_regressor": REGRESSION_MODEL_NAME,
    "randomforestregressor": REGRESSION_MODEL_NAME,
    "rfr": REGRESSION_MODEL_NAME,
}


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
            feature_dir=(
                Path(feature_dir) if feature_dir else base_dir / "data" / "feature"
            ),
            tuning_dir=(
                Path(tuning_dir)
                if tuning_dir
                else base_dir / "models" / "tuning_optuna"
            ),
        )


def tune_random_forest(
    paths: TuningPaths | None = None,
    model_names: list[str] | None = None,
    n_trials: int = 40,
    cv_splits: int = 5,
    random_state: int = SEED,
    timeout: int | None = None,
    classification_scoring: str = DEFAULT_CLASSIFICATION_SCORING,
    regression_scoring: str = DEFAULT_REGRESSION_SCORING,
) -> dict[str, Any]:
    return tune_best_model(
        paths=paths,
        model_names=model_names,
        n_trials=n_trials,
        cv_splits=cv_splits,
        random_state=random_state,
        timeout=timeout,
        classification_scoring=classification_scoring,
        regression_scoring=regression_scoring,
    )


def tune_random_forest_classifier(
    paths: TuningPaths | None = None,
    n_trials: int = 40,
    cv_splits: int = 5,
    random_state: int = SEED,
    timeout: int | None = None,
    scoring: str = DEFAULT_CLASSIFICATION_SCORING,
) -> dict[str, Any]:
    paths = paths or TuningPaths.from_defaults()
    paths.tuning_dir.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train, y_train = load_training_features(paths.feature_dir, CLASSIFICATION_TASK)
    return tune_model(
        model_name=CLASSIFICATION_MODEL_NAME,
        X_train=X_train,
        y_train=y_train,
        paths=paths,
        n_trials=n_trials,
        cv_splits=cv_splits,
        random_state=random_state,
        timeout=timeout,
        scoring=scoring,
    )


def tune_random_forest_regressor(
    paths: TuningPaths | None = None,
    n_trials: int = 40,
    cv_splits: int = 5,
    random_state: int = SEED,
    timeout: int | None = None,
    scoring: str = DEFAULT_REGRESSION_SCORING,
) -> dict[str, Any]:
    paths = paths or TuningPaths.from_defaults()
    paths.tuning_dir.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train, y_train = load_training_features(paths.feature_dir, REGRESSION_TASK)
    return tune_model(
        model_name=REGRESSION_MODEL_NAME,
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
    scoring: str | None = None,
    classification_scoring: str = DEFAULT_CLASSIFICATION_SCORING,
    regression_scoring: str = DEFAULT_REGRESSION_SCORING,
) -> dict[str, Any]:
    paths = paths or TuningPaths.from_defaults()
    paths.tuning_dir.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    selected_models = normalize_model_names(model_names)
    if scoring:
        regression_scoring = scoring

    results = {}
    for model_name in selected_models:
        task = model_task(model_name)
        X_train, y_train = load_training_features(paths.feature_dir, task)
        task_scoring = (
            classification_scoring
            if task == CLASSIFICATION_TASK
            else regression_scoring
        )
        results[model_name] = tune_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            paths=paths,
            n_trials=n_trials,
            cv_splits=cv_splits,
            random_state=random_state,
            timeout=timeout,
            scoring=task_scoring,
        )

    best_params = training_params_config(results)
    summary = {
        "task": "classification_and_regression",
        "classification_model": CLASSIFICATION_MODEL_NAME,
        "regression_model": REGRESSION_MODEL_NAME,
        "classification_target": TARGET_CLASS_COLUMN,
        "regression_target": TARGET_REGRESSION_COLUMN,
        "classification_scoring": classification_scoring,
        "regression_scoring": regression_scoring,
        "scoring": {
            CLASSIFICATION_TASK: classification_scoring,
            REGRESSION_TASK: regression_scoring,
        },
        "n_trials_per_model": n_trials,
        "cv_splits": cv_splits,
        "random_state": random_state,
        "best_params": best_params,
        "models": {
            name: {
                "task": result["task"],
                "target_column": result["target_column"],
                "best_value": result["best_value"],
                "best_params": result["best_params"],
                "scoring": result["scoring"],
                "cv_splits": result["cv_splits"],
            }
            for name, result in results.items()
        },
    }

    output_paths = {
        "best_config": paths.tuning_dir / "best_model_config.json",
        "best_params": paths.tuning_dir / "best_model_params.json",
        "summary": paths.tuning_dir / "best_model_tuning_summary.json",
    }
    _write_json(output_paths["best_config"], summary)
    _write_json(output_paths["best_params"], best_params)
    _write_json(output_paths["summary"], summary)
    postgres_sync = _sync_tuning_outputs_to_postgres(
        output_paths=output_paths,
        summary=summary,
        model_name="random_forest",
    )

    return {
        **summary,
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
    task = model_task(model_name)
    target_column = target_column_for_task(task)
    cv = build_cv(task, y_train, cv_splits, random_state)

    def objective(trial: optuna.Trial) -> float:
        model = build_model(model_name, suggest_params(model_name, trial, random_state))
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )

    output_paths = {
        "best_params": paths.tuning_dir / f"{model_name}_best_params.json",
        "summary": paths.tuning_dir / f"{model_name}_tuning_summary.json",
        "study": paths.tuning_dir / f"{model_name}_study.joblib",
    }
    summary = {
        "task": task,
        "target_column": target_column,
        "model": model_name,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "scoring": scoring,
        "n_trials": len(study.trials),
        "cv_splits": cv.get_n_splits(),
        "requested_cv_splits": cv_splits,
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


def build_model(
    model_name: str,
    params: dict[str, Any],
) -> RandomForestClassifier | RandomForestRegressor:
    if model_name == CLASSIFICATION_MODEL_NAME:
        return RandomForestClassifier(**params)
    if model_name == REGRESSION_MODEL_NAME:
        return RandomForestRegressor(**params)
    raise ValueError(f"Unsupported model: {model_name}")


def suggest_params(
    model_name: str,
    trial: optuna.Trial,
    random_state: int,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_categorical(
            "max_depth",
            [None, 5, 10, 15, 20, 30],
        ),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "max_features",
            ["sqrt", "log2", None],
        ),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "random_state": random_state,
        "n_jobs": -1,
    }

    if model_name == CLASSIFICATION_MODEL_NAME:
        params["class_weight"] = trial.suggest_categorical(
            "class_weight",
            ["balanced", "balanced_subsample", None],
        )
        return params

    if model_name == REGRESSION_MODEL_NAME:
        return params

    raise ValueError(f"Unsupported model: {model_name}")


def normalize_model_names(model_names: list[str] | None) -> list[str]:
    if not model_names:
        return list(SUPPORTED_MODELS)

    normalized = []
    for name in model_names:
        model_key = name.strip().lower().replace("-", "_")
        if model_key in {"all", "both"}:
            candidates = SUPPORTED_MODELS
        else:
            model_name = MODEL_ALIASES.get(model_key)
            if not model_name:
                allowed = ", ".join(SUPPORTED_MODELS)
                raise ValueError(f"Unsupported model `{name}`. Choose from: {allowed}.")
            candidates = (model_name,)

        for candidate in candidates:
            if candidate not in normalized:
                normalized.append(candidate)

    return normalized


def model_task(model_name: str) -> str:
    if model_name == CLASSIFICATION_MODEL_NAME:
        return CLASSIFICATION_TASK
    if model_name == REGRESSION_MODEL_NAME:
        return REGRESSION_TASK
    raise ValueError(f"Unsupported model: {model_name}")


def target_column_for_task(task: str) -> str:
    if task == CLASSIFICATION_TASK:
        return TARGET_CLASS_COLUMN
    if task == REGRESSION_TASK:
        return TARGET_REGRESSION_COLUMN
    raise ValueError(f"Unsupported task: {task}")


def training_params_config(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if CLASSIFICATION_MODEL_NAME in results:
        config["classification_model_params"] = results[CLASSIFICATION_MODEL_NAME][
            "best_params"
        ]
    if REGRESSION_MODEL_NAME in results:
        config["regression_model_params"] = results[REGRESSION_MODEL_NAME]["best_params"]
    return config


def build_cv(
    task: str,
    y_train: pd.Series,
    cv_splits: int,
    random_state: int,
) -> KFold | StratifiedKFold:
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2.")

    if task == CLASSIFICATION_TASK:
        class_counts = y_train.value_counts()
        if class_counts.shape[0] < 2:
            raise ValueError("Classification tuning needs at least two classes.")
        resolved_splits = min(cv_splits, int(class_counts.min()))
        if resolved_splits < 2:
            raise ValueError(
                "Classification tuning needs at least two samples in each class."
            )
        return StratifiedKFold(
            n_splits=resolved_splits,
            shuffle=True,
            random_state=random_state,
        )

    resolved_splits = min(cv_splits, int(y_train.shape[0]))
    if resolved_splits < 2:
        raise ValueError("Regression tuning needs at least two training rows.")
    return KFold(n_splits=resolved_splits, shuffle=True, random_state=random_state)


def load_training_features(
    feature_dir: Path,
    task: str,
) -> tuple[pd.DataFrame, pd.Series]:
    target_column = target_column_for_task(task)
    if task == CLASSIFICATION_TASK:
        files = {
            "X_train": "X_train_class.csv",
            "y_train": "y_train_class.csv",
        }
        y_dtype = int
    elif task == REGRESSION_TASK:
        files = {
            "X_train": "X_train_reg.csv",
            "y_train": "y_train_reg.csv",
        }
        y_dtype = float
    else:
        raise ValueError(f"Unsupported task: {task}")

    missing = [
        name
        for name in files.values()
        if not (feature_dir / name).exists()
    ]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing feature files in {feature_dir}: {joined}. "
            "Run `python manage.py preprocess_data` first."
        )

    X_train = pd.read_csv(feature_dir / files["X_train"])
    y_train = pd.read_csv(feature_dir / files["y_train"])[target_column].astype(y_dtype)
    return X_train, y_train


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
