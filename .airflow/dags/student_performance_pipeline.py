from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_DIR = Path("/opt/airflow/project")
PYTHON = "python"

DEFAULT_ARGS = {
    "owner": "student-performance",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

COMMON_ENV = {
    "DJANGO_SETTINGS_MODULE": "student_performance_prediction.settings",
    "DB_HOST": "postgres",
    "DB_PORT": "5432",
    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
}


def django_command(task_id: str, command: str) -> BashOperator:
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} manage.py {command}",
        append_env=True,
        env=COMMON_ENV,
    )


with DAG(
    dag_id="student_performance_training_pipeline",
    description=(
        "One-time bootstrap pipeline to preprocess raw data, tune RandomForest "
        "models, train the first model pair, and promote them for prediction."
    ),
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule="@once",
    catchup=False,
    max_active_runs=1,
    tags=["student-performance", "mlflow", "bootstrap"],
) as dag:
    migrate = django_command("migrate_django", "migrate --noinput")

    preprocess = django_command("preprocess_data", "preprocess_data")

    tune = BashOperator(
        task_id="tune_best_model",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "TRIALS=${AIRFLOW_TUNING_TRIALS:-3} && "
            "CV_SPLITS=${AIRFLOW_CV_SPLITS:-3} && "
            f"{PYTHON} manage.py tune_best_model "
            "--n-trials \"$TRIALS\" "
            "--cv-splits \"$CV_SPLITS\" "
            "--tuning-dir models/tuning_optuna"
        ),
        append_env=True,
        env=COMMON_ENV,
    )

    train = django_command(
        "train_models",
        "train_models --params-path models/tuning_optuna/best_model_params.json",
    )

    promote = django_command("promote_best_model", "promote_best_model")

    migrate >> preprocess >> tune >> train >> promote

# Note: Tune trước để tìm hyperparameters tốt cho RandomForestClassifier và RandomForestRegressor. Sau đó train_models dùng các params đó để train model cuối cùng, evaluate test set, lưu model .joblib, log MLflow và đăng ký model registry.

# Vì tune_best_model dùng Optuna + cross-validation để tìm hyperparameters tốt cho RandomForestClassifier và RandomForestRegressor. Sau đó train_models dùng các params đó để train model cuối cùng, evaluate test set, lưu model .joblib, log MLflow và đăng ký model registry.

# Nếu không tune trước thì train_models vẫn chạy được, nhưng dùng default params trong training.py. Tune trước là để training cuối dùng bộ tham số tốt hơn thay vì mặc định.