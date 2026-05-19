from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_DIR = Path(os.getenv("AIRFLOW_PROJECT_DIR", "/opt/airflow/project"))
PYTHON = os.getenv("AIRFLOW_PYTHON", "python")
LOCAL_TZ = pendulum.timezone("Asia/Ho_Chi_Minh")

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
    dag_id="student_performance_weekly_retraining_pipeline",
    description=(
        "Reload features from raw CSV data, tune models, retrain, and promote "
        "the best model every week."
    ),
    default_args=DEFAULT_ARGS,
    start_date=pendulum.datetime(2026, 5, 19, tz=LOCAL_TZ),
    schedule=os.getenv("AIRFLOW_WEEKLY_RETRAIN_SCHEDULE", "0 2 * * 1"),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=12),
    tags=["student-performance", "weekly", "retraining", "mlflow"],
) as dag:
    migrate = django_command("migrate_django", "migrate --noinput")

    reload_data = BashOperator(
        task_id="reload_data_from_raw",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "RAW_DIR=${AIRFLOW_RAW_DIR:-data/raw} && "
            "PREPROCESSED_DIR=${AIRFLOW_PREPROCESSED_DIR:-data/preprocessed} && "
            "FEATURE_DIR=${AIRFLOW_FEATURE_DIR:-data/feature} && "
            f"{PYTHON} manage.py preprocess_data "
            "--raw-dir \"$RAW_DIR\" "
            "--preprocessed-dir \"$PREPROCESSED_DIR\" "
            "--feature-dir \"$FEATURE_DIR\""
        ),
        append_env=True,
        env=COMMON_ENV,
    )

    tune = BashOperator(
        task_id="tune_best_model",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "FEATURE_DIR=${AIRFLOW_FEATURE_DIR:-data/feature} && "
            "TUNING_DIR=${AIRFLOW_TUNING_DIR:-models/tuning_optuna} && "
            "TRIALS=${AIRFLOW_WEEKLY_TUNING_TRIALS:-${AIRFLOW_TUNING_TRIALS:-3}} && "
            "CV_SPLITS=${AIRFLOW_WEEKLY_CV_SPLITS:-${AIRFLOW_CV_SPLITS:-3}} && "
            f"{PYTHON} manage.py tune_best_model "
            "--feature-dir \"$FEATURE_DIR\" "
            "--tuning-dir \"$TUNING_DIR\" "
            "--n-trials \"$TRIALS\" "
            "--cv-splits \"$CV_SPLITS\""
        ),
        append_env=True,
        env=COMMON_ENV,
    )

    retrain = BashOperator(
        task_id="retrain_models",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "FEATURE_DIR=${AIRFLOW_FEATURE_DIR:-data/feature} && "
            "MODEL_DIR=${AIRFLOW_MODEL_DIR:-models} && "
            "TUNING_DIR=${AIRFLOW_TUNING_DIR:-models/tuning_optuna} && "
            "PARAMS_PATH=${AIRFLOW_BEST_MODEL_CONFIG:-$TUNING_DIR/best_model_config.json} && "
            f"{PYTHON} manage.py train_models "
            "--feature-dir \"$FEATURE_DIR\" "
            "--model-dir \"$MODEL_DIR\" "
            "--params-path \"$PARAMS_PATH\""
        ),
        append_env=True,
        env=COMMON_ENV,
    )

    promote = django_command("promote_best_model", "promote_best_model")

    migrate >> reload_data >> tune >> retrain >> promote
