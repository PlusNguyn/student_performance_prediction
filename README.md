# student_performance_prediction

## Chay Django UI bang Docker Compose

Sao chep cau hinh moi truong neu chua co file `.env`:

```powershell
Copy-Item .env.example .env
```

Build va chay giao dien Django kem Postgres:

```powershell
docker compose up --build django
```

Mo giao dien:

- Django UI: <http://localhost:8000>
- Django admin: <http://localhost:8000/admin/>

Service `django` se tu dong chay `python manage.py migrate --noinput` truoc khi start server.

Muon chay day du UI + MLflow + Airflow:

```powershell
docker compose up --build django mlflow airflow
```

## Chay pipeline bang Airflow

Project da co pipeline tu dong cho toan bo quy trinh ML:

1. `preprocess_data`: tien xu ly du lieu raw va tao feature.
2. `tune_best_model`: dung Optuna tune model tot hon.
3. `train_models`: train model voi tham so tot nhat va log len MLflow.
4. `promote_best_model`: chon version co metric tot nhat va gan alias `production`.

Chay cac service:

```powershell
docker compose up -d postgres mlflow airflow
```

Mo giao dien:

- Airflow: <http://localhost:8080> voi user/password mac dinh `admin` / `admin`
- MLflow: <http://localhost:5000>

DAG full pipeline hien co la `student_performance_training_pipeline`. Co the trigger thu cong tren UI Airflow, hoac dung CLI:

```powershell
docker compose exec airflow airflow dags trigger student_performance_training_pipeline
```

## Weekly retraining tu raw data

DAG `student_performance_weekly_retraining_pipeline` doc lai du lieu hien co trong `data/raw`,
tao lai feature, chay Optuna tuning, train lai model va promote model tot nhat.
Lich mac dinh la `0 2 * * 1` (02:00 thu Hai hang tuan, timezone `Asia/Ho_Chi_Minh`).

Trigger thu cong:

```powershell
docker compose exec airflow airflow dags trigger student_performance_weekly_retraining_pipeline
```

Co the chinh so lan tune va duong dan du lieu trong `.env`:

```env
AIRFLOW_TUNING_TRIALS=3
AIRFLOW_CV_SPLITS=3
AIRFLOW_RAW_DIR=data/raw
AIRFLOW_PREPROCESSED_DIR=data/preprocessed
AIRFLOW_FEATURE_DIR=data/feature
AIRFLOW_TUNING_DIR=models/tuning_optuna
AIRFLOW_MODEL_DIR=models
AIRFLOW_WEEKLY_RETRAIN_SCHEDULE="0 2 * * 1"
AIRFLOW_WEEKLY_TUNING_TRIALS=
AIRFLOW_WEEKLY_CV_SPLITS=
```

## Du lieu luu trong Postgres

Pipeline khong luu raw data vao Postgres. Cac artifact sau duoc sync tu dong sau khi chay command tuong ung:

- `preprocessing_processeddataset`: metadata cua preprocessed/feature datasets.
- `preprocessing_processeddatasetrow`: tung dong du lieu da xu ly va feature o dang JSON.
- `preprocessing_pipelinefileartifact`: feature metadata nhu columns, medians, encoders.
- `tuning_optuna_tuningartifact`: ket qua Optuna, best params va study file.
- `model_training_trainedmodelartifact`: model binary, metrics, model selection va MLflow run id.
- `mlflow_tracking_modelpromotion`: lich su model version duoc promote len alias `production`.

Tao/cap nhat bang Django:

```powershell
docker compose run --rm airflow bash -c "cd /opt/airflow/project && python manage.py migrate --noinput"
```
