# Hệ thống dự đoán hiệu suất học tập sinh viên

Dự án này xây dựng một hệ thống MLOps nhỏ cho bài toán dự đoán hiệu suất học tập của sinh viên học trực tuyến, dựa trên cấu trúc dữ liệu OULAD. Hệ thống bao gồm xử lý dữ liệu, tạo đặc trưng, tối ưu siêu tham số, huấn luyện hai mô hình Random Forest, tracking bằng MLflow, orchestration bằng Airflow, lưu metadata vào Postgres và giao diện Django để xem dashboard/prediction demo.

## Mục tiêu đề tài

Bài toán chính là dự đoán sớm tình trạng học tập của sinh viên tại một thời điểm đang diễn ra trong kỳ học, thay vì chờ tới cuối khóa mới biết kết quả.

Hệ thống dự đoán hai đầu ra:

- `target_pass`: bài toán classification, dự đoán sinh viên có khả năng Pass/Distinction hay Fail/Withdrawn.
- `learning_percentage`: bài toán regression, dự đoán phần trăm học tập/đạt điểm kỳ vọng trong khoảng 0-100.

Điểm quan trọng của demo hiện tại: dữ liệu demo là snapshot tại thời điểm hiện tại, không chứa kết quả cuối khóa trong `studentInfo.csv`. Dashboard build feature từ click, assessment, đăng ký học và thông tin nhân khẩu học rồi gọi model để predict.

## Kiến trúc tổng quan

Luồng xử lý đầy đủ:

```text
Raw OULAD / Demo OULAD-like CSV
-> preprocess_data
-> tune_best_model
-> train_models
-> promote_best_model
-> Django dashboard / prediction demo
```

Các thành phần chính:

- Django: giao diện web, phân quyền Student/Lecturer/Admin, dashboard prediction.
- Preprocessing: đọc CSV OULAD, làm sạch, feature engineering, tạo train/test split, tạo biểu đồ.
- Optuna: tune siêu tham số cho RandomForestClassifier và RandomForestRegressor.
- Model training: train hai mô hình chính, evaluate, lưu model local và log MLflow.
- MLflow: tracking metrics/artifacts/model registry, gắn alias `production`.
- Airflow: tự động hóa pipeline theo DAG.
- Postgres: database cho Django, metadata artifact và MLflow backend store.

## Cấu trúc thư mục

```text
accounts/                       # User, role, dashboard, demo prediction
preprocessing/                  # Pipeline xử lý dữ liệu và tạo demo OULAD-like
model_training/                 # Huấn luyện RandomForest classifier/regressor
tuning_optuna/                  # Tối ưu siêu tham số bằng Optuna
mlflow_tracking/                # Promote model trong MLflow registry
student_performance_prediction/ # Django settings, urls, wsgi/asgi
.airflow/dags/                  # DAG Airflow
.mlflow/                        # Dockerfile và requirements cho MLflow server
templates/                      # Giao diện Django
static/                         # CSS/JS
data/raw/                       # Dữ liệu OULAD gốc nếu có
data/demo_oulad/                # Dữ liệu demo giả lập, bị .gitignore
data/preprocessed/              # Output sau preprocessing, bị .gitignore
data/feature/                   # Feature train/test, bị .gitignore
models/                         # Model artifacts, bị .gitignore
```

## Dữ liệu đầu vào

Pipeline dùng schema tương tự OULAD với 7 file CSV:

- `studentInfo.csv`: thông tin sinh viên, khóa học, giới tính, vùng, học vấn, IMD, tuổi, tín chỉ, disability.
- `studentRegistration.csv`: ngày đăng ký và ngày hủy đăng ký nếu có.
- `studentAssessment.csv`: điểm bài assessment đã nộp.
- `studentVle.csv`: click activity trên môi trường học trực tuyến.
- `assessments.csv`: metadata assessment, ngày assessment, trọng số.
- `vle.csv`: metadata hoạt động học tập online.
- `courses.csv`: module, presentation và độ dài kỳ học.

Với dữ liệu train gốc OULAD, `studentInfo.csv` có `final_result` để tạo target train. Với dữ liệu demo prediction, `studentInfo.csv` không có `final_result` vì đây là snapshot hiện tại.

## Dữ liệu demo OULAD-like

Dự án có generator để tạo dữ liệu demo học trực tuyến theo schema OULAD:

```powershell
python -m preprocessing.demo_data --output-dir data/demo_oulad --students 200 --snapshot-date 2026-05-23 --presentation-start-date 2026-01-01
```

Hoặc qua Django management command:

```powershell
python manage.py create_oulad_demo_data --output-dir data/demo_oulad --students 200 --snapshot-date 2026-05-23 --presentation-start-date 2026-01-01
```

Bộ demo hiện tại được tạo với:

- `student_count`: 200 sinh viên.
- `code_presentation`: `2026B`.
- `presentation_start_date`: `2026-01-01`.
- `snapshot_date`: `2026-05-23`.
- `observation_day`: 142.
- `studentInfo.csv`: không có cột kết quả cuối khóa.
- `studentVle.csv`: chỉ có click tới ngày 142.
- `studentAssessment.csv`: chỉ có bài nộp tới ngày 142.

File `metadata.json` trong `data/demo_oulad` chỉ mô tả bộ demo được tạo như thế nào. File này không được model dùng để predict.

Lưu ý: thư mục `data/` đang nằm trong `.gitignore`, vì vậy dữ liệu demo có trên máy local/container nhưng không được commit vào git. Khi clone project mới, hãy chạy lại lệnh tạo demo ở trên.

## Feature engineering

Các feature chính dùng cho model:

```python
MODEL_FEATURE_COLUMNS = [
    "code_module",
    "code_presentation",
    "gender_m",
    "region",
    "edu_level",
    "imd_num",
    "age_num",
    "num_of_prev_attempts",
    "studied_credits",
    "disability_flag",
    "total_score",
    "avg_score",
    "num_assessments",
    "total_clicks",
    "registration_lead_days",
    "registered_flag",
    "registration_duration",
    "clicks_per_credit",
    "score_per_assess",
    "clicks_per_day",
    "clicks_per_week",
    "engagement_intensity",
    "credit_load_category",
    "assessment_density",
    "score_per_credit",
    "activity_efficiency",
    "low_activity_flag",
    "late_registration_flag",
    "high_click_but_low_score_flag",
]
```

Một số feature được mã hóa từ dữ liệu gốc:

- `gender_m`: từ `gender`.
- `edu_level`: từ `highest_education`.
- `imd_num`: từ `imd_band`.
- `age_num`: từ `age_band`.
- `disability_flag`: từ `disability`.

Một số feature được tổng hợp:

- `total_score`, `avg_score`, `num_assessments`: từ `studentAssessment`.
- `total_clicks`, `active_days`, `avg_daily_clicks`: từ `studentVle`.

Một số feature được tạo mới:

- `clicks_per_credit`
- `score_per_assess`
- `clicks_per_day`
- `clicks_per_week`
- `engagement_intensity`
- `credit_load_category`
- `assessment_density`
- `score_per_credit`
- `activity_efficiency`
- `low_activity_flag`
- `late_registration_flag`
- `high_click_but_low_score_flag`

## Mô hình

Hệ thống dùng hai mô hình chính:

- `RandomForestClassifier`: dự đoán `target_pass`.
- `RandomForestRegressor`: dự đoán `learning_percentage`.

Lý do dùng hai mô hình riêng:

- Classification trả lời câu hỏi sinh viên có khả năng pass hay không.
- Regression trả lời mức độ học tập/điểm kỳ vọng theo phần trăm.
- Hai target có bản chất khác nhau nên tách model giúp metric, tuning và registry rõ ràng hơn.

## Tune trước rồi train sau

Thứ tự đúng:

```text
preprocess_data
-> tune_best_model
-> train_models --params-path models/tuning_optuna/best_model_params.json
-> promote_best_model
```

`tune_best_model` dùng Optuna để tìm params tốt cho hai model:

- `classification_model_params`
- `regression_model_params`

Kết quả được lưu tại:

```text
models/tuning_optuna/best_model_params.json
```

Sau đó `train_models` train lại hai model cuối cùng bằng params này, evaluate trên test split, lưu model `.joblib`, log MLflow và sync metadata vào Postgres.

Nếu không tune trước, `train_models` vẫn chạy được nhưng dùng params mặc định trong code.

## MLflow tracking

MLflow tracking URI mặc định trong Docker:

```text
http://mlflow:5000
```

Giao diện local:

```text
http://localhost:5000
```

Các run chính:

- `preprocess-data`: log profile, feature files, preprocessed files và biểu đồ.
- `train-classification-regression`: log params, metrics, model artifacts và registered model.

Biểu đồ preprocessing được log ở:

```text
Artifacts -> plots/
```

Các biểu đồ gồm:

- `pass_fail_distribution.png`
- `final_result_distribution.png`
- `learning_percentage_distribution.png`
- `feature_correlation.png`

Registered models:

- `student-performance-best-model-classification`
- `student-performance-best-model-regression`

Lệnh promote:

```powershell
python manage.py promote_best_model
```

Mặc định lệnh này promote cả hai model lên alias:

```text
production
```

Có thể promote riêng:

```powershell
python manage.py promote_best_model --task classification
python manage.py promote_best_model --task regression
```

## Airflow DAGs

Dự án có hai DAG:

### `student_performance_training_pipeline`

DAG bootstrap, chạy một lần:

```text
migrate_django
-> preprocess_data
-> tune_best_model
-> train_models
-> promote_best_model
```

### `student_performance_weekly_retraining_pipeline`

DAG retraining định kỳ:

```text
migrate_django
-> reload_data_from_raw
-> tune_best_model
-> retrain_models
-> promote_best_model
```

Lịch mặc định:

```text
0 2 * * 1
```

Tức 02:00 thứ Hai hằng tuần theo timezone `Asia/Ho_Chi_Minh`.

## Docker Compose services

File `docker-compose.yml` có các service:

| Service | Công dụng |
|---|---|
| `postgres` | Database chung cho Django, metadata pipeline và MLflow backend store. |
| `django` | Web app chính tại `http://localhost:8000`. Chạy migrate rồi runserver. |
| `mlflow` | MLflow Tracking Server tại `http://localhost:5000`. |
| `airflow` | Airflow webserver/scheduler tại `http://localhost:8080`. |

Chạy web Django:

```powershell
docker compose up --build django
```

Chạy MLflow + Airflow:

```powershell
docker compose up -d postgres mlflow airflow
```

Chạy full hệ thống:

```powershell
docker compose up --build django mlflow airflow
```

Airflow không gọi HTTP sang container `django`; nó mount source code vào `/opt/airflow/project` và tự chạy `python manage.py ...`.

## Cấu hình môi trường

Các biến môi trường quan trọng:

```env
DJANGO_DEBUG=True
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1
DJANGO_SECRET_KEY=django-insecure-dev-only-key

POSTGRES_DB=student_performance
POSTGRES_USER=student_user
POSTGRES_PASSWORD=student_password
POSTGRES_PORT=5432
DB_HOST=postgres
DB_PORT=5432

MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=student-performance
MLFLOW_REGISTERED_MODEL_NAME=student-performance-best-model

DEMO_DATA_DIR=data/demo_oulad

AIRFLOW_TUNING_TRIALS=3
AIRFLOW_CV_SPLITS=3
AIRFLOW_RAW_DIR=data/raw
AIRFLOW_PREPROCESSED_DIR=data/preprocessed
AIRFLOW_FEATURE_DIR=data/feature
AIRFLOW_TUNING_DIR=models/tuning_optuna
AIRFLOW_MODEL_DIR=models
AIRFLOW_BEST_MODEL_PARAMS=models/tuning_optuna/best_model_params.json
AIRFLOW_WEEKLY_RETRAIN_SCHEDULE="0 2 * * 1"
```

## Chạy local không Docker

Cài thư viện:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Tạo demo data:

```powershell
python manage.py create_oulad_demo_data --students 200 --snapshot-date 2026-05-23 --presentation-start-date 2026-01-01
```

Chạy migration:

```powershell
python manage.py migrate
```

Chạy preprocessing:

```powershell
python manage.py preprocess_data
```

Tune model:

```powershell
python manage.py tune_best_model
```

Train model bằng params đã tune:

```powershell
python manage.py train_models --params-path models/tuning_optuna/best_model_params.json
```

Chạy Django:

```powershell
python manage.py runserver
```

## Rà soát thư viện

Các thư viện đang dùng trong `requirements.txt`:

| Thư viện | Vai trò |
|---|---|
| `Django` | Web framework, auth, admin, management commands. |
| `mlflow` | Tracking, artifacts, model registry. Được include qua `.mlflow/requirements.txt`. |
| `psycopg[binary]` | Django kết nối PostgreSQL. |
| `psycopg2-binary` | MLflow server dùng SQLAlchemy PostgreSQL backend. |
| `python-dotenv` | Đọc file `.env`. |
| `pandas` | Đọc CSV, merge, aggregate, xử lý feature. |
| `numpy` | Numeric processing, clipping, safe operations. |
| `scikit-learn` | RandomForest, train/test split, metrics, cross-validation. |
| `optuna` | Tối ưu siêu tham số. |
| `joblib` | Lưu model và Optuna study. |
| `matplotlib` | Vẽ biểu đồ preprocessing. |
| `tzdata` | Hỗ trợ timezone trên Windows/container nhẹ. |
| `asgiref`, `sqlparse`, `typing_extensions` | Phụ thuộc nền của Django/Python, giữ để môi trường ổn định. |

Trong `.airflow/requirements.txt`, không cần khai báo `apache-airflow` vì image base đã là `apache/airflow:2.10.5-python3.10`. File này chỉ cài thêm dependency của project để DAG chạy được `python manage.py ...`.

## Dữ liệu lưu trong Postgres

Pipeline không lưu raw CSV vào Postgres. Các metadata/artifact được sync:

- `preprocessing_processeddataset`: metadata dataset sau xử lý.
- `preprocessing_processeddatasetrow`: từng dòng preprocessed ở dạng JSON.
- `preprocessing_pipelinefileartifact`: metadata files, feature columns, medians, encoders.
- `tuning_optuna_tuningartifact`: best params, tuning summary, Optuna study.
- `model_training_trainedmodelartifact`: model binary, metrics, model selection, MLflow run id.
- `mlflow_tracking_modelpromotion`: lịch sử promote model lên alias `production`.

Nếu Postgres hoặc MLflow chưa chạy, command vẫn tạo file output local và báo phần sync/tracking bị skip.

## Dashboard và phân quyền

Các role chính:

- Student: xem prediction của chính mình theo `account_code` là Student ID.
- Lecturer: xem danh sách demo prediction, risk level, engagement, learning percentage.
- Admin: quản lý user và xem dashboard quản trị.

Để student demo xem được prediction, profile của user cần có `account_code` trùng với một `id_student` trong `data/demo_oulad/studentInfo.csv`, ví dụ:

```text
900000
900001
...
```

## Ghi chú vận hành

Nếu Docker Hub pull image lỗi EOF, thử pull base image trước:

```powershell
docker pull python:3.10-slim
docker compose build --pull django mlflow
```

Hoặc dùng mirror trong `.env`:

```env
PYTHON_BASE_IMAGE=mirror.gcr.io/library/python:3.10-slim
AIRFLOW_BASE_IMAGE=mirror.gcr.io/apache/airflow:2.10.5-python3.10
```

Nếu Django container báo thiếu model artifacts khi mở dashboard, hãy chạy:

```powershell
python manage.py preprocess_data
python manage.py tune_best_model
python manage.py train_models --params-path models/tuning_optuna/best_model_params.json
```

Nếu muốn xem biểu đồ trên MLflow, cần chạy lại preprocessing sau khi MLflow đã bật:

```powershell
docker compose up -d postgres mlflow
python manage.py preprocess_data
```

Sau đó vào:

```text
http://localhost:5000
```

và mở run `preprocess-data`, tab `Artifacts`, thư mục `plots`.
