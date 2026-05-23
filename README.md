# Hệ thống dự đoán hiệu suất học tập sinh viên

Đề tài xây dựng một hệ thống MLOps demo end-to-end cho bài toán dự đoán sớm hiệu suất học tập của sinh viên học trực tuyến. Hệ thống sử dụng dữ liệu có cấu trúc tương tự OULAD, xử lý dữ liệu, tạo đặc trưng, tối ưu siêu tham số, huấn luyện mô hình, tracking bằng MLflow, orchestration bằng Airflow, quản lý dữ liệu/mô hình bằng DVC và cung cấp dashboard Django cho Student, Lecturer và Admin.

Đây là hệ thống phục vụ đồ án và demo MLOps, không phải bản production hoàn chỉnh.

## Mục Tiêu

Bài toán chính là dự đoán tình trạng học tập của sinh viên trong khi khóa học vẫn đang diễn ra, thay vì chờ tới kết quả cuối kỳ.

Hệ thống dự đoán hai đầu ra:

- `target_pass`: classification, dự đoán sinh viên có khả năng `Pass/Distinction` hay `Fail/Withdrawn`.
- `learning_percentage`: regression, dự đoán phần trăm tiếp thu/kết quả học tập trong khoảng `0-100`.

Trên dashboard, mỗi sinh viên chỉ xem dự đoán của chính mình theo mã sinh viên được gán trong profile. Sinh viên không được tự thay đổi mã số sinh viên; admin là người quản lý ánh xạ tài khoản với `id_student`.

## Kiến Trúc

Luồng MLOps chính:

```text
Raw OULAD CSV / Demo OULAD-like CSV
-> Preprocessing + Feature Engineering
-> Optuna Hyperparameter Tuning
-> Random Forest Training
-> MLflow Tracking + Model Registry
-> Airflow Pipeline / Weekly Retraining
-> Django Dashboard / Demo Prediction
```

Các thành phần:

- `Django`: web app, authentication, phân quyền Student/Lecturer/Admin, dashboard dự đoán.
- `Preprocessing`: đọc CSV, làm sạch dữ liệu, tạo feature, train/test split, biểu đồ thống kê.
- `Optuna`: tối ưu siêu tham số cho RandomForestClassifier và RandomForestRegressor.
- `Model Training`: train, evaluate, lưu model `.joblib`, log artifact/metric.
- `MLflow`: experiment tracking, artifact tracking, model registry, alias `production`.
- `Airflow`: orchestration cho pipeline bootstrap và retraining định kỳ.
- `Postgres`: database cho Django, metadata artifact và MLflow backend store.
- `DVC`: versioning cho thư mục `data/` và `models/`.
- `Docker Compose`: chạy local các service chính.

## Cấu Trúc Thư Mục

```text
accounts/                       # Auth, role, dashboard, demo prediction
preprocessing/                  # Pipeline xử lý dữ liệu và tạo demo dataset
model_training/                 # Train/evaluate Random Forest models
tuning_optuna/                  # Tối ưu siêu tham số bằng Optuna
mlflow_tracking/                # Promote model trong MLflow registry
student_performance_prediction/ # Django settings, urls, wsgi/asgi
.airflow/dags/                  # Airflow DAGs
.mlflow/                        # Dockerfile/requirements cho MLflow server
templates/                      # Django templates
static/                         # CSS/JS
data/raw/                       # Raw OULAD CSV
data/demo_oulad/                # Demo prediction dataset
data/preprocessed/              # Output preprocessing
data/feature/                   # Feature train/test
models/                         # Model, metrics, tuning artifacts
```

`data/` và `models/` được ignore khỏi Git và được track bằng DVC thông qua `data.dvc` và `models.dvc`.

## Dữ Liệu

Pipeline dùng schema tương tự OULAD với 7 file CSV:

- `studentInfo.csv`
- `studentRegistration.csv`
- `studentAssessment.csv`
- `studentVle.csv`
- `assessments.csv`
- `vle.csv`
- `courses.csv`

Với dữ liệu train gốc, `studentInfo.csv` cần có `final_result` để tạo target. Với dữ liệu demo prediction, `studentInfo.csv` không có `final_result` vì dữ liệu mô phỏng snapshot hiện tại của sinh viên.

Tạo dữ liệu demo:

```powershell
python manage.py create_oulad_demo_data --students 200 --snapshot-date 2026-05-23 --presentation-start-date 2026-01-01
```

Mặc định dữ liệu demo được lưu ở:

```text
data/demo_oulad/
```

## Feature Engineering

Một số feature chính:

- Thông tin nền: `gender_m`, `edu_level`, `imd_num`, `age_num`, `disability_flag`
- Điểm đánh giá: `total_score`, `avg_score`, `num_assessments`, `score_per_assess`
- Tương tác học tập: `total_clicks`, `clicks_per_credit`, `clicks_per_day`, `clicks_per_week`
- Đăng ký học: `registration_lead_days`, `registered_flag`, `registration_duration`
- Feature rủi ro: `low_activity_flag`, `late_registration_flag`, `high_click_but_low_score_flag`

Target:

- `target_pass = 1` với `Pass/Distinction`, `0` với `Fail/Withdrawn`
- `learning_percentage` được tính từ điểm assessment có trọng số và giới hạn trong khoảng `0-100`

## Mô Hình

Hệ thống train hai mô hình:

- `RandomForestClassifier`: dự đoán `target_pass`
- `RandomForestRegressor`: dự đoán `learning_percentage`

Metric classification:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC nếu dữ liệu test có đủ hai lớp

Metric regression:

- RMSE
- MAE
- R2
- Within 10 percent

## Pipeline Chính

Chạy theo thứ tự:

```powershell
python manage.py preprocess_data
python manage.py tune_best_model
python manage.py train_models --params-path models/tuning_optuna/best_model_params.json
python manage.py promote_best_model
```

Ý nghĩa:

- `preprocess_data`: tạo dữ liệu sạch, feature files, profile và biểu đồ.
- `tune_best_model`: dùng Optuna để tìm tham số tốt.
- `train_models`: train model cuối cùng, evaluate, lưu model và log MLflow.
- `promote_best_model`: chọn model tốt nhất trong MLflow registry và gắn alias `production`.

Nếu chưa tune, `train_models` vẫn chạy với tham số mặc định.

## MLflow

MLflow tracking server trong Docker:

```text
http://localhost:5000
```

Các run chính:

- `preprocess-data`: log preprocessing profile, feature files, plots.
- `train-classification-regression`: log params, metrics, model artifacts.

Registered models:

```text
student-performance-best-model-classification
student-performance-best-model-regression
```

Alias mặc định khi promote:

```text
production
```

## Airflow

Dự án có 2 DAG:

```text
student_performance_training_pipeline
```

Pipeline bootstrap:

```text
migrate_django
-> preprocess_data
-> tune_best_model
-> train_models
-> promote_best_model
```

```text
student_performance_weekly_retraining_pipeline
```

Pipeline retraining định kỳ:

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

## DVC

Repo đã có DVC tracking cho dữ liệu và model:

```text
data.dvc
models.dvc
```

Kiểm tra trạng thái:

```powershell
dvc status
```

Nếu muốn làm việc nhiều máy hoặc deploy có kéo artifact, cần cấu hình DVC remote:

```powershell
dvc remote add -d storage <remote-path>
dvc push
```

Lưu ý: `.dvc/config` hiện để trống trong repo, vì remote storage phụ thuộc môi trường triển khai của từng máy/nhóm.

Nếu `dvc status` báo lỗi liên quan database/cache trên Windows, hãy kiểm tra quyền ghi trong `.dvc/tmp`, `.dvc/cache` hoặc khởi tạo lại cache DVC trước khi push/pull artifact. Phần DVC hiện phục vụ minh họa versioning dữ liệu/mô hình cho đồ án; remote storage là bước cấu hình thêm.

## Dashboard Và Phân Quyền

Role chính:

- `Student`: xem kết quả dự đoán cá nhân, learning percentage, risk level và nhắc nhở học tập.
- `Lecturer`: xem danh sách demo prediction của sinh viên, phân bố rủi ro và thống kê lớp.
- `Admin`: quản lý user, role, trạng thái tài khoản và dashboard vận hành.

Student dashboard có:

- Student ID
- Prediction: `Pass` hoặc `Fail / Withdrawn`
- Learning percentage
- Risk level: `Low`, `Medium`, `High`
- Gợi ý học tập cá nhân như `Cần cải thiện`, `Tiếp tục phát huy`, `Tăng tương tác học tập`

Để tài khoản student xem được prediction, `account_code` trong profile phải trùng với một `id_student` trong `data/demo_oulad/studentInfo.csv`, ví dụ:

```text
900000
900001
900002
```

## Cài Đặt Local

Tạo môi trường:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Tạo `.env` từ mẫu:

```powershell
copy .env.example .env
```

Nếu muốn chạy nhanh bằng SQLite khi test/dev:

```powershell
$env:DJANGO_DATABASE_ENGINE='sqlite'
```

Chạy migration:

```powershell
python manage.py migrate
```

Tạo demo data:

```powershell
python manage.py create_oulad_demo_data --students 200 --snapshot-date 2026-05-23 --presentation-start-date 2026-01-01
```

Chạy pipeline ML:

```powershell
python manage.py preprocess_data
python manage.py tune_best_model --n-trials 3 --cv-splits 3
python manage.py train_models --params-path models/tuning_optuna/best_model_params.json
```

Chạy Django:

```powershell
python manage.py runserver
```

Mở:

```text
http://localhost:8000
```

## Chạy Bằng Docker Compose

Chạy database, MLflow, Airflow và Django:

```powershell
docker compose up --build django mlflow airflow
```

Các service:

| Service | URL | Vai trò |
|---|---|---|
| Django | `http://localhost:8000` | Web dashboard |
| MLflow | `http://localhost:5000` | Tracking server |
| Airflow | `http://localhost:8080` | Pipeline orchestration |
| Postgres | `localhost:5432` | Database |

Tài khoản Airflow mặc định lấy từ `.env.example`:

```text
admin / admin
```

## Test Suite

Dự án có test cho các phần chính:

- `accounts`: phân quyền, khóa Student ID trong profile.
- `preprocessing`: feature engineering, target, encoding.
- `model_training`: classification/regression metrics.
- `tuning_optuna`: mapping task, model name, training config.
- `mlflow_tracking`: chọn metric và resolve model registry name.

Chạy test nhanh bằng SQLite:

```powershell
$env:DJANGO_DATABASE_ENGINE='sqlite'
venv\Scripts\python.exe manage.py test accounts preprocessing model_training tuning_optuna mlflow_tracking
```

Kết quả hiện tại:

```text
17 tests OK
```

Kiểm tra Django:

```powershell
venv\Scripts\python.exe manage.py check
```

## Deployment

Với đồ án, nên triển khai theo một trong hai hướng:

- Demo local bằng Docker Compose hoặc Cloudflare Tunnel/ngrok.
- Deploy bản web demo gọn lên Render/Railway/VPS, chỉ mang theo `data/demo_oulad` và model cần inference.

Không nên đưa toàn bộ `data/raw`, `data/preprocessed`, Airflow logs và artifacts lớn lên PaaS miễn phí.

Lưu ý production:

- Dockerfile hiện dùng `python manage.py runserver`, phù hợp demo/dev.
- Nếu deploy production thật, nên đổi sang Gunicorn/Uvicorn, cấu hình static files, HTTPS, secret key, allowed hosts và persistent storage.
- Monitoring hiện là dashboard demo, chưa có drift detection/alerting production như Evidently, Prometheus hoặc Grafana.

## Phạm Vi MLOps Hiện Tại

Có thể trình bày đề tài là:

```text
Hệ thống MLOps demo end-to-end cho dự đoán hiệu suất học tập sinh viên.
```

Đã có:

- Data preprocessing
- Feature engineering
- Hyperparameter tuning
- Model training/evaluation
- Experiment tracking
- Model registry promotion
- Pipeline orchestration
- Data/model versioning bằng DVC
- Dashboard phân quyền
- Test suite cơ bản

Chưa phải production MLOps hoàn chỉnh vì còn thiếu:

- CI/CD thật
- Data quality validation tự động
- Drift monitoring thật
- Model serving tách riêng khỏi Django
- Production deployment hardening
- DVC remote mặc định trong repo

## Ghi Chú Vận Hành

Nếu dashboard báo thiếu artifact:

```text
Missing demo prediction artifacts
```

Hãy tạo demo data và train model lại:

```powershell
python manage.py create_oulad_demo_data --students 200 --snapshot-date 2026-05-23 --presentation-start-date 2026-01-01
python manage.py preprocess_data
python manage.py tune_best_model --n-trials 3 --cv-splits 3
python manage.py train_models --params-path models/tuning_optuna/best_model_params.json
```

Nếu MLflow chưa chạy, pipeline vẫn tạo file local nhưng phần MLflow tracking sẽ được báo là disabled hoặc skipped.

Nếu chạy Airflow trong Docker, DAG sẽ mount source code vào:

```text
/opt/airflow/project
```

và gọi trực tiếp:

```text
python manage.py ...
```
