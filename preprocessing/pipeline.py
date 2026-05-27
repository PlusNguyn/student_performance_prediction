from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.model_selection import train_test_split


matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 42
STUDENT_KEY = ["code_module", "code_presentation", "id_student"]
TARGET_CLASS_COLUMN = "target_pass"
TARGET_REGRESSION_COLUMN = "learning_percentage"
TARGET_COLUMN = TARGET_REGRESSION_COLUMN

CATEGORICAL_COLUMNS = [
    "code_module",
    "code_presentation",
    "region",
    "credit_load_category",
]

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
    "clicks_per_credit", # new feature engineering
    "score_per_assess", # new feature engineering
    "clicks_per_day",
    "clicks_per_week",
    "engagement_intensity", # new feature engineering
    "credit_load_category",
    "assessment_density",
    "score_per_credit",
    "activity_efficiency",
    "low_activity_flag", # new feature engineering
    "late_registration_flag",
    "high_click_but_low_score_flag",
]


@dataclass(frozen=True) 
# dataclass giúp Python tự tạo sẵn các hàm như __init__, __repr__, __eq__... 
# và frozen=True làm cho instance của class này trở nên bất biến (immutable), \
# nghĩa là sau khi được tạo ra, bạn không thể thay đổi giá trị của các thuộc tính của nó.
class PreprocessingPaths:
    '''
    Chứa các đường dẫn thư mục được sử dụng trong toàn bộ quy trình tiền xử lý.
    '''
    raw_dir: Path
    preprocessed_dir: Path
    feature_dir: Path
    plot_dir: Path



    @classmethod
    def from_defaults(
        cls,
        raw_dir: str | Path | None = None,
        preprocessed_dir: str | Path | None = None,
        feature_dir: str | Path | None = None,
        plot_dir: str | Path | None = None,
    ) -> "PreprocessingPaths":
        '''
        Tạo cấu hình đường dẫn từ tham số đầu vào hoặc từ cấu trúc thư mục mặc định.
        '''
        base_dir = Path(settings.BASE_DIR)
        resolved_preprocessed_dir = (
            Path(preprocessed_dir)
            if preprocessed_dir
            else base_dir / "data" / "preprocessed"
        )
        return cls(
            raw_dir=Path(raw_dir) if raw_dir else base_dir / "data" / "raw",
            preprocessed_dir=resolved_preprocessed_dir,
            feature_dir=Path(feature_dir) if feature_dir else base_dir / "data" / "feature",
            plot_dir=Path(plot_dir) if plot_dir else resolved_preprocessed_dir / "plots",
        )


def run_preprocessing(
    paths: PreprocessingPaths | None = None,
    test_size: float = 0.2,
    random_state: int = SEED,
) -> dict[str, Any]:
    '''
    Thực thi toàn bộ quy trình tiền xử lý dữ liệu và lưu các kết quả đầu ra.
    Hàm trả về thống kê dataset, trạng thái MLflow/PostgreSQL và đường dẫn artifact.
    '''
    paths = paths or PreprocessingPaths.from_defaults()
    _ensure_directories(paths)

    # Đọc và làm sạch dữ liệu nguồn trước khi tổng hợp đặc trưng cho từng sinh viên.
    raw_tables = load_raw_tables(paths.raw_dir)
    missing_before = _missing_summary(raw_tables)
    cleaned_tables = clean_raw_tables(raw_tables)
    missing_after = _missing_summary(cleaned_tables)

    students = _add_targets(cleaned_tables["student_info"])
    student_reg = cleaned_tables["student_reg"]
    student_assess = cleaned_tables["student_assess"]
    student_vle = cleaned_tables["student_vle"]
    assessments = cleaned_tables["assessments"]

    assessment_features = build_assessment_features(student_assess, assessments)
    vle_features = build_vle_features(student_vle)
    modeling_dataset = build_modeling_dataset(
        students=students,
        student_reg=student_reg,
        assessment_features=assessment_features,
        vle_features=vle_features,
    )
    encoded_dataset, X, y_class, y_reg, medians, category_levels = prepare_features(
        modeling_dataset
    )

    (
        X_train_class,
        X_test_class,
        y_train_class,
        y_test_class,
        X_train_reg,
        X_test_reg,
        y_train_reg,
        y_test_reg,
    ) = split_feature_sets(
        X=X,
        y_class=y_class,
        y_reg=y_reg,
        test_size=test_size,
        random_state=random_state,
    )

    # Tạo báo cáo trực quan và lưu toàn bộ artifact dùng cho huấn luyện mô hình.
    plot_paths = save_preprocessing_plots(
        modeling_dataset=modeling_dataset,
        encoded_dataset=encoded_dataset,
        paths=paths,
    )
    profile = build_preprocessing_profile(
        raw_tables=raw_tables,
        missing_before=missing_before,
        missing_after=missing_after,
        modeling_dataset=modeling_dataset,
        encoded_dataset=encoded_dataset,
        test_size=test_size,
        random_state=random_state,
        plot_paths=plot_paths,
    )
    output_paths = save_outputs(
        paths=paths,
        students=students,
        student_reg=student_reg,
        assessment_features=assessment_features,
        vle_features=vle_features,
        modeling_dataset=modeling_dataset,
        encoded_dataset=encoded_dataset,
        X=X,
        y_class=y_class,
        y_reg=y_reg,
        X_train_class=X_train_class,
        X_test_class=X_test_class,
        y_train_class=y_train_class,
        y_test_class=y_test_class,
        X_train_reg=X_train_reg,
        X_test_reg=X_test_reg,
        y_train_reg=y_train_reg,
        y_test_reg=y_test_reg,
        medians=medians,
        category_levels=category_levels,
        profile=profile,
        plot_paths=plot_paths,
    )

    mlflow_tracking = log_preprocessing_to_mlflow(
        paths=paths,
        output_paths=output_paths,
        profile=profile,
        test_size=test_size,
        random_state=random_state,
    )
    postgres_sync = _safe_sync_outputs_to_postgres(
        output_paths=output_paths,
        rows=int(encoded_dataset.shape[0]),
        features=len(X.columns),
        train_rows=int(X_train_reg.shape[0]),
        test_rows=int(X_test_reg.shape[0]),
        class_positive_rate=float(y_class.mean()),
        target_mean=float(y_reg.mean()),
        target_min=float(y_reg.min()),
        target_max=float(y_reg.max()),
        test_size=test_size,
        random_state=random_state,
        mlflow_tracking=mlflow_tracking,
    )

    return {
        "rows": int(encoded_dataset.shape[0]),
        "features": len(X.columns),
        "classification_train_rows": int(X_train_class.shape[0]),
        "classification_test_rows": int(X_test_class.shape[0]),
        "regression_train_rows": int(X_train_reg.shape[0]),
        "regression_test_rows": int(X_test_reg.shape[0]),
        "classification_target": TARGET_CLASS_COLUMN,
        "regression_target": TARGET_REGRESSION_COLUMN,
        "target": TARGET_REGRESSION_COLUMN,
        "class_positive_rate": float(y_class.mean()),
        "target_mean": float(y_reg.mean()),
        "target_min": float(y_reg.min()),
        "target_max": float(y_reg.max()),
        "mlflow": mlflow_tracking,
        "postgres_sync": postgres_sync,
        "paths": output_paths,
    }


def load_raw_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    '''
    Đọc các bảng dữ liệu OULAD thô cần thiết từ thư mục nguồn.
    '''
    return {
        "student_info": _read_csv(raw_dir / "studentInfo.csv"),
        "student_reg": _read_csv(raw_dir / "studentRegistration.csv"),
        "student_assess": _read_csv(raw_dir / "studentAssessment.csv"),
        "student_vle": _read_csv(raw_dir / "studentVle.csv"),
        "assessments": _read_csv(raw_dir / "assessments.csv"),
        "vle": _read_csv(raw_dir / "vle.csv"),
        "courses": _read_csv(raw_dir / "courses.csv"),
    }


def clean_raw_tables(raw_tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    '''
    Thực hiện các bước làm sạch dữ liệu cơ bản cho từng bảng, bao gồm:
    '''
    tables = {name: frame.copy() for name, frame in raw_tables.items()}
    # Sao chép dữ liệu gốc để tránh thay đổi trực tiếp, đảm bảo tính toàn vẹn của dữ liệu gốc trong quá trình làm sạch.

    student_info = tables["student_info"]
    student_reg = tables["student_reg"]
    student_assess = tables["student_assess"]
    assessments = tables["assessments"]
    vle = tables["vle"]

    student_info["imd_band"] = student_info["imd_band"].fillna(
        _mode_or_default(student_info["imd_band"], "Unknown")
    ) # Điền giá trị thiếu trong cột imd_band bằng giá trị mode của cột đó, hoặc "Unknown" nếu không có mode nào tồn tại.

    student_reg["date_registration"] = pd.to_numeric(
        student_reg["date_registration"],
        errors="coerce",
    ) # Chuyển đổi cột date_registration sang kiểu số, nếu có lỗi trong quá trình chuyển đổi sẽ được thay thế bằng NaN.

    student_reg["date_registration"] = student_reg["date_registration"].fillna(
        student_reg["date_registration"].median()
    ) # Điền giá trị thiếu trong cột date_registration bằng giá trị trung vị của cột đó, giúp giảm thiểu ảnh hưởng của các giá trị ngoại lai.

    student_reg["date_unregistration"] = pd.to_numeric(
        student_reg["date_unregistration"],
        errors="coerce",
    ).fillna(-1) # Chuyển đổi cột date_unregistration sang kiểu số, nếu có lỗi trong quá trình chuyển đổi sẽ được thay thế bằng -1.

    student_assess["score"] = pd.to_numeric(
        student_assess["score"],
        errors="coerce",
    ).fillna(0) # Chuyển đổi cột score sang kiểu số, nếu có lỗi trong quá trình chuyển đổi sẽ được thay thế bằng 0.

    student_assess["is_banked"] = pd.to_numeric(
        student_assess["is_banked"],
        errors="coerce",
    ).fillna(0) # Chuyển đổi cột is_banked sang kiểu số, nếu có lỗi trong quá trình chuyển đổi sẽ được thay thế bằng 0.

    assessments["date"] = pd.to_numeric(assessments["date"], errors="coerce")
    assessments["date"] = assessments["date"].fillna(assessments["date"].median())
    assessments["weight"] = pd.to_numeric(assessments["weight"], errors="coerce").fillna(1)

    for column in ["week_from", "week_to"]:
        if column in vle.columns:
            vle[column] = pd.to_numeric(vle[column], errors="coerce").fillna(-1)

    return tables


def build_assessment_features(
    student_assessment: pd.DataFrame,
    assessments: pd.DataFrame,
) -> pd.DataFrame:
    '''
    Tính toán các đặc trưng liên quan đến đánh giá cho mỗi sinh viên, bao gồm:
    - Tổng điểm
    - Điểm trung bình
    - Điểm thấp nhất
    - Điểm cao nhất
    - Số lượng đánh giá
    - Số lần nộp bài trễ
    '''
    merged = student_assessment.merge(
        assessments[STUDENT_KEY[:2] + ["id_assessment", "weight"]],
        on="id_assessment",
        how="left",
    ) # Kết hợp dữ liệu giữa student_assessment và assessments dựa trên cột id_assessment để lấy thông tin về trọng số của từng đánh giá.

    merged["weight_filled"] = merged["weight"].fillna(1) 
    # Điền giá trị thiếu trong cột weight bằng 1, giả định rằng nếu không có trọng số nào được cung cấp thì trọng số mặc định là 1.
    merged["weighted_score"] = merged["score"] * merged["weight_filled"]
    # Tính toán điểm đã được trọng số hóa bằng cách nhân điểm số của sinh viên với trọng số tương ứng của đánh giá đó.

    features = (
        merged.groupby(STUDENT_KEY, as_index=False)
        .agg(
            total_score=("score", "sum"),
            avg_score=("score", "mean"),
            min_score=("score", "min"),
            max_score=("score", "max"),
            num_assessments=("id_assessment", "count"),
            late_submissions=("is_banked", "sum"),
            weighted_score_sum=("weighted_score", "sum"),
            weight_sum=("weight_filled", "sum"),
        )
    ) # Nhóm dữ liệu theo STUDENT_KEY và tính toán các đặc trưng tổng hợp cho mỗi sinh viên, bao gồm tổng điểm, điểm trung bình, điểm thấp nhất, điểm cao nhất, số lượng đánh giá, số lần nộp bài trễ, tổng điểm đã được trọng số hóa và tổng trọng số.

    features[TARGET_REGRESSION_COLUMN] = safe_divide(
        features["weighted_score_sum"],
        features["weight_sum"],
    ).fillna(features["avg_score"]) # Tính toán cột learning_percentage bằng cách chia tổng điểm đã được trọng số hóa cho tổng trọng số, nếu có giá trị NaN thì sẽ được thay thế bằng điểm trung bình của sinh viên đó.


    features[TARGET_REGRESSION_COLUMN] = (
        features[TARGET_REGRESSION_COLUMN].fillna(0).clip(lower=0, upper=100)
    )
    # Đảm bảo rằng giá trị trong cột learning_percentage nằm trong khoảng [0, 100]

    return features.drop(columns=["weighted_score_sum", "weight_sum"])


def build_vle_features(student_vle: pd.DataFrame) -> pd.DataFrame:
    '''
    Tính toán các đặc trưng liên quan đến hoạt động trên nền tảng học tập điện tử (VLE) cho mỗi sinh viên, bao gồm:
    - Tổng số lần click
    - Số ngày hoạt động
    - Số lần click trung bình mỗi ngày
    - Số lần click cao nhất trong một ngày
    '''
    return (
        student_vle.groupby(STUDENT_KEY, as_index=False)
        .agg(
            total_clicks=("sum_click", "sum"),
            active_days=("date", "nunique"),
            avg_daily_clicks=("sum_click", "mean"),
            max_clicks_day=("sum_click", "max"),
        )
    )

def build_modeling_dataset(
    *,
    students: pd.DataFrame,
    student_reg: pd.DataFrame,
    assessment_features: pd.DataFrame,
    vle_features: pd.DataFrame,
) -> pd.DataFrame:
    '''
    Kết hợp dữ liệu từ các bảng khác nhau để tạo ra một dataset hoàn chỉnh cho việc modeling, bao gồm:
    - Thông tin sinh viên
    - Thông tin đăng ký
    - Đặc trưng đánh giá
    - Đặc trưng hoạt động VLE
    '''
    merged = students.merge(student_reg, on=STUDENT_KEY, how="left")
    merged = merged.merge(assessment_features, on=STUDENT_KEY, how="left")
    merged = merged.merge(vle_features, on=STUDENT_KEY, how="left")

    numeric_fill_zero = [
        "total_score",
        "avg_score",
        "min_score",
        "max_score",
        "num_assessments",
        "late_submissions",
        "learning_percentage",
        "total_clicks",
        "active_days",
        "avg_daily_clicks",
        "max_clicks_day",
    ]

    merged[numeric_fill_zero] = merged[numeric_fill_zero].fillna(0)\

    merged = add_features(merged)
    return merged[
        [
            "id_student",
            "final_result",
            *MODEL_FEATURE_COLUMNS,
            TARGET_CLASS_COLUMN,
            TARGET_REGRESSION_COLUMN,
        ]
    ].copy()


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    '''
    Thêm các đặc trưng mới vào DataFrame đầu vào. Các đặc trưng này bao gồm:
    - registration_lead_days: Số ngày từ khi đăng ký đến ngày bắt đầu khóa học
    - registered_flag: Cờ cho biết sinh viên đã đăng ký hay chưa
    - registration_duration: Thời gian đăng ký (từ ngày đăng ký đến ngày hủy đăng ký hoặc ngày hiện tại nếu chưa hủy)
    - imd_num: Chuyển đổi imd_band thành số
    - age_num: Chuyển đổi age_band thành số
    - edu_level: Chuyển đổi highest_education thành cấp
    - gender_m: Chuyển đổi gender thành cờ giới tính nam
    - disability_flag: Chuyển đổi disability thành cờ khuyết tật
    - clicks_per_credit: Số lần click trung bình mỗi tín chỉ đã học
    - score_per_assess: Điểm trung bình mỗi đánh giá
    - clicks_per_day: Số lần click trung bình mỗi ngày hoạt động
    - clicks_per_week: Số lần click trung bình mỗi tuần hoạt động
    - engagement_intensity: Cường độ tương tác dựa trên số lần click so với trung bình
    - credit_load_category: Phân loại khối lượng tín chỉ đã học thành "light", "medium", "heavy"
    - assessment_density: Số lượng đánh giá trên mỗi tín chỉ đã học
    - score_per_credit: Điểm trung bình mỗi tín chỉ đã học
    - activity_efficiency: Hiệu quả hoạt động dựa trên điểm trung bình mỗi đánh giá so với số lần click trung bình mỗi tín chỉ
    - low_activity_flag: Cờ cho biết sinh viên có số lần click thấp hơn trung vị hay không
    - late_registration_flag: Cờ cho biết sinh viên có đăng ký muộn hơn trung vị hay không
    - high_click_but_low_score_flag: Cờ cho biết sinh viên có số lần click cao hơn trung vị nhưng điểm thấp hơn trung vị hay không
    '''
    df = frame.copy()

    df["registration_lead_days"] = -pd.to_numeric(
        df["date_registration"],
        errors="coerce",
    ).fillna(0)
    df["registered_flag"] = df["date_unregistration"].eq(-1).astype(int)
    # Cột registered_flag được tạo ra bằng cách kiểm tra xem cột date_unregistration có bằng -1 hay không. Nếu bằng -1, sinh viên vẫn đang đăng ký; nếu khác -1, sinh viên đã hủy đăng ký.
    df["registration_duration"] = np.where(
        df["date_unregistration"].ne(-1),
        df["date_unregistration"] - df["date_registration"],
        -1,
    )

    df["imd_num"] = df["imd_band"].astype(str).apply(imd_to_mid)
    df["imd_num"] = df["imd_num"].fillna(df["imd_num"].median()).fillna(0)

    age_map = {
        "0-35": 17.5,
        "35-55": 45.0,
        "55<=": 60.0,
    }
    age_band = df["age_band"].astype(str).str.strip()
    placeholder_age = age_band.str.extract(r"(\d+)")[0].astype(float).fillna(30.0)
    df["age_num"] = age_band.map(age_map).fillna(placeholder_age)

    edu_map = {
        "No Formal quals": 0,
        "Lower Than A Level": 1,
        "A Level or Equivalent": 2,
        "HE Qualification": 3,
        "Postgraduate Qualification": 4,
    }
    df["edu_level"] = df["highest_education"].map(edu_map).fillna(2).astype(int)
    df["gender_m"] = df["gender"].map({"M": 1, "F": 0}).fillna(0).astype(int)
    df["disability_flag"] = df["disability"].map({"Y": 1, "N": 0}).fillna(0).astype(int)

    df["clicks_per_credit"] = safe_divide(df["total_clicks"], df["studied_credits"])
    df["score_per_assess"] = safe_divide(df["total_score"], df["num_assessments"])

    positive_duration = df["registration_duration"].where(df["registration_duration"] > 0)
    df["clicks_per_day"] = safe_divide(df["total_clicks"], positive_duration)
    df["clicks_per_week"] = df["clicks_per_day"] * 7
    total_clicks_std = df["total_clicks"].std()
    if total_clicks_std and not np.isnan(total_clicks_std):
        df["engagement_intensity"] = (
            df["total_clicks"] - df["total_clicks"].mean()
        ) / total_clicks_std
    else:
        df["engagement_intensity"] = 0

    df["credit_load_category"] = pd.cut(
        df["studied_credits"],
        bins=[-np.inf, 60, 90, np.inf],
        labels=["light", "medium", "heavy"],
    ).astype(str)
    df["assessment_density"] = safe_divide(df["num_assessments"], df["studied_credits"])
    df["score_per_credit"] = safe_divide(df["total_score"], df["studied_credits"])
    df["activity_efficiency"] = safe_divide(
        df["score_per_assess"],
        df["clicks_per_credit"],
    )
    df["low_activity_flag"] = (df["total_clicks"] < df["total_clicks"].median()).astype(int)
    df["late_registration_flag"] = (
        df["registration_lead_days"] < df["registration_lead_days"].median()
    ).astype(int)
    df["high_click_but_low_score_flag"] = (
        (df["total_clicks"] > df["total_clicks"].median())
        & (df["total_score"] < df["total_score"].median())
    ).astype(int)

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def prepare_features(
    modeling_dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, float], dict[str, list[str]]]:
    '''
    Chuẩn bị các đặc trưng cho mô hình hóa từ dữ liệu đầu vào. Các bước bao gồm:
    - Điền giá trị thiếu cho các cột số bằng giá trị trung vị
    - Điền giá trị thiếu cho các cột phân loại bằng giá trị mode hoặc "Unknown"
    - Mã hóa one-hot cho các cột phân loại
    - Tách các đặc trưng và mục tiêu thành X, y_class, y_reg
    - Trả về dataset đã mã hóa, X, y_class, y_reg, dictionary chứa giá trị trung vị cho các cột số và dictionary chứa các mức phân loại cho các cột phân loại

    '''
    df_clean = modeling_dataset.copy()
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    medians = df_clean[numeric_columns].median(numeric_only=True).to_dict()
    df_clean[numeric_columns] = df_clean[numeric_columns].fillna(medians)

    category_levels: dict[str, list[str]] = {}
    for column in CATEGORICAL_COLUMNS:
        mode = _mode_or_default(df_clean[column], "Unknown")
        df_clean[column] = df_clean[column].fillna(mode).astype(str)
        category_levels[column] = sorted(df_clean[column].unique().tolist())

    encoded_dataset = pd.get_dummies(
        df_clean,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
        dtype=int,
    )
    ignored_columns = [
        TARGET_CLASS_COLUMN,
        TARGET_REGRESSION_COLUMN,
        "id_student",
        "final_result",
    ]
    feature_columns = [
        column
        for column in encoded_dataset.columns
        if column not in ignored_columns
    ]
    X = encoded_dataset[feature_columns].copy()
    X_medians = X.median(numeric_only=True).to_dict()
    X = X.fillna(X_medians)
    y_class = encoded_dataset[TARGET_CLASS_COLUMN].astype(int)
    y_reg = encoded_dataset[TARGET_REGRESSION_COLUMN].astype(float)
    return encoded_dataset, X, y_class, y_reg, X_medians, category_levels


def split_feature_sets(
    *,
    X: pd.DataFrame,
    y_class: pd.Series,
    y_reg: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    '''
    Chia tập đặc trưng thành tập train/test riêng cho bài toán phân loại và hồi quy.
    Tập phân loại được stratify theo nhãn khi có nhiều hơn một lớp.
    '''
    stratify = y_class if y_class.nunique() > 1 else None
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X,
        y_class,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X,
        y_reg,
        test_size=test_size,
        random_state=random_state,
    )
    return (
        X_train_class,
        X_test_class,
        y_train_class,
        y_test_class,
        X_train_reg,
        X_test_reg,
        y_train_reg,
        y_test_reg,
    )


def save_preprocessing_plots(
    *,
    modeling_dataset: pd.DataFrame,
    encoded_dataset: pd.DataFrame,
    paths: PreprocessingPaths,
) -> dict[str, str]:
    '''
    Tạo và lưu các biểu đồ mô tả phân phối mục tiêu và tương quan đặc trưng.
    Trả về dictionary ánh xạ tên biểu đồ tới đường dẫn file đã lưu.
    '''
    paths.plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {
        "final_result_distribution": paths.plot_dir / "final_result_distribution.png",
        "pass_fail_distribution": paths.plot_dir / "pass_fail_distribution.png",
        "learning_percentage_distribution": paths.plot_dir / "learning_percentage_distribution.png",
        "feature_correlation": paths.plot_dir / "feature_correlation.png",
    }

    _plot_bar(
        modeling_dataset[TARGET_CLASS_COLUMN].map({0: "Fail/Withdrawn", 1: "Pass/Distinction"}).value_counts(),
        plot_paths["pass_fail_distribution"],
        title="Pass/Fail Target Distribution",
        ylabel="Students",
    )
    if "final_result" in modeling_dataset.columns:
        final_result_counts = modeling_dataset["final_result"].value_counts()
    else:
        final_result_counts = pd.Series(dtype=int)
    _plot_bar(
        final_result_counts,
        plot_paths["final_result_distribution"],
        title="Final Result Distribution",
        ylabel="Students",
    )
    _plot_histogram(
        modeling_dataset[TARGET_REGRESSION_COLUMN],
        plot_paths["learning_percentage_distribution"],
        title="Learning Percentage Distribution",
        xlabel="Learning percentage",
    )
    _plot_correlation_heatmap(encoded_dataset, plot_paths["feature_correlation"])

    return {name: str(path) for name, path in plot_paths.items()}


def build_preprocessing_profile(
    *,
    raw_tables: dict[str, pd.DataFrame],
    missing_before: dict[str, dict[str, int]],
    missing_after: dict[str, dict[str, int]],
    modeling_dataset: pd.DataFrame,
    encoded_dataset: pd.DataFrame,
    test_size: float,
    random_state: int,
    plot_paths: dict[str, str],
) -> dict[str, Any]:
    '''
    Tổng hợp hồ sơ tiền xử lý gồm kích thước dữ liệu, dữ liệu thiếu, mục tiêu
    mô hình, cấu hình chia tập và đường dẫn biểu đồ.
    '''
    return {
        "raw_shapes": {
            name: {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])}
            for name, frame in raw_tables.items()
        },
        "missing_before": missing_before,
        "missing_after": missing_after,
        "modeling_rows": int(modeling_dataset.shape[0]),
        "encoded_rows": int(encoded_dataset.shape[0]),
        "encoded_columns": int(encoded_dataset.shape[1]),
        "feature_columns": [
            column
            for column in encoded_dataset.columns
            if column
            not in [TARGET_CLASS_COLUMN, TARGET_REGRESSION_COLUMN, "id_student", "final_result"]
        ],
        "classification_target": TARGET_CLASS_COLUMN,
        "regression_target": TARGET_REGRESSION_COLUMN,
        "class_distribution": {
            str(key): int(value)
            for key, value in modeling_dataset[TARGET_CLASS_COLUMN].value_counts().items()
        },
        "learning_percentage": {
            "mean": float(modeling_dataset[TARGET_REGRESSION_COLUMN].mean()),
            "min": float(modeling_dataset[TARGET_REGRESSION_COLUMN].min()),
            "max": float(modeling_dataset[TARGET_REGRESSION_COLUMN].max()),
        },
        "test_size": test_size,
        "random_state": random_state,
        "plots": plot_paths,
    }


def save_outputs(
    *,
    paths: PreprocessingPaths,
    students: pd.DataFrame,
    student_reg: pd.DataFrame,
    assessment_features: pd.DataFrame,
    vle_features: pd.DataFrame,
    modeling_dataset: pd.DataFrame,
    encoded_dataset: pd.DataFrame,
    X: pd.DataFrame,
    y_class: pd.Series,
    y_reg: pd.Series,
    X_train_class: pd.DataFrame,
    X_test_class: pd.DataFrame,
    y_train_class: pd.Series,
    y_test_class: pd.Series,
    X_train_reg: pd.DataFrame,
    X_test_reg: pd.DataFrame,
    y_train_reg: pd.Series,
    y_test_reg: pd.Series,
    medians: dict[str, float],
    category_levels: dict[str, list[str]],
    profile: dict[str, Any],
    plot_paths: dict[str, str],
) -> dict[str, str]:
    '''
    Lưu dataset đã xử lý, tập đặc trưng, target, bộ chia train/test và metadata.
    Trả về toàn bộ đường dẫn artifact để phục vụ tracking và đồng bộ dữ liệu.
    '''
    # Các file này mô tả dữ liệu sau khi làm sạch và xây dựng đặc trưng.
    preprocessed_files = {
        "students_labeled": paths.preprocessed_dir / "students_labeled.csv",
        "student_registration_cleaned": paths.preprocessed_dir / "student_registration_cleaned.csv",
        "vle_features": paths.preprocessed_dir / "vle_features.csv",
        "assessment_features": paths.preprocessed_dir / "assessment_features.csv",
        "modeling_dataset": paths.preprocessed_dir / "modeling_dataset.csv",
        "preprocessed_dataset": paths.preprocessed_dir / "preprocessed_dataset.csv",
        "preprocessing_profile": paths.preprocessed_dir / "preprocessing_profile.json",
    }
    students.to_csv(preprocessed_files["students_labeled"], index=False)
    student_reg.to_csv(preprocessed_files["student_registration_cleaned"], index=False)
    vle_features.to_csv(preprocessed_files["vle_features"], index=False)
    assessment_features.to_csv(preprocessed_files["assessment_features"], index=False)
    modeling_dataset.to_csv(preprocessed_files["modeling_dataset"], index=False)
    encoded_dataset.to_csv(preprocessed_files["preprocessed_dataset"], index=False)
    _write_json(preprocessed_files["preprocessing_profile"], profile)

    # Các file này là đầu vào trực tiếp và metadata phục vụ huấn luyện mô hình.
    feature_files = {
        "X": paths.feature_dir / "X.csv",
        "y": paths.feature_dir / "y.csv",
        "X_class": paths.feature_dir / "X_class.csv",
        "y_class": paths.feature_dir / "y_class.csv",
        "X_reg": paths.feature_dir / "X_reg.csv",
        "y_reg": paths.feature_dir / "y_reg.csv",
        "X_train": paths.feature_dir / "X_train.csv",
        "X_test": paths.feature_dir / "X_test.csv",
        "y_train": paths.feature_dir / "y_train.csv",
        "y_test": paths.feature_dir / "y_test.csv",
        "X_train_class": paths.feature_dir / "X_train_class.csv",
        "X_test_class": paths.feature_dir / "X_test_class.csv",
        "y_train_class": paths.feature_dir / "y_train_class.csv",
        "y_test_class": paths.feature_dir / "y_test_class.csv",
        "X_train_reg": paths.feature_dir / "X_train_reg.csv",
        "X_test_reg": paths.feature_dir / "X_test_reg.csv",
        "y_train_reg": paths.feature_dir / "y_train_reg.csv",
        "y_test_reg": paths.feature_dir / "y_test_reg.csv",
        "feature_columns": paths.feature_dir / "feature_columns.json",
        "classification_feature_columns": paths.feature_dir / "classification_feature_columns.json",
        "regression_feature_columns": paths.feature_dir / "regression_feature_columns.json",
        "medians": paths.feature_dir / "medians.json",
        "category_levels": paths.feature_dir / "category_levels.json",
        "encoders": paths.feature_dir / "encoders.joblib",
        "metadata": paths.feature_dir / "metadata.json",
    }
    X.to_csv(feature_files["X"], index=False)
    y_reg.rename(TARGET_REGRESSION_COLUMN).to_csv(feature_files["y"], index=False)
    X.to_csv(feature_files["X_class"], index=False)
    y_class.rename(TARGET_CLASS_COLUMN).to_csv(feature_files["y_class"], index=False)
    X.to_csv(feature_files["X_reg"], index=False)
    y_reg.rename(TARGET_REGRESSION_COLUMN).to_csv(feature_files["y_reg"], index=False)

    X_train_reg.to_csv(feature_files["X_train"], index=False)
    X_test_reg.to_csv(feature_files["X_test"], index=False)
    y_train_reg.rename(TARGET_REGRESSION_COLUMN).to_csv(feature_files["y_train"], index=False)
    y_test_reg.rename(TARGET_REGRESSION_COLUMN).to_csv(feature_files["y_test"], index=False)

    X_train_class.to_csv(feature_files["X_train_class"], index=False)
    X_test_class.to_csv(feature_files["X_test_class"], index=False)
    y_train_class.rename(TARGET_CLASS_COLUMN).to_csv(feature_files["y_train_class"], index=False)
    y_test_class.rename(TARGET_CLASS_COLUMN).to_csv(feature_files["y_test_class"], index=False)
    X_train_reg.to_csv(feature_files["X_train_reg"], index=False)
    X_test_reg.to_csv(feature_files["X_test_reg"], index=False)
    y_train_reg.rename(TARGET_REGRESSION_COLUMN).to_csv(feature_files["y_train_reg"], index=False)
    y_test_reg.rename(TARGET_REGRESSION_COLUMN).to_csv(feature_files["y_test_reg"], index=False)

    feature_columns = X.columns.tolist()
    _write_json(feature_files["feature_columns"], feature_columns)
    _write_json(feature_files["classification_feature_columns"], feature_columns)
    _write_json(feature_files["regression_feature_columns"], feature_columns)
    _write_json(feature_files["medians"], medians)
    _write_json(feature_files["category_levels"], category_levels)
    joblib.dump({"category_levels": category_levels}, feature_files["encoders"])
    _write_json(
        feature_files["metadata"],
        {
            "pipeline": "oulad_preprocessing",
            "classification_target": TARGET_CLASS_COLUMN,
            "regression_target": TARGET_REGRESSION_COLUMN,
            "target_column": TARGET_REGRESSION_COLUMN,
            "target_description": (
                "Learning percentage derived from weighted assessment scores and "
                "clipped to the 0-100 range. Students without assessment scores "
                "are assigned 0."
            ),
            "rows": int(encoded_dataset.shape[0]),
            "features": len(feature_columns),
            "classification_train_rows": int(X_train_class.shape[0]),
            "classification_test_rows": int(X_test_class.shape[0]),
            "regression_train_rows": int(X_train_reg.shape[0]),
            "regression_test_rows": int(X_test_reg.shape[0]),
            "class_positive_rate": float(y_class.mean()),
            "target_mean": float(y_reg.mean()),
            "target_min": float(y_reg.min()),
            "target_max": float(y_reg.max()),
            "categorical_columns": CATEGORICAL_COLUMNS,
            "base_model_feature_columns": MODEL_FEATURE_COLUMNS,
            "encoded_feature_columns": feature_columns,
            "plots": plot_paths,
        },
    )

    return {
        name: str(path)
        for name, path in (preprocessed_files | feature_files | plot_paths).items()
    }


def log_preprocessing_to_mlflow(
    *,
    paths: PreprocessingPaths,
    output_paths: dict[str, str],
    profile: dict[str, Any],
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    '''
    Ghi tham số, metric và artifact tiền xử lý lên MLflow nếu tracking được bật.
    Khi MLflow không khả dụng hoặc logging thất bại, trả về lý do tương ứng.
    '''
    tracking_uri = getattr(settings, "MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        return {"enabled": False, "reason": "MLFLOW_TRACKING_URI is not configured."}

    try:
        import mlflow
    except ImportError as exc:
        return {"enabled": False, "reason": f"MLflow is not installed: {exc}"}

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(run_name="preprocess-data") as run:
                mlflow.log_param("pipeline", "oulad_preprocessing")
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("random_state", random_state)
                mlflow.log_param("classification_target", TARGET_CLASS_COLUMN)
                mlflow.log_param("regression_target", TARGET_REGRESSION_COLUMN)
                mlflow.log_metric("rows", profile["encoded_rows"])
                mlflow.log_metric("features", len(profile["feature_columns"]))
                mlflow.log_metric(
                    "class_positive_rate",
                    profile["class_distribution"].get("1", 0) / profile["encoded_rows"],
                )
                mlflow.log_metric(
                    "learning_percentage.mean",
                    profile["learning_percentage"]["mean"],
                )
                mlflow.log_metric(
                    "learning_percentage.min",
                    profile["learning_percentage"]["min"],
                )
                mlflow.log_metric(
                    "learning_percentage.max",
                    profile["learning_percentage"]["max"],
                )
                mlflow.log_artifacts(str(paths.preprocessed_dir), artifact_path="preprocessed")
                mlflow.log_artifacts(str(paths.feature_dir), artifact_path="features")
                plot_artifacts = [
                    str(path)
                    for path in sorted(paths.plot_dir.glob("*"))
                    if path.is_file()
                ]
                if plot_artifacts:
                    mlflow.log_artifacts(str(paths.plot_dir), artifact_path="plots")

                return {
                    "enabled": True,
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "tracking_uri": tracking_uri,
                    "plot_artifacts": plot_artifacts,
                }
    except Exception as exc:
        return {
            "enabled": True,
            "tracking_uri": tracking_uri,
            "error": str(exc),
        }


def imd_to_mid(value: Any) -> float:
    '''
    Chuyển đổi giá trị imd_band thành một số trung bình tương ứng. Các giá trị có thể là một khoảng (ví dụ: "0-10%") 
    hoặc một giá trị duy nhất (ví dụ: "20%"). Nếu giá trị không hợp lệ hoặc thiếu, trả về NaN.
    '''
    try:
        if pd.isna(value) or str(value) == "Unknown":
            return np.nan
        text = str(value).replace("%", "").replace("<=", "").strip()
        parts = [part for part in text.split("-") if part]
        if len(parts) == 1:
            return float(parts[0])
        return (float(parts[0]) + float(parts[-1])) / 2.0
    except Exception:
        return np.nan


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    '''
    Thực hiện phép chia giữa hai Series một cách an toàn, tránh lỗi chia cho zero và xử lý các giá trị vô hạn.
    '''
    denominator = denominator.replace(0, np.nan)
    return numerator.div(denominator).replace([np.inf, -np.inf], np.nan).fillna(0)


def _add_targets(students: pd.DataFrame) -> pd.DataFrame:
    '''
    Tạo nhãn phân loại pass/fail từ kết quả cuối cùng của sinh viên.
    '''
    students = students.copy()
    students[TARGET_CLASS_COLUMN] = students["final_result"].map(
        {
            "Pass": 1,
            "Distinction": 1,
            "Fail": 0,
            "Withdrawn": 0,
        }
    ).fillna(0).astype(int)
    return students


def _plot_bar(values: pd.Series, path: Path, *, title: str, ylabel: str) -> None:
    '''
    Lưu biểu đồ cột của một Series tại đường dẫn được chỉ định.
    '''
    fig, ax = plt.subplots(figsize=(8, 5))
    values.sort_index().plot(kind="bar", ax=ax, color="#2E86AB")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_histogram(values: pd.Series, path: Path, *, title: str, xlabel: str) -> None:
    '''
    Lưu histogram biểu diễn phân phối của một biến số.
    '''
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values.dropna(), bins=24, color="#39A96B", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Students")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_correlation_heatmap(encoded_dataset: pd.DataFrame, path: Path) -> None:
    '''
    Lưu heatmap tương quan của các đặc trưng quan trọng có trong dataset.
    '''
    preferred_columns = [
        "learning_percentage",
        "target_pass",
        "total_score",
        "avg_score",
        "num_assessments",
        "total_clicks",
        "studied_credits",
        "registration_lead_days",
        "clicks_per_credit",
        "score_per_assess",
        "engagement_intensity",
        "activity_efficiency",
    ]
    columns = [column for column in preferred_columns if column in encoded_dataset.columns]
    corr = encoded_dataset[columns].corr(numeric_only=True).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Feature Correlation")
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _read_csv(path: Path) -> pd.DataFrame:
    '''
    Đọc file CSV nguồn và coi ký hiệu `?` là giá trị thiếu.
    '''
    if not path.exists():
        raise FileNotFoundError(f"Missing raw CSV file: {path}")
    return pd.read_csv(path, na_values=["?"])


def _mode_or_default(series: pd.Series, default: Any) -> Any:
    '''
    Tính mode của một Series, trả về giá trị mặc định nếu không có mode nào tồn tại.
    '''
    mode = series.dropna().mode()
    if mode.empty:
        return default
    return mode.iloc[0]


def _missing_summary(tables: dict[str, pd.DataFrame]) -> dict[str, dict[str, int]]:
    '''Tính toán số lượng giá trị thiếu trong mỗi cột của mỗi bảng.'''
    return {
        name: {str(column): int(value) for column, value in frame.isna().sum().items()}
        for name, frame in tables.items()
    }



def _ensure_directories(paths: PreprocessingPaths) -> None:
    '''
    Tạo các thư mục đầu ra cần thiết nếu chúng chưa tồn tại.
    '''
    paths.preprocessed_dir.mkdir(parents=True, exist_ok=True)
    paths.feature_dir.mkdir(parents=True, exist_ok=True)
    paths.plot_dir.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    '''
    Ghi payload thành file JSON UTF-8 với định dạng dễ đọc.
    '''
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _safe_sync_outputs_to_postgres(
    *,
    output_paths: dict[str, str],
    rows: int,
    features: int,
    train_rows: int,
    test_rows: int,
    class_positive_rate: float,
    target_mean: float,
    target_min: float,
    target_max: float,
    test_size: float,
    random_state: int,
    mlflow_tracking: dict[str, Any],
) -> dict[str, Any]:
    '''
    Thực hiện đồng bộ artifact sang PostgreSQL và chuyển lỗi thành kết quả trả về.
    '''
    try:
        return _sync_outputs_to_postgres(
            output_paths=output_paths,
            rows=rows,
            features=features,
            train_rows=train_rows,
            test_rows=test_rows,
            class_positive_rate=class_positive_rate,
            target_mean=target_mean,
            target_min=target_min,
            target_max=target_max,
            test_size=test_size,
            random_state=random_state,
            mlflow_tracking=mlflow_tracking,
        )
    except Exception as exc:
        return {
            "datasets": [],
            "files": [],
            "error": str(exc),
        }


def _sync_outputs_to_postgres(
    *,
    output_paths: dict[str, str],
    rows: int,
    features: int,
    train_rows: int,
    test_rows: int,
    class_positive_rate: float,
    target_mean: float,
    target_min: float,
    target_max: float,
    test_size: float,
    random_state: int,
    mlflow_tracking: dict[str, Any],
) -> dict[str, Any]:
    '''
    Chuyển artifact cùng metadata của pipeline sang hàm lưu trữ PostgreSQL.
    '''
    from preprocessing.storage import sync_preprocessing_outputs_to_postgres

    return sync_preprocessing_outputs_to_postgres(
        output_paths,
        metadata={
            "pipeline": "oulad_preprocessing",
            "rows": rows,
            "features": features,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "classification_target": TARGET_CLASS_COLUMN,
            "regression_target": TARGET_REGRESSION_COLUMN,
            "target_column": TARGET_REGRESSION_COLUMN,
            "class_positive_rate": class_positive_rate,
            "target_mean": target_mean,
            "target_min": target_min,
            "target_max": target_max,
            "test_size": test_size,
            "random_state": random_state,
            "mlflow": mlflow_tracking,
        },
    )
