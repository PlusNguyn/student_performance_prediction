from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from django.conf import settings


CLASSIFICATION_TASK = "classification"
REGRESSION_TASK = "regression"
MODEL_TASKS = (CLASSIFICATION_TASK, REGRESSION_TASK)
TASK_METRIC_CANDIDATES = {
    CLASSIFICATION_TASK: (
        "classification.f1",
        "classification.roc_auc",
        "classification.accuracy",
        "classification.precision",
        "classification.recall",
    ),
    REGRESSION_TASK: (
        "regression.rmse",
        "regression.mae",
        "regression.r2",
        "regression.within_10_percent",
    ),
}
DEFAULT_METRIC_CANDIDATES = (
    *TASK_METRIC_CANDIDATES[REGRESSION_TASK],
    *TASK_METRIC_CANDIDATES[CLASSIFICATION_TASK],
)


@dataclass(frozen=True)
class PromotionResult:
    model_name: str
    alias: str
    version: str
    run_id: str
    metric_name: str
    metric_value: float
    previous_version: str | None
    dry_run: bool


def promote_best_model_version(
    *,
    model_name: str | None = None,
    task: str | None = None,
    alias: str = "production",
    metric_name: str | None = None,
    higher_is_better: bool | None = None,
    tracking_uri: str | None = None,
    dry_run: bool = False,
) -> PromotionResult:
    client = build_mlflow_client(tracking_uri)
    resolved_task = _resolve_task(task) or _infer_task_from_metric(metric_name)
    model_name = _resolve_model_name(model_name, resolved_task)
    resolved_task = resolved_task or _infer_task_from_model_name(model_name)
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No versions found for registered model `{model_name}`.")

    selected_metric = metric_name or _select_metric_name(client, versions, resolved_task)
    reverse = (
        higher_is_better
        if higher_is_better is not None
        else not _metric_lower_is_better(selected_metric)
    )
    scored_versions = [
        (version, _score_version(client, version, selected_metric))
        for version in versions
    ]
    scored_versions = [
        (version, score)
        for version, score in scored_versions
        if score is not None
    ]
    if not scored_versions:
        raise ValueError(
            f"No model versions for `{model_name}` contain metric `{selected_metric}`."
        )

    best_version, best_score = sorted(
        scored_versions,
        key=lambda item: item[1]["value"],
        reverse=reverse,
    )[0]
    previous_version = get_alias_version(client, model_name, alias)

    if not dry_run:
        client.set_registered_model_alias(model_name, alias, best_version.version)

    result = PromotionResult(
        model_name=model_name,
        alias=alias,
        version=best_version.version,
        run_id=best_version.run_id,
        metric_name=best_score["name"],
        metric_value=best_score["value"],
        previous_version=previous_version,
        dry_run=dry_run,
    )
    _record_model_promotion(result, source="best_metric")
    return result


def promote_specific_model_version(
    *,
    version: str,
    model_name: str | None = None,
    task: str | None = None,
    alias: str = "production",
    tracking_uri: str | None = None,
    dry_run: bool = False,
) -> PromotionResult:
    '''
    chuyển một version cụ thể của model lên production. Không đánh giá version đó so với các version khác, chỉ đơn thuần promote version được chỉ định.
    '''
    client = build_mlflow_client(tracking_uri)
    resolved_task = _resolve_task(task)
    model_name = _resolve_model_name(model_name, resolved_task)
    resolved_task = resolved_task or _infer_task_from_model_name(model_name)
    model_version = client.get_model_version(model_name, version)
    run = client.get_run(model_version.run_id)
    score = _pick_metric(run.data.metrics, None, resolved_task)
    previous_version = get_alias_version(client, model_name, alias)

    if not dry_run:
        client.set_registered_model_alias(model_name, alias, version)

    result = PromotionResult(
        model_name=model_name,
        alias=alias,
        version=model_version.version,
        run_id=model_version.run_id,
        metric_name=score["name"],
        metric_value=score["value"],
        previous_version=previous_version,
        dry_run=dry_run,
    )
    _record_model_promotion(result, source="specific_version")
    return result


def load_production_model_uri(
    model_name: str | None = None,
    task: str | None = None,
    alias: str = "production",
) -> str:
    '''
    Tạo URI cho model đang được lưu trữ trong production. URI này có thể được sử dụng để tải model bằng mlflow.pyfunc.load_model() hoặc các hàm tương tự.
    '''
    model_name = _resolve_model_name(model_name, _resolve_task(task))
    return f"models:/{model_name}@{alias}"


def promote_default_model_versions(
    *,
    model_name: str | None = None,
    alias: str = "production",
    tracking_uri: str | None = None,
    dry_run: bool = False,
) -> tuple[PromotionResult, ...]:
    '''
    Promote the best versions of a model for each task. Điều này sẽ tìm kiếm các version tốt nhất của model cho cả classification và regression (nếu có) và promote chúng lên production dưới alias được chỉ định. Kết quả trả về là một tuple chứa kết quả promotion cho mỗi task.
    '''
    return tuple(
        promote_best_model_version(
            model_name=model_name,
            task=task,
            alias=alias,
            tracking_uri=tracking_uri,
            dry_run=dry_run,
        )
        for task in MODEL_TASKS
    )


def build_mlflow_client(tracking_uri: str | None = None) -> Any:
    try:
        import mlflow
        from mlflow import MlflowClient
    except ImportError as exc:
        raise ImportError(
            "MLflow is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    resolved_tracking_uri = tracking_uri or settings.MLFLOW_TRACKING_URI
    if not resolved_tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is not configured.")

    mlflow.set_tracking_uri(resolved_tracking_uri)
    return MlflowClient(tracking_uri=resolved_tracking_uri)


def get_alias_version(client: Any, model_name: str, alias: str) -> str | None:
    try:
        return client.get_model_version_by_alias(model_name, alias).version
    except Exception:
        return None


def _score_version(
    client: Any,
    model_version: Any,
    metric_name: str,
) -> dict[str, float | str] | None:
    run = client.get_run(model_version.run_id)
    try:
        return _pick_metric(run.data.metrics, metric_name)
    except ValueError:
        return None


def _pick_metric(
    metrics: dict[str, float],
    metric_name: str | None,
    task: str | None = None,
) -> dict[str, float | str]:
    if metric_name:
        if metric_name not in metrics:
            raise ValueError(f"Metric `{metric_name}` is missing.")
        return {"name": metric_name, "value": float(metrics[metric_name])}

    for candidate in _metric_candidates(task):
        if candidate in metrics:
            return {"name": candidate, "value": float(metrics[candidate])}

    raise ValueError("No default promotion metric found.")


def _select_metric_name(client: Any, versions: list[Any], task: str | None = None) -> str:
    metrics_by_version = [
        client.get_run(version.run_id).data.metrics
        for version in versions
    ]
    candidates = _metric_candidates(task)
    for candidate in candidates:
        if any(candidate in metrics for metrics in metrics_by_version):
            return candidate
    raise ValueError(
        "No model version contains a promotion metric. "
        f"Tried: {', '.join(candidates)}."
    )


def _metric_lower_is_better(metric_name: str) -> bool:
    lowered = metric_name.lower()
    return any(token in lowered for token in ("rmse", "mae", "mse", "loss", "error"))


def _metric_candidates(task: str | None) -> tuple[str, ...]:
    resolved_task = _resolve_task(task)
    if resolved_task:
        return TASK_METRIC_CANDIDATES[resolved_task]
    return DEFAULT_METRIC_CANDIDATES


def _resolve_task(task: str | None) -> str | None:
    if task is None:
        return None
    if task not in MODEL_TASKS:
        raise ValueError(
            f"Unknown model task `{task}`. Expected one of: {', '.join(MODEL_TASKS)}."
        )
    return task


def _resolve_model_name(model_name: str | None, task: str | None) -> str:
    base_name = model_name or settings.MLFLOW_REGISTERED_MODEL_NAME
    if not task:
        return base_name

    suffix = f"-{task}"
    if base_name.endswith(suffix):
        return base_name
    for existing_task in MODEL_TASKS:
        existing_suffix = f"-{existing_task}"
        if base_name.endswith(existing_suffix):
            return f"{base_name[: -len(existing_suffix)]}{suffix}"
    return f"{base_name}{suffix}"


def _infer_task_from_model_name(model_name: str) -> str | None:
    for task in MODEL_TASKS:
        if model_name.endswith(f"-{task}"):
            return task
    return None


def _infer_task_from_metric(metric_name: str | None) -> str | None:
    if not metric_name:
        return None
    for task in MODEL_TASKS:
        if metric_name.startswith(f"{task}."):
            return task
    return None


def _record_model_promotion(result: PromotionResult, *, source: str) -> None:
    from mlflow_tracking.models import ModelPromotion

    ModelPromotion.objects.create(
        model_name=result.model_name,
        alias=result.alias,
        version=result.version,
        run_id=result.run_id,
        metric_name=result.metric_name,
        metric_value=result.metric_value,
        previous_version=result.previous_version or "",
        dry_run=result.dry_run,
        metadata={"source": source},
    )
