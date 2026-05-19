from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from model_training.models import TrainedModelArtifact


MODEL_ARTIFACTS = {
    "xgboost_model": "xgboost",
    "lightgbm_model": "lightgbm",
    "best_model": "",
}
METADATA_ARTIFACTS = {
    "metrics",
    "model_selection",
    "features",
    "medians",
}
EXPLAINER_ARTIFACTS = {
    "xgboost_explainer": "xgboost",
}


def sync_training_outputs_to_postgres(
    *,
    output_paths: dict[str, str | Path],
    metrics: dict[str, dict[str, float]],
    best_model_name: str,
    model_selection: dict[str, Any],
    mlflow_tracking: dict[str, Any],
) -> dict[str, list[str]]:
    synced = {"models": [], "metadata": [], "explainers": []}

    for name, raw_path in output_paths.items():
        path = Path(raw_path)
        if not path.exists():
            continue

        if name in MODEL_ARTIFACTS:
            model_name = MODEL_ARTIFACTS[name] or best_model_name
            _sync_artifact(
                artifact_type=TrainedModelArtifact.MODEL,
                name=name,
                model_name=model_name,
                path=path,
                metrics=metrics.get(model_name, {}),
                parameters=_model_parameters(model_selection, model_name),
                metadata={
                    "pipeline": "model_training",
                    "best_model": best_model_name,
                    "model_selection": model_selection,
                },
                mlflow_tracking=mlflow_tracking,
                is_best=name == "best_model" or model_name == best_model_name,
            )
            synced["models"].append(name)
            continue

        if name in METADATA_ARTIFACTS:
            _sync_artifact(
                artifact_type=TrainedModelArtifact.METADATA,
                name=name,
                model_name=best_model_name,
                path=path,
                metrics=metrics.get(best_model_name, {}),
                parameters={},
                metadata={
                    "pipeline": "model_training",
                    "best_model": best_model_name,
                    "json_content": _read_json_if_available(path),
                },
                mlflow_tracking=mlflow_tracking,
                is_best=False,
            )
            synced["metadata"].append(name)
            continue

        if name in EXPLAINER_ARTIFACTS:
            model_name = EXPLAINER_ARTIFACTS[name]
            _sync_artifact(
                artifact_type=TrainedModelArtifact.EXPLAINER,
                name=name,
                model_name=model_name,
                path=path,
                metrics=metrics.get(model_name, {}),
                parameters=_model_parameters(model_selection, model_name),
                metadata={
                    "pipeline": "model_training",
                    "best_model": best_model_name,
                },
                mlflow_tracking=mlflow_tracking,
                is_best=model_name == best_model_name,
            )
            synced["explainers"].append(name)

    return synced


def _sync_artifact(
    *,
    artifact_type: str,
    name: str,
    model_name: str,
    path: Path,
    metrics: dict[str, float],
    parameters: dict[str, Any],
    metadata: dict[str, Any],
    mlflow_tracking: dict[str, Any],
    is_best: bool,
) -> TrainedModelArtifact:
    payload = path.read_bytes()
    text_content = ""
    binary_content = None

    if path.suffix.lower() in {".json", ".txt"}:
        text_content = payload.decode("utf-8")
    else:
        binary_content = payload

    artifact, _ = TrainedModelArtifact.objects.update_or_create(
        artifact_type=artifact_type,
        name=name,
        defaults={
            "model_name": model_name,
            "source_path": str(path),
            "file_format": path.suffix.lstrip("."),
            "file_size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "metrics": metrics,
            "parameters": parameters,
            "metadata": metadata,
            "mlflow_run_id": mlflow_tracking.get("run_id", ""),
            "mlflow_experiment_id": mlflow_tracking.get("experiment_id", ""),
            "mlflow_registered_model_name": mlflow_tracking.get(
                "registered_model_name",
                "",
            ),
            "is_best": is_best,
            "text_content": text_content,
            "binary_content": binary_content,
        },
    )
    return artifact


def _model_parameters(
    model_selection: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    tuning_config = model_selection.get("tuning_config") or {}
    model_results = tuning_config.get("models") or {}
    if model_name in model_results:
        return model_results[model_name].get("best_params", {})
    if tuning_config.get("best_model") == model_name:
        return tuning_config.get("best_params", {})
    return {}


def _read_json_if_available(path: Path) -> Any:
    if path.suffix.lower() != ".json":
        return None
    return json.loads(path.read_text(encoding="utf-8"))
