from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tuning_optuna.models import TuningArtifact


def sync_tuning_outputs_to_postgres(
    *,
    output_paths: dict[str, str | Path],
    summary: dict[str, Any],
    model_name: str,
) -> dict[str, list[str]]:
    synced = {"artifacts": []}

    for name, raw_path in output_paths.items():
        path = Path(raw_path)
        if not path.exists():
            continue

        artifact_type = _artifact_type(name)
        _sync_artifact(
            artifact_type=artifact_type,
            name=_artifact_name(name, model_name),
            model_name=model_name,
            path=path,
            summary=summary,
        )
        synced["artifacts"].append(name)

    return synced


def _sync_artifact(
    *,
    artifact_type: str,
    name: str,
    model_name: str,
    path: Path,
    summary: dict[str, Any],
) -> TuningArtifact:
    payload = path.read_bytes()
    text_content = ""
    binary_content = None

    if path.suffix.lower() in {".json", ".txt"}:
        text_content = payload.decode("utf-8")
    else:
        binary_content = payload

    artifact, _ = TuningArtifact.objects.update_or_create(
        artifact_type=artifact_type,
        name=name,
        defaults={
            "model_name": model_name,
            "source_path": str(path),
            "file_format": path.suffix.lstrip("."),
            "file_size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "scoring": summary.get("scoring", ""),
            "best_value": summary.get("best_value"),
            "best_params": summary.get("best_params", {}),
            "metadata": {
                "pipeline": "tuning_optuna",
                "json_content": _read_json_if_available(path),
                "summary": {
                    key: value
                    for key, value in summary.items()
                    if key not in {"paths"}
                },
            },
            "text_content": text_content,
            "binary_content": binary_content,
        },
    )
    return artifact


def _artifact_type(name: str) -> str:
    if "params" in name:
        return TuningArtifact.PARAMETERS
    if "study" in name:
        return TuningArtifact.STUDY
    return TuningArtifact.SUMMARY


def _artifact_name(name: str, model_name: str) -> str:
    if model_name == "best_model":
        return name
    return f"{model_name}_{name}"


def _read_json_if_available(path: Path) -> Any:
    if path.suffix.lower() != ".json":
        return None
    return json.loads(path.read_text(encoding="utf-8"))
