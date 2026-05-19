from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from django.db import transaction

from preprocessing.models import (
    PipelineFileArtifact,
    ProcessedDataset,
    ProcessedDatasetRow,
)


PREPROCESSED_DATASETS = {
    "vle_features",
    "assessment_features",
    "preprocessed_dataset",
}
FEATURE_DATASETS = {
    "X",
    "y",
    "X_train",
    "X_test",
    "y_train",
    "y_test",
}
FEATURE_FILE_ARTIFACTS = {
    "feature_columns",
    "medians",
    "metadata",
    "encoders",
}


def sync_preprocessing_outputs_to_postgres(
    output_paths: dict[str, str | Path],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    synced = {"datasets": [], "files": []}
    metadata = metadata or {}

    for name, raw_path in output_paths.items():
        path = Path(raw_path)
        if not path.exists():
            continue

        if name in PREPROCESSED_DATASETS:
            _sync_tabular_dataset(
                kind=ProcessedDataset.PREPROCESSED,
                name=name,
                path=path,
                metadata=metadata,
            )
            synced["datasets"].append(name)
            continue

        if name in FEATURE_DATASETS:
            _sync_tabular_dataset(
                kind=ProcessedDataset.FEATURE,
                name=name,
                path=path,
                metadata=metadata,
            )
            synced["datasets"].append(name)
            continue

        if name in FEATURE_FILE_ARTIFACTS:
            _sync_file_artifact(
                kind=PipelineFileArtifact.FEATURE_METADATA,
                name=name,
                path=path,
                metadata=metadata,
            )
            synced["files"].append(name)

    return synced


def _sync_tabular_dataset(
    *,
    kind: str,
    name: str,
    path: Path,
    metadata: dict[str, Any],
) -> ProcessedDataset:
    dataframe = pd.read_csv(path)
    columns = [str(column) for column in dataframe.columns]

    with transaction.atomic():
        dataset, _ = ProcessedDataset.objects.update_or_create(
            kind=kind,
            name=name,
            defaults={
                "source_path": str(path),
                "row_count": int(dataframe.shape[0]),
                "column_count": int(dataframe.shape[1]),
                "columns": columns,
                "metadata": {
                    **metadata,
                    "file_format": path.suffix.lstrip("."),
                },
            },
        )
        ProcessedDatasetRow.objects.filter(dataset=dataset).delete()
        _bulk_insert_rows(dataset, dataframe)

    return dataset


def _bulk_insert_rows(dataset: ProcessedDataset, dataframe: pd.DataFrame) -> None:
    rows: list[ProcessedDatasetRow] = []
    normalized = dataframe.where(pd.notnull(dataframe), None)

    for row_index, record in enumerate(normalized.to_dict(orient="records")):
        rows.append(
            ProcessedDatasetRow(
                dataset=dataset,
                row_index=row_index,
                data={str(key): _json_safe(value) for key, value in record.items()},
            )
        )
        if len(rows) >= 1000:
            ProcessedDatasetRow.objects.bulk_create(rows, batch_size=1000)
            rows.clear()

    if rows:
        ProcessedDatasetRow.objects.bulk_create(rows, batch_size=1000)


def _sync_file_artifact(
    *,
    kind: str,
    name: str,
    path: Path,
    metadata: dict[str, Any],
) -> PipelineFileArtifact:
    payload = path.read_bytes()
    text_content = ""
    binary_content = None

    if path.suffix.lower() in {".json", ".txt", ".csv"}:
        text_content = payload.decode("utf-8")
    else:
        binary_content = payload

    artifact, _ = PipelineFileArtifact.objects.update_or_create(
        kind=kind,
        name=name,
        defaults={
            "source_path": str(path),
            "file_format": path.suffix.lstrip("."),
            "file_size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "metadata": {
                **metadata,
                "json_content": _read_json_if_available(path),
            },
            "text_content": text_content,
            "binary_content": binary_content,
        },
    )
    return artifact


def _read_json_if_available(path: Path) -> Any:
    if path.suffix.lower() != ".json":
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
