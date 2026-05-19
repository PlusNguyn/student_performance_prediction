from django.db import models


class ProcessedDataset(models.Model):
    PREPROCESSED = "preprocessed"
    FEATURE = "feature"

    KIND_CHOICES = [
        (PREPROCESSED, "Preprocessed"),
        (FEATURE, "Feature"),
    ]

    kind = models.CharField(max_length=32, choices=KIND_CHOICES)
    name = models.CharField(max_length=128)
    source_path = models.CharField(max_length=512, blank=True)
    row_count = models.PositiveIntegerField(default=0)
    column_count = models.PositiveIntegerField(default=0)
    columns = models.JSONField(default=list, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["kind", "name"],
                name="unique_processed_dataset_kind_name",
            ),
        ]
        indexes = [
            models.Index(fields=["kind", "name"]),
        ]

    def __str__(self) -> str:
        return f"{self.kind}:{self.name}"


class ProcessedDatasetRow(models.Model):
    dataset = models.ForeignKey(
        ProcessedDataset,
        on_delete=models.CASCADE,
        related_name="records",
    )
    row_index = models.PositiveIntegerField()
    data = models.JSONField()

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["dataset", "row_index"],
                name="unique_processed_dataset_row_index",
            ),
        ]
        indexes = [
            models.Index(fields=["dataset", "row_index"]),
        ]

    def __str__(self) -> str:
        return f"{self.dataset}#{self.row_index}"


class PipelineFileArtifact(models.Model):
    FEATURE_METADATA = "feature_metadata"
    PREPROCESSING_METADATA = "preprocessing_metadata"

    KIND_CHOICES = [
        (FEATURE_METADATA, "Feature metadata"),
        (PREPROCESSING_METADATA, "Preprocessing metadata"),
    ]

    kind = models.CharField(max_length=64, choices=KIND_CHOICES)
    name = models.CharField(max_length=128)
    source_path = models.CharField(max_length=512, blank=True)
    file_format = models.CharField(max_length=32, blank=True)
    file_size_bytes = models.PositiveBigIntegerField(default=0)
    sha256 = models.CharField(max_length=64, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    text_content = models.TextField(blank=True)
    binary_content = models.BinaryField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["kind", "name"],
                name="unique_pipeline_file_artifact_kind_name",
            ),
        ]
        indexes = [
            models.Index(fields=["kind", "name"]),
        ]

    def __str__(self) -> str:
        return f"{self.kind}:{self.name}"
