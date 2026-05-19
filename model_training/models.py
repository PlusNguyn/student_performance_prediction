from django.db import models


class TrainedModelArtifact(models.Model):
    MODEL = "model"
    EXPLAINER = "explainer"
    METADATA = "metadata"

    ARTIFACT_TYPE_CHOICES = [
        (MODEL, "Model"),
        (EXPLAINER, "Explainer"),
        (METADATA, "Metadata"),
    ]

    artifact_type = models.CharField(max_length=32, choices=ARTIFACT_TYPE_CHOICES)
    name = models.CharField(max_length=128)
    model_name = models.CharField(max_length=64, blank=True)
    source_path = models.CharField(max_length=512, blank=True)
    file_format = models.CharField(max_length=32, blank=True)
    file_size_bytes = models.PositiveBigIntegerField(default=0)
    sha256 = models.CharField(max_length=64, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    parameters = models.JSONField(default=dict, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    mlflow_run_id = models.CharField(max_length=128, blank=True)
    mlflow_experiment_id = models.CharField(max_length=128, blank=True)
    mlflow_registered_model_name = models.CharField(max_length=256, blank=True)
    is_best = models.BooleanField(default=False)
    text_content = models.TextField(blank=True)
    binary_content = models.BinaryField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["artifact_type", "name"],
                name="unique_trained_model_artifact_type_name",
            ),
        ]
        indexes = [
            models.Index(fields=["artifact_type", "name"]),
            models.Index(fields=["model_name", "is_best"]),
            models.Index(fields=["mlflow_run_id"]),
        ]

    def __str__(self) -> str:
        return f"{self.artifact_type}:{self.name}"
