from django.db import models


class TuningArtifact(models.Model):
    SUMMARY = "summary"
    PARAMETERS = "parameters"
    STUDY = "study"

    ARTIFACT_TYPE_CHOICES = [
        (SUMMARY, "Summary"),
        (PARAMETERS, "Parameters"),
        (STUDY, "Study"),
    ]

    artifact_type = models.CharField(max_length=32, choices=ARTIFACT_TYPE_CHOICES)
    name = models.CharField(max_length=128)
    model_name = models.CharField(max_length=64, blank=True)
    source_path = models.CharField(max_length=512, blank=True)
    file_format = models.CharField(max_length=32, blank=True)
    file_size_bytes = models.PositiveBigIntegerField(default=0)
    sha256 = models.CharField(max_length=64, blank=True)
    scoring = models.CharField(max_length=128, blank=True)
    best_value = models.FloatField(null=True, blank=True)
    best_params = models.JSONField(default=dict, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    text_content = models.TextField(blank=True)
    binary_content = models.BinaryField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["artifact_type", "name"],
                name="unique_tuning_artifact_type_name",
            ),
        ]
        indexes = [
            models.Index(fields=["artifact_type", "name"]),
            models.Index(fields=["model_name"]),
        ]

    def __str__(self) -> str:
        return f"{self.artifact_type}:{self.name}"
