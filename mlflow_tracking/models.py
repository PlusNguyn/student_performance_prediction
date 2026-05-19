from django.db import models


class ModelPromotion(models.Model):
    model_name = models.CharField(max_length=256)
    alias = models.CharField(max_length=128)
    version = models.CharField(max_length=64)
    run_id = models.CharField(max_length=128)
    metric_name = models.CharField(max_length=128)
    metric_value = models.FloatField()
    previous_version = models.CharField(max_length=64, blank=True)
    dry_run = models.BooleanField(default=False)
    metadata = models.JSONField(default=dict, blank=True)
    promoted_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["model_name", "alias"]),
            models.Index(fields=["run_id"]),
            models.Index(fields=["promoted_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.model_name}@{self.alias}=v{self.version}"
