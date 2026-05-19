from django.contrib import admin

from model_training.models import TrainedModelArtifact


@admin.register(TrainedModelArtifact)
class TrainedModelArtifactAdmin(admin.ModelAdmin):
    list_display = (
        "artifact_type",
        "name",
        "model_name",
        "is_best",
        "file_size_bytes",
        "updated_at",
    )
    list_filter = ("artifact_type", "model_name", "is_best")
    search_fields = ("name", "model_name", "source_path", "mlflow_run_id", "sha256")
