from django.contrib import admin

from tuning_optuna.models import TuningArtifact


@admin.register(TuningArtifact)
class TuningArtifactAdmin(admin.ModelAdmin):
    list_display = (
        "artifact_type",
        "name",
        "model_name",
        "scoring",
        "best_value",
        "updated_at",
    )
    list_filter = ("artifact_type", "model_name", "scoring")
    search_fields = ("name", "model_name", "source_path", "sha256")
