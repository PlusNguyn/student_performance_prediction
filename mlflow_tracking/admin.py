from django.contrib import admin

from mlflow_tracking.models import ModelPromotion


@admin.register(ModelPromotion)
class ModelPromotionAdmin(admin.ModelAdmin):
    list_display = (
        "model_name",
        "alias",
        "version",
        "metric_name",
        "metric_value",
        "dry_run",
        "promoted_at",
    )
    list_filter = ("alias", "metric_name", "dry_run")
    search_fields = ("model_name", "version", "run_id")
