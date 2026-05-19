from django.contrib import admin

try:
    from prediction.models import PredictionArtifact, PredictionBatch, PredictionRecord
except ImportError:
    PredictionArtifact = PredictionBatch = PredictionRecord = None


if PredictionBatch:
    @admin.register(PredictionBatch)
    class PredictionBatchAdmin(admin.ModelAdmin):
        list_display = ("name", "model_name", "row_count", "updated_at")
        search_fields = ("name", "model_name", "description")


if PredictionRecord:
    @admin.register(PredictionRecord)
    class PredictionRecordAdmin(admin.ModelAdmin):
        list_display = (
            "batch",
            "id_student",
            "actual_value",
            "predicted_value",
            "absolute_error",
        )
        list_filter = ("batch",)
        search_fields = ("id_student",)


if PredictionArtifact:
    @admin.register(PredictionArtifact)
    class PredictionArtifactAdmin(admin.ModelAdmin):
        list_display = ("batch", "artifact_type", "name", "file_format", "updated_at")
        list_filter = ("batch", "artifact_type")
        search_fields = ("name", "source_path")
