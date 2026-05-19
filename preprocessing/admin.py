from django.contrib import admin

from preprocessing.models import (
    PipelineFileArtifact,
    ProcessedDataset,
    ProcessedDatasetRow,
)


@admin.register(ProcessedDataset)
class ProcessedDatasetAdmin(admin.ModelAdmin):
    list_display = ("kind", "name", "row_count", "column_count", "updated_at")
    list_filter = ("kind",)
    search_fields = ("name", "source_path")


@admin.register(ProcessedDatasetRow)
class ProcessedDatasetRowAdmin(admin.ModelAdmin):
    list_display = ("dataset", "row_index")
    search_fields = ("dataset__name",)


@admin.register(PipelineFileArtifact)
class PipelineFileArtifactAdmin(admin.ModelAdmin):
    list_display = ("kind", "name", "file_format", "file_size_bytes", "updated_at")
    list_filter = ("kind", "file_format")
    search_fields = ("name", "source_path", "sha256")
