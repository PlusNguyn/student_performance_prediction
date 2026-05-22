from django.core.management.base import BaseCommand, CommandError

from mlflow_tracking.registry import (
    MODEL_TASKS,
    promote_default_model_versions,
    promote_best_model_version,
    promote_specific_model_version,
)


class Command(BaseCommand):
    help = "Promote the best MLflow model version to a production alias."

    def add_arguments(self, parser):
        parser.add_argument("--model-name", default=None)
        parser.add_argument(
            "--task",
            choices=[*MODEL_TASKS, "both"],
            default="both",
        )
        parser.add_argument("--alias", default="production")
        parser.add_argument("--metric", default=None)
        parser.add_argument("--model-version", default=None)
        parser.add_argument("--tracking-uri", default=None)
        parser.add_argument("--lower-is-better", action="store_true")
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **options):
        try:
            if options["task"] == "both" and not options["model_version"]:
                if options["metric"]:
                    raise CommandError(
                        "Use --task classification or --task regression when passing --metric."
                    )
                if options["lower_is_better"]:
                    raise CommandError(
                        "Use --task classification or --task regression when passing --lower-is-better."
                    )
                results = promote_default_model_versions(
                    model_name=options["model_name"],
                    alias=options["alias"],
                    tracking_uri=options["tracking_uri"],
                    dry_run=options["dry_run"],
                )
            elif options["model_version"]:
                tasks = MODEL_TASKS if options["task"] == "both" else (options["task"],)
                results = tuple(
                    promote_specific_model_version(
                        model_name=options["model_name"],
                        task=task,
                        version=options["model_version"],
                        alias=options["alias"],
                        tracking_uri=options["tracking_uri"],
                        dry_run=options["dry_run"],
                    )
                    for task in tasks
                )
            else:
                result = promote_best_model_version(
                    model_name=options["model_name"],
                    task=options["task"],
                    alias=options["alias"],
                    metric_name=options["metric"],
                    higher_is_better=False if options["lower_is_better"] else None,
                    tracking_uri=options["tracking_uri"],
                    dry_run=options["dry_run"],
                )
                results = (result,)
        except Exception as exc:
            raise CommandError(str(exc)) from exc

        for result in results:
            action = "Would promote" if result.dry_run else "Promoted"
            self.stdout.write(
                self.style.SUCCESS(
                    f"{action} `{result.model_name}` version {result.version} "
                    f"to alias `{result.alias}`."
                )
            )
            self.stdout.write(f"Run ID: {result.run_id}")
            self.stdout.write(f"Metric: {result.metric_name}={result.metric_value:.6f}")
            if result.previous_version:
                self.stdout.write(
                    f"Previous {result.alias}: version {result.previous_version}"
                )
