from django.core.management.base import BaseCommand, CommandError

from mlflow_tracking.registry import (
    promote_best_model_version,
    promote_specific_model_version,
)


class Command(BaseCommand):
    help = "Promote the best MLflow model version to a production alias."

    def add_arguments(self, parser):
        parser.add_argument("--model-name", default=None)
        parser.add_argument("--alias", default="production")
        parser.add_argument("--metric", default=None)
        parser.add_argument("--model-version", default=None)
        parser.add_argument("--tracking-uri", default=None)
        parser.add_argument("--lower-is-better", action="store_true")
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **options):
        try:
            if options["model_version"]:
                result = promote_specific_model_version(
                    model_name=options["model_name"],
                    version=options["model_version"],
                    alias=options["alias"],
                    tracking_uri=options["tracking_uri"],
                    dry_run=options["dry_run"],
                )
            else:
                result = promote_best_model_version(
                    model_name=options["model_name"],
                    alias=options["alias"],
                    metric_name=options["metric"],
                    higher_is_better=False if options["lower_is_better"] else None,
                    tracking_uri=options["tracking_uri"],
                    dry_run=options["dry_run"],
                )
        except Exception as exc:
            raise CommandError(str(exc)) from exc

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
            self.stdout.write(f"Previous {result.alias}: version {result.previous_version}")
