from django.core.management.base import BaseCommand

from model_training.training import TrainingPaths, train_models


class Command(BaseCommand):
    help = "Train XGBoost and LightGBM models from generated feature files."

    def add_arguments(self, parser):
        parser.add_argument("--feature-dir", default=None)
        parser.add_argument("--model-dir", default=None)
        parser.add_argument("--params-path", default=None)
        parser.add_argument("--seed", type=int, default=42)

    def handle(self, *args, **options):
        paths = TrainingPaths.from_defaults(
            feature_dir=options["feature_dir"],
            model_dir=options["model_dir"],
            params_path=options["params_path"],
        )
        summary = train_models(paths=paths, random_state=options["seed"])

        self.stdout.write(self.style.SUCCESS("Model training completed."))
        for name, metrics in summary["metrics"].items():
            self.stdout.write(
                f"{name}: RMSE={metrics['rmse']:.4f} "
                f"MAE={metrics['mae']:.4f} "
                f"R2={metrics['r2']:.4f} "
                f"Within10={metrics['within_10_percent']:.2%}"
            )
        self.stdout.write(f"Best model: {summary['best_model']}")
        mlflow_status = summary["mlflow"]
        if mlflow_status.get("run_id"):
            self.stdout.write(
                self.style.SUCCESS(
                    f"MLflow run: {mlflow_status['run_id']} "
                    f"({mlflow_status['tracking_uri']})"
                )
            )
        elif mlflow_status.get("error"):
            self.stdout.write(
                self.style.WARNING(
                    f"MLflow tracking skipped with error: {mlflow_status['error']}"
                )
            )
        else:
            self.stdout.write(
                self.style.WARNING(
                    f"MLflow tracking disabled: {mlflow_status.get('reason')}"
                )
            )
        postgres_sync = summary["postgres_sync"]
        self.stdout.write(
            self.style.SUCCESS(
                "Postgres sync: "
                f"{len(postgres_sync['models'])} models, "
                f"{len(postgres_sync['metadata'])} metadata files, "
                f"{len(postgres_sync['explainers'])} explainers"
            )
        )
        self.stdout.write(f"Model dir: {paths.model_dir}")
