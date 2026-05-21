from django.core.management.base import BaseCommand

from preprocessing.pipeline import PreprocessingPaths, run_preprocessing


class Command(BaseCommand):
    help = "Preprocess OULAD raw CSV files and create train/test feature files."

    def add_arguments(self, parser):
        parser.add_argument("--raw-dir", default=None)
        parser.add_argument("--preprocessed-dir", default=None)
        parser.add_argument("--feature-dir", default=None)
        parser.add_argument("--plot-dir", default=None)
        parser.add_argument("--test-size", type=float, default=0.2)
        parser.add_argument("--seed", type=int, default=42)

    def handle(self, *args, **options):
        paths = PreprocessingPaths.from_defaults(
            raw_dir=options["raw_dir"],
            preprocessed_dir=options["preprocessed_dir"],
            feature_dir=options["feature_dir"],
            plot_dir=options["plot_dir"],
        )
        summary = run_preprocessing(
            paths=paths,
            test_size=options["test_size"],
            random_state=options["seed"],
        )

        self.stdout.write(self.style.SUCCESS("Preprocessing completed."))
        self.stdout.write(f"Rows: {summary['rows']}")
        self.stdout.write(f"Features: {summary['features']}")
        self.stdout.write(f"Classification target: {summary['classification_target']}")
        self.stdout.write(
            "Classification split: "
            f"{summary['classification_train_rows']} train / "
            f"{summary['classification_test_rows']} test"
        )
        self.stdout.write(
            f"Pass/Distinction rate: {summary['class_positive_rate']:.2%}"
        )
        self.stdout.write(f"Regression target: {summary['regression_target']}")
        self.stdout.write(
            "Regression split: "
            f"{summary['regression_train_rows']} train / "
            f"{summary['regression_test_rows']} test"
        )
        self.stdout.write(f"Learning percentage mean: {summary['target_mean']:.2f}%")
        self.stdout.write(
            "Learning percentage range: "
            f"{summary['target_min']:.2f}% - {summary['target_max']:.2f}%"
        )
        mlflow_status = summary["mlflow"]
        if mlflow_status.get("run_id"):
            self.stdout.write(
                self.style.SUCCESS(
                    f"MLflow preprocessing run: {mlflow_status['run_id']} "
                    f"({mlflow_status['tracking_uri']})"
                )
            )
        elif mlflow_status.get("error"):
            self.stdout.write(
                self.style.WARNING(
                    f"MLflow preprocessing tracking skipped with error: "
                    f"{mlflow_status['error']}"
                )
            )
        else:
            self.stdout.write(
                self.style.WARNING(
                    f"MLflow preprocessing tracking disabled: "
                    f"{mlflow_status.get('reason')}"
                )
            )
        postgres_sync = summary["postgres_sync"]
        if postgres_sync.get("error"):
            self.stdout.write(
                self.style.WARNING(
                    f"Postgres sync skipped with error: {postgres_sync['error']}"
                )
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    "Postgres sync: "
                    f"{len(postgres_sync['datasets'])} datasets, "
                    f"{len(postgres_sync['files'])} files"
                )
            )
        self.stdout.write(f"Preprocessed dir: {paths.preprocessed_dir}")
        self.stdout.write(f"Feature dir: {paths.feature_dir}")
        self.stdout.write(f"Plot dir: {paths.plot_dir}")
