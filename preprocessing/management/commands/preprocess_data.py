from django.core.management.base import BaseCommand

from preprocessing.pipeline import PreprocessingPaths, run_preprocessing


class Command(BaseCommand):
    help = "Preprocess OULAD raw CSV files and create train/test feature files."

    def add_arguments(self, parser):
        parser.add_argument("--raw-dir", default=None)
        parser.add_argument("--preprocessed-dir", default=None)
        parser.add_argument("--feature-dir", default=None)
        parser.add_argument("--test-size", type=float, default=0.2)
        parser.add_argument("--seed", type=int, default=42)

    def handle(self, *args, **options):
        paths = PreprocessingPaths.from_defaults(
            raw_dir=options["raw_dir"],
            preprocessed_dir=options["preprocessed_dir"],
            feature_dir=options["feature_dir"],
        )
        summary = run_preprocessing(
            paths=paths,
            test_size=options["test_size"],
            random_state=options["seed"],
        )

        self.stdout.write(self.style.SUCCESS("Preprocessing completed."))
        self.stdout.write(f"Rows: {summary['rows']}")
        self.stdout.write(f"Features: {summary['features']}")
        self.stdout.write(f"Train rows: {summary['train_rows']}")
        self.stdout.write(f"Test rows: {summary['test_rows']}")
        self.stdout.write(f"Target: {summary['target']}")
        self.stdout.write(f"Target mean: {summary['target_mean']:.2f}%")
        self.stdout.write(
            f"Target range: {summary['target_min']:.2f}% - {summary['target_max']:.2f}%"
        )
        postgres_sync = summary["postgres_sync"]
        self.stdout.write(
            self.style.SUCCESS(
                "Postgres sync: "
                f"{len(postgres_sync['datasets'])} datasets, "
                f"{len(postgres_sync['files'])} files"
            )
        )
        self.stdout.write(f"Preprocessed dir: {paths.preprocessed_dir}")
        self.stdout.write(f"Feature dir: {paths.feature_dir}")
