from django.core.management.base import BaseCommand

from tuning_optuna.tuning import TuningPaths, tune_best_model


class Command(BaseCommand):
    help = "Tune candidate models with Optuna and select the highest-scoring one."

    def add_arguments(self, parser):
        parser.add_argument("--feature-dir", default=None)
        parser.add_argument("--tuning-dir", default=None)
        parser.add_argument("--models", default="xgboost,lightgbm")
        parser.add_argument("--n-trials", type=int, default=40)
        parser.add_argument("--cv-splits", type=int, default=5)
        parser.add_argument("--scoring", default="neg_root_mean_squared_error")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--timeout", type=int, default=None)

    def handle(self, *args, **options):
        paths = TuningPaths.from_defaults(
            feature_dir=options["feature_dir"],
            tuning_dir=options["tuning_dir"],
        )
        model_names = [
            name.strip()
            for name in options["models"].split(",")
            if name.strip()
        ]
        summary = tune_best_model(
            paths=paths,
            model_names=model_names,
            n_trials=options["n_trials"],
            cv_splits=options["cv_splits"],
            random_state=options["seed"],
            timeout=options["timeout"],
            scoring=options["scoring"],
        )

        self.stdout.write(self.style.SUCCESS("Optuna model selection completed."))
        for model_name, result in summary["models"].items():
            value = result["best_value"]
            display_metric = "rmse" if summary["scoring"] == "neg_root_mean_squared_error" else summary["scoring"]
            display_value = -value if summary["scoring"] == "neg_root_mean_squared_error" else value
            self.stdout.write(
                f"{model_name}: {display_metric}={display_value:.4f}"
            )
        best_display_metric = "rmse" if summary["scoring"] == "neg_root_mean_squared_error" else summary["scoring"]
        best_display_value = (
            -summary["best_value"]
            if summary["scoring"] == "neg_root_mean_squared_error"
            else summary["best_value"]
        )
        self.stdout.write(
            f"Best model: {summary['best_model']} "
            f"({best_display_metric}={best_display_value:.4f})"
        )
        postgres_sync = summary["postgres_sync"]
        self.stdout.write(
            self.style.SUCCESS(
                "Postgres sync: "
                f"{len(postgres_sync['artifacts'])} best-model tuning artifacts"
            )
        )
        self.stdout.write(f"Tuning dir: {paths.tuning_dir}")
