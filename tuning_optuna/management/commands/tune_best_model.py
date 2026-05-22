from django.core.management.base import BaseCommand

from tuning_optuna.tuning import (
    DEFAULT_CLASSIFICATION_SCORING,
    DEFAULT_REGRESSION_SCORING,
    SUPPORTED_MODELS,
    TuningPaths,
    tune_best_model,
)


class Command(BaseCommand):
    help = "Tune RandomForest classification and regression models with Optuna."

    def add_arguments(self, parser):
        parser.add_argument("--feature-dir", default=None)
        parser.add_argument("--tuning-dir", default=None)
        parser.add_argument("--models", default=",".join(SUPPORTED_MODELS))
        parser.add_argument("--n-trials", type=int, default=40)
        parser.add_argument("--cv-splits", type=int, default=5)
        parser.add_argument(
            "--classification-scoring",
            default=DEFAULT_CLASSIFICATION_SCORING,
        )
        parser.add_argument(
            "--regression-scoring",
            default=DEFAULT_REGRESSION_SCORING,
        )
        parser.add_argument(
            "--scoring",
            default=None,
            help="Deprecated alias for --regression-scoring.",
        )
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
            classification_scoring=options["classification_scoring"],
            regression_scoring=options["regression_scoring"],
        )

        self.stdout.write(self.style.SUCCESS("Optuna RandomForest tuning completed."))
        for model_name, result in summary["models"].items():
            value = result["best_value"]
            display_metric = display_metric_name(result["scoring"])
            display_value = display_metric_value(result["scoring"], value)
            self.stdout.write(
                f"{model_name}: {display_metric}={display_value:.4f}"
            )
        self.stdout.write(
            "Training params: "
            f"{summary['paths']['best_params']}"
        )
        postgres_sync = summary["postgres_sync"]
        self.stdout.write(
            self.style.SUCCESS(
                "Postgres sync: "
                f"{len(postgres_sync['artifacts'])} best-model tuning artifacts"
            )
        )
        self.stdout.write(f"Tuning dir: {paths.tuning_dir}")


def display_metric_name(scoring: str) -> str:
    if scoring == "neg_root_mean_squared_error":
        return "rmse"
    return scoring


def display_metric_value(scoring: str, value: float) -> float:
    if scoring == "neg_root_mean_squared_error":
        return -value
    return value
