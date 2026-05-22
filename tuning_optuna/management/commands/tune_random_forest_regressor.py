from django.core.management.base import BaseCommand

from tuning_optuna.tuning import (
    DEFAULT_REGRESSION_SCORING,
    TuningPaths,
    tune_random_forest_regressor,
)


class Command(BaseCommand):
    help = "Tune RandomForestRegressor hyperparameters with Optuna."

    def add_arguments(self, parser):
        parser.add_argument("--feature-dir", default=None)
        parser.add_argument("--tuning-dir", default=None)
        parser.add_argument("--n-trials", type=int, default=40)
        parser.add_argument("--cv-splits", type=int, default=5)
        parser.add_argument("--scoring", default=DEFAULT_REGRESSION_SCORING)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--timeout", type=int, default=None)

    def handle(self, *args, **options):
        paths = TuningPaths.from_defaults(
            feature_dir=options["feature_dir"],
            tuning_dir=options["tuning_dir"],
        )
        summary = tune_random_forest_regressor(
            paths=paths,
            n_trials=options["n_trials"],
            cv_splits=options["cv_splits"],
            random_state=options["seed"],
            timeout=options["timeout"],
            scoring=options["scoring"],
        )

        self.stdout.write(
            self.style.SUCCESS("Optuna RandomForestRegressor tuning completed.")
        )
        display_metric = display_metric_name(summary["scoring"])
        display_value = display_metric_value(summary["scoring"], summary["best_value"])
        self.stdout.write(f"Best {display_metric}: {display_value:.4f}")
        self.stdout.write(f"Best params: {summary['best_params']}")
        postgres_sync = summary["postgres_sync"]
        self.stdout.write(
            self.style.SUCCESS(
                "Postgres sync: "
                f"{len(postgres_sync['artifacts'])} tuning artifacts"
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
