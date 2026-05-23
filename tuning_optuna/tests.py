from django.test import SimpleTestCase

from .tuning import (
    normalize_model_names,
    target_column_for_task,
    training_params_config,
)


class TuningConfigurationTests(SimpleTestCase):
    def test_normalize_model_names_defaults_to_supported_models(self):
        models = normalize_model_names(None)

        self.assertIn("random_forest_classifier", models)
        self.assertIn("random_forest_regressor", models)

    def test_normalize_model_names_rejects_unknown_model(self):
        with self.assertRaises(ValueError):
            normalize_model_names(["unknown_model"])

    def test_target_column_for_task(self):
        self.assertEqual(target_column_for_task("classification"), "target_pass")
        self.assertEqual(target_column_for_task("regression"), "learning_percentage")

    def test_training_params_config_splits_model_params_by_task(self):
        config = training_params_config(
            {
                "random_forest_classifier": {
                    "best_params": {"n_estimators": 50, "max_depth": 4},
                    "task": "classification",
                },
                "random_forest_regressor": {
                    "best_params": {"n_estimators": 80, "max_depth": 6},
                    "task": "regression",
                },
            }
        )

        self.assertEqual(config["classification_model_params"]["n_estimators"], 50)
        self.assertEqual(config["regression_model_params"]["max_depth"], 6)
