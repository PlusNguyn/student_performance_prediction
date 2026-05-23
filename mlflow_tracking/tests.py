from django.test import SimpleTestCase, override_settings

from .registry import (
    _metric_lower_is_better,
    _pick_metric,
    _resolve_model_name,
    _resolve_task,
)


class RegistryHelperTests(SimpleTestCase):
    def test_resolve_model_name_adds_task_suffix(self):
        with override_settings(MLFLOW_REGISTERED_MODEL_NAME="student-model"):
            self.assertEqual(
                _resolve_model_name(None, "classification"),
                "student-model-classification",
            )
            self.assertEqual(
                _resolve_model_name("student-model-regression", "classification"),
                "student-model-classification",
            )

    def test_metric_direction_detects_error_metrics(self):
        self.assertTrue(_metric_lower_is_better("regression.rmse"))
        self.assertTrue(_metric_lower_is_better("validation_loss"))
        self.assertFalse(_metric_lower_is_better("classification.f1"))

    def test_pick_metric_uses_task_specific_priority(self):
        metrics = {
            "classification.accuracy": 0.8,
            "classification.f1": 0.7,
            "regression.rmse": 12.5,
        }

        classification_score = _pick_metric(metrics, None, "classification")
        regression_score = _pick_metric(metrics, None, "regression")

        self.assertEqual(classification_score["name"], "classification.f1")
        self.assertEqual(classification_score["value"], 0.7)
        self.assertEqual(regression_score["name"], "regression.rmse")
        self.assertEqual(regression_score["value"], 12.5)

    def test_unknown_task_is_rejected(self):
        with self.assertRaises(ValueError):
            _resolve_task("ranking")
