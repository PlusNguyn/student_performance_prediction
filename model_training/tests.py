import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from .training import evaluate_classifier, evaluate_regressor, load_training_config


class DummyProbabilisticClassifier:
    def predict_proba(self, X):
        return np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.1, 0.9],
            ]
        )


class ModelEvaluationTests(SimpleTestCase):
    def test_evaluate_classifier_returns_core_metrics_and_auc(self):
        X_test = pd.DataFrame({"feature": [1, 2, 3, 4]})
        y_test = pd.Series([0, 1, 0, 1])
        predictions = np.array([0, 1, 0, 1])

        metrics = evaluate_classifier(
            DummyProbabilisticClassifier(),
            X_test,
            y_test,
            predictions,
        )

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_evaluate_regressor_returns_error_metrics(self):
        y_test = pd.Series([50.0, 70.0, 90.0])
        predictions = np.array([55.0, 65.0, 100.0])

        metrics = evaluate_regressor(y_test, predictions)

        self.assertAlmostEqual(metrics["mae"], 20 / 3)
        self.assertGreater(metrics["rmse"], 0)
        self.assertEqual(metrics["within_10_percent"], 1.0)

    def test_missing_training_config_returns_none(self):
        self.assertIsNone(load_training_config(None))
