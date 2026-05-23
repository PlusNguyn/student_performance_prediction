import pandas as pd
from django.test import SimpleTestCase

from .pipeline import (
    TARGET_REGRESSION_COLUMN,
    add_features,
    build_assessment_features,
    prepare_features,
)


class FeatureEngineeringTests(SimpleTestCase):
    def test_registered_flag_marks_active_students_as_one(self):
        frame = pd.DataFrame(
            {
                "date_registration": [-12, -5],
                "date_unregistration": [-1, 45],
                "imd_band": ["20-30%", "40-50%"],
                "age_band": ["0-35", "35-55"],
                "highest_education": ["A Level or Equivalent", "HE Qualification"],
                "gender": ["M", "F"],
                "disability": ["N", "Y"],
                "total_clicks": [120, 10],
                "studied_credits": [60, 90],
                "total_score": [160, 45],
                "num_assessments": [2, 1],
            }
        )

        result = add_features(frame)

        self.assertEqual(result.loc[0, "registered_flag"], 1)
        self.assertEqual(result.loc[1, "registered_flag"], 0)
        self.assertGreater(result.loc[0, "clicks_per_credit"], 0)
        self.assertIn(result.loc[1, "credit_load_category"], {"medium", "heavy"})

    def test_weighted_assessment_features_are_clipped_to_percentage_range(self):
        student_assessment = pd.DataFrame(
            {
                "id_student": [900001, 900001],
                "id_assessment": [1, 2],
                "date_submitted": [10, 20],
                "is_banked": [0, 1],
                "score": [80, 140],
            }
        )
        assessments = pd.DataFrame(
            {
                "code_module": ["DDD", "DDD"],
                "code_presentation": ["2026B", "2026B"],
                "id_assessment": [1, 2],
                "weight": [40, 60],
            }
        )

        result = build_assessment_features(student_assessment, assessments)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "num_assessments"], 2)
        self.assertEqual(result.loc[0, "late_submissions"], 1)
        self.assertEqual(result.loc[0, TARGET_REGRESSION_COLUMN], 100)

    def test_prepare_features_excludes_targets_and_student_identifier(self):
        modeling_dataset = pd.DataFrame(
            {
                "id_student": [900001, 900002],
                "final_result": ["Pass", "Fail"],
                "target_pass": [1, 0],
                "learning_percentage": [82.5, 41.0],
                "code_module": ["DDD", "EEE"],
                "code_presentation": ["2026B", "2026B"],
                "region": ["East", "West"],
                "credit_load_category": ["light", "medium"],
                "gender_m": [1, 0],
                "edu_level": [2, 3],
                "imd_num": [25.0, 45.0],
                "age_num": [20.0, 40.0],
                "num_of_prev_attempts": [0, 1],
                "studied_credits": [60, 90],
                "disability_flag": [0, 1],
                "total_score": [160.0, 45.0],
                "avg_score": [80.0, 45.0],
                "num_assessments": [2, 1],
                "total_clicks": [120, 10],
                "registration_lead_days": [12, 5],
                "registered_flag": [1, 0],
                "registration_duration": [-1, 50],
                "clicks_per_credit": [2.0, 0.1],
                "score_per_assess": [80.0, 45.0],
                "clicks_per_day": [0.0, 0.2],
                "clicks_per_week": [0.0, 1.4],
                "engagement_intensity": [1.0, -1.0],
                "assessment_density": [0.03, 0.01],
                "score_per_credit": [2.7, 0.5],
                "activity_efficiency": [40.0, 450.0],
                "low_activity_flag": [0, 1],
                "late_registration_flag": [0, 1],
                "high_click_but_low_score_flag": [0, 0],
            }
        )

        encoded_dataset, X, y_class, y_reg, medians, category_levels = prepare_features(
            modeling_dataset
        )

        self.assertNotIn("id_student", X.columns)
        self.assertNotIn("target_pass", X.columns)
        self.assertNotIn("learning_percentage", X.columns)
        self.assertEqual(y_class.tolist(), [1, 0])
        self.assertEqual(y_reg.tolist(), [82.5, 41.0])
        self.assertIn("code_module", category_levels)
        self.assertTrue(medians)
        self.assertEqual(encoded_dataset.shape[0], 2)
