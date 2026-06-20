from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pipeline import apply_event_adjustments, observed_public_mse


class EventProfileTests(unittest.TestCase):
    def test_stronger_profile_adjusts_only_sandy_rows(self) -> None:
        test_df = pd.DataFrame(
            {
                "dteday": [
                    "2012-10-29",
                    "2012-10-30",
                    "2012-10-30",
                    "2012-10-31",
                ],
                "hr": [0, 13, 19, 9],
            }
        )
        pred = np.array([100.0, 200.0, 300.0, 400.0])

        adjusted, meta = apply_event_adjustments(test_df, pred, profile="stronger")

        np.testing.assert_allclose(adjusted, np.array([20.0, 60.0, 135.0, 400.0]))
        self.assertEqual(meta["profile"], "stronger")
        self.assertEqual(meta["changed_rows"], 3)

    def test_empirical_storm_profile_uses_train_derived_core_factors(self) -> None:
        test_df = pd.DataFrame(
            {
                "dteday": [
                    "2012-10-29",
                    "2012-10-30",
                    "2012-10-30",
                    "2012-10-31",
                ],
                "hr": [0, 13, 19, 9],
            }
        )
        pred = np.array([100.0, 200.0, 300.0, 400.0])

        adjusted, meta = apply_event_adjustments(test_df, pred, profile="empirical_storm")

        np.testing.assert_allclose(adjusted, np.array([20.0, 50.0, 150.0, 400.0]))
        self.assertEqual(meta["profile"], "empirical_storm")
        self.assertEqual(meta["changed_rows"], 3)

    def test_strong_core_0p33_profile_is_local_strong_variant(self) -> None:
        test_df = pd.DataFrame(
            {
                "dteday": [
                    "2012-10-29",
                    "2012-10-30",
                    "2012-10-30",
                    "2012-10-31",
                ],
                "hr": [0, 13, 19, 9],
            }
        )
        pred = np.array([100.0, 200.0, 300.0, 400.0])

        adjusted, meta = apply_event_adjustments(test_df, pred, profile="strong_core_0p33")

        np.testing.assert_allclose(adjusted, np.array([25.0, 66.0, 150.0, 400.0]))
        self.assertEqual(meta["profile"], "strong_core_0p33")
        self.assertEqual(meta["changed_rows"], 3)

    def test_score_rebound_mid_raises_only_core_storm_rows_from_strong(self) -> None:
        test_df = pd.DataFrame(
            {
                "dteday": [
                    "2012-10-29",
                    "2012-10-30",
                    "2012-10-30",
                    "2012-10-31",
                ],
                "hr": [0, 13, 19, 9],
            }
        )
        pred = np.array([100.0, 200.0, 300.0, 400.0])

        adjusted, meta = apply_event_adjustments(test_df, pred, profile="score_rebound_mid")

        np.testing.assert_allclose(adjusted, np.array([30.0, 90.0, 150.0, 400.0]))
        self.assertEqual(meta["profile"], "score_rebound_mid")
        self.assertEqual(meta["changed_rows"], 3)

    def test_observed_public_mse_records_known_submission_score(self) -> None:
        self.assertEqual(observed_public_mse("raw_count_0p42764_event_strong"), 2886.37549)
        self.assertEqual(observed_public_mse("raw_count_0p42764_event_empirical_storm"), 2889.86619)
        self.assertIsNone(observed_public_mse("raw_count_0p42764_event_empirical_shutdown"))


if __name__ == "__main__":
    unittest.main()
