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

    def test_observed_public_mse_records_known_submission_score(self) -> None:
        self.assertEqual(observed_public_mse("raw_count_0p42764_event_strong"), 2886.37549)
        self.assertIsNone(observed_public_mse("raw_count_0p42764_event_stronger"))


if __name__ == "__main__":
    unittest.main()
