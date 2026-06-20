from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline import save_candidate_submissions


class CandidateSummaryTests(unittest.TestCase):
    def test_condition_adjustment_updates_validation_mse(self) -> None:
        test_df = pd.DataFrame(
            {
                "ID": [10, 11],
                "dteday": ["2012-10-02", "2012-10-02"],
                "hr": [12, 13],
                "weathersit": [3, 2],
            }
        )
        valid_df = pd.DataFrame(
            {
                "ID": [1, 2],
                "dteday": ["2012-05-14", "2012-05-14"],
                "hr": [12, 13],
                "weathersit": [3, 2],
            }
        )
        components = {
            "main_calibrated": np.array([100.0, 100.0]),
            "main_calibrated_0p4": np.array([100.0, 100.0]),
            "main_calibrated_1p0": np.array([100.0, 100.0]),
            "main_raw": np.array([100.0, 100.0]),
        }
        y_valid = np.array([94.0, 100.0])

        with tempfile.TemporaryDirectory() as tmp:
            rows = save_candidate_submissions(
                Path(tmp),
                test_df,
                valid_df,
                components,
                components,
                y_valid,
                default_count_weight=0.45,
                event_enabled=True,
            )

        rows_by_name = {str(row["candidate"]): row for row in rows}
        self.assertEqual(rows_by_name["main_raw"]["validation_mse"], 18.0)
        self.assertEqual(rows_by_name["main_raw_event_score_rebound_fit_weather3_soft"]["validation_mse"], 0.0)
        self.assertEqual(
            rows_by_name["main_raw_event_score_rebound_fit_weather3_soft"]["validation_adjusted_rows"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
