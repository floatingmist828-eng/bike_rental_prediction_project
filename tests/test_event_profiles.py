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

    def test_score_rebound_1p50_uses_parametric_rebound_factors(self) -> None:
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

        adjusted, meta = apply_event_adjustments(test_df, pred, profile="score_rebound_1p50")

        np.testing.assert_allclose(adjusted, np.array([32.5, 100.0, 150.0, 400.0]))
        self.assertEqual(meta["profile"], "score_rebound_1p50")
        self.assertEqual(meta["changed_rows"], 3)

    def test_score_rebound_fit_holiday_soft_adds_winter_holiday_factors(self) -> None:
        test_df = pd.DataFrame(
            {
                "dteday": [
                    "2012-10-29",
                    "2012-10-30",
                    "2012-11-22",
                    "2012-12-24",
                    "2012-12-25",
                    "2012-12-31",
                ],
                "hr": [0, 13, 12, 18, 12, 12],
            }
        )
        pred = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])

        adjusted, meta = apply_event_adjustments(test_df, pred, profile="score_rebound_fit_holiday_soft")

        np.testing.assert_allclose(adjusted, np.array([35.0, 110.0, 225.0, 320.0, 350.0, 600.0]))
        self.assertEqual(meta["profile"], "score_rebound_fit_holiday_soft")
        self.assertEqual(meta["changed_rows"], 5)

    def test_score_rebound_fit_holiday_strong_uses_larger_winter_holiday_cuts(self) -> None:
        test_df = pd.DataFrame(
            {
                "dteday": [
                    "2012-10-29",
                    "2012-10-30",
                    "2012-11-22",
                    "2012-12-24",
                    "2012-12-25",
                    "2012-12-31",
                ],
                "hr": [0, 13, 12, 18, 12, 12],
            }
        )
        pred = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])

        adjusted, meta = apply_event_adjustments(test_df, pred, profile="score_rebound_fit_holiday_strong")

        np.testing.assert_allclose(adjusted, np.array([35.0, 110.0, 195.0, 280.0, 300.0, 600.0]))
        self.assertEqual(meta["profile"], "score_rebound_fit_holiday_strong")
        self.assertEqual(meta["changed_rows"], 5)

    def test_score_rebound_fit_holiday_extended_covers_adjacent_low_demand_days(self) -> None:
        test_df = pd.DataFrame(
            {
                "dteday": [
                    "2012-11-22",
                    "2012-11-23",
                    "2012-12-24",
                    "2012-12-25",
                    "2012-12-26",
                    "2012-12-31",
                ],
                "hr": [12, 12, 18, 12, 12, 12],
            }
        )
        pred = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])

        adjusted, meta = apply_event_adjustments(test_df, pred, profile="score_rebound_fit_holiday_extended")

        np.testing.assert_allclose(adjusted, np.array([65.0, 170.0, 210.0, 240.0, 400.0, 510.0]))
        self.assertEqual(meta["profile"], "score_rebound_fit_holiday_extended")
        self.assertEqual(meta["changed_rows"], 6)

    def test_score_rebound_fit_weather3_soft_adjusts_rain_without_overriding_sandy(self) -> None:
        test_df = pd.DataFrame(
            {
                "dteday": [
                    "2012-10-02",
                    "2012-10-29",
                    "2012-10-30",
                    "2012-10-02",
                ],
                "hr": [12, 12, 13, 13],
                "weathersit": [3, 3, 3, 2],
            }
        )
        pred = np.array([100.0, 200.0, 300.0, 400.0])

        adjusted, meta = apply_event_adjustments(test_df, pred, profile="score_rebound_fit_weather3_soft")

        np.testing.assert_allclose(adjusted, np.array([94.0, 70.0, 165.0, 400.0]))
        self.assertEqual(meta["profile"], "score_rebound_fit_weather3_soft")
        self.assertEqual(meta["changed_rows"], 3)

    def test_late_2012_calendar_rebalance_applies_months_and_overrides_event_dates(self) -> None:
        test_df = pd.DataFrame(
            {
                "dteday": [
                    "2012-08-08",
                    "2012-12-30",
                    "2012-12-31",
                    "2012-10-30",
                ],
                "mnth": [8, 12, 12, 10],
                "hr": [12, 12, 12, 20],
                "weathersit": [1, 1, 1, 1],
            }
        )
        pred = np.array([100.0, 100.0, 100.0, 100.0])

        adjusted, meta = apply_event_adjustments(test_df, pred, profile="late_2012_calendar_rebalance")

        np.testing.assert_allclose(
            adjusted,
            np.array([88.38223041755854, 92.20084335570969, 58.08801677491103, 25.56039193079757]),
        )
        self.assertEqual(meta["profile"], "late_2012_calendar_rebalance")
        self.assertEqual(meta["changed_rows"], 4)

    def test_workday_bad_weather_hour_rebalance_uses_group_factors_and_event_overrides(self) -> None:
        test_df = pd.DataFrame(
            {
                "dteday": [
                    "2012-08-08",
                    "2012-08-08",
                    "2012-08-05",
                    "2012-12-18",
                    "2012-12-31",
                ],
                "mnth": [8, 8, 8, 12, 12],
                "workingday": [1, 1, 0, 1, 1],
                "hr": [12, 12, 0, 21, 21],
                "weathersit": [2, 1, 1, 2, 2],
            }
        )
        pred = np.array([100.0, 100.0, 100.0, 100.0, 100.0])

        adjusted, meta = apply_event_adjustments(
            test_df,
            pred,
            profile="late_2012_workday_bad_weather_hour_rebalance",
        )

        np.testing.assert_allclose(
            adjusted,
            np.array(
                [
                    88.49562414573704,
                    91.60678799509973,
                    101.99953632369396,
                    93.45334059709067,
                    58.19195451942899,
                ]
            ),
        )
        self.assertEqual(meta["profile"], "late_2012_workday_bad_weather_hour_rebalance")
        self.assertEqual(meta["changed_rows"], 5)

    def test_observed_public_mse_records_known_submission_score(self) -> None:
        self.assertEqual(observed_public_mse("raw_count_0p44000_event_score_rebound_fit_weather3_soft"), 2841.38813)
        self.assertEqual(observed_public_mse("raw_count_0p45000_event_score_rebound_fit_weather3_soft"), 2843.40079)
        self.assertEqual(observed_public_mse("legacy_raw_count_0p42764_event_strong"), 2886.37549)
        self.assertEqual(observed_public_mse("legacy_raw_count_0p42764_event_empirical_storm"), 2889.86619)
        self.assertIsNone(observed_public_mse("raw_count_0p42764_event_strong"))
        self.assertIsNone(observed_public_mse("raw_count_0p42764_event_empirical_shutdown"))


if __name__ == "__main__":
    unittest.main()
