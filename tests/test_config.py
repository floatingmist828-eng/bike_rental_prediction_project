from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import ExperimentConfig
from src.pipeline import parse_args


class ConfigDefaultTests(unittest.TestCase):
    def test_default_event_profile_uses_late_2012_calendar_rebalance(self) -> None:
        cfg = ExperimentConfig(
            train_path=Path("data/train.csv"),
            test_path=Path("data/test.csv"),
            output_dir=Path("outputs"),
            model_dir=Path("models"),
        )

        self.assertEqual(cfg.event_adjustment_profile, "late_2012_calendar_rebalance")

    def test_default_count_weight_uses_late_2012_calendar_rebalance(self) -> None:
        cfg = ExperimentConfig(
            train_path=Path("data/train.csv"),
            test_path=Path("data/test.csv"),
            output_dir=Path("outputs"),
            model_dir=Path("models"),
        )

        self.assertAlmostEqual(cfg.count_blend_weight, 0.8860577001342274)

    def test_cli_default_count_weight_uses_late_2012_calendar_rebalance(self) -> None:
        with patch("sys.argv", ["run.py"]):
            args = parse_args()

        self.assertAlmostEqual(args.count_blend_weight, 0.8860577001342274)

    def test_cli_default_event_profile_uses_late_2012_calendar_rebalance(self) -> None:
        with patch("sys.argv", ["run.py"]):
            args = parse_args()

        self.assertEqual(args.event_adjustment_profile, "late_2012_calendar_rebalance")

    def test_cli_can_still_select_reproducible_2681_candidate(self) -> None:
        with patch(
            "sys.argv",
            [
                "run.py",
                "--count-blend-weight",
                "0.4",
                "--event-adjustment-profile",
                "score_rebound_fit_holiday_extended",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.count_blend_weight, 0.4)
        self.assertEqual(args.event_adjustment_profile, "score_rebound_fit_holiday_extended")


if __name__ == "__main__":
    unittest.main()
