from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import ExperimentConfig
from src.pipeline import parse_args


class ConfigDefaultTests(unittest.TestCase):
    def test_default_event_profile_uses_validation_supported_weather_adjustment(self) -> None:
        cfg = ExperimentConfig(
            train_path=Path("data/train.csv"),
            test_path=Path("data/test.csv"),
            output_dir=Path("outputs"),
            model_dir=Path("models"),
        )

        self.assertEqual(cfg.event_adjustment_profile, "score_rebound_fit_weather3_soft")

    def test_default_count_weight_matches_current_public_best(self) -> None:
        cfg = ExperimentConfig(
            train_path=Path("data/train.csv"),
            test_path=Path("data/test.csv"),
            output_dir=Path("outputs"),
            model_dir=Path("models"),
        )

        self.assertEqual(cfg.count_blend_weight, 0.44)

    def test_cli_default_count_weight_matches_current_public_best(self) -> None:
        with patch("sys.argv", ["run.py"]):
            args = parse_args()

        self.assertEqual(args.count_blend_weight, 0.44)


if __name__ == "__main__":
    unittest.main()
