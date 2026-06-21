from __future__ import annotations

import unittest
from pathlib import Path

from src.config import ExperimentConfig


class ConfigDefaultTests(unittest.TestCase):
    def test_default_event_profile_uses_validation_supported_weather_adjustment(self) -> None:
        cfg = ExperimentConfig(
            train_path=Path("data/train.csv"),
            test_path=Path("data/test.csv"),
            output_dir=Path("outputs"),
            model_dir=Path("models"),
        )

        self.assertEqual(cfg.event_adjustment_profile, "score_rebound_fit_weather3_soft")


if __name__ == "__main__":
    unittest.main()
