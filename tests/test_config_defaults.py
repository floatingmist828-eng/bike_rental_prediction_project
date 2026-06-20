from __future__ import annotations

import unittest
from pathlib import Path

from src.config import ExperimentConfig


class ConfigDefaultTests(unittest.TestCase):
    def test_default_submission_uses_current_validation_best_candidate(self) -> None:
        cfg = ExperimentConfig(
            train_path=Path("data/train.csv"),
            test_path=Path("data/test.csv"),
            output_dir=Path("outputs"),
            model_dir=Path("models"),
        )

        self.assertEqual(cfg.count_blend_weight, 1.075)
        self.assertEqual(cfg.event_adjustment_profile, "score_rebound_fit_weather3_overcount_late_hour")


if __name__ == "__main__":
    unittest.main()
