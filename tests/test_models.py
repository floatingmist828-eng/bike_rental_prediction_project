from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from src.config import TARGET_COL
from src.features import make_features, make_sample_weight
from src.models import ModelSpec, fit_model, make_model_specs
from src.utils import load_data


class RecordingEstimator:
    def __init__(self) -> None:
        self.fit_kwargs: dict[str, object] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs: object) -> "RecordingEstimator":
        self.fit_kwargs = kwargs
        return self


class ModelSpecTests(unittest.TestCase):
    def test_fit_model_uses_custom_sample_weight_parameter(self) -> None:
        estimator = RecordingEstimator()
        spec = ModelSpec("recording", estimator, "raw", "reg__sample_weight")
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
        y = np.array([1.0, 2.0, 3.0])
        sample_weight = np.array([1.0, 1.5, 2.0])

        fit_model(spec, X, y, sample_weight=sample_weight)

        np.testing.assert_allclose(estimator.fit_kwargs["reg__sample_weight"], sample_weight)

    def test_default_model_set_includes_trend_components(self) -> None:
        specs = make_model_specs("default", seed=42, n_jobs=1)
        names = {spec.name: spec for spec in specs}

        self.assertIn("ridge_log_trend_a1", names)
        self.assertIn("poisson_trend_a1", names)
        self.assertIn("tweedie_trend_a0p1", names)
        self.assertEqual(names["ridge_log_trend_a1"].sample_weight_fit_param, "reg__sample_weight")

    def test_trend_pipeline_can_be_serialized_after_fit(self) -> None:
        train_df, _ = load_data(Path("data/train.csv"), Path("data/test.csv"))
        small_train = train_df.head(96)
        [X_train], _ = make_features([small_train], small_train, "base")
        y_train = small_train[TARGET_COL].to_numpy(dtype=float)
        sample_weight = make_sample_weight(small_train)
        spec = next(s for s in make_model_specs("default", seed=42, n_jobs=1) if s.name == "ridge_log_trend_a1")

        fit_model(spec, X_train, y_train, sample_weight=sample_weight)

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trend.joblib"
            joblib.dump(spec.estimator, path)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
