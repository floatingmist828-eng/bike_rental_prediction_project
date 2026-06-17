from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


TargetTransform = Literal["raw", "log1p"]
CalibrationMode = Literal["none", "scale", "affine"]


@dataclass
class ModelSpec:
    name: str
    estimator: object
    target_transform: TargetTransform = "raw"


@dataclass(frozen=True)
class PredictionCalibrator:
    mode: CalibrationMode = "none"
    slope: float = 1.0
    intercept: float = 0.0
    strength: float = 1.0

    def apply(self, pred: np.ndarray) -> np.ndarray:
        values = np.asarray(pred, dtype=float)
        if self.mode == "none":
            calibrated = values
        elif self.mode in {"scale", "affine"}:
            full_correction = self.slope * values + self.intercept
            calibrated = values + self.strength * (full_correction - values)
        else:
            raise ValueError(f"Unknown calibration mode: {self.mode}")
        calibrated = np.nan_to_num(calibrated, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(calibrated, 0.0, None)


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def make_model_specs(model_set: str, seed: int, n_jobs: int) -> list[ModelSpec]:
    """Create deterministic model configurations.

    default: LightGBM + XGBoost ensemble if available, with sklearn fallback.
    quick: smaller models for fast smoke tests.
    sklearn: only sklearn models.
    """
    specs: list[ModelSpec] = []
    n_jobs = int(n_jobs)

    if model_set not in {"default", "quick", "sklearn"}:
        raise ValueError(f"Unknown model_set: {model_set}")

    if model_set in {"default", "quick"} and _has_module("lightgbm"):
        import lightgbm as lgb

        if model_set == "quick":
            specs.extend(
                [
                    ModelSpec(
                        "lgbm_raw_quick",
                        lgb.LGBMRegressor(
                            objective="regression_l2",
                            n_estimators=350,
                            learning_rate=0.045,
                            num_leaves=31,
                            min_child_samples=15,
                            subsample=0.85,
                            subsample_freq=1,
                            colsample_bytree=0.85,
                            reg_alpha=0.05,
                            reg_lambda=1.0,
                            random_state=seed,
                            n_jobs=n_jobs,
                            verbose=-1,
                        ),
                        "raw",
                    ),
                    ModelSpec(
                        "lgbm_log_quick",
                        lgb.LGBMRegressor(
                            objective="regression_l2",
                            n_estimators=350,
                            learning_rate=0.045,
                            num_leaves=31,
                            min_child_samples=15,
                            subsample=0.85,
                            subsample_freq=1,
                            colsample_bytree=0.90,
                            reg_alpha=0.05,
                            reg_lambda=1.0,
                            random_state=seed + 1,
                            n_jobs=n_jobs,
                            verbose=-1,
                        ),
                        "log1p",
                    ),
                ]
            )
        else:
            specs.extend(
                [
                    ModelSpec(
                        "lgbm_tweedie_count_a",
                        lgb.LGBMRegressor(
                            objective="tweedie",
                            n_estimators=1400,
                            learning_rate=0.025,
                            num_leaves=15,
                            min_child_samples=5,
                            subsample=0.80,
                            subsample_freq=1,
                            colsample_bytree=0.90,
                            reg_alpha=0.05,
                            reg_lambda=0.50,
                            tweedie_variance_power=1.3,
                            random_state=seed + 4980,
                            n_jobs=n_jobs,
                            verbose=-1,
                        ),
                        "raw",
                    ),
                    ModelSpec(
                        "lgbm_poisson_count_a",
                        lgb.LGBMRegressor(
                            objective="poisson",
                            n_estimators=600,
                            learning_rate=0.080,
                            num_leaves=23,
                            min_child_samples=40,
                            subsample=0.90,
                            subsample_freq=1,
                            colsample_bytree=0.90,
                            reg_alpha=0.05,
                            reg_lambda=0.50,
                            random_state=seed + 4988,
                            n_jobs=n_jobs,
                            verbose=-1,
                        ),
                        "raw",
                    ),
                    ModelSpec(
                        "lgbm_poisson_count_b",
                        lgb.LGBMRegressor(
                            objective="poisson",
                            n_estimators=1200,
                            learning_rate=0.035,
                            num_leaves=23,
                            min_child_samples=40,
                            subsample=0.80,
                            subsample_freq=1,
                            colsample_bytree=0.90,
                            reg_alpha=0.05,
                            reg_lambda=0.50,
                            random_state=seed + 4994,
                            n_jobs=n_jobs,
                            verbose=-1,
                        ),
                        "raw",
                    ),
                    ModelSpec(
                        "lgbm_tweedie_count_b",
                        lgb.LGBMRegressor(
                            objective="tweedie",
                            n_estimators=1400,
                            learning_rate=0.025,
                            num_leaves=15,
                            min_child_samples=40,
                            subsample=0.80,
                            subsample_freq=1,
                            colsample_bytree=0.90,
                            reg_alpha=0.05,
                            reg_lambda=1.00,
                            tweedie_variance_power=1.3,
                            random_state=seed + 5060,
                            n_jobs=n_jobs,
                            verbose=-1,
                        ),
                        "raw",
                    ),
                    ModelSpec(
                        "lgbm_tweedie_count_c",
                        lgb.LGBMRegressor(
                            objective="tweedie",
                            n_estimators=1400,
                            learning_rate=0.025,
                            num_leaves=15,
                            min_child_samples=40,
                            subsample=0.90,
                            subsample_freq=1,
                            colsample_bytree=0.90,
                            reg_alpha=0.05,
                            reg_lambda=2.00,
                            tweedie_variance_power=1.5,
                            random_state=seed + 5080,
                            n_jobs=n_jobs,
                            verbose=-1,
                        ),
                        "raw",
                    ),
                    ModelSpec(
                        "lgbm_tweedie_count_d",
                        lgb.LGBMRegressor(
                            objective="tweedie",
                            n_estimators=1200,
                            learning_rate=0.035,
                            num_leaves=15,
                            min_child_samples=80,
                            subsample=0.80,
                            subsample_freq=1,
                            colsample_bytree=0.90,
                            reg_alpha=0.05,
                            reg_lambda=2.00,
                            tweedie_variance_power=1.3,
                            random_state=seed + 5069,
                            n_jobs=n_jobs,
                            verbose=-1,
                        ),
                        "raw",
                    ),
                    ModelSpec(
                        "lgbm_poisson_count_c",
                        lgb.LGBMRegressor(
                            objective="poisson",
                            n_estimators=1000,
                            learning_rate=0.045,
                            num_leaves=15,
                            min_child_samples=40,
                            subsample=0.80,
                            subsample_freq=1,
                            colsample_bytree=0.90,
                            reg_alpha=0.05,
                            reg_lambda=2.00,
                            random_state=seed + 5042,
                            n_jobs=n_jobs,
                            verbose=-1,
                        ),
                        "raw",
                    ),
                    ModelSpec(
                        "lgbm_tweedie_count_e",
                        lgb.LGBMRegressor(
                            objective="tweedie",
                            n_estimators=1000,
                            learning_rate=0.045,
                            num_leaves=23,
                            min_child_samples=40,
                            subsample=0.80,
                            subsample_freq=1,
                            colsample_bytree=0.90,
                            reg_alpha=0.05,
                            reg_lambda=2.00,
                            tweedie_variance_power=1.1,
                            random_state=seed + 4996,
                            n_jobs=n_jobs,
                            verbose=-1,
                        ),
                        "raw",
                    ),
                ]
            )

    if model_set in {"default", "quick"} and _has_module("xgboost"):
        from xgboost import XGBRegressor

        if model_set == "quick":
            specs.append(
                ModelSpec(
                    "xgb_raw_quick",
                    XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=350,
                        learning_rate=0.050,
                        max_depth=5,
                        min_child_weight=3.0,
                        subsample=0.90,
                        colsample_bytree=0.90,
                        reg_lambda=2.0,
                        reg_alpha=0.05,
                        random_state=seed + 3,
                        n_jobs=n_jobs,
                        tree_method="hist",
                        eval_metric="rmse",
                    ),
                    "raw",
                )
            )
        else:
            specs.extend(
                [
                    ModelSpec(
                        "xgb_raw_horizon",
                        XGBRegressor(
                            objective="reg:squarederror",
                            n_estimators=800,
                            learning_rate=0.050,
                            max_depth=4,
                            min_child_weight=6.0,
                            subsample=0.90,
                            colsample_bytree=0.90,
                            reg_lambda=2.0,
                            reg_alpha=0.05,
                            random_state=seed + 64,
                            n_jobs=n_jobs,
                            tree_method="hist",
                            eval_metric="rmse",
                        ),
                        "raw",
                    ),
                    ModelSpec(
                        "xgb_log",
                        XGBRegressor(
                            objective="reg:squarederror",
                            n_estimators=350,
                            learning_rate=0.050,
                            max_depth=5,
                            min_child_weight=2.0,
                            subsample=0.90,
                            colsample_bytree=0.90,
                            reg_lambda=1.5,
                            reg_alpha=0.02,
                            random_state=seed + 4,
                            n_jobs=n_jobs,
                            tree_method="hist",
                            eval_metric="rmse",
                        ),
                        "log1p",
                    ),
                ]
            )

    if model_set == "sklearn" or not specs:
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

        specs.extend(
            [
                ModelSpec(
                    "hist_gbdt_raw",
                    HistGradientBoostingRegressor(
                        max_iter=250 if model_set == "quick" else 400,
                        learning_rate=0.045,
                        max_leaf_nodes=31,
                        min_samples_leaf=15,
                        l2_regularization=0.10,
                        random_state=seed + 10,
                    ),
                    "raw",
                ),
                ModelSpec(
                    "extra_trees_raw",
                    ExtraTreesRegressor(
                        n_estimators=200 if model_set == "quick" else 350,
                        max_features=0.80,
                        min_samples_leaf=1,
                        random_state=seed + 11,
                        n_jobs=n_jobs,
                    ),
                    "raw",
                ),
            ]
        )

    return specs


def fit_model(spec: ModelSpec, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> object:
    y_fit = np.log1p(y) if spec.target_transform == "log1p" else y
    try:
        spec.estimator.fit(X, y_fit, sample_weight=sample_weight)
    except TypeError:
        spec.estimator.fit(X, y_fit)
    return spec.estimator


def predict_model(spec: ModelSpec, X: pd.DataFrame) -> np.ndarray:
    pred = np.asarray(spec.estimator.predict(X), dtype=float)
    if spec.target_transform == "log1p":
        pred = np.expm1(pred)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(pred, 0.0, None)


def optimize_ensemble_weights(
    prediction_map: dict[str, np.ndarray],
    y_true: np.ndarray,
    seed: int,
    calibration_mode: CalibrationMode = "none",
    calibration_strength: float = 0.0,
) -> dict[str, float]:
    names = list(prediction_map.keys())
    if not names:
        raise ValueError("No predictions supplied for ensemble weighting.")
    if len(names) == 1:
        return {names[0]: 1.0}

    P = np.vstack([prediction_map[name] for name in names]).T

    try:
        from scipy.optimize import minimize

        def objective(weights: np.ndarray) -> float:
            pred = P @ weights
            if calibration_mode != "none" and calibration_strength > 0.0:
                calibrator = fit_prediction_calibrator(pred, y_true, calibration_mode, calibration_strength)
                pred = calibrator.apply(pred)
            return float(mean_squared_error(y_true, pred))

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * len(names)
        starts = [np.ones(len(names), dtype=float) / len(names)]
        starts.extend(np.eye(len(names), dtype=float))
        rng = np.random.default_rng(seed)
        starts.extend(rng.dirichlet(np.ones(len(names)), size=20))
        best_weights = starts[0]
        best_mse = objective(best_weights)
        for x0 in starts:
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000},
            )
            if result.success:
                weights = np.clip(result.x, 0.0, 1.0)
                weights = weights / weights.sum()
                mse = objective(weights)
                if mse < best_mse:
                    best_mse = mse
                    best_weights = weights
        return {name: float(best_weights[i]) for i, name in enumerate(names)}
    except Exception:
        pass

    rng = np.random.default_rng(seed)
    best_weights = np.ones(len(names), dtype=float) / len(names)
    best_pred = P @ best_weights
    if calibration_mode != "none" and calibration_strength > 0.0:
        best_pred = fit_prediction_calibrator(best_pred, y_true, calibration_mode, calibration_strength).apply(best_pred)
    best_mse = float(mean_squared_error(y_true, best_pred))
    for _ in range(10000):
        weights = rng.dirichlet(np.ones(len(names)))
        pred = P @ weights
        if calibration_mode != "none" and calibration_strength > 0.0:
            pred = fit_prediction_calibrator(pred, y_true, calibration_mode, calibration_strength).apply(pred)
        mse = float(mean_squared_error(y_true, pred))
        if mse < best_mse:
            best_mse = mse
            best_weights = weights
    return {name: float(best_weights[i]) for i, name in enumerate(names)}


def weighted_average_predictions(prediction_map: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    pred = None
    for name, values in prediction_map.items():
        weight = float(weights.get(name, 0.0))
        if pred is None:
            pred = weight * values
        else:
            pred = pred + weight * values
    if pred is None:
        raise ValueError("No predictions available.")
    return np.clip(pred, 0.0, None)


def fit_prediction_calibrator(
    pred_valid: np.ndarray,
    y_valid: np.ndarray,
    mode: CalibrationMode = "affine",
    strength: float = 1.0,
) -> PredictionCalibrator:
    """Fit a simple validation-based correction for future-horizon bias."""
    pred = np.asarray(pred_valid, dtype=float)
    target = np.asarray(y_valid, dtype=float)
    strength = float(np.clip(strength, 0.0, 1.0))
    if mode == "none":
        return PredictionCalibrator(mode="none", strength=0.0)
    if pred.shape != target.shape:
        raise ValueError("pred_valid and y_valid must have the same shape.")

    if mode == "scale":
        denom = float(np.dot(pred, pred))
        slope = 1.0 if denom <= 0.0 else float(np.dot(pred, target) / denom)
        return PredictionCalibrator(
            mode=mode,
            slope=float(np.clip(slope, 0.85, 1.25)),
            intercept=0.0,
            strength=strength,
        )

    if mode == "affine":
        design = np.vstack([pred, np.ones_like(pred)]).T
        slope, intercept = np.linalg.lstsq(design, target, rcond=None)[0]
        return PredictionCalibrator(
            mode=mode,
            slope=float(np.clip(slope, 0.85, 1.25)),
            intercept=float(np.clip(intercept, -30.0, 30.0)),
            strength=strength,
        )

    raise ValueError(f"Unknown calibration mode: {mode}")
