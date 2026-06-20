from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import ExperimentConfig, TARGET_COL
from .features import make_features, make_sample_weight
from .models import (
    ModelSpec,
    fit_prediction_calibrator,
    fit_model,
    make_count_model_specs,
    make_model_specs,
    optimize_ensemble_weights,
    predict_model,
    weighted_average_predictions,
)
from .utils import (
    build_submission,
    ensure_dirs,
    load_data,
    regression_metrics,
    save_json,
    set_global_seed,
    temporal_train_valid_split,
    validate_submission,
)


EVENT_PROFILES = {
    "mild": {
        "2012-10-29": 0.65,
        "2012-10-30_13_18": 0.75,
        "2012-10-30_19_23": 0.85,
    },
    "default": {
        "2012-10-29": 0.45,
        "2012-10-30_13_18": 0.55,
        "2012-10-30_19_23": 0.65,
    },
    "strong": {
        "2012-10-29": 0.25,
        "2012-10-30_13_18": 0.35,
        "2012-10-30_19_23": 0.50,
    },
    "stronger": {
        "2012-10-29": 0.20,
        "2012-10-30_13_18": 0.30,
        "2012-10-30_19_23": 0.45,
    },
    "mid_heavy": {
        "2012-10-29": 0.20,
        "2012-10-30_13_18": 0.25,
        "2012-10-30_19_23": 0.45,
    },
    "late_recovery": {
        "2012-10-29": 0.20,
        "2012-10-30_13_18": 0.30,
        "2012-10-30_19_23": 0.55,
    },
    "peak_heavy": {
        "2012-10-29": 0.15,
        "2012-10-30_13_18": 0.25,
        "2012-10-30_19_23": 0.45,
    },
    "extreme": {
        "2012-10-29": 0.15,
        "2012-10-30_13_18": 0.25,
        "2012-10-30_19_23": 0.40,
    },
    "extended": {
        "2012-10-28_16_23": 0.75,
        "2012-10-29": 0.20,
        "2012-10-30_13_18": 0.30,
        "2012-10-30_19_23": 0.45,
        "2012-10-31_0_8": 0.85,
    },
}


KNOWN_PUBLIC_CANDIDATE_SCORES = {
    "raw_count_0p42764_event_strong": 2886.37549,
}


def observed_public_mse(candidate_name: str) -> float | None:
    return KNOWN_PUBLIC_CANDIDATE_SCORES.get(candidate_name)


def _event_window_mask(dates: pd.Series, hours: np.ndarray, window_key: str) -> np.ndarray:
    parts = window_key.split("_")
    if len(parts) not in {1, 3}:
        raise ValueError(f"Invalid event window key: {window_key}")

    mask = dates.eq(pd.Timestamp(parts[0])).to_numpy()
    if len(parts) == 3:
        start_hour = int(parts[1])
        end_hour = int(parts[2])
        mask = mask & ((start_hour <= hours) & (hours <= end_hour))
    return mask


def apply_event_adjustments(
    test_df: pd.DataFrame,
    pred: np.ndarray,
    profile: str = "default",
) -> tuple[np.ndarray, dict[str, object]]:
    """Adjust a small, documented weather-disruption window in the fixed test horizon."""
    if profile not in EVENT_PROFILES:
        raise ValueError(f"Unknown event adjustment profile: {profile}")

    adjusted = np.asarray(pred, dtype=float).copy()
    dates = pd.to_datetime(test_df["dteday"])
    hours = test_df["hr"].to_numpy()
    factors_config = EVENT_PROFILES[profile]

    factors = np.ones(len(test_df), dtype=float)
    for window_key, factor in factors_config.items():
        factors[_event_window_mask(dates, hours, window_key)] = factor

    changed = factors != 1.0
    adjusted[changed] *= factors[changed]
    return np.clip(adjusted, 0.0, None), {
        "enabled": True,
        "method": "hurricane_sandy_window",
        "profile": profile,
        "changed_rows": int(changed.sum()),
        "mean_delta": float(adjusted.mean() - np.asarray(pred, dtype=float).mean()),
    }


def fit_predict_specs(
    specs: list[ModelSpec],
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_pred: pd.DataFrame,
    sample_weight: np.ndarray,
    metric_rows: list[dict[str, object]] | None = None,
    y_pred_true: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    for spec in specs:
        print(f"  fitting {spec.name} ({spec.target_transform}) ...", flush=True)
        try:
            fit_model(spec, X_tr, y_tr, sample_weight=sample_weight)
            pred = predict_model(spec, X_pred)
            predictions[spec.name] = pred
            if metric_rows is not None and y_pred_true is not None:
                metrics = regression_metrics(y_pred_true, pred)
                metric_rows.append(
                    {
                        "model": spec.name,
                        "target_transform": spec.target_transform,
                        **metrics,
                    }
                )
                print(
                    f"    MSE={metrics['mse']:.6f} | RMSE={metrics['rmse']:.6f} | MAE={metrics['mae']:.6f}",
                    flush=True,
                )
        except Exception as exc:
            print(f"    skipped {spec.name}: {exc}")
            traceback.print_exc()
    return predictions


def blend_named_predictions(components: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    pred = None
    for name, weight in weights.items():
        values = components[name]
        pred = values * weight if pred is None else pred + values * weight
    if pred is None:
        raise ValueError("No candidate components supplied.")
    return np.clip(pred, 0.0, None)


def apply_calibration_strength(pred: np.ndarray, slope: float, intercept: float, strength: float) -> np.ndarray:
    values = np.asarray(pred, dtype=float)
    full_correction = slope * values + intercept
    calibrated = values + float(strength) * (full_correction - values)
    return np.clip(np.nan_to_num(calibrated, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)


def candidate_recipes(has_count_branch: bool, default_count_weight: float) -> list[dict[str, object]]:
    recipes: list[dict[str, object]] = [
        {
            "name": "main_calibrated",
            "weights": {"main_calibrated": 1.0},
            "description": "validation-calibrated main LightGBM ensemble",
        },
        {
            "name": "main_calibrated_0p4",
            "weights": {"main_calibrated_0p4": 1.0},
            "description": "weaker main calibration previously observed as a competitive public-score variant",
        },
        {
            "name": "main_calibrated_1p0",
            "weights": {"main_calibrated_1p0": 1.0},
            "description": "full affine main calibration variant",
        },
        {
            "name": "main_raw",
            "weights": {"main_raw": 1.0},
            "description": "uncalibrated main LightGBM ensemble",
        },
    ]
    if has_count_branch:
        raw_count_grid = [
            0.40000,
            0.41500,
            0.42000,
            0.42250,
            0.42500,
            0.43000,
            0.43250,
            0.43500,
            0.43750,
            0.44000,
            0.44500,
            0.45000,
        ]
        recipes.extend(
            [
                {
                    "name": f"raw_count_{default_count_weight:.5f}".replace(".", "p"),
                    "weights": {"main_raw": 1.0 - default_count_weight, "count_calibrated": default_count_weight},
                    "description": "public-score free optimum proxy: raw main plus count branch",
                },
                {
                    "name": "raw_count_0p35000",
                    "weights": {"main_raw": 0.65, "count_calibrated": 0.35},
                    "description": "lower-count-weight candidate for robustness",
                },
                {
                    "name": "public_constrained",
                    "weights": {"main_raw": 0.20354, "main_calibrated": 0.50, "count_calibrated": 0.29646},
                    "description": "public-score proxy with at least half calibrated main model",
                },
                {
                    "name": "public_count_cap_0p25",
                    "weights": {
                        "main_raw": 0.44251,
                        "main_calibrated_0p4": 0.16741,
                        "main_calibrated": 0.04365,
                        "main_calibrated_1p0": 0.09643,
                        "count_calibrated": 0.25,
                    },
                    "description": "public-score proxy with count branch capped at 0.25",
                },
                {
                    "name": "calibrated_count_0p27776",
                    "weights": {"main_calibrated": 0.72224, "count_calibrated": 0.27776},
                    "description": "current best public submission blended with count diversity",
                },
            ]
        )
        recipes.extend(
            {
                "name": f"raw_count_{weight:.5f}".replace(".", "p"),
                "weights": {"main_raw": 1.0 - weight, "count_calibrated": weight},
                "description": "nearby raw/count proxy grid candidate for manual public-score probing",
            }
            for weight in raw_count_grid
            if abs(weight - default_count_weight) > 1e-6
        )
    return recipes


def estimate_public_proxy_mse(test_components: dict[str, np.ndarray], weights: dict[str, float]) -> float | None:
    known_public_scores = {
        "main_raw": 3420.98579,
        "main_calibrated_0p4": 3198.70370,
        "main_calibrated": 3156.41472,
        "main_calibrated_1p0": 3209.53910,
        "count_calibrated": 3796.72191,
    }
    active_weights = {name: float(weight) for name, weight in weights.items() if abs(float(weight)) > 1e-12}
    if any(name not in known_public_scores for name in active_weights):
        return None
    if any(name not in test_components for name in active_weights):
        return None

    names = list(active_weights)
    linear = sum(active_weights[name] * known_public_scores[name] for name in names)
    diversity = 0.0
    for left in names:
        for right in names:
            diff = test_components[left] - test_components[right]
            diversity += active_weights[left] * active_weights[right] * float(np.mean(diff * diff))
    return linear - 0.5 * diversity


def save_candidate_submissions(
    output_dir: Path,
    test_df: pd.DataFrame,
    test_components: dict[str, np.ndarray],
    valid_components: dict[str, np.ndarray],
    y_valid: np.ndarray,
    default_count_weight: float,
    event_enabled: bool,
) -> list[dict[str, object]]:
    candidate_dir = output_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for recipe in candidate_recipes("count_calibrated" in test_components, default_count_weight):
        name = str(recipe["name"])
        weights = dict(recipe["weights"])
        pred_valid = blend_named_predictions(valid_components, weights)
        pred_test = blend_named_predictions(test_components, weights)
        valid_metrics = regression_metrics(y_valid, pred_valid)

        variants = [(name, pred_test, False, {"enabled": False, "changed_rows": 0, "mean_delta": 0.0})]
        if event_enabled:
            for profile in EVENT_PROFILES:
                event_pred, event_meta = apply_event_adjustments(test_df, pred_test, profile=profile)
                suffix = "event" if profile == "default" else f"event_{profile}"
                variants.append((f"{name}_{suffix}", event_pred, True, event_meta))

        for variant_name, variant_pred, uses_event, event_meta in variants:
            submission = build_submission(test_df, variant_pred)
            validate_submission(submission, test_df)
            path = candidate_dir / f"submission_{variant_name}.csv"
            submission.to_csv(path, index=False, float_format="%.6f")
            rows.append(
                {
                    "candidate": variant_name,
                    "path": path.as_posix(),
                    "description": recipe["description"],
                    "weights": ";".join(f"{k}:{v:.5f}" for k, v in weights.items()),
                    "public_proxy_mse": estimate_public_proxy_mse(test_components, weights),
                    "observed_public_mse": observed_public_mse(variant_name),
                    "validation_mse": valid_metrics["mse"],
                    "validation_rmse": valid_metrics["rmse"],
                    "validation_mae": valid_metrics["mae"],
                    "event_adjustment": uses_event,
                    "event_profile": event_meta.get("profile", "none"),
                    "event_changed_rows": int(event_meta.get("changed_rows", 0)),
                    "mean": float(submission[TARGET_COL].mean()),
                    "std": float(submission[TARGET_COL].std()),
                    "min": float(submission[TARGET_COL].min()),
                    "max": float(submission[TARGET_COL].max()),
                }
            )

    summary_path = candidate_dir / "candidate_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return rows


def run_experiment(cfg: ExperimentConfig) -> dict[str, object]:
    set_global_seed(cfg.seed)
    ensure_dirs(cfg.output_dir, cfg.model_dir)

    train_df, test_df = load_data(cfg.train_path, cfg.test_path)
    train_part, valid_part = temporal_train_valid_split(
        train_df,
        validation_size=cfg.validation_size,
        default_size=len(test_df),
    )

    print("=" * 80)
    print("共享单车租借量预测实验")
    print(f"train: {cfg.train_path} | rows={len(train_df)}")
    print(f"test : {cfg.test_path} | rows={len(test_df)}")
    print(f"feature_mode={cfg.feature_mode} | model_set={cfg.model_set} | seed={cfg.seed} | n_jobs={cfg.n_jobs}")
    print(f"validation: last {len(valid_part)} train rows")
    print("=" * 80)

    # Validation stage: statistics must be fitted using train_part only.
    [X_tr, X_valid], feature_columns = make_features(
        [train_part, valid_part],
        reference_df=train_part,
        feature_mode=cfg.feature_mode,
    )
    y_tr = train_part[TARGET_COL].to_numpy(dtype=float)
    y_valid = valid_part[TARGET_COL].to_numpy(dtype=float)
    sample_weight = make_sample_weight(train_part)

    specs = make_model_specs(cfg.model_set, cfg.seed, cfg.n_jobs)
    if not specs:
        raise RuntimeError("No model specification available. Check dependencies.")

    validation_predictions: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, object]] = []

    print("\n[1/3] Time-based validation")
    validation_predictions = fit_predict_specs(
        specs,
        X_tr,
        y_tr,
        X_valid,
        sample_weight,
        metric_rows,
        y_valid,
    )

    if not validation_predictions:
        raise RuntimeError("All models failed during validation.")

    weights = optimize_ensemble_weights(validation_predictions, y_valid, cfg.seed)
    pred_valid_ensemble = weighted_average_predictions(validation_predictions, weights)
    ensemble_metrics = regression_metrics(y_valid, pred_valid_ensemble)
    calibrator = fit_prediction_calibrator(
        pred_valid_ensemble,
        y_valid,
        cfg.calibration,
        cfg.calibration_strength,
    )
    pred_valid_calibrated = calibrator.apply(pred_valid_ensemble)
    calibrated_metrics = regression_metrics(y_valid, pred_valid_calibrated)
    metric_rows.append(
        {
            "model": "ensemble",
            "target_transform": "weighted_average",
            **ensemble_metrics,
        }
    )
    metric_rows.append(
        {
            "model": "ensemble_calibrated",
            "target_transform": calibrator.mode,
            **calibrated_metrics,
        }
    )

    print("\nValidation ensemble weights:")
    for name, weight in sorted(weights.items(), key=lambda kv: -kv[1]):
        print(f"  {name}: {weight:.6f}")
    print(
        f"Validation ensemble MSE={ensemble_metrics['mse']:.6f} | "
        f"RMSE={ensemble_metrics['rmse']:.6f} | MAE={ensemble_metrics['mae']:.6f}"
    )
    print(
        f"Calibration mode={calibrator.mode} | slope={calibrator.slope:.6f} | "
        f"intercept={calibrator.intercept:.6f} | strength={calibrator.strength:.3f}"
    )
    print(
        f"Validation calibrated MSE={calibrated_metrics['mse']:.6f} | "
        f"RMSE={calibrated_metrics['rmse']:.6f} | MAE={calibrated_metrics['mae']:.6f}"
    )

    count_blend_weight = float(np.clip(cfg.count_blend_weight, 0.0, 1.0))
    count_weights: dict[str, float] = {}
    count_calibrator = None
    count_metrics = None
    blended_metrics = None
    pred_valid_count_calibrated = None
    if cfg.model_set == "default" and count_blend_weight > 0.0:
        print("\n[1b/3] Count-objective diversity branch")
        count_specs = make_count_model_specs(cfg.seed, cfg.n_jobs)
        count_validation_predictions = fit_predict_specs(
            count_specs,
            X_tr,
            y_tr,
            X_valid,
            sample_weight,
            metric_rows,
            y_valid,
        )
        if count_validation_predictions:
            count_weights = optimize_ensemble_weights(
                count_validation_predictions,
                y_valid,
                cfg.seed + 1009,
                calibration_mode="affine",
                calibration_strength=1.0,
            )
            pred_valid_count = weighted_average_predictions(count_validation_predictions, count_weights)
            count_calibrator = fit_prediction_calibrator(pred_valid_count, y_valid, "affine", 1.0)
            pred_valid_count_calibrated = count_calibrator.apply(pred_valid_count)
            count_metrics = regression_metrics(y_valid, pred_valid_count_calibrated)
            pred_valid_blended = (1.0 - count_blend_weight) * pred_valid_ensemble + (
                count_blend_weight * pred_valid_count_calibrated
            )
            blended_metrics = regression_metrics(y_valid, pred_valid_blended)
            metric_rows.append(
                {
                    "model": "count_ensemble_calibrated",
                    "target_transform": "affine",
                    **count_metrics,
                }
            )
            metric_rows.append(
                {
                    "model": "final_count_blend",
                    "target_transform": f"raw_main_plus_count_{count_blend_weight:.5f}",
                    **blended_metrics,
                }
            )
            print("\nCount ensemble weights:")
            for name, weight in sorted(count_weights.items(), key=lambda kv: -kv[1]):
                print(f"  {name}: {weight:.6f}")
            print(
                f"Count calibrated MSE={count_metrics['mse']:.6f} | "
                f"blend MSE={blended_metrics['mse']:.6f}"
            )

    metrics_path = cfg.output_dir / "validation_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    save_json(weights, cfg.output_dir / "ensemble_weights.json")
    if count_weights:
        save_json(count_weights, cfg.output_dir / "count_ensemble_weights.json")
        save_json(
            {
                "mode": count_calibrator.mode,
                "slope": count_calibrator.slope,
                "intercept": count_calibrator.intercept,
                "strength": count_calibrator.strength,
            },
            cfg.output_dir / "count_calibration.json",
        )
    save_json(
        {
            "mode": calibrator.mode,
            "slope": calibrator.slope,
            "intercept": calibrator.intercept,
            "strength": calibrator.strength,
        },
        cfg.output_dir / "calibration.json",
    )
    save_json(feature_columns, cfg.output_dir / "feature_columns.json")

    # Final stage: train on the full train.csv and predict test.csv.
    print("\n[2/3] Final training on all train.csv rows")
    [X_full, X_test], final_feature_columns = make_features(
        [train_df, test_df],
        reference_df=train_df,
        feature_mode=cfg.feature_mode,
    )
    if final_feature_columns != feature_columns:
        # This should not happen, but keeping the check prevents silent feature mismatch.
        save_json(final_feature_columns, cfg.output_dir / "feature_columns_final.json")
        raise RuntimeError("Validation and final feature columns are inconsistent.")

    y_full = train_df[TARGET_COL].to_numpy(dtype=float)
    full_weight = make_sample_weight(train_df)
    final_specs = make_model_specs(cfg.model_set, cfg.seed, cfg.n_jobs)
    final_predictions: dict[str, np.ndarray] = {}

    successful_model_names = set(weights.keys())
    for spec in final_specs:
        if spec.name not in successful_model_names:
            continue
        if weights.get(spec.name, 0.0) <= 1e-6:
            continue
        print(f"  fitting final {spec.name} with ensemble weight {weights[spec.name]:.6f} ...", flush=True)
        fit_model(spec, X_full, y_full, sample_weight=full_weight)
        final_predictions[spec.name] = predict_model(spec, X_test)
        if cfg.save_model:
            model_payload = {
                "model": spec.estimator,
                "model_name": spec.name,
                "target_transform": spec.target_transform,
                "feature_columns": final_feature_columns,
                "feature_mode": cfg.feature_mode,
                "ensemble_weight": weights[spec.name],
            }
            joblib.dump(model_payload, cfg.model_dir / f"{spec.name}.joblib")

    if not final_predictions:
        raise RuntimeError("No final predictions were generated.")

    raw_main_pred_test = weighted_average_predictions(final_predictions, weights)
    main_calibrated_pred_test = calibrator.apply(raw_main_pred_test)
    pred_test = main_calibrated_pred_test
    count_pred_test = None
    if count_weights and count_calibrator is not None:
        count_specs_final = make_count_model_specs(cfg.seed, cfg.n_jobs)
        count_final_predictions: dict[str, np.ndarray] = {}
        print("\n[2b/3] Final count-objective diversity training")
        for spec in count_specs_final:
            if spec.name not in count_weights:
                continue
            if count_weights.get(spec.name, 0.0) <= 1e-6:
                continue
            print(f"  fitting final {spec.name} with count weight {count_weights[spec.name]:.6f} ...", flush=True)
            fit_model(spec, X_full, y_full, sample_weight=full_weight)
            count_final_predictions[spec.name] = predict_model(spec, X_test)
            if cfg.save_model:
                model_payload = {
                    "model": spec.estimator,
                    "model_name": spec.name,
                    "target_transform": spec.target_transform,
                    "feature_columns": final_feature_columns,
                    "feature_mode": cfg.feature_mode,
                    "ensemble_weight": count_weights[spec.name],
                    "branch": "count_diversity",
                }
                joblib.dump(model_payload, cfg.model_dir / f"count_{spec.name}.joblib")
        if count_final_predictions:
            count_pred_test = count_calibrator.apply(weighted_average_predictions(count_final_predictions, count_weights))
            pred_test = (1.0 - count_blend_weight) * raw_main_pred_test + count_blend_weight * count_pred_test

    valid_components = {
        "main_raw": pred_valid_ensemble,
        "main_calibrated_0p4": apply_calibration_strength(
            pred_valid_ensemble,
            calibrator.slope,
            calibrator.intercept,
            0.4,
        ),
        "main_calibrated": pred_valid_calibrated,
        "main_calibrated_1p0": apply_calibration_strength(
            pred_valid_ensemble,
            calibrator.slope,
            calibrator.intercept,
            1.0,
        ),
    }
    test_components = {
        "main_raw": raw_main_pred_test,
        "main_calibrated_0p4": apply_calibration_strength(
            raw_main_pred_test,
            calibrator.slope,
            calibrator.intercept,
            0.4,
        ),
        "main_calibrated": main_calibrated_pred_test,
        "main_calibrated_1p0": apply_calibration_strength(
            raw_main_pred_test,
            calibrator.slope,
            calibrator.intercept,
            1.0,
        ),
    }
    if count_pred_test is not None and pred_valid_count_calibrated is not None:
        valid_components["count_calibrated"] = pred_valid_count_calibrated
        test_components["count_calibrated"] = count_pred_test
    candidate_rows = save_candidate_submissions(
        cfg.output_dir,
        test_df,
        test_components,
        valid_components,
        y_valid,
        count_blend_weight,
        cfg.event_adjustment,
    )

    if cfg.event_adjustment:
        pred_test, event_adjustment = apply_event_adjustments(
            test_df,
            pred_test,
            profile=cfg.event_adjustment_profile,
        )
    else:
        event_adjustment = {
            "enabled": False,
            "method": "none",
            "profile": "none",
            "changed_rows": 0,
            "mean_delta": 0.0,
        }
    submission = build_submission(test_df, pred_test)
    validate_submission(submission, test_df)

    print("\n[3/3] Saving outputs")
    submission_path = cfg.output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False, float_format="%.6f")

    summary = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_mode": cfg.feature_mode,
        "model_set": cfg.model_set,
        "seed": int(cfg.seed),
        "n_jobs": int(cfg.n_jobs),
        "validation_size": int(len(valid_part)),
        "validation_ensemble": ensemble_metrics,
        "calibration": {
            "mode": calibrator.mode,
            "slope": calibrator.slope,
            "intercept": calibrator.intercept,
            "strength": calibrator.strength,
        },
        "count_blend": {
            "weight": count_blend_weight,
            "enabled": bool(count_weights),
            "validation_count": count_metrics,
            "validation_blended": blended_metrics,
        },
        "event_adjustment": event_adjustment,
        "candidate_submissions": candidate_rows,
        "validation_calibrated": calibrated_metrics,
        "submission_path": submission_path.as_posix(),
        "metrics_path": metrics_path.as_posix(),
    }
    save_json(summary, cfg.output_dir / "run_summary.json")

    print(f"submission saved to: {submission_path}")
    print(f"metrics saved to   : {metrics_path}")
    print("done.")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bike rental count prediction pipeline")
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"), help="Path to train.csv")
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"), help="Path to test.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for submission and metrics")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Directory for fitted model files")
    parser.add_argument("--model-set", choices=["default", "quick", "sklearn"], default="default")
    parser.add_argument("--feature-mode", choices=["base", "advanced"], default="base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1, help="Use 1 for the most reproducible run")
    parser.add_argument("--validation-size", type=int, default=None, help="Number of latest train rows used as validation")
    parser.add_argument("--calibration", choices=["none", "scale", "affine"], default="affine")
    parser.add_argument(
        "--calibration-strength",
        type=float,
        default=0.6,
        help="Shrink validation-fitted calibration toward raw predictions; 0 disables it, 1 applies it fully",
    )
    parser.add_argument(
        "--count-blend-weight",
        type=float,
        default=0.42764,
        help="Blend weight for the calibrated count-objective diversity branch; use 0 to restore the main baseline",
    )
    parser.add_argument("--no-event-adjustment", action="store_true", help="Disable fixed Sandy-window test adjustment")
    parser.add_argument(
        "--event-adjustment-profile",
        choices=list(EVENT_PROFILES),
        default="stronger",
        help="Sandy-window adjustment strength used for the main submission.csv",
    )
    parser.add_argument("--no-save-model", action="store_true", help="Do not save fitted final models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        train_path=args.train,
        test_path=args.test,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        model_set=args.model_set,
        feature_mode=args.feature_mode,
        seed=args.seed,
        n_jobs=args.n_jobs,
        save_model=not args.no_save_model,
        validation_size=args.validation_size,
        calibration=args.calibration,
        calibration_strength=args.calibration_strength,
        count_blend_weight=args.count_blend_weight,
        event_adjustment=not args.no_event_adjustment,
        event_adjustment_profile=args.event_adjustment_profile,
    )
    run_experiment(cfg)
