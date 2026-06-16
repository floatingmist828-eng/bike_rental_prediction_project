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
    fit_model,
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
    for spec in specs:
        print(f"  fitting {spec.name} ({spec.target_transform}) ...", flush=True)
        try:
            fit_model(spec, X_tr, y_tr, sample_weight=sample_weight)
            pred_valid = predict_model(spec, X_valid)
            metrics = regression_metrics(y_valid, pred_valid)
            validation_predictions[spec.name] = pred_valid
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

    if not validation_predictions:
        raise RuntimeError("All models failed during validation.")

    weights = optimize_ensemble_weights(validation_predictions, y_valid, cfg.seed)
    pred_valid_ensemble = weighted_average_predictions(validation_predictions, weights)
    ensemble_metrics = regression_metrics(y_valid, pred_valid_ensemble)
    metric_rows.append(
        {
            "model": "ensemble",
            "target_transform": "weighted_average",
            **ensemble_metrics,
        }
    )

    print("\nValidation ensemble weights:")
    for name, weight in sorted(weights.items(), key=lambda kv: -kv[1]):
        print(f"  {name}: {weight:.6f}")
    print(
        f"Validation ensemble MSE={ensemble_metrics['mse']:.6f} | "
        f"RMSE={ensemble_metrics['rmse']:.6f} | MAE={ensemble_metrics['mae']:.6f}"
    )

    metrics_path = cfg.output_dir / "validation_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    save_json(weights, cfg.output_dir / "ensemble_weights.json")
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

    pred_test = weighted_average_predictions(final_predictions, weights)
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
        "submission_path": str(submission_path),
        "metrics_path": str(metrics_path),
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
    )
    run_experiment(cfg)
