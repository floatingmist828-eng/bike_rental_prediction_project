from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


FeatureMode = Literal["base", "advanced"]
ModelSet = Literal["default", "quick", "sklearn"]
CalibrationMode = Literal["none", "scale", "affine"]


@dataclass(frozen=True)
class ExperimentConfig:
    train_path: Path
    test_path: Path
    output_dir: Path
    model_dir: Path
    model_set: ModelSet = "default"
    feature_mode: FeatureMode = "base"
    seed: int = 42
    n_jobs: int = 1
    save_model: bool = True
    validation_size: int | None = None
    calibration: CalibrationMode = "affine"


TARGET_COL = "cnt"
ID_COL = "ID"
DATE_COL = "dteday"

TRAIN_REQUIRED_COLUMNS = [
    "ID",
    "dteday",
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "cnt",
]

TEST_REQUIRED_COLUMNS = [c for c in TRAIN_REQUIRED_COLUMNS if c != TARGET_COL]
