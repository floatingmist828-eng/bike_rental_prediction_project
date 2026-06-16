from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import ID_COL, TARGET_COL, TEST_REQUIRED_COLUMNS, TRAIN_REQUIRED_COLUMNS


def set_global_seed(seed: int) -> None:
    """Set common random seeds for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_data(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    validate_input_schema(train_df, test_df)
    return train_df, test_df


def validate_input_schema(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    missing_train = [c for c in TRAIN_REQUIRED_COLUMNS if c not in train_df.columns]
    missing_test = [c for c in TEST_REQUIRED_COLUMNS if c not in test_df.columns]
    if missing_train:
        raise ValueError(f"train.csv 缺少字段: {missing_train}")
    if missing_test:
        raise ValueError(f"test.csv 缺少字段: {missing_test}")
    if TARGET_COL in test_df.columns:
        raise ValueError("test.csv 不应包含目标列 cnt。")
    if train_df[TRAIN_REQUIRED_COLUMNS].isna().any().any():
        bad = train_df[TRAIN_REQUIRED_COLUMNS].columns[train_df[TRAIN_REQUIRED_COLUMNS].isna().any()].tolist()
        raise ValueError(f"train.csv 存在缺失值字段: {bad}")
    if test_df[TEST_REQUIRED_COLUMNS].isna().any().any():
        bad = test_df[TEST_REQUIRED_COLUMNS].columns[test_df[TEST_REQUIRED_COLUMNS].isna().any()].tolist()
        raise ValueError(f"test.csv 存在缺失值字段: {bad}")
    if train_df[ID_COL].duplicated().any():
        raise ValueError("train.csv 中 ID 存在重复。")
    if test_df[ID_COL].duplicated().any():
        raise ValueError("test.csv 中 ID 存在重复。")


def temporal_train_valid_split(
    train_df: pd.DataFrame,
    validation_size: int | None,
    default_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the latest rows as validation to mimic future-time prediction."""
    n_total = len(train_df)
    n_valid = default_size if validation_size is None else int(validation_size)
    if not (1 <= n_valid < n_total):
        raise ValueError(f"validation_size 必须在 [1, {n_total - 1}] 范围内，当前为 {n_valid}")
    train_part = train_df.iloc[: n_total - n_valid].copy()
    valid_part = train_df.iloc[n_total - n_valid :].copy()
    return train_part, valid_part


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"mse": mse, "rmse": rmse, "mae": mae}


def save_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_submission(test_df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    pred = np.asarray(pred, dtype=float)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    pred = np.clip(pred, 0.0, None)
    return pd.DataFrame({ID_COL: test_df[ID_COL].values, TARGET_COL: pred})


def validate_submission(submission: pd.DataFrame, test_df: pd.DataFrame) -> None:
    if list(submission.columns) != [ID_COL, TARGET_COL]:
        raise ValueError("submission.csv 必须只包含两列: ID,cnt")
    if len(submission) != len(test_df):
        raise ValueError("submission.csv 行数必须与 test.csv 完全一致。")
    if not np.array_equal(submission[ID_COL].values, test_df[ID_COL].values):
        raise ValueError("submission.csv 的 ID 必须与 test.csv 一一对应且顺序不变。")
    if submission[TARGET_COL].isna().any():
        raise ValueError("submission.csv 中 cnt 存在缺失值。")
    if (submission[TARGET_COL] < 0).any():
        raise ValueError("submission.csv 中 cnt 不应为负值。")
