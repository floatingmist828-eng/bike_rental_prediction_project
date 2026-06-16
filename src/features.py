from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .config import DATE_COL, ID_COL, TARGET_COL

FeatureMode = Literal["base", "advanced"]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic date/time, periodic and weather-interaction features."""
    out = df.copy()
    dt = pd.to_datetime(out[DATE_COL])
    origin = pd.Timestamp("2011-01-01")

    out["day"] = dt.dt.day.astype(np.int16)
    out["dayofyear"] = dt.dt.dayofyear.astype(np.int16)
    out["weekofyear"] = dt.dt.isocalendar().week.astype(np.int16)
    out["quarter"] = dt.dt.quarter.astype(np.int8)
    out["is_month_start"] = dt.dt.is_month_start.astype(np.int8)
    out["is_month_end"] = dt.dt.is_month_end.astype(np.int8)
    out["days_since_start"] = (dt - origin).dt.days.astype(np.int16)
    out["timestamp_hour"] = ((dt - origin).dt.days * 24 + out["hr"]).astype(np.int32)
    out["year_month_index"] = (out["yr"] * 12 + out["mnth"]).astype(np.int16)

    out["is_weekend"] = out["weekday"].isin([0, 6]).astype(np.int8)
    out["rush_hour"] = out["hr"].isin([7, 8, 9, 16, 17, 18, 19]).astype(np.int8)
    out["commute_morning"] = ((out["workingday"] == 1) & out["hr"].isin([7, 8, 9])).astype(np.int8)
    out["commute_evening"] = ((out["workingday"] == 1) & out["hr"].isin([16, 17, 18, 19])).astype(np.int8)
    out["night"] = out["hr"].isin([0, 1, 2, 3, 4, 5]).astype(np.int8)
    out["midday"] = out["hr"].isin([11, 12, 13, 14]).astype(np.int8)
    out["work_rush"] = ((out["workingday"] == 1) & (out["rush_hour"] == 1)).astype(np.int8)
    out["leisure_hour"] = ((out["workingday"] == 0) & out["hr"].between(10, 18)).astype(np.int8)

    cyclic_specs = [("hr", 24), ("weekday", 7), ("mnth", 12), ("dayofyear", 366)]
    for col, period in cyclic_specs:
        out[f"{col}_sin"] = np.sin(2.0 * np.pi * out[col] / period)
        out[f"{col}_cos"] = np.cos(2.0 * np.pi * out[col] / period)

    out["temp_hum"] = out["temp"] * out["hum"]
    out["atemp_hum"] = out["atemp"] * out["hum"]
    out["temp_wind"] = out["temp"] * out["windspeed"]
    out["hum_wind"] = out["hum"] * out["windspeed"]
    out["temp_minus_atemp"] = out["temp"] - out["atemp"]
    out["bad_weather"] = (out["weathersit"] >= 3).astype(np.int8)
    out["very_bad_weather"] = (out["weathersit"] >= 4).astype(np.int8)
    out["hot"] = (out["temp"] >= 0.72).astype(np.int8)
    out["cold"] = (out["temp"] <= 0.24).astype(np.int8)
    out["humid"] = (out["hum"] >= 0.80).astype(np.int8)

    out["hr_workingday"] = (out["hr"] * 2 + out["workingday"]).astype(np.int16)
    out["hr_weekday"] = (out["hr"] * 7 + out["weekday"]).astype(np.int16)
    out["month_hr"] = (out["mnth"] * 24 + out["hr"]).astype(np.int16)
    out["season_hr"] = (out["season"] * 24 + out["hr"]).astype(np.int16)

    return out


def _multi_index_from_frame(df: pd.DataFrame, columns: list[str]) -> pd.Index | pd.MultiIndex:
    if len(columns) == 1:
        return pd.Index(df[columns[0]])
    return pd.MultiIndex.from_frame(df[columns])


def _fill_by_group_fallback(
    df: pd.DataFrame,
    values: np.ndarray,
    fallback_maps: list[tuple[list[str], pd.Series]],
    global_mean: float,
) -> np.ndarray:
    filled = pd.Series(values, index=df.index, dtype="float64")
    for group_cols, group_mean in fallback_maps:
        mask = filled.isna()
        if not mask.any():
            break
        idx = _multi_index_from_frame(df.loc[mask, group_cols], group_cols)
        filled.loc[mask] = group_mean.reindex(idx).values
    return filled.fillna(global_mean).values


def add_historical_reference_features(frames: list[pd.DataFrame], reference_df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Add previous-year count references computed only from the provided reference dataframe.

    These features are disabled in the default pipeline because they can be split-sensitive,
    but they are useful for subsequent experiments.
    """
    if TARGET_COL not in reference_df.columns:
        raise ValueError("reference_df must include cnt to build historical reference features.")

    ref = reference_df.copy()
    ref0 = ref[ref["yr"] == 0].copy()
    ref1 = ref[ref["yr"] == 1].copy()
    global_mean = float(ref[TARGET_COL].mean())

    exact_prev_year = ref0.groupby(["mnth", "day", "hr"])[TARGET_COL].mean()
    timestamp_map = ref.groupby("timestamp_hour")[TARGET_COL].mean()

    fallback_groups = [
        ["mnth", "hr", "workingday"],
        ["mnth", "hr"],
        ["season", "hr", "workingday"],
        ["hr", "workingday"],
        ["hr"],
    ]
    fallback_maps = [(g, ref.groupby(g)[TARGET_COL].mean()) for g in fallback_groups]

    if len(ref0) > 0 and len(ref1) > 0:
        pairs = ref1[["mnth", "day", "hr", "workingday", TARGET_COL]].merge(
            ref0[["mnth", "day", "hr", TARGET_COL]],
            on=["mnth", "day", "hr"],
            how="inner",
            suffixes=("_y1", "_y0"),
        )
        if len(pairs) > 0:
            global_ratio = float((pairs[f"{TARGET_COL}_y1"].sum() + 100.0) / (pairs[f"{TARGET_COL}_y0"].sum() + 100.0))
            hr_sum = pairs.groupby("hr")[[f"{TARGET_COL}_y1", f"{TARGET_COL}_y0"]].sum()
            ratio_by_hr = (hr_sum[f"{TARGET_COL}_y1"] + 20.0) / (hr_sum[f"{TARGET_COL}_y0"] + 20.0)
            hw_sum = pairs.groupby(["hr", "workingday"])[[f"{TARGET_COL}_y1", f"{TARGET_COL}_y0"]].sum()
            ratio_by_hr_workingday = (hw_sum[f"{TARGET_COL}_y1"] + 30.0) / (hw_sum[f"{TARGET_COL}_y0"] + 30.0)
        else:
            global_ratio = 1.0
            ratio_by_hr = pd.Series(dtype="float64")
            ratio_by_hr_workingday = pd.Series(dtype="float64")
    else:
        global_ratio = 1.0
        ratio_by_hr = pd.Series(dtype="float64")
        ratio_by_hr_workingday = pd.Series(dtype="float64")

    output_frames: list[pd.DataFrame] = []
    for frame in frames:
        out = frame.copy()
        exact_idx = pd.MultiIndex.from_frame(out[["mnth", "day", "hr"]])
        exact_values = exact_prev_year.reindex(exact_idx).values
        out["prev_year_calendar_cnt"] = _fill_by_group_fallback(out, exact_values, fallback_maps, global_mean)

        same_weekday_values = timestamp_map.reindex(out["timestamp_hour"] - 364 * 24).values
        out["prev_year_same_weekday_cnt"] = _fill_by_group_fallback(out, same_weekday_values, fallback_maps, global_mean)

        out["prev_year_calendar_log"] = np.log1p(out["prev_year_calendar_cnt"])
        out["prev_year_same_weekday_log"] = np.log1p(out["prev_year_same_weekday_cnt"])

        if len(ratio_by_hr) > 0:
            out["ratio_hr"] = ratio_by_hr.reindex(out["hr"]).values
        else:
            out["ratio_hr"] = np.nan
        if len(ratio_by_hr_workingday) > 0:
            hw_idx = pd.MultiIndex.from_frame(out[["hr", "workingday"]])
            out["ratio_hr_workingday"] = ratio_by_hr_workingday.reindex(hw_idx).values
        else:
            out["ratio_hr_workingday"] = np.nan

        out["ratio_hr"] = pd.Series(out["ratio_hr"]).fillna(global_ratio).values
        out["ratio_hr_workingday"] = (
            pd.Series(out["ratio_hr_workingday"]).fillna(pd.Series(out["ratio_hr"])).fillna(global_ratio).values
        )
        out["yoy_global_ratio"] = global_ratio
        out["prev_year_scaled_global"] = out["prev_year_calendar_cnt"] * out["yoy_global_ratio"]
        out["prev_year_scaled_hr"] = out["prev_year_calendar_cnt"] * out["ratio_hr"]
        out["prev_year_scaled_hr_workingday"] = out["prev_year_calendar_cnt"] * out["ratio_hr_workingday"]
        out["prev_same_weekday_scaled"] = out["prev_year_same_weekday_cnt"] * out["ratio_hr_workingday"]
        output_frames.append(out)

    return output_frames


def add_target_statistics(frames: list[pd.DataFrame], reference_df: pd.DataFrame) -> list[pd.DataFrame]:
    """Add smoothed target-mean statistics computed only from reference_df."""
    if TARGET_COL not in reference_df.columns:
        raise ValueError("reference_df must include cnt to build target statistics.")

    groups_list = [
        ["hr"],
        ["hr", "workingday"],
        ["hr", "weekday"],
        ["mnth", "hr"],
        ["mnth", "hr", "workingday"],
        ["season", "hr", "workingday"],
        ["weathersit", "hr", "workingday"],
        ["weekday", "hr", "season"],
        ["yr", "mnth", "hr", "workingday"],
        ["day", "hr"],
        ["dayofyear", "hr"],
        ["hr", "bad_weather"],
    ]

    global_mean = float(reference_df[TARGET_COL].mean())
    output_frames = [frame.copy() for frame in frames]
    smoothing = 15.0

    for group_cols in groups_list:
        stats = reference_df.groupby(group_cols)[TARGET_COL].agg(["mean", "count"])
        encoded = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
        feature_name = "te_" + "_".join(group_cols)
        for i, frame in enumerate(output_frames):
            idx = _multi_index_from_frame(frame, group_cols)
            output_frames[i][feature_name] = encoded.reindex(idx).fillna(global_mean).values

    return output_frames


def make_features(
    frames: Iterable[pd.DataFrame],
    reference_df: pd.DataFrame,
    feature_mode: FeatureMode = "base",
) -> tuple[list[pd.DataFrame], list[str]]:
    """
    Build aligned feature matrices.

    Parameters
    ----------
    frames:
        Any train/validation/test dataframes to transform, in the same order desired by caller.
    reference_df:
        The training portion used for statistics that require cnt. For validation this must be
        only the earlier training split; for final prediction it can be the full training set.
    feature_mode:
        base: stable deterministic features only.
        advanced: base + historical reference + smoothed target statistics.
    """
    transformed = [add_calendar_features(df) for df in frames]
    reference = add_calendar_features(reference_df)

    if feature_mode == "advanced":
        transformed = add_historical_reference_features(transformed, reference)
        transformed = add_target_statistics(transformed, reference)
    elif feature_mode != "base":
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    drop_cols = [ID_COL, TARGET_COL, DATE_COL]
    feature_frames: list[pd.DataFrame] = []
    for frame in transformed:
        X = frame.drop(columns=[c for c in drop_cols if c in frame.columns])
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for col in X.columns:
            if str(X[col].dtype).startswith("UInt"):
                X[col] = X[col].astype(np.int16)
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise ValueError(f"Non-numeric feature columns found: {non_numeric}")
        feature_frames.append(X)

    feature_columns = feature_frames[0].columns.tolist()
    aligned_frames = [X.reindex(columns=feature_columns, fill_value=0.0) for X in feature_frames]
    return aligned_frames, feature_columns


def make_sample_weight(df: pd.DataFrame) -> np.ndarray:
    """Slightly emphasize recent and high-rush-hour observations for MSE-oriented fitting."""
    feat = add_calendar_features(df)
    day = feat["days_since_start"].to_numpy(dtype=float)
    denom = max(1.0, float(day.max() - day.min()))
    weight = 1.0 + 0.30 * feat["yr"].to_numpy(dtype=float) + 0.25 * (day - day.min()) / denom
    rush_mask = (feat["workingday"] == 1) & feat["hr"].isin([7, 8, 17, 18, 19])
    weight += 0.10 * rush_mask.astype(float).to_numpy()
    return weight.astype(float)
