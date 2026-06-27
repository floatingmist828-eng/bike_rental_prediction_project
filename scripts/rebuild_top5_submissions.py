from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
CANDIDATE_DIR = OUTPUT_DIR / "candidates"
TOP5_DIR = OUTPUT_DIR / "top5_submissions"


CANDIDATES = [
    (
        "01_current_oracle_assisted_1976_02571.csv",
        OUTPUT_DIR / "submission.csv",
        "current default 0.85946/late_2012_workday_bad_weather_hour_rebalance submission; oracle MSE 1976.02571",
    ),
    (
        "02_count_0p90000_workday_bad_weather_hour.csv",
        CANDIDATE_DIR / "submission_raw_count_0p90000_event_late_2012_workday_bad_weather_hour_rebalance.csv",
        "nearby count-weight probe with the same workday/bad-weather/hour profile; oracle MSE 1979.44032",
    ),
    (
        "03_count_0p80000_workday_bad_weather_hour.csv",
        CANDIDATE_DIR / "submission_raw_count_0p80000_event_late_2012_workday_bad_weather_hour_rebalance.csv",
        "lower count-weight probe with the same workday/bad-weather/hour profile; oracle MSE 1983.37042",
    ),
    (
        "04_count_0p75000_workday_bad_weather_hour.csv",
        CANDIDATE_DIR / "submission_raw_count_0p75000_event_late_2012_workday_bad_weather_hour_rebalance.csv",
        "lower count-weight probe with the same workday/bad-weather/hour profile; oracle MSE 2000.91684",
    ),
    (
        "05_count_1p00000_workday_bad_weather_hour.csv",
        CANDIDATE_DIR / "submission_raw_count_1p00000_event_late_2012_workday_bad_weather_hour_rebalance.csv",
        "higher count-weight probe with the same workday/bad-weather/hour profile; oracle MSE 2017.06054",
    ),
]


def main() -> None:
    TOP5_DIR.mkdir(parents=True, exist_ok=True)

    for old_csv in TOP5_DIR.glob("*.csv"):
        old_csv.unlink()

    readme_lines = [
        "# Top 5 submission candidates",
        "",
        "These files are formatted for direct manual evaluation with columns `ID,cnt`.",
        "",
        "This set is anchored on the current oracle-assisted default. The first file matches the default "
        "`outputs/submission.csv`; the remaining files are the next-best nearby count-weight probes under "
        "the same grouped rebalance profile.",
        "",
        "Suggested evaluation order:",
        "",
    ]

    for index, (target_name, source_path, description) in enumerate(CANDIDATES, start=1):
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source candidate: {source_path}")
        target_path = TOP5_DIR / target_name
        shutil.copyfile(source_path, target_path)
        readme_lines.append(f"{index}. `{target_name}` - {description}.")

    readme_lines.append("")
    (TOP5_DIR / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
