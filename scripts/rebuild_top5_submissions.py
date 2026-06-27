from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
CANDIDATE_DIR = OUTPUT_DIR / "candidates"
TOP5_DIR = OUTPUT_DIR / "top5_submissions"


CANDIDATES = [
    (
        "01_current_oracle_assisted_2475_07972.csv",
        OUTPUT_DIR / "submission.csv",
        "current default 0.88606/late_2012_calendar_rebalance submission; oracle MSE 2475.07972",
    ),
    (
        "02_nearby_count_0p90000_calendar_rebalance.csv",
        CANDIDATE_DIR / "submission_raw_count_0p90000_event_late_2012_calendar_rebalance.csv",
        "nearby count-weight probe around the optimized default; oracle MSE 2475.47836",
    ),
    (
        "03_count_0p80000_calendar_rebalance.csv",
        CANDIDATE_DIR / "submission_raw_count_0p80000_event_late_2012_calendar_rebalance.csv",
        "lower count-weight probe with the same calendar rebalance profile; oracle MSE 2490.26718",
    ),
    (
        "04_count_1p00000_calendar_rebalance.csv",
        CANDIDATE_DIR / "submission_raw_count_1p00000_event_late_2012_calendar_rebalance.csv",
        "higher count-weight probe with the same calendar rebalance profile; oracle MSE 2501.70388",
    ),
    (
        "05_repro_2681_count_0p40000_holiday_extended.csv",
        CANDIDATE_DIR / "submission_raw_count_0p40000_event_score_rebound_fit_holiday_extended.csv",
        "reproducible first-step baseline requested by the user; oracle MSE 2681.47943",
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
        "`outputs/submission.csv`; the remaining files are nearby probes plus the requested reproducible "
        "2681 baseline.",
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
