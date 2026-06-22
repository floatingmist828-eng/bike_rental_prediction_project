from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
CANDIDATE_DIR = OUTPUT_DIR / "candidates"
TOP5_DIR = OUTPUT_DIR / "top5_submissions"


CANDIDATES = [
    (
        "01_current_public_best_2841_38813.csv",
        OUTPUT_DIR / "submission.csv",
        "current default 0.440/weather3_soft submission; observed public MSE 2841.38813",
    ),
    (
        "02_previous_public_best_2843_40079.csv",
        CANDIDATE_DIR / "submission_raw_count_0p45000_event_score_rebound_fit_weather3_soft.csv",
        "previous default 0.450/weather3_soft submission; observed public MSE 2843.40079",
    ),
    (
        "03_nearby_count_0p445_weather3_soft.csv",
        CANDIDATE_DIR / "submission_raw_count_0p44500_event_score_rebound_fit_weather3_soft.csv",
        "nearest middle probe between the two public-scored defaults; validation MSE 2843.738708",
    ),
    (
        "04_calibrated_count_0p27776_weather3_soft.csv",
        CANDIDATE_DIR / "submission_calibrated_count_0p27776_event_score_rebound_fit_weather3_soft.csv",
        "lower-count calibrated blend; validation MSE 2894.974348",
    ),
    (
        "05_public_constrained_weather3_soft.csv",
        CANDIDATE_DIR / "submission_public_constrained_event_score_rebound_fit_weather3_soft.csv",
        "conservative blend with at least half calibrated main model; validation MSE 2890.986889",
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
        "This set is anchored on the current observed public best. The first file matches the default "
        "`outputs/submission.csv`; the remaining files are backups or narrow probes around the same "
        "weather-adjusted count-blend family.",
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
