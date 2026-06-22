from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
CANDIDATE_DIR = OUTPUT_DIR / "candidates"
TOP5_DIR = OUTPUT_DIR / "top5_submissions"


CANDIDATES = [
    (
        "01_current_public_best_2843_40079.csv",
        OUTPUT_DIR / "submission.csv",
        "current rollback default; observed public MSE 2843.40079",
    ),
    (
        "02_lower_count_0p445_weather3_soft.csv",
        CANDIDATE_DIR / "submission_raw_count_0p44500_event_score_rebound_fit_weather3_soft.csv",
        "nearest lower count-weight probe; validation MSE 2843.738708",
    ),
    (
        "03_lower_count_0p440_weather3_soft.csv",
        CANDIDATE_DIR / "submission_raw_count_0p44000_event_score_rebound_fit_weather3_soft.csv",
        "slightly stronger guard against high-count public overfit; validation MSE 2845.564620",
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
        "This set is intentionally risk-constrained after prior local-validation winners regressed online. "
        "The first file remains the observed public best; the rest are narrow probes around that baseline, "
        "focused on reducing high-count overfit rather than chasing lower local validation MSE.",
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
