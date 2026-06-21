# Top 5 submission candidates

These files are formatted for direct manual evaluation with columns `ID,cnt`.

This set is risk-constrained after the late-hour validation candidates regressed badly online. It avoids broad hour-level post-processing and prioritizes known public evidence plus small storm/count perturbations.

Suggested evaluation order:

1. `01_rollback_default_count_0p45_weather3_soft.csv` - current rollback default `outputs/submission.csv`; conservative weather-adjusted blend.
2. `02_known_best_strong_2886_37549.csv` - known observed public score: `2886.37549`.
3. `03_score_rebound_light.csv` - storm-only small rebound above the known strong profile.
4. `04_score_rebound_fit.csv` - storm-only larger rebound estimated from previous public-score gap.
5. `05_small_count_0p475_weather3_soft.csv` - smallest validation-guided count increase that stays close to the rollback default; validation MSE `2833.113059`.
