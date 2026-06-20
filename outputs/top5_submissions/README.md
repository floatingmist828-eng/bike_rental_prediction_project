# Top 5 submission candidates

These files are formatted for direct manual evaluation with columns `ID,cnt`.

Suggested evaluation order:

1. `01_current_default_count_1p075_late_hour.csv` - current default `outputs/submission.csv`; validation MSE `2683.346736`.
2. `02_validation_second_count_1p05_late_hour.csv` - nearly tied late-hour candidate; validation MSE `2683.348008`.
3. `03_validation_third_count_1p025_late_hour.csv` - slightly less aggressive count extrapolation; validation MSE `2683.741068`.
4. `04_validation_fourth_count_1p00_late_hour.csv` - no count extrapolation beyond the calibrated count branch; validation MSE `2684.525916`.
5. `05_validation_fifth_count_0p90_late_hour.csv` - lower count weight fallback; validation MSE `2691.583188`.

The late-hour profile is the most aggressive validation-driven candidate family. If public/private score regresses, compare against `outputs/candidates/*overcount_hum_rush.csv` and earlier conservative candidates.
