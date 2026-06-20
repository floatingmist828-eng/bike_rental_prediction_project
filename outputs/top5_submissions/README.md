# Top 5 submission candidates

These files are formatted for direct manual evaluation with columns `ID,cnt`.

Suggested evaluation order:

1. `01_current_default_count_0p45_weather3_soft.csv` - current default `outputs/submission.csv`; conservative weather-adjusted blend.
2. `02_validation_best_count_0p80_weather3_soft.csv` - lowest adjusted validation MSE candidate in the current grid; highest public-score risk because it relies more on the count branch.
3. `03_validation_best_count_0p80_weather3_holiday_soft.csv` - same aggressive blend plus light Thanksgiving/Christmas cuts.
4. `04_validation_second_count_0p75_weather3_soft.csv` - slightly less aggressive validation-best family candidate.
5. `05_known_best_strong_2886_37549.csv` - legacy known public-score baseline for comparison.
