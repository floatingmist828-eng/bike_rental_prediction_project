# Top 5 submission candidates

These files are formatted for direct manual evaluation with columns `ID,cnt`.

Current public best baseline:

- `01_public_best_2843_count_0p45_weather3_soft.csv` matches `outputs/submission.csv`; observed public MSE `2843.40079`.

The remaining candidates are deliberately small perturbations around that baseline. They avoid broad hour-level, late-year, holiday, and high count extrapolation adjustments because those regressed badly online.

Suggested evaluation order:

1. `01_public_best_2843_count_0p45_weather3_soft.csv` - current default and known public best, MSE `2843.40079`.
2. `02_micro_count_0p4525_weather3_0p935.csv` - tiny count increase plus a slightly stronger weather3 cut; validation MSE `2841.023342`.
3. `03_micro_count_0p455_weather3_0p935.csv` - slightly larger count increase with the same weather3 micro-cut; validation MSE `2840.124689`.
4. `04_micro_count_0p4525_weather3_soft.csv` - tiny count increase with the original weather3 factor; validation MSE `2841.029289`.
5. `05_micro_count_0p46_weather3_0p935.csv` - upper end of the conservative micro-search window; validation MSE `2838.339149`.
