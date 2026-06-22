# Top 5 submission candidates

These files are formatted for direct manual evaluation with columns `ID,cnt`.

This set is anchored on the current observed public best. The first file matches the default `outputs/submission.csv`; the remaining files are backups or narrow probes around the same weather-adjusted count-blend family.

Suggested evaluation order:

1. `01_current_public_best_2841_38813.csv` - current default 0.440/weather3_soft submission; observed public MSE 2841.38813.
2. `02_previous_public_best_2843_40079.csv` - previous default 0.450/weather3_soft submission; observed public MSE 2843.40079.
3. `03_nearby_count_0p445_weather3_soft.csv` - nearest middle probe between the two public-scored defaults; validation MSE 2843.738708.
4. `04_calibrated_count_0p27776_weather3_soft.csv` - lower-count calibrated blend; validation MSE 2894.974348.
5. `05_public_constrained_weather3_soft.csv` - conservative blend with at least half calibrated main model; validation MSE 2890.986889.
