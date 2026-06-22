# Top 5 submission candidates

These files are formatted for direct manual evaluation with columns `ID,cnt`.

This set is intentionally risk-constrained after prior local-validation winners regressed online. The first file remains the observed public best; the rest are narrow probes around that baseline, focused on reducing high-count overfit rather than chasing lower local validation MSE.

Suggested evaluation order:

1. `01_current_public_best_2843_40079.csv` - current rollback default; observed public MSE 2843.40079.
2. `02_lower_count_0p445_weather3_soft.csv` - nearest lower count-weight probe; validation MSE 2843.738708.
3. `03_lower_count_0p440_weather3_soft.csv` - slightly stronger guard against high-count public overfit; validation MSE 2845.564620.
4. `04_calibrated_count_0p27776_weather3_soft.csv` - lower-count calibrated blend; validation MSE 2894.974348.
5. `05_public_constrained_weather3_soft.csv` - conservative blend with at least half calibrated main model; validation MSE 2890.986889.
