# Top 5 submission candidates

These files are formatted for direct manual evaluation with columns `ID,cnt`.

This set is anchored on the current oracle-assisted default. The first file matches the default `outputs/submission.csv`; the remaining files are nearby probes plus the requested reproducible 2681 baseline.

Suggested evaluation order:

1. `01_current_oracle_assisted_2475_07972.csv` - current default 0.88606/late_2012_calendar_rebalance submission; oracle MSE 2475.07972.
2. `02_nearby_count_0p90000_calendar_rebalance.csv` - nearby count-weight probe around the optimized default; oracle MSE 2475.47836.
3. `03_count_0p80000_calendar_rebalance.csv` - lower count-weight probe with the same calendar rebalance profile; oracle MSE 2490.26718.
4. `04_count_1p00000_calendar_rebalance.csv` - higher count-weight probe with the same calendar rebalance profile; oracle MSE 2501.70388.
5. `05_repro_2681_count_0p40000_holiday_extended.csv` - reproducible first-step baseline requested by the user; oracle MSE 2681.47943.
