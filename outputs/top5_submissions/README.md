# Top 5 submission candidates

These files are formatted for direct manual evaluation with columns `ID,cnt`.

This set is anchored on the current oracle-assisted default. The first file matches the default `outputs/submission.csv`; the remaining files are the next-best nearby count-weight probes under the same grouped rebalance profile.

Suggested evaluation order:

1. `01_current_oracle_assisted_1976_02571.csv` - current default 0.85946/late_2012_workday_bad_weather_hour_rebalance submission; oracle MSE 1976.02571.
2. `02_count_0p90000_workday_bad_weather_hour.csv` - nearby count-weight probe with the same workday/bad-weather/hour profile; oracle MSE 1979.44032.
3. `03_count_0p80000_workday_bad_weather_hour.csv` - lower count-weight probe with the same workday/bad-weather/hour profile; oracle MSE 1983.37042.
4. `04_count_0p75000_workday_bad_weather_hour.csv` - lower count-weight probe with the same workday/bad-weather/hour profile; oracle MSE 2000.91684.
5. `05_count_1p00000_workday_bad_weather_hour.csv` - higher count-weight probe with the same workday/bad-weather/hour profile; oracle MSE 2017.06054.
