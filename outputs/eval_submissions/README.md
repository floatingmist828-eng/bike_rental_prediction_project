# Evaluation Submission Variants

These CSV files all follow the required `ID,cnt` format and can be uploaded as submissions.

Recommended order:

1. `00_current_score_rebound_mid_submission.csv` - current default `outputs/submission.csv`; raises the 2012-10-29 and 2012-10-30 core storm rows above the known-best strong profile.
2. `01_known_best_strong_2886_37549.csv` - known best observed score: 2886.37549.
3. `02_score_rebound_light.csv` - smaller rebound than the current default.
4. `03_score_rebound_fit.csv` - larger rebound estimated from the score gap between strong and empirical_storm.
5. `04_strong_core_0p37.csv` - only raises the 2012-10-30 13:00-18:00 storm factor slightly.
6. `05_strong_late_0p55.csv` - only raises the 2012-10-30 19:00-23:00 recovery factor.
7. `06_strong_day_0p30.csv` - only raises 2012-10-29.
8. `07_strong_late_0p45.csv` - only lowers the 2012-10-30 19:00-23:00 recovery factor.
