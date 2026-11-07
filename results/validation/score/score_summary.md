# Score summary

- Scorer version: 0.1.1
- Schema version: 1.0
- Submission: forecast_backtest.csv
- Actuals: actuals_validation.csv
- Spike margin: 50.00 EUR/MWh
- Rows in submission: 26880
- Rows scored (price metrics): 26700; skipped (no actual): 180
- Rows scored (exceptional prices): 26700
- Rows scored (direction): 26700

## Overall results

| Metric | Value | Use |
|---|---|---|
| MAE (p50) | 43.7217 | Level 1: accuracy |
| Pinball loss p10 | 7.5410 | |
| Pinball loss p90 | 17.5979 | |
| Mean pinball loss p10 and p90 | 12.5694 | Level 1: calibration |
| P10-P90 coverage | 0.8080 | supporting metric (target about 0.80) |
| Brier p_spike_high | 0.1017 | |
| Brier p_spike_low | 0.0888 | |
| Mean Brier, exceptional prices | 0.0953 | Level 1: exceptional prices |
| Three-class Brier, direction | 0.4192 | Level 1: direction |
| Pinball loss p50 | 21.8608 | for information |
| Observed share of high / low quarter-hours | 0.1213 / 0.1296 | for information |
| Baseline MAE (p50) | 66.2033 | for information |
| MAE difference vs baseline (team - baseline) | -22.4817 | for information |
| MAE ratio vs baseline (team / baseline) | 0.6604 | for information |

## Results by forecast horizon

| Hour | Steps | Rows | MAE | Pinball p10/p90 mean | Coverage | Brier spikes | Brier direction |
|---|---|---|---|---|---|---|---|
| 1 | 1-4 | 2688 | 42.0046 | 12.1830 | 0.8240 | 0.0901 | 0.4010 |
| 2 | 5-8 | 2684 | 42.8831 | 12.2724 | 0.8193 | 0.0917 | 0.4119 |
| 3 | 9-12 | 2680 | 43.4001 | 12.4071 | 0.8149 | 0.0932 | 0.4175 |
| 4 | 13-16 | 2676 | 43.5687 | 12.5222 | 0.8090 | 0.0944 | 0.4165 |
| 5 | 17-20 | 2672 | 43.7388 | 12.5885 | 0.8076 | 0.0954 | 0.4173 |
| 6 | 21-24 | 2668 | 43.9762 | 12.6528 | 0.8028 | 0.0961 | 0.4207 |
| 7 | 25-28 | 2664 | 44.1095 | 12.7367 | 0.8026 | 0.0966 | 0.4220 |
| 8 | 29-32 | 2660 | 44.2675 | 12.7605 | 0.7985 | 0.0976 | 0.4248 |
| 9 | 33-36 | 2656 | 44.5715 | 12.7408 | 0.7986 | 0.0985 | 0.4293 |
| 10 | 37-40 | 2652 | 44.7284 | 12.8391 | 0.8020 | 0.0991 | 0.4312 |

Lower is better for MAE, pinball loss and Brier scores. Points are calculated with `score.py --points` after all submissions have been scored.
