# US Family A — County Presidential Core

| Item | Value |
|---|---|
| Unit | US county (~3100) |
| Target | Dem **two-party** share |
| Test years | 2012, 2016, 2020, 2024 |

## Mean OOT MAE

| Model | MAE | RMSE |
|---|---:|---:|
| Previous-election baseline | 0.0345 | 0.0432 |
| Random Forest | 0.0529 | 0.0626 |
| Ridge | 0.0539 | 0.0631 |
| HistGradientBoosting | 0.0554 | 0.0654 |

## Conformal coverage (target 0.90)

| Model | Mean coverage |
|---|---:|
| HistGradientBoosting | 0.8810 |
| Previous-election baseline | 0.8343 |
| Random Forest | 0.8735 |
| Ridge | 0.9099 |

## Feature weights
`FEATURE_WEIGHTS.md` + `results/feature_weights_*.csv`

## Gaps
- No county poll_index (not invented).
- ACS key / ERS-BLS bulk blocked; yearly unemployment kept; income/edu/density wave-static.

**model dependence ≠ causal**
