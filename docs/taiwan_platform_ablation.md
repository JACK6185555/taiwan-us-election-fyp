# Taiwan platform (政綱) ablation — 2026-10-27

## Setup
- **Unit**: county × presidential election, expanding-window OOT (same as poll ablation).
- **Test years (ML)**: 2012, 2016, 2020, 2024 (2004/2008 skipped: lag features incomplete).
- **Target**: `target_DPP_vote_share`.
- **History**: lag + change features from `train_taiwan_forecast.py`.
- **Poll**: 60-day national DPP support share + swing vs previous national DPP (same as `run_taiwan_poll_ablation.py`).
- **Platform**: from `inputs/taiwan_platforms_long.csv`, **only** `pre_election_flag==1`.
  - Camps: DPP nominee vs final KMT rival (2016 = 朱立倫, not 洪秀柱).
  - **Coverage-normalised** issue shares within each camp → `plat_diff_*` = DPP share − rival share for 兩岸/國防, 經濟/產業, 能源/環境, 社福/長照/年金, 司法/政治改革, 教育/文化.
  - Keyword diffs (台獨/九二/核能綠能/長照).
  - Interactions: `plat_diff_* × (prev_DPP − 0.5)` so national text meets county lean.
  - Raw row counts **not** used as model features (only `plat_coverage_balance`).
- **Leakage**: features use that year’s pre-election text only; no vote outcomes; OOT expanding window.

## Coverage (pre_election rows)
| year | DPP n | rival n | balance |
|---|---|---|---|
| 2000 | 5 | 5 | 0.50 |
| 2004 | 0 | 0 | — (NaN features) |
| 2008 | 39 | 93 | 0.30 |
| 2012 | 22 | 24 | 0.48 |
| 2016 | 12 | 11 | 0.48 |
| 2020 | 19 | 24 | 0.44 |
| 2024 | 47 | 21 | 0.31 |

## Mean MAE (2012–2024)

| Experiment | Baseline | Ridge | RF | HGB |
|---|---|---|---|---|
| History | 0.0793 | 0.1776 | 0.1041 | 0.1257 |
| History+Platform | 0.0793 | 0.1113 | 0.1092 | 0.1103 |
| History+Poll | 0.0793 | **0.0611** | 0.1036 | 0.1235 |
| History+Poll+Platform | 0.0793 | 0.1064 | 0.1097 | 0.1103 |
| UniformPollSwing | **0.0266** | — | — | — |

## Per-year MAE (Ridge)

| | 2012 | 2016 | 2020 | 2024 |
|---|---|---|---|---|
| History | 0.0433 | 0.1256 | 0.3956 | 0.1459 |
| History+Platform | 0.1705 | 0.0741 | 0.0442 | 0.1563 |
| History+Poll | 0.0433 | 0.0796 | 0.0555 | 0.0661 |
| History+Poll+Platform | 0.1705 | 0.0716 | 0.0444 | 0.1393 |

## Interpretation (honest)
1. **Platform alone** lowers History-Ridge mean (0.18→0.11) but **hurts 2012** badly (0.04→0.17); gains in 2016/2020 look unstable with 4 test years.
2. **Poll still dominates** among learned models (History+Poll Ridge 0.061). Adding Platform on top **worsens** Ridge (0.061→0.106)—national text features likely collinear / overfit with tiny T.
3. **UniformPollSwing (0.027)** remains the strongest simple reference; platforms do not beat it.
4. Caveats: secondary/unverified text; uneven coverage; 2004 empty after filter; national features only weakly identified at county level even with lean interactions.
5. Treat as **exploratory**; do not claim platforms improve forecasting until denser, verified, preferably local platforms exist.

## Outputs
- `scripts/run_taiwan_platform_ablation.py`
- `outputs/metrics/taiwan_platform_ablation_metrics.csv`
- `outputs/metrics/taiwan_platform_year_features.csv`
- `outputs/metrics/taiwan_platform_ablation_summary.json`
