# Taiwan FYP Core — Regional Vote-Share Forecasting

Real-data implementation of the FYP guide (sections 4.1, 6, 10–16, Demo 6) for **Taiwan only**.

## Unit & target

- **Unit**: township/district × election year (`country=TW`)
- **Primary target**: `target_vote_share` = DPP presidential candidate vote share
- **Secondary**: `swing` = V(t) − V(t−1)
- **Years modeled**: 2012, 2016, 2020, 2024 (2008 used only for lags)

## Feature mapping (section 16 schema)

| Schema column | Taiwan source | Notes |
|---------------|---------------|-------|
| country | constant `TW` | |
| region | `縣市名+鄉鎮市區名` | e.g. `臺北市松山區` |
| year | election year | 2012–2024 |
| prev_vote_share | CEC prior presidential DPP share | lag 4 years |
| prev_turnout | CEC `elprof` ballots/eligible | prior election |
| unemployment | **NaN** | no consistent township series — **excluded from models** |
| median_income | 綜所稅鄉鎮 `平均數` (mean taxable income, 千元) | proxy for median; ROC years 105/109/112 |
| urbanization | min-max of log1p(人口密度) | density proxy in [0,1] |
| education_rate | ODRP024 戶長專科以上占比 | ROC 108→2012/2016; 112→2020/2024 |
| poll_index | **NaN** | no township poll series — **excluded**; do not invent |
| target_vote_share | CEC DPP presidential share | |

Extra columns kept: `prev_swing`, `swing`, `log_people`, `geo_region`, vote counts, ROC year tags.

## Missing / N/A

- **Public opinion / poll_index**: omitted (no consistent township-level polls).
- **Unemployment**: omitted.
- **TEDS / party ID / identity**: not in this ecological model.

## Methods

- Models: Previous-election baseline, Ridge (+StandardScaler), Random Forest, HistGradientBoosting
- Evaluation: **out-of-time only** (train years < test year); never random 80/20 as main eval
- Ablation: History; History+Economy; History+Demographics; History+Economy+Demo; All_available
- Importance: RF impurity importance, train on years < 2024
- Conformal: split conformal (Demo 6), α=0.10; calibration = last year before test

## National DPP share check

 year  national_dpp     n
 2012      0.461576 355.0
 2016      0.561234 368.0
 2020      0.571307 368.0
 2024      0.400503 368.0

## Backtest MAE (from this run)

```
test_year           2016    2020    2024
model                                   
HistGB            0.0570  0.0726  0.2104
PreviousElection  0.0900  0.0220  0.1487
RandomForest      0.0525  0.0694  0.2096
Ridge             0.0491  0.0833  0.2379
```

## Ablation ranking (2024 holdout)

            features    MAE   RMSE      R2
History+Economy+Demo 0.2096 0.2172 -2.0520
       All_available 0.2096 0.2172 -2.0520
     History+Economy 0.2106 0.2179 -2.0712
History+Demographics 0.2172 0.2238 -2.2391
             History 0.2191 0.2252 -2.2797

## Top features (RF, train < 2024)

        feature  importance  test_year country
prev_vote_share    0.956103       2024      TW
     prev_swing    0.016868       2024      TW
   prev_turnout    0.009755       2024      TW
  median_income    0.009300       2024      TW
   urbanization    0.002960       2024      TW
 education_rate    0.002625       2024      TW
     log_people    0.002389       2024      TW

## Conformal (2024, α=0.10)

{
  "calibration_year": 2020,
  "test_year": 2024,
  "alpha": 0.1,
  "quantile": 0.10840270304544689,
  "coverage": 0.008152173913043478,
  "avg_width": 0.21680540609089374,
  "n_train": 598,
  "n_calib": 368,
  "n_test": 368,
  "target_coverage": 0.9
}

## Caveats

1. **Ecological fallacy**: township correlations ≠ individual voter behavior.
2. **No TEDS party ID / 統獨** — literature often finds those stronger than socioeconomics.
3. **2024 three-candidate race** (DPP/KMT/TPP) breaks share levels vs 2012–2020 two-bloc contests; baseline lag can beat ML on MAE.
4. Income is **mean** taxable income, not true median household income.
5. Urbanization is a **density transform**, not an official urban classification.
6. Education rate is **household-head** college+ share (ODRP024), not population 15+ attainment.
7. Geographic boundary / name changes may leave a few unmatched townships.

## Reproduce

```bash
/workspace/taiwan-election-data/.venv/bin/python \
  /workspace/taiwan-election-data/fyp_taiwan/scripts/run_taiwan_fyp_core.py
```

## Outputs

- `harmonized_tw_regions.csv`
- `backtest_results.csv`
- `ablation_results.csv`
- `feature_importance.csv`
- `conformal_intervals_2024.csv`
- `conformal_meta_2024.json`
- `charts/*.png`


## Conformal coverage failure (important)

Empirical coverage on **2024** was **~0.8%** vs target 90% (α=0.10). Calibration used **2020** residuals (q≈0.108). The 2024 three-candidate race shifted DPP township shares downward by ~0.15 on average, so absolute residuals far exceed the 2020-calibrated quantile. This is a central **RQ4** finding: naive split conformal **does not** transfer under strong temporal distribution shift. Do not treat the 2024 intervals as reliable uncertainty bands without adaptive/reweighted conformal or swing-target modeling.

## Swing (ΔV) follow-up

Primary-target swing experiments live in **`../fyp_taiwan_swing/`**:

- Target: `ΔV = V(t) − V(t−1)` (DPP share)
- Baselines: `ZeroSwing` (ΔV̂=0) and `PreviousSwing` (ΔV̂=prev_swing)
- Same out-of-time / ablation / RF importance / conformal α=0.10 protocol
- Key contrast: level RF importance is ~96% `prev_vote_share`; swing RF redistributes toward `median_income` and `prev_swing` (see that folder’s README)

Reproduce: `fyp_taiwan_swing/scripts/run_taiwan_swing.py`
