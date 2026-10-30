# Taiwan FYP Core — Swing (ΔV) Forecasting

Primary target: **ΔV(i,t) = V(i,t) − V(i,t−1)** where V = DPP presidential share.
Panel: `../fyp_taiwan/harmonized_tw_regions.csv` (same township × year).

## Baselines (documented)

| Model | Definition |
|-------|------------|
| **ZeroSwing** | Predict ΔV̂ = 0 (no change). Primary naive baseline for swing. Equivalent errors to level PreviousElection. |
| **PreviousSwing** | Predict ΔV̂(t) = prev_swing = ΔV(t−1). FYP “PreviousElection” analogue for swings (persistence of change). |
| Ridge / RandomForest / HistGB | Supervised models on available features |

## Features (no invented unemployment / poll)

prev_vote_share, prev_swing, prev_turnout, median_income, urbanization, education_rate, log_people

`prev_swing` missing on earliest train rows → median-imputed from training fold only.

## Swing distribution (this panel)

```
 year    mean    std  count
 2012  0.0411 0.0134    245
 2016  0.0899 0.0229    353
 2020  0.0002 0.0278    368
 2024 -0.1487 0.0415    368
```

## Methods

- Out-of-time only (train years < test year)
- Ablation: History / +Economy / +Demographics / combined
- RF impurity importance, train < 2024
- Split conformal α=0.10; calibration = last year before test; intervals on ΔV clipped to [-1,1]

## Backtest MAE (swing)

```
test_year        2016    2020    2024
model                                
HistGB         0.0495  0.0900  0.2134
PreviousSwing  0.0624  0.0893  0.1490
RandomForest   0.0493  0.0847  0.2044
Ridge          0.0487  0.0836  0.2381
ZeroSwing      0.0900  0.0220  0.1487
```

## Ablation ranking (2024 holdout, swing)

            features    MAE   RMSE       R2
     History+Economy 0.2000 0.2089 -24.4463
History+Economy+Demo 0.2043 0.2151 -25.9942
       All_available 0.2044 0.2152 -26.0057
             History 0.2162 0.2214 -27.6009
History+Demographics 0.2219 0.2277 -29.2370

## Top features (RF swing, train < 2024)

        feature  importance  test_year country target
  median_income    0.419843       2024      TW  swing
     prev_swing    0.296804       2024      TW  swing
prev_vote_share    0.102964       2024      TW  swing
   urbanization    0.051415       2024      TW  swing
 education_rate    0.048324       2024      TW  swing
   prev_turnout    0.045559       2024      TW  swing
     log_people    0.035089       2024      TW  swing

## Conformal (2024 swing, α=0.10)

{
  "target": "swing",
  "calibration_year": 2020,
  "test_year": 2024,
  "alpha": 0.1,
  "quantile": 0.11252192537983897,
  "coverage": 0.002717391304347826,
  "avg_width": 0.22504385075967798,
  "n_train": 598,
  "n_calib": 368,
  "n_test": 368,
  "target_coverage": 0.9,
  "mean_actual_swing": -0.14872504338071132,
  "mean_pred_swing": 0.07604373100063153
}


## Comparison: swing vs level prediction

Level-model MAE (from `fyp_taiwan/backtest_results.csv`):

```
test_year           2016    2020    2024
model                                   
HistGB            0.0570  0.0726  0.2104
PreviousElection  0.0900  0.0220  0.1487
RandomForest      0.0525  0.0694  0.2096
Ridge             0.0491  0.0833  0.2379
```

Key identities / findings:

1. **ZeroSwing MAE on ΔV equals PreviousElection MAE on V** (same absolute errors: |V−V_prev| = |ΔV−0|).
2. Predicting **level** is dominated by `prev_vote_share` (importance ~0.96). Predicting **swing** removes that mechanical lag; importance redistributes toward `prev_swing`, demographics, and income — or remains weak if ΔV is mostly a national shock.
3. **2024** national DPP drop (~−0.15) is a large common shock: township features explain little of the *level shift*; ML on swing still struggles vs ZeroSwing when the mean swing is far from historical train means.
4. If ML beats ZeroSwing on swing MAE, features add value beyond persistence; if not, ecological covariates are insufficient for out-of-time ΔV.

## Caveats

1. Ecological / township model — not individual TEDS.
2. 2024 three-candidate race creates large negative mean swing vs 2012–2020.
3. Naive conformal calibrated on 2020 may undercover 2024 if residual scale shifts.
4. Same data gaps as level model (no unemployment, no poll_index).

## Reproduce

```bash
/workspace/taiwan-election-data/.venv/bin/python \
  /workspace/taiwan-election-data/fyp_taiwan_swing/scripts/run_taiwan_swing.py
```

## Outputs

- `backtest_results.csv`
- `ablation_results.csv`
- `feature_importance.csv`
- `conformal_intervals_2024.csv`
- `conformal_meta_2024.json`
- `charts/*.png`
