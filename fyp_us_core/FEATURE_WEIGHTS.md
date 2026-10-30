# US Family A — Feature weights (model dependence, not causal)

Not causal. No county `poll_index` (Guide gap — not invented).

## Ridge standardized coefficients (signed, train < 2024)

- **Dem share two elections ago** (`prev2_dem_two_party_share`): weight=0.02805, |w| share=13.3%
- **GOP share two elections ago** (`prev2_gop_two_party_share`): weight=-0.02805, |w| share=13.3%
- **Previous GOP two-party share** (`prev_gop_two_party_share`): weight=-0.02769, |w| share=13.2%
- **Previous Dem–GOP margin** (`prev_dem_gop_margin`): weight=0.02769, |w| share=13.2%
- **Previous Dem two-party share** (`prev_dem_two_party_share`): weight=0.02769, |w| share=13.2%
- **Log-votes change** (`log_votes_change_1`): weight=0.02214, |w| share=10.5%
- **Bachelor's+ rate (ERS wave)** (`education_rate`): weight=0.01309, |w| share=6.2%
- **Abs Dem previous swing** (`abs_dem_change_1`): weight=-0.006409, |w| share=3.0%

## Random Forest MDI impurity (train < 2024)

- **Previous GOP two-party share** (`prev_gop_two_party_share`): weight=0.5605, |w| share=56.1%
- **Previous Dem two-party share** (`prev_dem_two_party_share`): weight=0.1867, |w| share=18.7%
- **Previous Dem–GOP margin** (`prev_dem_gop_margin`): weight=0.1766, |w| share=17.7%
- **Log-votes change** (`log_votes_change_1`): weight=0.03217, |w| share=3.2%
- **Bachelor's+ rate (ERS wave)** (`education_rate`): weight=0.008741, |w| share=0.9%
- **Unemployment rate** (`unemployment`): weight=0.007248, |w| share=0.7%
- **Abs Dem previous swing** (`abs_dem_change_1`): weight=0.005235, |w| share=0.5%
- **Log median income** (`log_median_income`): weight=0.004967, |w| share=0.5%

## Permutation MAE ↑ on 2024 subsample


### Ridge

- **Dem share two elections ago** (`prev2_dem_two_party_share`): weight=0.01476, |w| share=20.9%
- **GOP share two elections ago** (`prev2_gop_two_party_share`): weight=0.01476, |w| share=20.9%
- **Previous GOP two-party share** (`prev_gop_two_party_share`): weight=0.01262, |w| share=17.9%
- **Previous Dem–GOP margin** (`prev_dem_gop_margin`): weight=0.01262, |w| share=17.9%
- **Previous Dem two-party share** (`prev_dem_two_party_share`): weight=0.01262, |w| share=17.9%
- **Bachelor's+ rate (ERS wave)** (`education_rate`): weight=0.001091, |w| share=1.5%
- **Log population density (ACS)** (`log_density`): weight=0.0009353, |w| share=1.3%
- **Previous log total votes** (`prev_log_total_votes`): weight=-0.0003609, |w| share=0.5%

### Random Forest

- **Previous GOP two-party share** (`prev_gop_two_party_share`): weight=0.06226, |w| share=68.0%
- **Previous Dem two-party share** (`prev_dem_two_party_share`): weight=0.01209, |w| share=13.2%
- **Previous Dem–GOP margin** (`prev_dem_gop_margin`): weight=0.01088, |w| share=11.9%
- **GOP share two elections ago** (`prev2_gop_two_party_share`): weight=-0.002237, |w| share=2.4%
- **Dem share two elections ago** (`prev2_dem_two_party_share`): weight=-0.001958, |w| share=2.1%
- **Log-votes change** (`log_votes_change_1`): weight=-0.0006238, |w| share=0.7%
- **Log median income** (`log_median_income`): weight=-0.0005507, |w| share=0.6%
- **Log population density (ACS)** (`log_density`): weight=0.000399, |w| share=0.4%

### HistGradientBoosting

- **Previous GOP two-party share** (`prev_gop_two_party_share`): weight=0.05591, |w| share=55.9%
- **Previous Dem two-party share** (`prev_dem_two_party_share`): weight=0.03729, |w| share=37.3%
- **Dem share two elections ago** (`prev2_dem_two_party_share`): weight=-0.001992, |w| share=2.0%
- **Log population density (ACS)** (`log_density`): weight=0.001499, |w| share=1.5%
- **Log-votes change** (`log_votes_change_1`): weight=-0.001382, |w| share=1.4%
- **GOP share two elections ago** (`prev2_gop_two_party_share`): weight=-0.0006095, |w| share=0.6%
- **Previous log total votes** (`prev_log_total_votes`): weight=0.0003601, |w| share=0.4%
- **Dem previous swing** (`dem_change_1`): weight=-0.0003326, |w| share=0.3%

## Files
- `feature_weights_all.csv`
- `feature_weights_ridge_signed.csv`
- `feature_weights_rf_impurity.csv`
- `feature_weights_permutation.csv`


## Socio / polls gap
- Census ACS API requires key.
- ERS/BLS bulk blocked/404 from this host.
- Panel: yearly unemployment; income/edu/density wave-static.
- **No county poll_index**.
