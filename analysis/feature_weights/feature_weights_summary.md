# Feature weights summary (recomputed)

**Recomputed:** 2026-10-30 14:24 Asia/Hong_Kong (UTC+8)

**Interpretation:** weights = model dependence in out-of-time folds — **not causal**, **not** reasons a party wins.

Aggregation TW: **mean across OOT folds** 2012, 2016, 2020, 2024.
US: Ridge/RF coef on train<2024; permutation on 2024 subsample (n≤800) — same protocol as `us_core_weights_conformal.py`.

## Core specification electoral-only mean OOT MAE (side-check)

                     model      mae     rmse
Previous-election baseline 0.079265 0.081417
             Random Forest 0.103277 0.116403
      HistGradientBoosting 0.118865 0.126775
                     Ridge 0.178580 0.185480

## Core+ (enriched) enriched mean OOT MAE

                     model      mae     rmse
Previous-election baseline 0.079265 0.081417
             Random Forest 0.104918 0.117369
      HistGradientBoosting 0.118867 0.126579
                     Ridge 0.142500 0.147859

## A+ + education Extension mean OOT MAE (NOT locked Core specification)

                     model      mae     rmse
Previous-election baseline 0.079265 0.081417
             Random Forest 0.105594 0.117527
      HistGradientBoosting 0.119513 0.128222
                     Ridge 0.142188 0.147653

## Top Ridge standardized coefficients (electoral)

                       feature                 feature_label  coef_mean_signed  coef_mean_abs  rank_by_abs
          prev2_DPP_vote_share   DPP share two elections ago          0.024433       0.024433            1
           prev_KMT_vote_share            Previous KMT share         -0.020908       0.020908            2
           prev_DPP_KMT_margin       Previous DPP-KMT margin          0.020897       0.020897            3
           prev_DPP_vote_share            Previous DPP share          0.020735       0.020735            4
         prev_DPP_Other_margin     Previous DPP-other margin          0.019650       0.019650            5
                Other_change_1          Other previous swing          0.010615       0.010615            6
        prev2_Other_vote_share Other share two elections ago         -0.010435       0.010435            7
                  KMT_change_1            KMT previous swing         -0.009052       0.009052            8
         prev_Other_vote_share          Previous other share          0.008733       0.008733            9
previous_party_change_abs_mean    Mean party swing magnitude         -0.008395       0.008395           10

## Top RF permutation (electoral)

                       feature                 feature_label  permutation_mae_increase_mean  impurity_mean  rank
          prev2_DPP_vote_share   DPP share two elections ago                       0.020047       0.372946     1
           prev_KMT_vote_share            Previous KMT share                       0.007709       0.204174     2
           prev_DPP_KMT_margin       Previous DPP-KMT margin                       0.003290       0.069496     3
         prev_DPP_Other_margin     Previous DPP-other margin                       0.001623       0.024635     4
           prev_DPP_vote_share            Previous DPP share                       0.001496       0.079345     5
     previous_party_change_std    Dispersion of party swings                       0.000318       0.043828     6
previous_party_change_abs_mean    Mean party swing magnitude                       0.000079       0.049396     7
                  KMT_change_1            KMT previous swing                       0.000068       0.040249     8
                  DPP_change_1            DPP previous swing                       0.000026       0.001340     9
        prev2_Other_vote_share Other share two elections ago                       0.000000       0.041976    10

## Top HGB permutation (electoral)

                       feature               feature_label  permutation_mae_increase_mean  rank
          prev2_DPP_vote_share DPP share two elections ago                       0.014707     1
           prev_DPP_vote_share          Previous DPP share                       0.009838     2
           prev_KMT_vote_share          Previous KMT share                       0.008446     3
                  prev_turnout            Previous turnout                       0.001258     4
previous_party_change_abs_mean  Mean party swing magnitude                       0.000339     5
       DPP_KMT_margin_change_1       DPP-KMT margin change                       0.000224     6
         prev_DPP_Other_margin   Previous DPP-other margin                       0.000090     7
     previous_party_change_std  Dispersion of party swings                       0.000034     8
                  DPP_change_1          DPP previous swing                       0.000005     9
                Other_change_1        Other previous swing                       0.000000    10

## Top Ridge (A+ enriched)

               feature                           feature_label  coef_mean_signed  coef_mean_abs
  prev2_DPP_vote_share             DPP share two elections ago          0.023685       0.023685
 prev_DPP_Other_margin               Previous DPP-other margin          0.021500       0.021500
   prev_DPP_vote_share                      Previous DPP share          0.021321       0.021321
   prev_DPP_KMT_margin                 Previous DPP-KMT margin          0.020846       0.020846
   prev_KMT_vote_share                      Previous KMT share         -0.020254       0.020254
  prev2_KMT_vote_share             KMT share two elections ago         -0.011821       0.011821
       lag_both_id_pct      Lag national both ID % (ESC PROXY)         -0.010423       0.010423
  lag_taiwanese_id_pct Lag national Taiwanese ID % (ESC PROXY)          0.008855       0.008855
    lag_chinese_id_pct   Lag national Chinese ID % (ESC PROXY)         -0.006863       0.006863
prev2_Other_vote_share           Other share two elections ago         -0.006328       0.006328

## Top Ridge (A+ + education Extension)

                       feature                           feature_label  coef_mean_signed  coef_mean_abs
          prev2_DPP_vote_share             DPP share two elections ago          0.023629       0.023629
         prev_DPP_Other_margin               Previous DPP-other margin          0.021318       0.021318
           prev_DPP_vote_share                      Previous DPP share          0.021151       0.021151
           prev_DPP_KMT_margin                 Previous DPP-KMT margin          0.020684       0.020684
           prev_KMT_vote_share                      Previous KMT share         -0.020098       0.020098
          prev2_KMT_vote_share             KMT share two elections ago         -0.011732       0.011732
               lag_both_id_pct      Lag national both ID % (ESC PROXY)         -0.010457       0.010457
          lag_taiwanese_id_pct Lag national Taiwanese ID % (ESC PROXY)          0.008884       0.008884
            lag_chinese_id_pct   Lag national Chinese ID % (ESC PROXY)         -0.006884       0.006884
previous_party_change_abs_mean              Mean party swing magnitude         -0.006426       0.006426
        prev2_Other_vote_share           Other share two elections ago         -0.006384       0.006384
                Other_change_1                    Other previous swing          0.006039       0.006039

## US Core side-check mean OOT MAE

                     model      mae     rmse
Previous-election baseline 0.034484 0.043151
             Random Forest 0.052480 0.062087
                     Ridge 0.053872 0.063107
      HistGradientBoosting 0.056068 0.066161

## US Ridge permutation top (2024 subsample)

                  feature                feature_label    weight
prev2_dem_two_party_share  Dem share two elections ago  0.014755
prev2_gop_two_party_share  GOP share two elections ago  0.014755
 prev_gop_two_party_share Previous GOP two-party share  0.012621
      prev_dem_gop_margin      Previous Dem–GOP margin  0.012621
 prev_dem_two_party_share Previous Dem two-party share  0.012621
           education_rate  Bachelor's+ rate (ERS wave)  0.001091
              log_density Log population density (ACS)  0.000935
     prev_log_total_votes     Previous log total votes -0.000361
         abs_dem_change_1       Abs Dem previous swing  0.000190
             dem_change_1           Dem previous swing -0.000103

## Extension national associations (NOT Core specification) — refreshed

                          layer country                       feature                                                feature_label                  target  n_years  pearson_r  ridge_std_coef      mae                                                                  method                                                            interpretation                              years
Extension_national_NOT_Core      US      dem_two_party_share_poll        538 national poll Dem two-party (election-year as-of) nat_dem_two_party_share        7   0.412952        0.006602 0.011635             national series only; Ridge(alpha=0.1) after StandardScaler  correlational / predictive association — NOT causal; NOT county Core specification 2000,2004,2008,2012,2016,2020,2024
Extension_national_NOT_Core      TW     poll_dpp_two_party_approx TW national poll DPP/(DPP+KMT) crude mean (Wikipedia tables)               dpp_share        3   0.141669        0.010728 0.071717 national series only; crude poll average by year×side; Ridge(alpha=0.1) correlational — NOT causal; NOT county Core specification; target mapping heuristic                     2016,2020,2024
Extension_national_NOT_Core      TW x_sentiment_gap_dpp_minus_kmt                 X camp mean sentiment DPP−KMT (pilot sample)               dpp_share        2   1.000000             NaN      NaN                            pilot X API sample aggregates; national only              correlational / demo only — NOT causal; NOT Core specification; tiny n                          2020,2024
Extension_national_NOT_Core      US x_sentiment_gap_dem_minus_gop                 X camp mean sentiment DEM−GOP (pilot sample) nat_dem_two_party_share        2   1.000000             NaN      NaN                            pilot X API sample aggregates; national only              correlational / demo only — NOT causal; NOT Core specification; tiny n                          2020,2024


## Files
- `taiwan_county_*.csv`, `taiwan_county_plus_*.csv`, `taiwan_county_plus_edu_*.csv`
- `us_county_feature_weights_*.csv`
- `extension_national_feature_associations.csv`
- `tw_us_feature_weights_comparison.csv`
- `docs/特徵權重說明.md`, `docs/權重重算說明.md`

