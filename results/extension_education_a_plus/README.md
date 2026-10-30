# TW education_rate Extension A+ (NOT Family A)

**Built:** 2026-10-30 14:05 Asia/Hong_Kong (UTC+8)

## Definition
- `lag_education_rate` = MOI 15歲以上大專以上比率(%)，取最近一個 **ce_year < 選舉年** 的縣市值。
- 來源：data.gov.tw dataset 14219 / MOI micst `c0110405`（`raw/moi/education_rate/moi_c0110405.csv`）。
- 對齊美國 ERS `education_rate`（人口學歷），**唔係**教育長政治傾向。

## Suites (OOT leave-one-election 2012/2016/2020/2024)
1. `family_a_electoral_only_rerun` — 對照重跑（唔取代鎖定主表）
2. `family_a_plus_enriched_rerun` — 既有 A+ 特徵重跑
3. `extension_a_plus_education` — A+ + `lag_education_rate`

## Mean OOT MAE (this run)
```
                        suite                      model      mae     rmse
   extension_a_plus_education Previous-election baseline 0.079265 0.081417
   extension_a_plus_education              Random Forest 0.105945 0.118248
   extension_a_plus_education       HistGradientBoosting 0.119513 0.128222
   extension_a_plus_education                      Ridge 0.142188 0.147653
family_a_electoral_only_rerun Previous-election baseline 0.079265 0.081417
family_a_electoral_only_rerun              Random Forest 0.103398 0.116545
family_a_electoral_only_rerun       HistGradientBoosting 0.118865 0.126775
family_a_electoral_only_rerun                      Ridge 0.178580 0.185480
 family_a_plus_enriched_rerun Previous-election baseline 0.079265 0.081417
 family_a_plus_enriched_rerun              Random Forest 0.104359 0.117164
 family_a_plus_enriched_rerun       HistGradientBoosting 0.118867 0.126579
 family_a_plus_enriched_rerun                      Ridge 0.142500 0.147859
```

## Locked published reference (unchanged)
- Family A Prev MAE **0.0793**; Family A Ridge **0.1786**
- Family A+ Ridge **0.1425** (published enriched)

## Caveats
- 權重／係數**唔係**因果。
- 教育率滯後對早期選舉覆蓋較佳（MOI 自民國86年起）；金門／連江亦有列。
- **禁止**把本 Extension MAE 寫入 Family A 主表。
