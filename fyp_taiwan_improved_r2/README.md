# Taiwan FYP — 第二輪改進（fyp_taiwan_improved_r2）

建基於 `fyp_taiwan_improved/`。誠實指標；唔造假縣市黨性。

## 本輪做了什麼

1. **更強空間特徵**：同縣 LOO `spatial_lag_*` + 距離無關 `county_mean_*`；ablation：NoSpatial / LOO / CountyMean / Both（總統 DPP／KMT 嘅 swing **同** demeaned）。
2. **一致 pipeline**：總統＋區域立委 DPP／KMT（及有滯後嘅 TPP）都跑 swing／demeaned／level。
3. **Demeaned → Level**：預測相對全國殘差後加返 `prev_nat`／`ESC_nat`／`oracle_nat`（oracle 標明洩漏）。
4. **不確定性**：喺 **demeaned** 上做 split-conformal + leave-one-election 殘差 q90。
5. **ESC 縣市級**：仍不可得 → `proxies/ESC_COUNTY_UNAVAILABLE.md`；Ridge 加顯式 year FE（測試年 FE=0）。

## 2024 before/after

| race | party | target | baseline | base MAE | r2 model | r2 MAE | Δ | win? |
|------|-------|--------|----------|----------|----------|--------|---|------|
| legislative_district | DPP | demeaned | PreviousDemeaned | 0.1044 | Ridge | 0.1348 | -0.0304 | False |
| legislative_district | DPP | level | PreviousShare | 0.1048 | Ridge | 0.1300 | -0.0252 | False |
| legislative_district | DPP | swing | ZeroSwing | 0.1048 | HistGB | 0.1357 | -0.0309 | False |
| legislative_district | KMT | demeaned | PreviousDemeaned | 0.1330 | Ridge | 0.1371 | -0.0041 | False |
| legislative_district | KMT | level | PreviousShare | 0.1330 | Ridge | 0.1348 | -0.0019 | False |
| legislative_district | KMT | swing | ZeroSwing | 0.1330 | RandomForest | 0.1328 | +0.0002 | True |
| legislative_district | TPP | demeaned | PreviousDemeaned | 0.0465 | HistGB | 0.0483 | -0.0017 | False |
| legislative_district | TPP | level | PreviousShare | 0.0389 | HistGB | 0.0415 | -0.0026 | False |
| legislative_district | TPP | swing | ZeroSwing | 0.0389 | nan | nan | +nan | None |
| presidential | DPP | demeaned | PreviousDemeaned | 0.0365 | RandomForest | 0.0410 | -0.0045 | False |
| presidential | DPP | level | PreviousShare | 0.1487 | Ridge | 0.0713 | +0.0775 | True |
| presidential | DPP | swing | ZeroSwing | 0.1487 | Ridge | 0.0403 | +0.1085 | True |
| presidential | KMT | demeaned | PreviousDemeaned | 0.0247 | RandomForest | 0.0420 | -0.0173 | False |
| presidential | KMT | level | PreviousShare | 0.0598 | RandomForest | 0.0548 | +0.0050 | True |
| presidential | KMT | swing | ZeroSwing | 0.0598 | RandomForest | 0.0845 | -0.0247 | False |
| presidential | TPP | demeaned | ZeroDemeaned | 0.0389 | nan | nan | +nan | None |

## 空間 ablation（總統 RF）

- Pres DPP swing RF: NoSpatial=0.1706, Both=0.1656, Δ(NoSpat-Both)=+0.0050
- Pres DPP demeaned RF: NoSpatial=0.0405, Both=0.0410, Δ(NoSpat-Both)=-0.0005
- Pres KMT swing RF: NoSpatial=0.0785, Both=0.0845, Δ(NoSpat-Both)=-0.0060
- Pres KMT demeaned RF: NoSpatial=0.0441, Both=0.0420, Δ(NoSpat-Both)=+0.0021

## Demeaned→Level vs PreviousShare

- DPP PreviousShare: MAE_level=0.1487 (baseline)
- DPP Ridge_demeaned+prev_nat: MAE_level=0.0683 (honest: add previous election national mean)
- DPP Ridge_demeaned+esc_nat: MAE_level=0.0626 (honest PROXY: prev_nat + ESC pid delta)
- DPP Ridge_demeaned+oracle_nat: MAE_level=0.1056 (ORACLE leakage: true test-year national — ranking skill only)
- DPP RandomForest_demeaned+prev_nat: MAE_level=0.1865 (honest: add previous election national mean)
- DPP RandomForest_demeaned+esc_nat: MAE_level=0.1795 (honest PROXY: prev_nat + ESC pid delta)
- DPP RandomForest_demeaned+oracle_nat: MAE_level=0.0410 (ORACLE leakage: true test-year national — ranking skill only)
- KMT PreviousShare: MAE_level=0.0598 (baseline)
- KMT Ridge_demeaned+prev_nat: MAE_level=0.1604 (honest: add previous election national mean)
- KMT Ridge_demeaned+esc_nat: MAE_level=0.1344 (honest PROXY: prev_nat + ESC pid delta)
- KMT Ridge_demeaned+oracle_nat: MAE_level=0.1092 (ORACLE leakage: true test-year national — ranking skill only)
- KMT RandomForest_demeaned+prev_nat: MAE_level=0.0255 (honest: add previous election national mean)
- KMT RandomForest_demeaned+esc_nat: MAE_level=0.0268 (honest PROXY: prev_nat + ESC pid delta)
- KMT RandomForest_demeaned+oracle_nat: MAE_level=0.0420 (ORACLE leakage: true test-year national — ranking skill only)

## Conformal（demeaned）

- presidential DPP demeaned: split cover=0.848 (halfwidth=0.1405); yearQ cover=0.8478260869565217 (halfwidth=0.14050647398031316)
- presidential KMT demeaned: split cover=1.000 (halfwidth=0.2306); yearQ cover=1.0 (halfwidth=0.23063927900004452)
- legislative_district DPP demeaned: split cover=0.957 (halfwidth=0.3537); yearQ cover=0.9565217391304348 (halfwidth=0.35366851992181)
- legislative_district KMT demeaned: split cover=0.750 (halfwidth=0.1828); yearQ cover=0.75 (halfwidth=0.18284546128028664)

## 重跑

```bash
cd /workspace/taiwan-election-data
.venv/bin/python fyp_taiwan_improved_r2/scripts/run_taiwan_improved_r2.py
```

## 仍然欠

- TEDS 微資料／縣市黨性
- 真實共邊 shapefile GCN（見 `fyp_taiwan_spatial_gnn/` 用同縣圖近似）
- 跨國 US transfer
