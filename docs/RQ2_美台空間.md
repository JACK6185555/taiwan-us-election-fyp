# RQ2 美台空間對照（縣市／縣級總統票）

> 主表分開：**臺灣** `fyp_taiwan_spatial_gnn/rq2_county/`；**美國** `fyp_us_taiwan_transfer/rq2_us_county/`。  
> **唔好**把兩邊 MAE 寫入 Core specification／對方表。數字以各目錄 CSV 為準（唔發明）。  
> **老師／口試誠實框住（含 RQ3／RQ4）：** `docs/RQ2_RQ3_RQ4_誠實框住.md`

## 共同協議

| 項目 | 內容 |
|---|---|
| 測試年 | 2012／2016／2020／2024（expanding：`train year < test`） |
| Seed | 42；CPU |
| 模型族 | PreviousShare；NonSpatial Ridge／RF；（TW 另有 HGB）；SGC |
| 圖 | Queen 共邊 + k-NN + 距離衰減敏感度 |

## 臺灣（已完成）

- 單位：22 縣市；特徵 = Core specification 16 個歷史特徵；目標 DPP share  
- Queen：33 邊，平均度 3.0  
- **結論**：PreviousShare mean MAE **0.0793** 最好；最佳 SGC（dist_decay_bin + RF）**0.1029** 只略好過 NonSpatial RF 0.1043；trainable GCN／GAT mixed／多數更差。  
- 詳見 `docs/空間共邊GCN.md`、`rq2_county_mean_mae_dpp.csv`

## 美國（本輪完成 — SGC 頭條）

- 單位：**3111** counties；目標 Dem two-party `party_share_A`  
- 特徵：`prev_share_A`, `log_total_votes`, `unemployment`, `education_rate`, `median_income`, `log_density`  
- 幾何：plotly `geojson-counties-fips`（Census TIGER ZIP HTTP **403**）→ Queen **9072** 邊，平均度 **5.83**；k-NN k=5；dist-decay σ=80 km  
- **Trainable full-US GCN**：runtime 仍 **deferred**；**Census 2024 鄰接＋全縣 SGC＋Mid-Atlantic GCN-lite 已跑**（`docs/US_GCN_DEFERRED.md`）。舊 2012 部分數字見 `gcn_partial_note.json`

### Mean OOT MAE（2012–2024；`rq2_us_county_mean_mae.csv`）

| family | model | adjacency | mean MAE |
|---|---|---|---:|
| baseline | PreviousShare | none | **0.0345** |
| NonSpatial | Ridge | none | 0.0388 |
| NonSpatial | RandomForest | none | 0.0419 |
| SGC | SGC_L1+Ridge | queen | 0.0437 |
| SGC | SGC_L1+Ridge | knn | 0.0438 |
| SGC | SGC_L1+RF | queen | 0.0441 |
| SGC | SGC_L1+RF | knn | 0.0441 |

**結論（誠實）**：美國縣級 **空間 SGC 未贏** PreviousShare，亦未贏 NonSpatial Ridge／RF。時間慣性極強（PrevShare 0.0345）。距離衰減圖更差（~0.05）。

### NonSpatial 特徵權重（摘要）

Ridge mean |coef|：`prev_share_A` **0.130** ≫ education 0.015 ≫ 其餘 &lt;0.01。  
RF impurity：`prev_share_A` **0.928**。詳見 `feature_weights_ridge_summary.csv`／`feature_importance_rf_summary.csv`。

## 跨國一句

兩邊 RQ2：**PreviousShare 仍然難撼**；真共邊／SGC 係更誠實假設，但喺呢個 OOT 窗 **未穩定改善預測**。臺灣 N 細、美國 N 大，結論方向一致。

## 重跑

```bash
.venv/bin/python fyp_us_taiwan_transfer/scripts/run_rq2_us_county_sgc_only.py
.venv/bin/python fyp_us_taiwan_transfer/scripts/export_rq2_us_nonspatial_weights.py
```


## US 空間圖＋trainable（2026-10-31 更新）

- **官方鄰接：** Census County Adjacency **2024** → `data/us_county_adjacency/`（match **3103／3111＝99.74%**；CT 8 歷史縣因改 Planning Regions 唔入圖；HI 4 縣 degree-0）。詳見 `docs/US_GCN_DEFERRED.md`。  
- **全縣 SGC（Census queen）：** Prev **0.0345** 仍最好；SGC_L1+Ridge ≈**0.0438**（與舊 GeoJSON Queen ~0.0437 一致）。產物：`results/us_census_adjacency_pilot/`。  
- **全縣 trainable GCN：** 仍 **runtime DEFERRED**（已唔係「無鄰接數據」）。舊部分數字見 `gcn_partial_note.json`。  
- **區域 pilot：** Mid-Atlantic 150 縣 GCN-lite（無 PyG）— 同子集 Prev≈**0.0370** 最好；ResidualDenseGCN≈0.073；DenseGCN≈0.093。NE 67 縣舊 pilot 仍見 `results/us_gcn_pilot/`。
