# 全縣可訓練 US GCN：DEFERRED（＋ New England pilot）

**日期：** 2026-10-30（Asia/Hong_Kong, UTC+8）  
**標籤：** `us_gcn_pilot` ≠ 全縣可訓練 GCN；**唔好**同 Core specification／全縣 RQ2 MAE 混報。  
**綜合框住：** `docs/RQ2_RQ3_RQ4_誠實框住.md` §一（空間）。

## 點解全縣 trainable GCN deferred

- 全縣約 **3111** counties；Queen 邊約 **9072**；expanding OOT 四屆 × TrainableGCN／ResidualGCN × 多圖，CPU 上單次完整跑曾被 runtime 截斷。
- 既有部分數字（唔當定稿）：`fyp_us_taiwan_transfer/rq2_us_county/gcn_partial_note.json`  
  - 2012：TrainableGCN≈0.086、ResidualGCN≈0.072（都差過全縣 PrevShare **0.0345**）  
  - 2016 嘗試：更差／未完成全協議

## RQ2 已有咩（夠寫主結論）

全縣 **SGC＋NonSpatial＋PreviousShare** 已齊（`docs/RQ2_美台空間.md`）：

| family | model | mean MAE |
|--------|-------|---------:|
| baseline | PreviousShare | **0.0345** |
| NonSpatial | Ridge | 0.0388 |
| NonSpatial | RF | 0.0419 |
| SGC | 最佳約 | ~0.0437 |

**結論：** 空間擴散未穩贏歷史基準；trainable 全縣 GCN 唔係 RQ2 缺口嘅唯一證據。

## 本輪 pilot（已跑）

- 腳本：`fyp_us_taiwan_transfer/scripts/run_us_gcn_pilot_ne.py`  
- 子集：**New England** 6 州 FIPS（09/23/25/33/44/50），**67** counties；Queen＋kNN；`max_epochs=25`；SEED=42  
- 產物：`results/us_gcn_pilot/`（`us_gcn_pilot_backtest.csv`、`us_gcn_pilot_mean_mae.csv`、`us_gcn_pilot_protocol.json`）  
- **同子集** mean MAE（2012–2024）：

| family | model | mean MAE |
|--------|-------|---------:|
| baseline | PreviousShare | **0.0314** |
| SGC | SGC_L1+RF knn | 0.0421 |
| NonSpatial | RF | 0.0424 |
| NonSpatial | Ridge | 0.0457 |
| trainable_gnn | ResidualGCN_L1 queen | 0.0569 |
| trainable_gnn | TrainableGCN_L1 queen | 0.0685 |

Pilot 內 GCN／SGC **未贏** PreviousShare（與全縣 SGC 方向一致）。數字**只**在 New England 子集內比較。

## 之後要跑全縣 trainable 嘅精確下一步

```bash
cd /workspace/taiwan-election-data
.venv/bin/python fyp_us_taiwan_transfer/scripts/run_rq2_us_county_spatial.py
# 預期長跑；可先提高 MAX_EPOCHS／只跑 queen；完成後更新 gcn_partial_note.json → 正式 backtest CSV
```

或先擴 pilot 至 Mid-Atlantic（NY/NJ/PA，~150 縣）再決定是否全縣。
