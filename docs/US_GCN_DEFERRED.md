# 全縣可訓練 US GCN：狀態更新（Census 鄰接已 unblock 圖；全縣 trainable 仍限）

**日期：** 2026-10-31（Asia/Hong_Kong, UTC+8）  
**標籤：** `us_census_adjacency_pilot`／`us_gcn_pilot` ≠ Core specification 主表；**唔好**同全縣 Prev **0.0345** 混報子集 MAE。  
**綜合框住：** `docs/RQ2_RQ3_RQ4_誠實框住.md` §一（空間）。

## 狀態一句

| 項目 | 狀態 |
|------|------|
| 官方縣鄰接圖（Census County Adjacency **2024**） | **已取得並對齊 panel** |
| 全縣 SGC（Census queen）OOT | **已跑**（Prev 仍贏） |
| Mid-Atlantic 可訓練 GCN-lite（唔用 torch-geometric） | **已跑**（Prev 仍贏） |
| 全縣可訓練 Dense／PyG GCN（四屆 × 多圖完整協議） | **仍 DEFERRED（runtime／範圍）** — 已唔再係「無鄰接數據」 |

**部分解除 DEFERRED：** 圖資料同 SGC／區域 trainable pilot 已齊；**全縣 trainable GCN 定稿表**仍然未交。

## 點解以前空白／而家唔再空白

- 以前：Plotly GeoJSON Queen 可用，但官方 Census 鄰接未入庫；全縣 trainable PyG 跑被 runtime 截斷 → 文件寫 DEFERRED。  
- **2026-10-31：** 下載 [county_adjacency2024.txt](https://www2.census.gov/geo/docs/reference/county_adjacency/county_adjacency2024.txt)；腳本 `fyp_us_taiwan_transfer/scripts/run_us_census_adjacency_pilot.py`；產物 `results/us_census_adjacency_pilot/`、`data/us_county_adjacency/`。

## FIPS 對齊（圖 ∩ panel）

| 指標 | 數值 |
|------|-----:|
| Panel 共同 OOT FIPS（2012–2024 ∩ prior） | 3111 |
| ∩ Census 2024 GEOID | **3103** |
| Match rate | **0.9974** |
| 子圖 undirected 邊 | **9188** |
| 平均度 | ~5.92 |
| Degree-0 islands | **4**（夏威夷 HI 縣：15001／15003／15007／15009 — 地理離島，預期） |
| Panel 有、Census 2024 無 | **8**（康州歷史縣 FIPS `09001`…`09015`） |

**康州說明：** Census **2024** 鄰接檔已改用 CT **Planning Regions**（唔再係 8 個傳統 county GEOID）。Panel 仍用歷史 county FIPS → 8 縣唔入圖。唔發明邊；誠實報告缺口。

來源說明：`data/us_county_adjacency/README.md`（另有副本 `fyp_us_core/data/`）。

## 全縣 SGC（Census queen；可同 locked Prev 對照）

同一 OOT 年 2012／2016／2020／2024；本跑 FullUS PreviousShare mean MAE ≈ **0.03448** ≈ locked Core **0.0345**（唔改 locked 數）。

| family | model | adjacency | mean MAE |
|--------|-------|-----------|---------:|
| baseline | PreviousShare | none | **0.0345**（≈ locked） |
| NonSpatial | Ridge | none | 0.0388 |
| NonSpatial | RandomForest | none | 0.0419 |
| SGC | SGC_L1+Ridge | census2024_queen | 0.0438 |
| SGC | SGC_L1+RF | census2024_queen | 0.0442 |

**結論：** 換官方 Census 鄰接後，全縣 SGC **仍然未贏** Prev／NonSpatial（與舊 Plotly Queen SGC ~0.0437 方向一致）。

## Mid-Atlantic trainable GCN-lite（NY／NJ／PA；子集內比較）

- **150** counties；Census 子圖；`DenseGCN_L1`／`ResidualDenseGCN_L1`（PyTorch `A_hat @ X`；**無** torch-geometric）  
- `max_epochs=30`，SEED=42  

| family | model | mean MAE（子集） |
|--------|-------|---------:|
| baseline | PreviousShare | **0.0370** |
| NonSpatial | Ridge | 0.0474 |
| NonSpatial | RF | 0.0491 |
| SGC | 最佳約 SGC_L1+RF | 0.0502 |
| trainable_gnn_lite | ResidualDenseGCN_L1 | 0.0730 |
| trainable_gnn_lite | DenseGCN_L1 | 0.0932 |

Pilot 內 GCN-lite／SGC **未贏** 同子集 Prev。**勿**與全縣 0.0345 混報。

## 舊 NE pilot／部分全縣數字（仍保留）

- NE 67 縣（GeoJSON Queen）：`results/us_gcn_pilot/`；同子集 Prev≈0.0314 最好。  
- 全縣部分 2012（舊 capped PyG）：`gcn_partial_note.json` — TrainableGCN≈0.086、ResidualGCN≈0.072（都差過全縣 Prev）。

## 依賴選擇

- **SGC：** `numpy` + `scipy.sparse` + sklearn Ridge／RF（唔需要 PyG）。  
- **GCN-lite：** `torch` dense matmul；環境無 `torch_geometric` 時仍可跑區域 pilot。  
- 全縣 trainable PyG：可選；非本輪必要條件。

## 之後若要「全縣 trainable 定稿」

```bash
cd /workspace/taiwan-us-election-fyp-push
python3 fyp_us_taiwan_transfer/scripts/run_us_census_adjacency_pilot.py   # 圖＋SGC＋MA pilot
# 全縣 DenseGCN：需加長 epochs／可能 GPU；完成後先寫正式 backtest CSV 再改本檔「仍 DEFERRED」句
```

或擴 pilot 至更多州；**唔好**在未完成四屆協議前宣稱全縣 GCN 已驗證贏 Prev。
