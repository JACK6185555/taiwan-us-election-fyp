# RQ2：真共邊（Queen contiguity）vs 同縣完全圖 proxy

## 研究問題對齊

空間模型想捕捉「鄰近鄉鎮選舉行為互相影響」。舊版 `fyp_taiwan_spatial_gnn` 用 **同年同縣市完全圖（county clique）**：同縣內每個鄉鎮兩兩相連。呢個 proxy：

- 會把唔接壤嘅鄉鎮當成鄰居（例如縣嘅兩端）；
- 大縣平均度會被誇大（2024 平均度 ≈ **22.6**）。

本更新改用內政部國土測繪中心 **鄉鎮市區界線 TOWN_MOI_1140318** 衍生幾何，建 **Queen 共邊**圖（共點或共邊即鄰居），2024 平均度 ≈ **5.26**，同真實地理鄰居數接近。

## 實作

### A. SGC 風格擴散（numpy；先做）

1. 建 undirected adjacency `A`，加 self-loop → `Ã = A + I`
2. 對稱正規化 `Â = D^{-1/2} Ã D^{-1/2}`
3. 特徵 `H ← Â H` 重複 1–2 層（權重 **W = I**）
4. 用擴散後（或 Raw∥擴散）特徵訓練 Ridge／RandomForest

CSV 標成 `GCN_diffuse_L*`／`Raw+GCN_L2`。

### B. 可訓練 GCN／GAT（PyTorch + torch-geometric；後加）

- 2 層 `GCNConv`／`GATConv`（GAT heads=2→1），hidden=32，dropout=0.15
- Adam lr=5e-3，weight_decay=1e-4，max_epochs=250，patience=35，seed=42，**CPU**
- 特徵同目標用 train-fit `StandardScaler`；預測 inverse-transform 後先計 MAE
- Fit：所有 `year < test_year`；early-stop 監察最後一個訓練年 MAE（該年仍在 train，無 test leak）
- 腳本：`fyp_taiwan_spatial_gnn/scripts/run_taiwan_trainable_gcn_gat.py`

## 幾何與對齊

| 項目 | 說明 |
|---|---|
| 來源 | [data.gov.tw #7441](https://data.gov.tw/dataset/7441)；TGOS ZIP 本環境 403，改用 [pdmaptw v2.4.0](https://github.com/plotdb/pdmaptw/releases/tag/v2.4.0) `town-2025-03-18.geojson`（同一 TOWN_MOI_1140318） |
| 授權 | OGDL-Taiwan-1.0 |
| 對鍵 | panel `region` = geo `name`（`台→臺`）；2012 alias：員林鎮→員林市、頭份鎮→頭份市 |
| 2024 match rate | **100%**（368/368） |
| Islands | 11 個（綠島、蘭嶼、琉球、澎湖望安／七美、金門烈嶼／烏坵、連江四鄉等）—多數係真離島 |

詳見 `raw/geo/README.md`、`fyp_taiwan_spatial_gnn/graph_meta_contiguity.json`。

## Ablation 設計

同一 panel、同一特徵、同一 **2024 out-of-time** split，比較：

- `NonSpatial`（Ridge／RF／MLP）
- `SGC` Raw∥L2 + Ridge／RF（Contiguity；SGC 腳本另有 County）
- `TrainableGCN`／`TrainableGAT`（Contiguity；2024 另報 County sensitivity）

指標：MAE／RMSE／R²（總統 DPP／KMT × swing／demeaned）。另有 2020／2016 OOT（trainable）。

## 主要發現（2024；見 CSV，唔發明數字）

結果**混合**，唔好單講「共邊一定贏」：

- Contiguity 有時贏 NonSpatial／County（例如 DPP demeaned RF、DPP swing Ridge／RF、KMT demeaned RF）。
- Contiguity + Ridge 喺 demeaned 目標可以明顯差過 NonSpatial（擴散可能模糊單位固定效應）。
- KMT swing 上 County clique 有時仍然較好——同縣完全圖可能意外捕捉到「縣級政治機器／配票」訊號，而唔係純地理外溢。

解讀建議：真共邊係 **更誠實嘅空間假設**；預測力係 empirical question，以 `gnn_contiguity_vs_county_ablation_2024.csv` 同 `trainable_vs_sgc_ablation_2024.csv` 為準。

## Trainable vs SGC／NonSpatial（2024 MAE）

| party | target | NS Ridge | NS RF | SGC+Ridge | SGC+RF | GCN contig | GAT contig |
|------|--------|----------|-------|-----------|--------|------------|------------|
| DPP | demeaned | 0.0503 | 0.0506 | 0.0978 | **0.0490** | 0.0989 | 0.0558 |
| DPP | swing | 0.0973 | 0.1717 | **0.0918** | 0.1695 | 0.1351 | 0.2098 |
| KMT | demeaned | **0.0168** | 0.0535 | 0.0646 | 0.0495 | 0.1779 | 0.0377 |
| KMT | swing | 0.1193 | **0.0204** | 0.1237 | 0.0285 | 0.0236 | 0.0882 |

重點（誠實）：

- **DPP demeaned**：SGC+RF Contig（0.0490）略好過 NonSpatial；GAT Contig（0.0558）接近但未贏 SGC。
- **DPP swing**：SGC+Ridge（0.0918）最好；trainable GCN／GAT **明顯差過**。
- **KMT demeaned**：NonSpatial Ridge（0.0168）壓倒性最好；trainable GCN 差。
- **KMT swing**：NonSpatial RF（0.0204）最好；TrainableGCN Contig（0.0236）接近第二，County GCN 0.0233。

結論：**可訓練圖網絡已跑通並有 ablation**，但喺呢個細 panel／強 baseline 下 **唔保證贏** SGC 或 NonSpatial——口試應如實講。

## 重跑

```bash
cd /workspace/taiwan-election-data
.venv/bin/python fyp_taiwan_spatial_gnn/scripts/run_taiwan_spatial_contiguity_gcn.py
.venv/bin/python fyp_taiwan_spatial_gnn/scripts/run_taiwan_trainable_gcn_gat.py
```

產物：`trainable_gcn_gat_ablation_2024.csv`、`trainable_vs_sgc_ablation_2024.csv`、`trainable_gcn_gat_backtest.csv`、`charts/trainable_gcn_gat_vs_baselines_2024.png`。

---

## RQ2 頭條：縣市層總統票同窗實驗（Core specification 對齊）

**呢一節先係 RQ2 主表。** 鄉鎮 SGC／GCN／GAT 係補充線，**唔好把鄉鎮 MAE 寫入 Core specification／呢張縣市表。**

### 設計

| 項目 | 內容 |
|---|---|
| Panel | `data/county_panel_features_2000_2024.csv`（22 縣市） |
| 特徵 | 與 Core specification 相同 16 個歷史特徵（`FEATURES` in `taiwan_core_backtest.py`） |
| Protocol | Expanding window：`train year < test_year`；測試年 **2012／2016／2020／2024** |
| 目標 | `target_DPP_vote_share`（主）；`target_KMT_vote_share`（附） |
| 圖 | 鄉鎮 GeoJSON dissolve→縣市；**Queen**（33 邊，平均度 3.0；澎湖／金門／連江離島）；**k-NN k=3**；**距離衰減** σ=80 km（EPSG:3826） |
| 模型 | PreviousShare；NonSpatial Ridge／RF／HGB；SGC L1／L2 + Ridge／RF；Trainable GCN L1／L2、**ResidualGCN**（輸入 skip）、GAT |
| Seed | 42；CPU；腳本 `fyp_taiwan_spatial_gnn/scripts/run_rq2_county_spatial.py` |
| 產物目錄 | `fyp_taiwan_spatial_gnn/rq2_county/` |

### DPP 平均 OOT MAE（2012–2024；節錄自 `rq2_county_mean_mae_dpp.csv`）

| family | model | adjacency | mean MAE | mean RMSE |
|---|---|---|---:|---:|
| baseline | PreviousShare | none | **0.0793** | 0.0814 |
| SGC_diffusion | SGC_L2+RandomForest | dist_decay_bin | 0.1029 | 0.1152 |
| SGC_diffusion | SGC_L1+RandomForest | dist_decay_bin | 0.1033 | 0.1154 |
| SGC_diffusion | SGC_L1+RandomForest | queen | 0.1037 | 0.1152 |
| NonSpatial | RandomForest | none | 0.1043 | 0.1163 |
| NonSpatial | HistGradientBoosting | none | 0.1189 | 0.1268 |
| trainable_gnn | ResidualGCN_L2 | knn | 0.1289 | 0.1317 |
| trainable_gnn | ResidualGCN_L2 | dist_decay | 0.1328 | 0.1357 |
| trainable_gnn | TrainableGAT | queen | 0.1496 | 0.1764 |
| NonSpatial | Ridge | none | 0.1786 | 0.1855 |

（完整表見 CSV；KMT 見 `rq2_county_mean_mae_kmt.csv`——PreviousShare 0.1094 仍最好。）

### 逐年 DPP（Previous vs RF NonSpatial vs 該年最佳空間）

| test_year | PreviousShare | RF NonSpatial | best spatial MAE | best spatial |
|---:|---:|---:|---:|---|
| 2012 | 0.0402 | 0.0743 | 0.0233 | ResidualGCN_L2 / knn |
| 2016 | 0.0979 | 0.1322 | 0.0552 | TrainableGCN_L2 / dist_decay |
| 2020 | 0.0193 | 0.0460 | 0.0428 | SGC_L1+RF / dist_decay_bin |
| 2024 | 0.1596 | 0.1647 | 0.1526 | ResidualGCN_L2 / dist_decay |

解讀：個別年份 trainable／SGC 可以贏；但 **跨年平均** PreviousShare 仍穩贏，trainable **方差好大**（例如 ResidualGCN knn：2012 MAE 0.023、2020 卻 0.224）。

### RQ2 結論（誠實）

**混合／空間未穩定贏非空間；基準仍最好。**

1. **Headline（對齊 Core specification）**：縣市總統票 OOT 上，**PreviousShare（mean MAE 0.0793）最好**；空間模型唔改變呢個 Core specification 故事。
2. **SGC vs NonSpatial**：距離衰減／Queen 上嘅 **SGC+RF 平均 MAE ≈ 0.103，略好過 NonSpatial RF 0.104**（Δ ≈ −0.0015）——有微弱空間訊號，但幅度細，唔好誇大。
3. **可訓練 GCN／GAT／ResidualGCN**：已用淺層＋殘差 skip 減 over-smoothing；**單年可以好靚**，但跨年平均同穩定性 **差過 NonSpatial／SGC**，唔建議當主結果。
4. **邊類型**：Queen（真共邊）同 dist-decay／k-NN 結果接近；離島（澎湖／金門／連江）Queen 度數 0，dist／k-NN 可補連邊。
5. **同鄉鎮線關係**：鄉鎮 contiguity 實驗（上文）係 **supplement**；本節縣市同窗先係 RQ2 對 Guide 嘅主對照。

### 重跑（縣市 RQ2）

```bash
cd /workspace/taiwan-election-data
.venv/bin/python fyp_taiwan_spatial_gnn/scripts/run_rq2_county_spatial.py
```

產物：`rq2_county/rq2_county_backtest.csv`、`rq2_county_mean_mae_dpp.csv`、`rq2_county_mean_mae_kmt.csv`、`rq2_county_ablation_2024.csv`、`rq2_county_year_comparison_dpp.csv`、`graph_meta_county.json`、`charts/rq2_county_*.png`。

## 未做／限制（更新）

- 已做：縣市同窗 RQ2；Queen／k-NN／距離衰減；ResidualGCN
- 未做：GraphSAGE／人口流動邊權；GPU 更深網；跨國空間（另線 RQ3）
- 限制：N=22 縣市×4 測試年——圖模型自由度高、方差大；離島 Queen 無邊
---

## 美國 RQ2（縣級；SGC 頭條）

Guide RQ2 兩邊都要空間。美國產物：`fyp_us_taiwan_transfer/rq2_us_county/`。

- Queen（plotly FIPS geojson；TIGER 403）：3111 縣，9072 邊，avg degree 5.83
- Mean OOT MAE：PreviousShare **0.0345** &lt; NonSpatial Ridge 0.0388 &lt; 最佳 SGC（L1+Ridge queen）0.0437
- 全美 trainable GCN：**deferred**（runtime）；見 `gcn_partial_note.json`
- 跨國摘要：`docs/RQ2_美台空間.md`
