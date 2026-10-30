# Taiwan FYP — Spatial Graph Model（`fyp_taiwan_spatial_gnn`）

## 方法 / Method

兩層對照（同一特徵／同一 OOT split）：

1. **SGC 擴散（numpy）**：`H ← Â H`，`Â = D^{-1/2}(A+I)D^{-1/2}`，1–2 層，權重 **W=I**，再接 Ridge／RF。見 `run_taiwan_spatial_contiguity_gcn.py`。
2. **可訓練 GCN／GAT（PyTorch + torch-geometric）**：2×`GCNConv`／`GATConv`，CPU，seed=42。見 `run_taiwan_trainable_gcn_gat.py`。

鄰接：
1. **CountyClique** — 同年同縣市鄉鎮完全圖 + self-loop（舊 proxy）
2. **Contiguity** — MOI **TOWN_MOI_1140318** 衍生 GeoJSON 嘅 **Queen 共邊** + self-loop

**評估**：總統 DPP／KMT；目標 swing 同 demeaned；**2024 out-of-time**（trainable 另跑 2020／2016 OOT）。

## 幾何來源 / Geometry

見 `raw/geo/README.md`。檔案：`raw/geo/town-2025-03-18.geojson`
（[data.gov.tw #7441](https://data.gov.tw/dataset/7441) → pdmaptw v2.4.0 mirror；TGOS 直連 403）。

授權：政府資料開放授權條款第1版（OGDL-Taiwan-1.0）。

## 圖統計（2024）

```json
{
  "match_rate_2024": 1.0,
  "contiguity": {
    "n_nodes": 368,
    "n_matched": 368,
    "n_unmatched": 0,
    "match_rate": 1.0,
    "unmatched": [],
    "n_undirected_edges_excl_self": 968,
    "avg_degree_excl_self": 5.260869565217392,
    "n_isolates_degree0_excl_self": 11
  },
  "county_clique": {
    "n_nodes": 368,
    "n_undirected_edges_excl_self": 4155,
    "avg_degree_excl_self": 22.581521739130434
  },
  "islands_full_geo": [
    "臺東縣綠島鄉",
    "臺東縣蘭嶼鄉",
    "連江縣南竿鄉",
    "連江縣莒光鄉",
    "金門縣烏坵鄉",
    "屏東縣琉球鄉",
    "澎湖縣望安鄉",
    "澎湖縣七美鄉",
    "連江縣北竿鄉",
    "連江縣東引鄉",
    "金門縣烈嶼鄉"
  ]
}
```

Match rate 2024 = **100.0%**. 全圖 Queen islands（多數為真離島）：臺東縣綠島鄉, 臺東縣蘭嶼鄉, 連江縣南竿鄉, 連江縣莒光鄉, 金門縣烏坵鄉, 屏東縣琉球鄉, 澎湖縣望安鄉, 澎湖縣七美鄉, 連江縣北竿鄉, 連江縣東引鄉, 金門縣烈嶼鄉.

## 結果檔

| File | Content |
|---|---|
| `gnn_contiguity_ablation_2024.csv` | 全模型列（adjacency × repr × model） |
| `gnn_contiguity_vs_county_ablation_2024.csv` | NonSpatial / County / Contiguity MAE 對照 |
| `gnn_contiguity_backtest.csv` | 含 2016／2020／2024 OOT |
| `adjacency_contiguity.npz` | 2024 induced Queen A + regions |
| `adjacency_contiguity_edges_2024.csv` | undirected edge list |
| `graph_meta_contiguity.json` | 圖統計 + provenance |
| `geo_match_log.csv` | 各年 name match／unmatched |
| `charts/contiguity_vs_county_vs_nonspatial_2024.png` | MAE 柱狀圖 |
| `trainable_gcn_gat_ablation_2024.csv` | 2024 NonSpatial / SGC / TrainableGCN / TrainableGAT |
| `trainable_vs_sgc_ablation_2024.csv` | 2024 MAE pivot（含 county sensitivity） |
| `trainable_gcn_gat_backtest.csv` | 2016／2020／2024 OOT 全列 |
| `trainable_gcn_gat_meta.json` | hyperparams（hidden=32, lr=5e-3, …） |
| `charts/trainable_gcn_gat_vs_baselines_2024.png` | trainable vs baselines 柱狀圖 |

舊檔 `gnn_spatial_ablation_2024.csv`／`run_taiwan_spatial_gnn.py` 仍係 **僅 county-clique** 結果，保留作對照。

## 重跑

```bash
# SGC contiguity vs county
.venv/bin/python fyp_taiwan_spatial_gnn/scripts/run_taiwan_spatial_contiguity_gcn.py
# Trainable GCN / GAT (needs torch + torch-geometric in .venv)
.venv/bin/python fyp_taiwan_spatial_gnn/scripts/run_taiwan_trainable_gcn_gat.py
```

## 限制 / Limitations（誠實）

- pdmaptw GeoJSON **有簡化**；極窄共邊可能漏連。平均度 ≈ 5.3（合理），但唔係地籍級 contiguity。
- 離島／飛地形成 **disconnected components**；self-loop 令 isolate 仍可擴散（只係唔撈鄰居）。
- 2012：`員林鎮`→`員林市`、`頭份鎮`→`頭份市` 用 alias；其餘年 100% match。
- SGC 擴散權重 W=I；**另有**可訓練 GCN／GAT（見 `trainable_*`），但 2024 OOT **唔係一律贏** NonSpatial／SGC。
- 跨國 transfer 另線 US 對照（唔喺本目錄）。

## Trainable GCN／GAT（2024 OOT MAE）

來源：`trainable_vs_sgc_ablation_2024.csv`（誠實數字；trainable **唔係**一律最好）。

| party | target | NS Ridge | NS RF | SGC+Ridge | SGC+RF | GCN contig | GAT contig |
|------|--------|----------|-------|-----------|--------|------------|------------|
| DPP | demeaned | 0.0503 | 0.0506 | 0.0978 | **0.0490** | 0.0989 | 0.0558 |
| DPP | swing | 0.0973 | 0.1717 | **0.0918** | 0.1695 | 0.1351 | 0.2098 |
| KMT | demeaned | **0.0168** | 0.0535 | 0.0646 | 0.0495 | 0.1779 | 0.0377 |
| KMT | swing | 0.1193 | **0.0204** | 0.1237 | 0.0285 | 0.0236 | 0.0882 |

Hyperparams：hidden=32，2 layers，dropout=0.15，lr=5e-3，wd=1e-4，max_epochs=250，patience=35，GAT heads=2，seed=42，CPU；X／y StandardScaler（train-fit），early-stop 監察最後一個訓練年（仍納入訓練）。
