# RQ2／RQ3／RQ4 誠實框住（空間・跨域轉移・不確定性）

**日期：** 2026-10-30（Asia/Hong_Kong, UTC+8）  
**學生專案：** US–Taiwan Election Forecasting FYP  
**定位：** 方法論補充章——**不是**勝選機率產品，**不是**政治轉移結論，**不是**已校準的選舉信心區間產品。  
**主數字釘死（勿改）：** TW Core specification Prev MAE **0.0793**；US Core Prev **0.0345**。Importance ≠ 因果；Prev 贏＝地理慣性，唔等於「選民只跟上屆」嘅因果宣稱。  
**資料範圍：** 公開開票／社經 PROXY；**未申請** TEDS／ANES。

> 本檔係老師進度報告同口試用嘅**誠實框住**版。細表／腳本仍以各子目錄 CSV／JSON 為準；唔好把子集、DA、Conformal 數字同 Core specification 主表混報。

---

## 一、空間（RQ2）

**一句主張：** 喺美台縣市／縣級總統票同窗 OOT 下，**共邊／SGC／可訓練 GNN 未穩定打敗 PreviousShare**；美國已用 **Census 2024 縣鄰接**跑全縣 SGC＋Mid-Atlantic GCN-lite；**全縣可訓練 GCN 定稿**仍 DEFERRED（runtime），區域 pilot **只**可在子集內比較。

### 1.1 做咗咩

| 項目 | 內容 | 路徑 |
|------|------|------|
| 共同協議 | 測試年 2012／2016／2020／2024；expanding（train year &lt; test）；SEED=42 | `docs/RQ2_美台空間.md` |
| 臺灣 | 22 縣市；Core specification 歷史特徵；Queen／k-NN／距離衰減；SGC＋可訓練 GCN／GAT | `fyp_taiwan_spatial_gnn/rq2_county/` |
| 美國全縣 | 3111 counties；SGC＋NonSpatial＋Prev；Queen≈9072 邊 | `fyp_us_taiwan_transfer/rq2_us_county/` |
| 美國官方鄰接 | Census County Adjacency 2024；panel∩graph **3103／3111** | `data/us_county_adjacency/` |
| 美國全縣 SGC（Census） | 已跑；Prev 0.0345 仍贏；SGC≈0.0438 | `results/us_census_adjacency_pilot/` |
| 美國全縣 trainable GCN | **仍 DEFERRED**（runtime）；已唔係無圖 | `docs/US_GCN_DEFERRED.md`、`gcn_partial_note.json` |
| MA／NE pilot | Mid-Atlantic 150 縣 GCN-lite；NE 67 縣舊 pilot | `results/us_census_adjacency_pilot/`、`results/us_gcn_pilot/` |

### 1.2 證據表（真實 CSV）

**臺灣（`rq2_county_mean_mae_dpp.csv`，DPP share）**

| family | model（摘要） | mean MAE |
|--------|---------------|---------:|
| baseline | PreviousShare | **0.0793** |
| SGC | 最佳 SGC_L2+RF（dist_decay_bin） | 0.1029 |
| NonSpatial | RandomForest | 0.1043 |
| NonSpatial | Ridge | 0.1786 |
| trainable_gnn | ResidualGCN／GAT／TrainableGCN | 多數 ≥0.13，部分遠差 |

**美國全縣（`rq2_us_county_mean_mae.csv`）**

| family | model | mean MAE |
|--------|-------|---------:|
| baseline | PreviousShare | **0.0345** |
| NonSpatial | Ridge | 0.0388 |
| NonSpatial | RandomForest | 0.0419 |
| SGC | 最佳約 SGC_L1+Ridge（queen） | **0.0437** |

**New England pilot（`us_gcn_pilot_mean_mae.csv`；勿與全縣 0.0345 混報）**

| family | model | mean MAE |
|--------|-------|---------:|
| baseline | PreviousShare | **0.0314** |
| SGC | SGC_L1+RF knn | 0.0421 |
| trainable_gnn | ResidualGCN_L1 queen | 0.0569 |
| trainable_gnn | TrainableGCN_L1 queen | 0.0685 |

**Census 全縣 SGC＋Mid-Atlantic（`us_census_adj_mean_mae.csv`；2026-10-31）**

| subset | family | model | mean MAE |
|--------|--------|-------|---------:|
| FullUS | baseline | PreviousShare | **0.0345**（≈ locked） |
| FullUS | SGC | SGC_L1+Ridge census2024 | 0.0438 |
| MidAtlantic | baseline | PreviousShare | **0.0370** |
| MidAtlantic | trainable_gnn_lite | ResidualDenseGCN_L1 | 0.0730 |


### 1.3 限制（必須講）

1. **Prev 仍然最好**（TW／US 全縣／NE 子集方向一致）——空間擴散係更誠實嘅相鄰假設，但喺呢個 OOT 窗**未穩定改善預測**。  
2. TW 最佳 SGC（0.1029）只略好過非空間 RF（0.1043），**仍遠輸** Prev 0.0793。  
3. US 全縣 SGC ~0.0437 **差過** Prev 0.0345 同 NonSpatial Ridge 0.0388。  
4. **全縣可訓練 US GCN 定稿 = 仍 DEFERRED**（runtime）；**Census 鄰接＋全縣 SGC＋MA GCN-lite 已補**——已唔係「無數據」。部分 2012 GCN≈0.086／ResidualGCN≈0.072（`gcn_partial_note.json`）都差過全縣 Prev，**唔當定稿**。  
5. NE／Mid-Atlantic pilot：**子集 ≠ 全國**；pilot 內 GCN 亦未贏同子集 Prev。CT 8 縣因 Census 2024 Planning Regions 唔入圖。  
6. Importance／空間係數 ≠ 因果；唔好解讀成「鄰縣決定投票」。

---

## 二、跨域轉移（RQ3）

**一句主張：** 美台 **joint／transfer／CORAL／MMD／DANN** 管線已跑通，屬**方法論探索**；**唔可以**宣稱「美台政治表徵已證明可轉移」或得出跨國政治結論——構念唔同、特徵薄、美國 N 主導。

### 2.1 做咗咩

| 項目 | 內容 | 路徑 |
|------|------|------|
| 共享 schema | 美＝Dem 兩黨份額；台＝DPP 有效票份額；**同名欄 ≠ 同一政治構念**（Guide §19） | `docs/美台對照_RQ3.md` |
| Demo 1／7 | 合成＋真實 country-only／joint／transfer | `fyp_us_taiwan_transfer/` |
| 社經合併 | 國內 z-score（`z_income`, `z_log_density`）＋`prev_share_A`, `log_total_votes` | `real_joint_mae_mean_acs.csv` |
| I-K DA | CORAL、核 MMD、adversarial DANN、low-data k＝0…22 | `results/real/da_transfer_mae_mean.csv` |

### 2.2 證據表（`real_joint_mae_mean_acs.csv`／`da_transfer_mae_mean.csv`）

**Country-only／joint／transfer（含 ACS／ERS 社經）**

| protocol | 最佳／基準（摘要） | mean MAE |
|----------|-------------------|---------:|
| country_only_TW | Ridge ≈ **0.0787**；Prev **0.0793** | — |
| country_only_US | Prev **0.0345** | — |
| joint_pooled | Prev ≈ **0.0348**（美國 N 主導） | — |
| transfer_US→TW | HGB **0.0690**；Ridge **0.0705**（&lt; TW Prev 0.079） | — |
| transfer_TW→US | Ridge **0.0461**（&gt; US Prev 0.0345） | — |

**Domain adaptation（Ridge 系 mean OOT MAE）**

| direction | method | mean MAE |
|-----------|--------|---------:|
| US→TW | mmd | **0.0700** |
| US→TW | no_da | 0.0705 |
| US→TW | coral | 0.0784 |
| US→TW | adv_dann | 0.1178 |
| TW→US | no_da | **0.0461** |
| TW→US | mmd | 0.0632 |
| TW→US | coral | 0.0955 |
| TW→US | adv_dann | 0.1060 |

MMD 診斷（特徵空間距離）US→TW 各測試年 mmd² ≈ 0.35–0.40（`da_mmd_diagnostic.csv`），顯示源／目標分布有明顯差距。

### 2.3 限制（必須講）

1. **可講：**「跨國實驗管線已測試」「有真實 country-only／joint／transfer／DA MAE」。  
2. **唔可講：**「美台表徵已證明可轉移／同一構念」「美國模式適用於臺灣政治」。  
3. US→TW transfer MAE 低於 TW Prev，**唔等於**政治知識轉移成功——共享特徵極薄（上屆票＋規模＋國內相對所得／密度），美國樣本量大、構念（Dem vs DPP）唔對等。  
4. DA（CORAL／MMD／DANN）喺呢組特徵上**無穩贏** no_da（US→TW 只有 MMD 微邊緣）；DANN 明顯更差。  
5. Joint pooled 被美國縣數主導（3111 ≫ 22）。  
6. 台無對齊失業／學士率入 joint；所得定義唔同，只用國內 z。  
7. **唔申請** TEDS——無個人層認同／黨性，無法驗證「政治轉移」。

---

## 三、不確定性／Conformal（RQ4）

**一句主張：** Split conformal（α＝0.1，名義覆蓋 90%）已作為 **Demo 6／探索實驗**跑過美台；**唔係**校準好嘅選舉「信心產品」——覆蓋率跨年波動大，分布轉移下可能嚴重誤校準。

### 3.1 做咗咩

| 項目 | 內容 | 路徑 |
|------|------|------|
| 美國 Core | split conformal；calib＝上一屆；train&lt;calib；α＝0.1 | `fyp_us_core/results/conformal_*.csv`／`conformal_meta.json` |
| 臺灣縣市協議掃描 | A（prev-election calib）／B（demeaned）／C（year LOO）；electoral＋enriched | `analysis/conformal_rq4_improved/conformal_protocols_summary.csv` |
| 臺灣舊單點 | 2024 單次 meta（嚴重欠覆蓋） | `fyp_taiwan/conformal_meta_2024.json` |
| 去均值補充 | demeaned 總統 2024 | `fyp_taiwan_improved_r2/conformal_demeaned_meta.csv` |

### 3.2 證據表（真實檔）

**美國 Core 平均覆蓋／寬度（`conformal_coverage_summary.csv`；目標 0.90）**

| model | mean coverage | mean width |
|-------|--------------:|-----------:|
| Ridge | **0.910** | 0.221 |
| HistGradientBoosting | 0.881 | 0.225 |
| Random Forest | 0.873 | 0.210 |
| Previous-election baseline | 0.834 | 0.157 |

**美國按年（`conformal_coverage_by_year.csv`）——顯示不穩定**

| test←calib | Prev coverage | Ridge coverage | 備註 |
|------------|--------------:|---------------:|------|
| 2016←2012 | **0.581** | 0.787 | 明顯**欠覆蓋** |
| 2020←2016 | **0.996** | 0.995 | 接近「過闊／過覆蓋」 |
| 2024←2020 | 0.925 | 0.947 | 較接近名義 0.90 |

**臺灣（縣市協議 A，electoral；節錄 `conformal_protocols_summary.csv`）**

| model | calib→test | coverage | avg_width | 解讀 |
|-------|------------|---------:|----------:|------|
| Ridge | 2012→2016 | 0.955 | 0.344 | 達標但區間闊 |
| Ridge | 2016→2020 | 1.000 | 0.270 | 過覆蓋 |
| Ridge | 2020→2024 | 1.000 | **1.018** | 達標但寬度≈整段份額，**無產品價值** |
| RF | 2020→2024 | **0.273** | 0.228 | 嚴重欠覆蓋 |
| HGB | 2020→2024 | **0.409** | 0.317 | 欠覆蓋 |

**臺灣舊單點（`fyp_taiwan/conformal_meta_2024.json`）**  
coverage ≈ **0.008**（目標 0.90）、avg_width ≈ 0.217 —— 喺分布大變年（2024 三腳督等）下，**經典 split conformal 可完全失效**。

**去均值補充（總統 DPP 2024；`conformal_demeaned_meta.csv`）**  
empirical coverage ≈ **0.848**（名義 0.90）、half-width ≈ 0.141 —— 較接近，但仍屬探索、單位係 demeaned，**唔入 Core specification 主結論**。

### 3.3 限制（必須講）

1. **探索級，不是校準產品**——唔好對外報「90% 信心區間保證」。  
2. 覆蓋率**跨年波動大**（美 2016 欠覆蓋、2020 過覆蓋）；TW 2024 部分協議欠覆蓋、部分協議區間闊到冇用。  
3. 選舉有**結構斷裂／分布轉移**（三腳督、全國衝擊）時，exchangeability 假設易破，conformal **可能嚴重誤校準**。  
4. 區間寬度往往 ≥0.15–0.30 份額（甚至 &gt;1），對縣市實務決策幫助有限。  
5. 唔好把 RQ4 覆蓋率當「模型準唔準」嘅替代——準確度仍以 Core specification／US Core **MAE** 為主。  
6. Guide 狀態：**I-I PARTIAL**（有產物、未達產品級）。

---

## 四、三節對老師／口試的統一講法

| RQ | 可講 | 唔好講 |
|----|------|--------|
| **RQ2 空間** | 跑齊 TW／US SGC；Prev 難撼；全縣 GCN deferred；NE pilot 方向一致 | 「空間模型贏咗」「NE 數字＝全國」 |
| **RQ3 轉移** | 管線＋DA 已測；薄特徵上 DA 未必贏；構念唔同 | 「證明美台政治可轉移」 |
| **RQ4 不確定性** | 有 coverage／width 表；跨年不穩；探索 | 「已校準 90% 選舉信心產品」 |

**與主結果關係：** 三節**不改** Core specification Prev **0.0793**／US Core Prev **0.0345**；唔同 Core specification 訓練模型 MAE（RF 0.1041／HGB 0.1189／Ridge 0.1786）混報。

---

## 五、檔案索引

| 用途 | 路徑 |
|------|------|
| 本綜合框住 | `docs/RQ2_RQ3_RQ4_誠實框住.md` |
| RQ2 短表 | `docs/RQ2_美台空間.md` |
| RQ3 短表 | `docs/美台對照_RQ3.md` |
| US GCN deferred | `docs/US_GCN_DEFERRED.md` |
| 空間長文 | `docs/空間共邊GCN.md` |
| Guide 狀態 | `docs/GUIDE_COMPLETION_STATUS.md` |
| 老師報告（含濃縮章） | `docs/老師說明報告.md` |

---

*數字全部來自上表所列 CSV／JSON；無發明指標。Importance ≠ 因果。未申請 TEDS。*
