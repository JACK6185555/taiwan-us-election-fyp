# 臺灣總統選舉鄉鎮預測 — Feature Weight 分析說明

**產出時間**：2026-10-29（UTC+8）  
**性質**：本目錄為 **生態學／加總層級（鄉鎮）模型**，**不是** TEDS 個人層政黨認同係數。

## 資料與樣本

| 項目 | 內容 |
|------|------|
| 分析單位 | 鄉鎮市區（township） |
| 選舉年 | 2016、2020、2024（2012 僅作滯後） |
| 完整列數 | 1,047（約 354 鄉鄉鎮 × 3 年） |
| 依變數 | 民進黨總統候選人得票率 `dpp_share` |
| 特徵 | 上屆 DPP／KMT 得票率與差距；log 平均所得、人口密度、人口、納稅戶、15+人口；區域虛擬變數（北為對照） |
| 合併鍵 | `縣市名+鄉鎮名`（例：`臺北市松山區`） |

全國加總核對：2016 DPP 56.12%；2020 57.13%；2024 40.05%。

## 方法

1. **主模型**：Random Forest + LASSO（標準化特徵）+ OLS  
2. **驗證**：**Leave-One-Election-Out（LOEO）** — 每次留一屆測試；另報 train 2016+2020 → test 2024  
3. **重要性**：RF **permutation importance**（測試集 MAE 上升幅度）；LASSO 標準化係數；RF impurity  
4. **對照**：僅用上屆得票率的 lag baseline；縣市加總模型；同年內「僅社經、無滯後」截面模型

## 主要結論（數字來自實際跑分）

### A. 預測 DPP 得票率（含滯後）— LOEO 平均 RF perm

| 排名 | Feature | mean perm (ΔMAE) | mean LASSO coef | RF impurity |
|------|---------|------------------|-----------------|-------------|
| 1 | lag_dpp_share | **0.02114** | **+0.355** | **0.484** |
| 2 | dpp_kmt_margin_lag | 0.00245 | +0.007 | 0.083 |
| 3 | region_south | 0.00103 | +0.002 | 0.041 |
| 4 | region_islands | 0.00026 | −0.007 | 0.001 |
| 5 | region_central | 0.00025 | −0.001 | 0.002 |
| 6 | log_pop_density | 0.00008 | +0.021 | 0.012 |
| … | 其餘社經 | ≈0 或負 | 較小 | 較小 |

**上屆民進黨得票率幾乎壟斷預測力。** 社經變數在控制滯後後，樣本外貢獻很小。

### B. 預測表現（務必讀）

| Holdout | RF MAE | Lag-only MAE | 解讀 |
|---------|--------|--------------|------|
| 2020 | 0.116 | **0.022** | 兩屆得票極穩定；lag 已夠 |
| 2024 | 0.204 | **0.148** | 三腳督全國 DPP 大跌；完整模型**差過** lag |
| 2016 | 0.219 | **0.089** | 同樣 lag 較優 |

→ 有公開社經特徵時，**歷史版圖（lag）仍是最強**；2024 結構斷裂令水平預測困難。

### C. 同年截面、無滯後（社經地理相關）

見 `socio_only_within_year_mean.csv` 與 `charts/socio_only_importance.png`。  
此設定回答「空間上誰比較綠」，**不是**預測權重；區域（南／離島）與人口密度／所得會較顯眼。

## 輸出檔

| 檔案 | 說明 |
|------|------|
| `township_panel_2016_2024.csv` | 建模用鄉鎮 panel |
| `feature_weights.csv` | train 2016+2020 → 2024 權重 |
| `feature_weights_loeo_mean_dpp_share.csv` | **主排名**（LOEO 平均） |
| `feature_weights_loeo_mean_delta_dpp.csv` | 得票變化模型 |
| `model_metrics_loeo.csv` | 各 holdout 誤差 |
| `county_feature_weights_loeo_mean.csv` | 縣市級對照 |
| `socio_only_within_year_mean.csv` | 無滯後截面 |
| `charts/*.png` | 圖 |

腳本：`analysis/scripts/compute_feature_weights.py`

## 限制（必讀）

1. **無** TEDS／選研微資料 → 無個人政黨認同、統獨、族群；那些在文獻中通常更重要。  
2. 僅 3 屆總統；2024 候選人結構不同（民眾黨）。  
3. 生態謬誤：鄉鎮相關 ≠ 個人因果。  
4. 所得／人口密度對應年為可得最接近年（非完美同期）。  
5. 特徵共線（人口、納稅戶、密度、所得）會令單一係數不穩；以 permutation／多模型交叉為準。  
