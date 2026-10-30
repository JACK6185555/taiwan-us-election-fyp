# 臺灣立委三黨分析（區域立委・鄉鎮層）

## 單位與範圍
- **地理單位**：鄉鎮市區（與總統 FYP 對齊，方便比較）
- **選票**：中選會 **區域立委** 候選人得票，依政黨代碼加總（DPP=16、KMT=1、TPP=350）
- **年份**：2016、2020、2024（本資料包無 2012 區域立委）
- **不分區**：僅有全國 `elretks`（見 `partylist_national_from_elretks.csv`），**無**鄉鎮票檔，故不入預測模型

## 全國核對（CEC）
### 區域立委（鄉鎮加總）
見 `run_summary.json` → `district_national_2024`（約 DPP 45.2% / KMT 40.4% / TPP 3.0%——因民眾黨只在部分選區提名，遠低於不分區）。

### 不分區政黨票（elretks）
| 年 | DPP | KMT | TPP |
|----|-----|-----|-----|
| 2016 | 44.06% | 26.91% | — |
| 2020 | 33.98% | 33.36% | 11.22% |
| 2024 | 36.16% | 34.58% | 22.07% |

## 模型
PreviousShare、Ridge、RandomForest、HistGB；swing 有 lag 時加 ZeroSwing。
TPP：2016 無；2020 無上屆；**2024 可用 2020 區域立委 TPP 作 PreviousShare**（比總統 TPP 條件更好）。

## 輸出
- `harmonized_tw_legislative.csv`、`township_district_party_votes.csv`
- `backtest_results_parties.csv`、`feature_importance_by_party.csv`、`ablation_by_party.csv`
- `compare_pres_vs_leg_2024.csv`、`charts/`

## 注意
- 區域得票率 ≠ 不分區得票率（候選人結構／提名範圍不同）
- `tpp_contested`：該鄉鎮所屬選區是否有 TPP 候選人
- LOCO 標明 **NOT temporal**

## 2024 Level MAE（實跑）
| Party | PreviousShare | RF | Ridge | HistGB |
|-------|---------------|-----|-------|--------|
| DPP | 0.105 | 0.164 | 0.106 | 0.142 |
| KMT | 0.133 | 0.139 | 0.144 | 0.139 |
| TPP | 0.039 | **0.021** | 0.026 | 0.023 |

區域全國加總 2024：DPP 45.17% / KMT 40.42% / TPP 2.97%。
TPP RF 重要性主要由 `tpp_contested`（≈0.80）驅動。
