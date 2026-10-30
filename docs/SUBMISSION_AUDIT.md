# 提交前數字核對：分析家族與主結果路徑（I-F／I-A／I-C）

核對日期：2026-10-30  
範圍：`taiwan-election-github/`（本交付包）與 `/workspace/taiwan-election-data/` 鏡像

## 為什麼需要這份文件

本包並行保留鄉鎮三黨、立委、improved／r2、空間共邊、美台 transfer、香港／政綱／中間選民等多條線。若混用「基準 MAE」而不標家族，評審會把不同切分／單位的數字放在同一張表比較。**唯一主結果路徑 = 家族 A**；其餘一律標「補充／Extensions」。

## 一、分析家族（本包）

| 代號 | 分析 | 資料 | 切分 | 樣本 | 目標 | 代表數字 | 狀態 |
|---|---|---|---|---|---|---|---|
| **A** | FYP Core 嚴格時間外 | `data/county_panel_features_2000_2024.csv`（154 列） | 擴展窗；測試 2012／2016／2020／2024 | 每折 22 縣市；共 **88** region-year | 縣市 DPP 總統有效票份額 | Prev **0.0793**；RF **0.1041**；HGB **0.1189**；Ridge **0.1786** | **核心（RQ1）** |
| 鄉鎮三黨 | 總統／立委鄉鎮 panel | `harmonized_tw_*` | OOT 鄉鎮 | 單位＝鄉鎮 | DPP／KMT／TPP | 例：2024 總統 Prev DPP 0.149 | **補充** |
| improved／r2 | swing／demeaned／ESC PROXY／空間 LOO | `results/improved*` | 2024 OOT 為主 | 鄉鎮 | swing／demeaned／level | ESC 氣候可贏 DPP swing | **補充** |
| 空間共邊／SGC | Queen contiguity 或同縣圖 | `results/spatial_*`、`docs/空間共邊GCN.md` | 2024 OOT | 鄉鎮 | demeaned／swing | 共邊唔係一律贏 | **補充（RQ2）** |
| 美台 RQ3 | MEDSL＋TW 縣市 | `docs/美台對照_RQ3.md`、`results/us_taiwan_transfer` | country-only／joint／transfer | 美縣市 N 主導 | 主要政黨份額 | TW Prev≈0.0793 量級 | **補充（RQ3）** |
| HK／政綱／中間選民 | 敘事＋內容編碼 | `docs/*`、`data/hongkong_factor` 等 | — | — | — | 無鄉鎮因果係數 | **補充敘事** |

## 二、家族 A 定稿表（2026-10-30 乾淨重跑）

來源：`results/family_a_county_presidential/`（鏡像 `analysis/taiwan_core/`）  
腳本：`scripts/taiwan_core_backtest.py`  
規則：訓練年 **嚴格小於** 測試年；測試屆 2012／2016／2020／2024；每屆 22 縣市一齊測。  
可重現說明：`analysis/taiwan_core/REPRODUCIBILITY.md`、`family_a_mean_metrics.csv`

| 模型 | 四屆平均 MAE | 四屆平均 RMSE |
|---|---:|---:|
| Previous-election baseline | **0.0793** | 0.0814 |
| Random Forest | **0.1041** | 0.1165 |
| HistGradientBoosting | **0.1189** | 0.1268 |
| Ridge | **0.1786** | 0.1855 |

與舊 tracker（Prev 0.0793／RF 0.1040／HGB 0.1189／Ridge 0.1786）相比，僅 RF 4 位由 0.1040 → **0.1041**（sklearn 1.9.1 vs 舊 1.7.x）；結論不變：**基準優於所有訓練模型**。

口試一句（I-A）：在 4 個測試屆、88 個縣市×年列上，這是**歷史方法比較**，不是未來準確率保證，也不發布勝選機率。同屆 22 縣市**非獨立**，不可把 88 當獨立 N。

## 三、I-C：重要性／消融用語

- 圖表與正文只稱 **model dependence in OOT**、**permutation MAE increase**、**移除特徵組後 MAE 變化**。  
- **禁止**：造成勝選、政策影響、因果效應、複製文獻 mixed logit 係數。  
- Ablation／importance 描述「模型依賴什麼」，不是「什麼讓候選人勝出」。

## 四、提交前檢查清單

- [x] README 開頭有單一家族 A 主結果表；其餘標題含「補充」或 Extensions  
- [x] 每個對外主 MAE 附：資料集＋切分＋測試樣本（4 屆／88 列／縣市）  
- [x] 不把鄉鎮 0.149 與縣市家族 A 0.0793 混稱同一個「基準」  
- [x] importance／ablation 旁有非因果 caution（老師報告＋圖標題）  
- [x] conformal／區間標「未校準／探索」，不進主貢獻  
- [x] 美台／空間／HK／政綱標補充，不與家族 A 混報  
- [x] `summary.json`／`family_a_mean_metrics.csv` 與 README／老師報告數字一致（4 位）  
- [x] 可重跑：`python scripts/taiwan_core_backtest.py`
- [x] 文獻脈絡兩條線＋申請資料未取得說明（`docs/文獻脈絡路線.md`／`申請資料未取得時之報告說明.md`）
- [x] Guide 完成度總表：`docs/GUIDE_COMPLETION_STATUS.md`（I-A…I-K／Demo 1–7；2026-10-30）

## 五、已知易混數字（本包）

| 數字 | 是什麼 | 不是什麼 |
|---|---|---|
| **0.0793** | 家族 A：縣市 DPP、4 屆 OOT、Previous baseline | 不是鄉鎮 PreviousShare |
| **0.1041** | 家族 A RF（本重跑定稿） | 舊稿 0.1040 已作廢 |
| ~0.149 | 鄉鎮 DPP 2024 PreviousShare（三黨補充線） | 不是家族 A 四屆平均 |
| improved Ridge ~0.04x DPP swing | 補充線＋ESC 全國 PROXY | 不是家族 A 主結果 |
| US→TW HGB ~0.07 | RQ3 探索 transfer | 不可當政治轉移結論 |

## 六、重跑指令

```bash
/workspace/taiwan-election-data/.venv/bin/python scripts/taiwan_core_backtest.py
# 輸出：results/family_a_county_presidential/
# 鏡像：analysis/taiwan_core/
```


## 七、文獻脈絡路線（2026-10-30）

- 設計：`docs/文獻脈絡路線.md`  
- 缺申請資料時可貼：`docs/申請資料未取得時之報告說明.md`  
- 老師報告已加 §五-B  
- **無新主 MAE**：未因本輪改動家族 A／A+ 定稿數字（Prev 0.0793；A+ Ridge 0.1425）  
- 公開探查客家 CSV → 全國 only → DATA_GAPS；不入模  
- 檢查：不把解釋線文獻係數寫進家族 A 權重表；不因缺 TEDS 捏造個人效應  
