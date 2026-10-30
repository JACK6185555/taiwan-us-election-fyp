# Extension：全國選前民調／政綱 ablation（非核心設定）

**日期：** 2026-10-30（UTC+8）  
**定位：** Guide §3／§6.2 polling／議題特徵的 **Extension** 探索；**不是** Core specification 主結果。  
**來源：** 自 `/workspace/tw2` 既有產物複製入本交付包（Wikipedia／二次引用民調表＋政綱長表；見 provenance）。

## 重要警告

1. **勿與核心設定 MAE 混報。** 核心設定 定稿 PreviousShare **0.0793**／RF **0.1041**／HGB **0.1189**／Ridge **0.1786**。
2. 民調為**全國**層級（同屆所有縣市同一 poll swing）；**無**縣市民調面板。
3. UniformPollSwing／History+Poll 的低 MAE 屬**探索**：僅 4 個測試屆、來源未全面人工覆核、2004 稀疏；**不可**當產品預測或勝選機率。
4. 政綱為內容編碼／詞頻代理，來源不齊（公報 OCR 限制）；**非因果**。
5. I-C：importance／ablation = model dependence，≠ 因果。

## 2012–2024 四屆平均 MAE（節錄）

| 實驗 | 模型 | MAE |
|------|------|----:|
| UniformPollSwing | prev_DPP+poll_swing | **0.0266** |
| History+Poll | Ridge | 0.0611 |
| History（對照） | PreviousElectionBaseline | **0.0793** |
| History+Poll | RandomForest | 0.1036 |
| History+Platform | Ridge | 0.1113 |
| History+Poll+Platform | Ridge | 0.1064 |
| History | Ridge | 0.1776 |

完整表：`poll_ablation_mean_mae_2012_2024.csv`、`platform_ablation_mean_mae_2012_2024.csv`。

## 檔案

| 路徑 | 內容 |
|------|------|
| `taiwan_poll_ablation_metrics.csv` | 民調 ablation 逐屆 |
| `taiwan_platform_ablation_metrics.csv` | 政綱 ablation 逐屆 |
| `../../data/extension_poll_platform/taiwan_polls.csv` | 選前民調原始長表（爬取） |
| `../../data/extension_poll_platform/taiwan_polls_long.csv` | prepare 後可用列 |
| `../../data/extension_poll_platform/taiwan_platforms_long.csv` | 政綱長表 |
| `../../docs/taiwan_polls_QA_spotcheck.md` 等 | QA／來源說明 |

## Provenance

- 腳本原址：`tw2/report/src/scripts/run_taiwan_poll_ablation.py`、`run_taiwan_platform_ablation.py`、`scrape_taiwan_polls_wikipedia.py`
- 本包**未重跑爬蟲**（避免重複抓取）；只複製既有 CSV／說明。
- TEDS 個體民調／縣市層民調：**未取得**（見 `docs/申請資料未取得時之報告說明.md`）。

- 驗證狀態：`../../docs/extension_poll_platform_VERIFICATION.md`
- 縣市民調缺口：`../../docs/縣市民調缺口說明.md`
