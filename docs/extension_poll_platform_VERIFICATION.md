# Extension：民調／政綱驗證狀態表

**日期：** 2026-10-30（Asia/Hong_Kong, UTC+8）  
**範圍：** `data/extension_poll_platform/`＋`results/extension_poll_platform/`  
**定位：** Extension only——**唔升格**入 Core specification。

## 政綱 `taiwan_platforms_long.csv`（510 列）

| year | 列數 | 主要來源類型 | 驗證狀態 | 說明 |
|-----:|-----:|--------------|----------|------|
| 2000 | 17 | 遠見／華視／Taipei Times／學術（二次） | **auto／二次填補**；未逐筆 `verified=true` | 公報無政見欄；連十條全文仍缺 |
| 2004 | 51 | Wikipedia | **auto_wikipedia** | 議題敘事二次；待手核 |
| 2008 | 173 | Wikipedia | **auto_wikipedia** | 同上 |
| 2012 | 61 | Wikipedia | **auto_wikipedia** | 同上 |
| 2016 | 44 | Wikipedia | **auto_wikipedia** | 同上 |
| 2020 | 62 | 風傳媒／中央社 | **secondary_news_fill** | 已取代舊口號列；未全面手核 |
| 2024 | 102 | CEC 公報 OCR（95）＋wiki（7） | **auto_cec_ocr**（主） | OCR 可重跑；建議抽樣手核 |

- CSV 欄 `verified`：**全部空白**（誠實：未完成全面人工覆核）。  
- `pre_election_flag`：`1`≈411；`review`≈99。  
- 詳見 `data/extension_poll_platform/taiwan_platforms_README.md`、`docs/taiwan_platforms_2000_2020_fill.md`。

## 民調 `taiwan_polls_long.csv`

| 項目 | 狀態 |
|------|------|
| 總列／region | 5165 列；**全部 `national`**（無縣市） |
| 自動檢查 | 無選後列；value 在 0–100（見 `docs/taiwan_polls_QA_spotcheck.md`） |
| 人工抽樣 | 每屆約 2 列已抽（同上 QA 表） |
| `field_end`／`published_at` | 可解析 5165／5165 |
| `field_start` 缺 | **7 個 poll_id／14 列**（多為 2004 blackout／二次引用）；**不捏造**開始日——註記於 note（`QA:field_end_approx`／`QA:secondary_cite`） |
| 縣市民調 | **無** → `docs/縣市民調缺口說明.md` |

## Ablation 數字（勿混報核心設定）

見 `results/extension_poll_platform/README.md`：UniformPollSwing **0.0266**、History+Poll Ridge **0.0611** 等為 Extension 探索。

## 本輪改進（2026-10-30）

- 新增本驗證狀態表＋縣市民調缺口說明。  
- 確認 poll `region` 全 national；記錄 2004 `field_start` 缺口而不填假日期。  
- **未**重跑 Wikipedia 爬蟲；**未**把 poll／platform 併入 Core specification。
