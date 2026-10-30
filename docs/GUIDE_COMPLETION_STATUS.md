# Guide 完成度總表（US–Taiwan Election Forecasting FYP）

**日期：** 2026-10-30（Asia/Hong_Kong, UTC+8）— 本輪補強：RQ2／RQ3／RQ4 誠實框住綜合章＋老師報告第 6 節；另：老師 PDF、Extension 驗證、US GCN NE pilot、申請清單  
**對照：** 《US–Taiwan Election Forecasting FYP Research and Demo Implementation Guide》（筆電 docx 離線；以 `GUIDE_ALIGNED_ISSUE_TRACKER`／`GUIDE_ANSWERS` 為 RQ1–4、Core／Extension、Demo 1–7、§19 代理）  
**交付包：** `/workspace/taiwan-election-github`  
**定位：** 方法論 OOT 區域得票份額比較——**不是**勝選預測或因果歸因。

---

## 一、總結一句

家族 A（RQ1）與美國 Core、RQ2 縣市同窗、RQ3、文獻／申請限制已齊；民調／政綱為 **Extension**（已補驗證狀態表＋縣市缺口說明）；US trainable GCN 全縣仍 **DEFERRED**，但已有 **New England `us_gcn_pilot`**；老師報告 **PDF 已重產**；TEDS／ESC Excel／筆電抄 zip／Git remote 仍 blocked。

---

## 二、Issue I-A … I-K

| ID | Guide 對應 | 狀態 | 證據（路徑／數字） | 備註 |
|----|------------|------|-------------------|------|
| **I-A** 外推能力／樣本限制敘事 | RQ1；§6 OOT；§19 | **DONE** | README、`SUBMISSION_AUDIT`、老師報告 §〇：4 測試屆／**88** region-year；Prev **0.0793** | 勿把 88 當獨立 N |
| **I-B** 政綱／議題入模 | §3；§6.2 ablation | **PARTIAL**↑ | Extension＋`docs/extension_poll_platform_VERIFICATION.md`（來源／年驗證狀態）；History+Platform Ridge **0.1113** | 未進 Family A；`verified` 欄仍空；**非因果** |
| **I-C** 重要性≠因果 | Demo 5；§19 | **DONE** | 一律 model dependence | 口試防雷已寫 |
| **I-D** 2026 地方情境 | §19 scenario | **DONE**（降級） | 補充／scenario；非主結果 | 非 Family A |
| **I-E** 民調≠模型目標 | §3／§19 | **PARTIAL**↑ | `縣市民調缺口說明.md`：poll long 表 **5165＝national**；UniformPollSwing＝Extension | 縣市面板仍無 |
| **I-F** 多線混淆 | §4 Core vs Extension | **DONE** | 單一主入口＋家族表 | RF **0.1041** |
| **I-G** Git／可重現 | §17；§19 | **PARTIAL** | 本地 commit；腳本可重跑 | 無 remote |
| **I-H** 美台／跨國 | RQ3；Demo 1／7 | **DONE** | 合成＋真實；美 Prev **0.0345**；TW **0.0793** | 不作政治轉移結論 |
| **I-I** Conformal | RQ4；Demo 6 | **PARTIAL** | 探索、非產品；US Ridge 覆蓋約 **0.91**；TW 跨年不穩（詳 `RQ2_RQ3_RQ4_誠實框住.md` §三） | 未當校準產品 |
| **I-J** 空間 vs 非空間 | RQ2 | **DONE**↑ | TW＋US SGC；**NE pilot** Prev≈**0.0314**；全縣 GCN DEFERRED；誠實框住：`RQ2_RQ3_RQ4_誠實框住.md` §一 | 子集數字勿混全縣 |
| **I-K** Domain adaptation | RQ3 Extension | **PARTIAL** | CORAL／MMD／DANN 已跑；DA 未必贏 no_da；框住：`RQ2_RQ3_RQ4_誠實框住.md` §二 | 薄特徵；無政治轉移結論 |

**狀態圖例：** DONE＝齊；PARTIAL＝有產物未達理想／未進 Core；N/A-application＝需申請；DEFERRED＝刻意延後。↑＝本輪有文件／pilot 補強。

---

## 三、Demo 1–7

| Demo | 狀態 | 路徑／說明 |
|------|------|-----------|
| **1** 合成 US/TW | **DONE** | `scripts/make_demo_us_tw_synthetic.py` |
| **2** 時間外 backtest | **DONE** | Family A TW／US |
| **3** OOT 誤差圖 | **DONE** | `results/.../charts/` |
| **4** Feature ablation | **DONE** | Core＋Extension |
| **5** Importance | **DONE** | feature_weights；非因果 |
| **6** Conformal | **PARTIAL** | 探索 |
| **7** Joint US/TW | **DONE** | 合成＋真實 |

---

## 四、Core vs Extension 檢查清單

| 項目 | 狀態 |
|------|------|
| Previous-election baseline（TW／US） | DONE |
| Ridge／RF／HGB OOT | DONE |
| Expanding-window OOT | DONE |
| Feature ablation（歷史群） | DONE |
| Explainability（非因果） | DONE |
| Conformal 產品級 | **未達** → 探索 |
| 總統同窗 non-spatial vs GCN/SGC | DONE（TW＋US SGC；US 全縣 trainable DEFERRED；**NE pilot DONE**） |
| Shared encoder／DA／low-data | PARTIAL |
| TEDS／完整 ESC Excel | **N/A-application**（checklist 已刷新） |
| 縣市／鄉鎮歷史民調面板 | 缺 → Extension 僅全國 |
| 老師報告 PDF | **DONE**（LibreOffice 自 docx；`老師說明報告_PDF_BUILD.md`） |
| 文獻脈絡＋專家論文 | DONE |

---

## 五、主數字釘死（勿混報）

| 家族 | 數字 |
|------|------|
| **A（TW RQ1）** | Prev **0.0793**；RF **0.1041**；HGB **0.1189**；Ridge **0.1786** |
| **A+ enriched** | Ridge **0.1425** |
| **US Core** | Prev **0.0345** |
| **RQ2 TW** | Prev 0.0793 最好 |
| **RQ2 US 全縣** | Prev 0.0345；SGC ~0.0437 |
| **us_gcn_pilot NE** | Prev ≈**0.0314**（僅子集） |
| **Extension Poll** | UniformPollSwing **0.0266**（探索） |
| **RQ4** | 不引用單一「已校準」覆蓋率當產品；見 `RQ2_RQ3_RQ4_誠實框住.md` |

---

## 六、仍 blocked／DEFERRED

1. **TEDS／ESC Excel** — 見 `待申請資料清單.md`（本輪不代申請、不爬）。  
2. **筆電 zip 抄寫** — 用戶機離線；箱上 zip 本輪再刷新。  
3. **全縣可訓練 US GCN** — DEFERRED；NE pilot 已替代「完全無 trainable 數字」。  
4. **Git remote** — 需用戶授權。  
5. **政綱全面 verified=true** — 狀態表已列；逐筆手核仍可後補。

---

## 七、相關文件

- `docs/老師說明報告.pdf`＋`老師說明報告_PDF_BUILD.md`  
- `docs/extension_poll_platform_VERIFICATION.md`、`縣市民調缺口說明.md`  
- `docs/US_GCN_DEFERRED.md`、`results/us_gcn_pilot/`  
- `docs/待申請資料清單.md`、`TEDS_ACCESS.md`、`ESC_SOURCES_AND_ACCESS.md`  
- `docs/GUIDE_ALIGNED_ISSUE_TRACKER.md`、`GUIDE_ANSWERS.md`、`SUBMISSION_AUDIT.md`

## 六、美台資料對等（2026-10-30）

要求：TW／US **同一資料層級**。詳表：`docs/美台資料對等表.md`。

| 層 | 狀態 |
|----|------|
| 歷史票 prev／prev2 | 兩邊 DONE |
| 滯後失業／所得／密度 | 大致對等（定義異；US 密度靜態） |
| 教育 | **不對稱**（US 有；TW 長面板缺） |
| 全國民調 Extension | US 已補 538＋Wiki 年代理（2004 空）；TW 既有 long 表 |
| 政綱 Extension | US APP metadata＋主題；TW Wiki 長表 |
| 認同／黨性氣候 PROXY | TW ESC；US MEDSL homebrew lean（非 Cook PVI） |
| 縣市民調／TEDS／ANES | 兩邊皆缺／申請 |

新檔：`data/us_parity_extension/`。**未改 Family A MAE。**


---

## 六、老師進度報告回饋對應（2026-10-30 16:32 Asia/Hong_Kong (UTC+8)）

老師意見：第一版可作 progress report；需 (a) 更組織化；(b) 比較教育／經濟／媒體重要性；(c) 緒論／文獻／資料分析加引用；(d) **先基本分析再預測**。

| 回饋 | 狀態 | 證據 |
|------|------|------|
| 報告重組（Intro→Lit→Data→Basic analysis→Prediction→Limits→Next） | **DONE** | `docs/老師說明報告.md`／`.docx`／`.pdf` |
| 教育 vs 經濟 vs 媒體重要性（TW＋US；權重≠因果） | **DONE**（Extension） | `results/factor_importance_edu_econ_media/`；縣市 OOT：Prev≫經濟≳教育；媒體僅全國小樣本 |
| 引用（緒論／文獻／資料分析） | **DONE** | 報告內嵌 Linzer、Abramowitz、Erikson & Wlezien、Brito、小笠原、Achen & Wang、Batto、ethnoterritorial DOI 等；書目不捏造 |
| 預測置於基本分析之後 | **DONE** | 報告 §4 因素 → §5 家族 A／US Core（Prev **0.0793**／**0.0345** 未改） |

**實證排序（誠實）：** 歷史票壓倒；縣市層 **經濟代理略優於教育**（尤其 TW Prev+Economic Ridge MAE 0.0725 vs Prev 0.0793）；媒體不可與縣市因素並列宣稱。
