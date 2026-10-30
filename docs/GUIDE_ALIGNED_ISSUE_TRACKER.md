# Guide 對齊問題改進追蹤表（更新）

**原版日期：** 2026-10-27　｜　**本包更新日期：** 2026-10-30（UTC+8）  
**目的：** 以《US–Taiwan Election Forecasting FYP Research and Demo Implementation Guide》為框架追蹤 I-A…I-K 與 Demo 1–7。  
**完成度總表：** [`GUIDE_COMPLETION_STATUS.md`](GUIDE_COMPLETION_STATUS.md)  
**定位：** 方法論研究（OOT 區域得票份額）——**不是**勝選預測或因果歸因。

**主結果釘死（核心設定／RQ1）：** Previous-election **0.0793** ＜ RF **0.1041** ＜ HGB **0.1189** ＜ Ridge **0.1786**（2026-10-30 乾淨重跑）。

---

## 問題追蹤表（2026-10-30 狀態）

| Issue ID | Guide 對應 | 2026-10-27 | **2026-10-30** | 證據／動作 |
|---|---|---|---|---|
| **I-A** | RQ1；§6；§19 | 敘事待收斂 | **DONE** | README／SUBMISSION_AUDIT／老師報告：4 屆／88 region-year；歷史方法比較 |
| **I-B** | §3；§6.2 | 長表起步 | **PARTIAL** | Extension 政綱長表＋ablation 已入包；未進 Core specification；verified 未齊 |
| **I-C** | Demo 5；§19 | 易誤讀 | **DONE** | 一律 model dependence；禁因果／勝選因素 |
| **I-D** | §19 scenario | 需降級 | **DONE** | 2026／地方標補充；非主結果 |
| **I-E** | 民調構念 | 可比性文件 | **PARTIAL** | Extension 全國 poll；無縣市面板；不把 poll＝標籤準確率 |
| **I-F** | Core vs Extension | 多線易混 | **DONE** | 單一核心設定 入口；數字 0.1041 定稿 |
| **I-G** | 可重現 | 無 .git | **PARTIAL** | 本地 git init；Core 可重跑對齊四位；無 remote |
| **I-H** | RQ3；Demo 1／7 | 無美國面板 | **DONE** | 合成 harness＋MEDSL 真實 US；country-only／joint／transfer |
| **I-I** | RQ4；Demo 6 | 覆蓋 2.9–13.6% | **PARTIAL** | improved protocols＋US ~0.91 Ridge；**仍標探索** |
| **I-J** | RQ2 | 僅地方 GCN | **DONE** | 總統縣市同窗 TW＋US SGC；trainable US GCN deferred |
| **I-K** | DA／low-data | 未做 | **PARTIAL** | CORAL／MMD／DANN／low-data 已跑；DA 未穩贏 |

---

## Demo 完成度（2026-10-30）

| Demo | 2026-10-27 | **2026-10-30** | 路徑 |
|---|---|---|---|
| 1 合成 US/TW | 無 | **DONE** | `scripts/make_demo_us_tw_synthetic.py` |
| 2 OOT backtest | TW Core | **DONE**（TW＋US） | `results/family_a_*` |
| 3 OOT 圖 | 有 | **DONE** | `charts/` |
| 4 Ablation | 部分 | **DONE**＋Extension poll／platform | `results/extension_poll_platform/` |
| 5 Importance | 有 | **DONE**＋權重說明 | `results/feature_weights/` |
| 6 Conformal | 覆蓋不足 | **PARTIAL**（探索） | `results/conformal_rq4_improved/` 等 |
| 7 Joint | 無 | **DONE**（合成＋真實） | `docs/美台對照_RQ3.md` |

---

## 用語規範（Guide §19）— 維持

| 使用 | 避免 |
|---|---|
| OOT backtest、expanding window | 隨機 80/20 當主驗證 |
| vote-share MAE、feature ablation | 「預測邊個贏」「勝選機率」 |
| model dependence | 「某政策造成勝選」 |
| calibration／empirical coverage（誠實報失敗） | 未校準卻稱可靠區間 |
| transfer、joint vs country-only | 假設美台構念完全相同 |
| scenario／補充 | 補充 MAE 與核心設定 混表 |

---

## 相對 2026-10-27 tracker 的資料進度（已併入本包）

| 項 | 狀態 |
|---|---|
| 美國縣級面板（MEDSL＋社經） | **已有**（I-H 不再 blocker） |
| 總統 RQ2 同窗 | **已有** |
| History+Poll／Platform ablation | **已複製為 Extension**（見 `results/extension_poll_platform/README.md`） |
| TEDS | 仍 **N/A-application** |
| 全縣 trainable US GCN | **DEFERRED** |

### Teacher feedback (2026-10-30 16:32 Asia/Hong_Kong (UTC+8))
Reorganized teacher report; edu/econ/media factor Extension at `results/factor_importance_edu_econ_media/`; citations in intro/lit/analysis; prediction after §4. Core specification unchanged.
