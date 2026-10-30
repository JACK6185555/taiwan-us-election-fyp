# Guide 問題回答與完成度（2026-10-30 更新）

**原版：** 2026-10-27（僅台灣）　｜　**本版：** 納入美國 Core、RQ2／RQ3、文獻脈絡、Extension 民調／政綱。  
**Guide 原檔：** 筆電 `US_Taiwan_Election_FYP_Research_and_Demo_Guide.docx`（本輪離線未重開）。  
**總表：** [`GUIDE_COMPLETION_STATUS.md`](GUIDE_COMPLETION_STATUS.md)

## 給評審的總結

台灣核心設定：四個總統測試屆（2012–2024，88 縣市×年）上，**Previous-election baseline 平均 MAE 0.0793** 優於 RF 0.1041／HGB 0.1189／Ridge 0.1786。美國縣級 Core：PreviousShare **0.0345** 同樣優於訓練模型。RQ2 總統同窗空間模型未改寫此結論。RQ3 已有合成 Demo 與真實 country-only／joint／transfer／DA，但**不作「美台表徵已證明可轉移」政治結論**。RQ4 conformal 有改進與美國接近 90% 的覆蓋，台灣整體仍標**探索、非產品**。

## 四個研究問題（更新答案）

| Guide 問題 | 答案（2026-10-30） | 證據 |
|---|---|---|
| **RQ1** 時間泛化 | 可建 OOT 基準；所測 ML **未穩定勝過** PreviousShare（台／美皆然）。 | TW：`results/family_a_county_presidential/`；US：`results/family_a_us_county_presidential/` |
| **RQ2** 空間建模 | 總統縣市同窗已比：SGC 相對 NonSpatial 僅微弱或未贏；Prev 仍最好。全縣 trainable US GCN deferred。 | `docs/空間共邊GCN.md`、`docs/RQ2_美台空間.md` |
| **RQ3** 跨域轉移 | **已測試**管線與真實面板；joint 受美國 N 主導；DA 未穩贏 no_da。**未**宣稱構念同一或政治可轉移。 | `docs/美台對照_RQ3.md` |
| **RQ4** 不確定性 | 未達「產品級校準」；美國 Ridge 平均覆蓋約 0.91；台灣 improved 協議不穩定。 | `results/conformal_rq4_improved/`、US `conformal_coverage_summary.csv` |

## FYP Core

| 要求 | 狀態 |
|---|---|
| 前一屆基準 | **完成**（TW＋US） |
| Ridge／RF／GB | **完成** |
| OOT backtesting | **完成**（屆數少，保守解讀） |
| Feature ablation | **完成**（Core＋Extension poll／platform） |
| Explainability | **完成初版**（非因果） |
| Conformal | **探索；未當主貢獻** |

### 主結果表述（口試）

> 在台灣 2000–2024 縣市總統面板上，以 2012／2016／2020／2024 作四個整屆時間外測試時，Previous-election 平均 MAE 為 **0.0793**；RF／HGB／Ridge 為 **0.1041／0.1189／0.1786**。美國縣級 Core 上 PreviousShare 約 **0.0345** 亦優於訓練模型。此為小樣本歷史方法比較，不是未來準確率保證，也不推論勝選原因。

## Extension 與 Demo（更新）

見 `GUIDE_COMPLETION_STATUS.md` §三。重點：Demo 1／7 **已有**；民調／政綱 ablation 為 Extension（UniformPollSwing 0.0266 等**不可**取代核心設定 敘事）。

## 申請資料

TEDS／ESC Excel 未取得 → `docs/申請資料未取得時之報告說明.md`。本輪不要求用戶申請。

## 口試簡答

- **完成了什麼？** RQ1 台美 Core、RQ2 同窗、RQ3 管線＋真實、權重、文獻路徑、Extension 民調／政綱。  
- **能回答誰勝選嗎？** 不能；importance ≠ 因果。  
- **空間有幫助嗎？** 總統同窗未穩贏 Prev；微弱 SGC 訊號勿誇大。  
- **美台轉移成功嗎？** 實驗已跑；結果審慎，不作政治成功宣稱。  
- **勝選機率？** 不能；RQ4 非產品。
