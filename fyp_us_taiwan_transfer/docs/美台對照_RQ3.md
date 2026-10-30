# 美台對照（RQ3）短報告 — 教師／口試用

**日期：** 2026-10-30（UTC+8）  
**定位：** 方法論 — 時間外（OOT）區域得票份額 MAE；**不是**勝選預測或政治結論。  
**Guide：** Demo 1／7、I-H、**I-K**（CORAL／MMD／adversarial DA、low-data）。

## 做咗咩

1. **共享 schema**——美＝Dem 兩黨份額；台＝DPP 有效票份額。**同名欄位 ≠ 同一政治構念**（Guide §19）。
2. **Demo 1 合成**＋**Demo 7** country-only／joint／transfer。
3. **真實美國縣級** MEDSL 2000–2024；**台灣縣市** 2000–2024 對齊。
4. **社經合併（本輪完成）：**
   - 美：ERS 失業（按年）、ERS 學士比例波段、ERS 2022 中位家戶所得（或缺則 ACS2020）、ACS2020 人口密度  
   - 台：稅務平均所得＋人口密度（縣市靜態代理）  
   - Joint 共享：`prev_share_A`, `log_total_votes`, `z_income`, `z_log_density`（**國內 z-score**，避免美元／新台幣混用）
5. **I-K：** CORAL、核 MMD／KMM 重權、adversarial DANN（torch）、low-data 曲線（k＝0…22）。

## RQ3 狀態

| 講法 | 可唔可以 |
|---|---|
| 「跨國實驗管線已測試」 | ✅ |
| 「有真實 country-only／joint／transfer MAE（含社經）」 | ✅ |
| 「CORAL／MMD／DA／low-data 已跑」 | ✅（結果顯示 DA 未必贏） |
| 「美台表徵已證明可轉移／同一構念」 | ❌ |

## 真實數字（本 run 計算）

### 含 ACS／ERS 社經（`real_joint_mae_mean_acs.csv`）

- **台 country-only：** Ridge ≈ **0.0787**；PreviousShare ≈ **0.0793**
- **美 country-only：** PreviousShare ≈ **0.0345**
- **Joint pooled：** ≈ **0.035**（美國 N 主導）
- **Transfer US→TW：** HGB ≈ **0.0690**；Ridge ≈ **0.0705**（PrevShare 0.079）

### DA（`da_transfer_mae_mean.csv`，US→TW）

- no_da Ridge **0.0705** ≈ MMD **0.0700**；CORAL **0.0784**；DANN **0.118**（未贏 baseline）

### Low-data

- TW→US：加標籤目標縣可將 no_da MAE 由 ~0.046 降到 ~**0.039**（k=22）  
- US→TW：幾乎唔受 k 影響（源域已經好大）

## 限制

- 美國縣數 ≫ 台灣縣市；joint 被美國主導。  
- 台所得 ≠ 美中位家戶所得；只用國內相對位置（z）。  
- 台無對齊失業／學士率 → 唔入 joint 共享集。  
- DA 喺呢組薄特徵上**無穩贏** no_da——口試如實講。  
- 合成 MAE 唔好同真實混報；唔好同臺灣 Core 家族 A 混報。

## 檔案入口

- 英文 README：`fyp_us_taiwan_transfer/README.md`  
- 特徵清單：`cleaned/joint_shared_feature_list.csv`  
- 美國來源：`raw/us/PROVENANCE.md`
