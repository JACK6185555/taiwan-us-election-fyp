# Taiwan FYP — 本輪改進（fyp_taiwan_improved）

對照 FYP guide：真實中選會鄉鎮面板、時間外推、ablation、importance、conformal。  
本目錄係喺 `fyp_taiwan` / `fyp_taiwan_swing` / `fyp_taiwan_parties` / `fyp_taiwan_legislative` 之上嘅 **priority improvements**。

## 本輪改進做了什麼

1. **目標改為 SWING（ΔV）同 DEMEANED（得票率 − 全國加權平均）**  
   民進黨／國民黨總統＋區域立委；有滯後先做民眾黨。同 ZeroSwing／PreviousShare／PreviousDemeaned 比較 MAE。
2. **ESC 政大選研 PROXY（全國）**  
   `proxies/esc_nccu_national_proxy.csv`：選前窗口嘅政黨偏好／統獨 lean 近似值（圖表＋新聞引用）。  
   **唔係** TEDS 微資料；**冇**公開縣市級黨性。Excel 官方數值要申請。已標 `PROXY`。
3. **2024 三腳督分黨頭**（DPP／KMT／TPP）分開建模。
4. **Conformal 打喺 SWING**（唔只 level），並報告 leave-one-election residual quantile。
5. **空間滯後**：同縣市內其他鄉鎮嘅上屆得票 leave-one-out 平均 + ablation。

## 結果有冇好啲？（2024 holdout）

| 賽事 | 黨 | 目標 | 最佳基準 MAE | 改進模型 MAE | 改進有冇贏 |
|------|----|------|-------------|-------------|-----------|
| 總統 | DPP | swing | ZeroSwing 0.149 | **Ridge 0.042** | ✅ |
| 總統 | DPP | level | PreviousShare 0.149 | **Ridge 0.067** | ✅ |
| 總統 | DPP | demeaned | PreviousDemeaned 0.036 | RF 0.040 | ❌ |
| 總統 | KMT | swing | ZeroSwing 0.060 | RF 0.073 | ❌ |
| 總統 | KMT | level | PreviousShare 0.060 | RF 0.053 | ✅ 微贏 |
| 總統 | TPP | level | PrevThird 0.209 | ESC_TPP_pid PROXY 0.104 | ✅（全國常數） |
| 立委區域 | 多數 | — | 基準仍較穩 | ML 多數輸／微贏 | 見 `before_after_2024.csv` |

**解讀**
- Ridge 贏 DPP swing／level，主要靠 **ESC 全國氣候當「年分體制」指標**（`esc_*_delta` 訓練 impurity 最高）；同一年內 ESC 係常數，**唔會幫你排鄉鎮名次**。
- **相對版圖（demeaned）** 仍然係 PreviousDemeaned 較穩 → 地理結構黏性高。
- 空間滯後有細增益（DPP swing RF：All 0.166 vs NoSpatial 0.171）；ESC ablation 更大。
- Conformal（名義 90%）：DPP swing 實際覆蓋約 **0.3%**（2024 全國衝擊太大）；KMT 覆蓋 100% 但半寬約 0.35（幾乎冇資訊）。Naive conformal 喺分布偏移下仍然失效／無用。

## 重跑

```bash
cd /workspace/taiwan-election-data
.venv/bin/python fyp_taiwan_improved/scripts/run_taiwan_improved.py
```

## 主要輸出

- `backtest_results_improved.csv` — 全部回測
- `before_after_2024.csv` — 基準 vs 改進
- `ablation_spatial_esc.csv` — 空間／ESC ablation
- `feature_importance_improved.csv`
- `conformal_swing_meta.csv` + intervals
- `level_from_swing_comparison.csv`
- `charts/*.png`
- `proxies/esc_nccu_national_proxy.csv` + 本 README

## 仍然欠／限制

- TEDS／選研 **個人** 黨性、統獨微資料（要申請）
- 縣市級黨性公開序列不存在 → 只用全國 PROXY + 區域 dummy + 滯後結構
- 失業率、民調指數鄉鎮層一致資料仍缺（唔造假）
- 民眾黨總統只有 2024；立委提名範圍有限
- 未做完整 GCN／跨國 transfer（本輪只加 spatial lag）
