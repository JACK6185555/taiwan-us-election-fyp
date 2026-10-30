# 核心設定 核心回測重現（I-A／I-F）

檢查日期：2026-10-30 00:21:07 UTC+8
腳本：`scripts/taiwan_core_backtest.py`
面板：`data/county_panel_features_2000_2024.csv`
切分：expanding window；train year < test year；每折 22 縣市
測試屆：2012／2016／2020／2024；測試列 **88** region-year
環境：scikit-learn 1.9.1（`taiwan-election-data/.venv`）

## 四屆平均 MAE（定稿）

| 模型 | MAE（原始） | MAE（4 位） | 舊 tracker 4 位 | Δ |
|---|---:|---:|---:|---:|
| Previous-election baseline | 0.079265 | **0.0793** | 0.0793 | 0.0000 |
| Random Forest | 0.104084 | **0.1041** | 0.1040 | +0.0001 |
| HistGradientBoosting | 0.118865 | **0.1189** | 0.1189 | 0.0000 |
| Ridge | 0.178580 | **0.1786** | 0.1786 | 0.0000 |

**說明：** 與舊 tracker 唯一可見差異是 Random Forest 四捨五入由 0.1040 → **0.1041**（原始 0.104084 vs 舊環境約 0.104020）。文件一律改用本表；口試可註「sklearn 版本差異，不影響結論：基準仍優於所有訓練模型」。

完整輸出：`backtest_metrics.csv`、`backtest_predictions.csv`、`feature_ablation_*.csv`、`feature_importance_*.csv`、`summary.json`、`family_a_mean_metrics.csv`、`charts/`。

I-C：permutation／ablation／Ridge 權重僅描述 **model dependence in OOT**，**非因果**。
