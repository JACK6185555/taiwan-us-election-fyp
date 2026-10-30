# X（Twitter）情緒／聲量特徵 — Extension 試點說明

**日期：** 2026-10-30（Asia/Hong_Kong, UTC+8）  
**狀態：** **Extension 試點完成**；**唔改 Family A MAE**  
**問題來源：** 用戶問可否用 X 爬蟲攞政黨／人物／政綱評論做情緒 feature。

---

## 一句結論

**可以作為報告／demo 用嘅全國級 Extension feature（情緒均值、正負比、樣本聲量、陣營差）**；  
**唔可以**當縣市 Family A 特徵（冇可靠 geo 對齊縣市）；**唔等於**因果；**唔係** HTML 爬蟲（用 Grok Bot 內建 X MCP `search_posts_all` 公開讀取）。

本輪差分符號 vs 全國勝方：**4 屆入面 3 屆同向**（TW2020、US2020、US2024 同向；**TW2024 反向**）。見下表。

---

## 方法

| 項目 | 做法 |
|------|------|
| API | Grok Bot 內建 X MCP：`search_posts_all`（全檔案庫）；**唔** HTML scrape、**唔**瀏覽器登入、**唔**要用戶 API key |
| 粒度 | **選年 × 實體（政黨／人物／議題）** 全國聚合；**縣市無法 join**（公開搜尋無穩定鄉鎮／縣 geo） |
| 視窗（UTC） | TW2024：2023-12-01～2024-01-12；TW2020：2019-12-01～2020-01-10；US2024：2024-10-01～2024-11-04；US2020：2020-10-01～2020-11-02 |
| Query | 實體詞 + 選舉關鍵字；`lang:zh`（TW）／`lang:en`（US）；`-is:retweet -is:reply` 減洗版 |
| 分頁 | 每 query×視窗 **1～2 頁**，`max_results=25`（試點預算約 **24 次** 有效讀取，遠低於 1000／日） |
| 中文情緒 | SnowNLP（0～1→[-1,1]）+ 小量政治正／負詞 nudge（支持／凍蒜／賣台／拒投等） |
| 英文情緒 | `vaderSentiment` compound [-1,1] |
| 聚合 | `mean_sentiment`、`pct_pos`／`pct_neg`、`n_posts`、`engagement_sum`；陣營差＝體積加權 mean（DPP−KMT 或 DEM−GOP） |

腳本：`scripts/build_x_sentiment_extension.py`（可重跑評分；**重新拉文**要再呼叫 X MCP）。

---

## 主數字（試點樣本）

原始／評分：`data/x_sentiment_extension/`（`raw/`、`x_posts_scored_long.csv` ≈506 帖、`x_entity_year_aggregates.csv`）。  
結果：`results/x_sentiment_extension/`。

### 實體均值（節錄）

| 國 | 年 | 實體 | n | mean_sent | pct_pos | pct_neg |
|----|----|------|---|-----------|---------|---------|
| TW | 2024 | 賴清德 | 40 | −0.40 | 0.28 | 0.73 |
| TW | 2024 | 侯友宜 | 35 | −0.27 | 0.34 | 0.63 |
| TW | 2024 | 柯文哲 | 16 | −0.19 | 0.38 | 0.63 |
| TW | 2024 | 民進黨 | 44 | −0.66 | 0.16 | 0.84 |
| TW | 2024 | 國民黨 | 41 | −0.71 | 0.12 | 0.88 |
| TW | 2024 | 兩岸（議題） | 21 | −0.61 | 0.14 | 0.86 |
| TW | 2020 | 蔡英文 | 15 | −0.49 | 0.20 | 0.80 |
| TW | 2020 | 民進黨 | 21 | −0.45 | 0.24 | 0.76 |
| TW | 2020 | 國民黨 | 23 | −0.48 | 0.22 | 0.78 |
| US | 2024 | Trump | 48 | +0.19 | 0.58 | 0.27 |
| US | 2024 | Harris | 47 | +0.11 | 0.51 | 0.28 |
| US | 2024 | Democrats | 24 | +0.03 | 0.54 | 0.46 |
| US | 2024 | Republicans | 25 | +0.11 | 0.56 | 0.44 |
| US | 2024 | immigration | 25 | −0.12 | 0.36 | 0.56 |
| US | 2020 | Biden | 24 | +0.13 | 0.54 | 0.33 |
| US | 2020 | Trump | 25 | −0.06 | 0.36 | 0.52 |

（完整表見 CSV。TW 討論整體偏負＝平台／海外用戶結構常見現象，**唔好**當「全民情緒」。）

### 陣營差 vs 全國勝方

| 國 | 年 | 差（綠／Dem − 藍／GOP，體積加權） | 全國勝方 | 符號同向？ |
|----|-----|----------------------------------|----------|-----------|
| TW | 2020 | +0.011 | 蔡英文（DPP） | **是** |
| TW | 2024 | −0.026 | 賴清德（DPP） | **否**（藍側相對冇咁負） |
| US | 2020 | +0.121 | Biden（DEM） | **是** |
| US | 2024 | −0.075 | Trump（GOP） | **是** |

檔：`results/x_sentiment_extension/sentiment_differential_vs_winner.csv`。

**解讀：** 試點顯示「全國級情緒差」有時同勝方同向，但 **n 細、樣本有偏、TW2024 已反向** → 只適合作 Extension 敘事／敏感度，**禁止**寫入 Family A 或宣稱預測力。

---

## 點解唔入 Family A

1. **粒度唔對：** 縣市開票需要縣特徵；X 公開搜尋**無**穩定縣／鄉鎮地理欄可 join。  
2. **樣本偏誤：** 活躍用戶 ≠ 選民；TW 中文區海外／立場極化嚴重；洗版帳號即使用 `-is:reply` 仍殘留。  
3. **評分誤差：** SnowNLP 主練產品評論；VADER 對諷刺／口號弱；政治詞表僅 nudge。  
4. **試點體量：** 每實體數十帖，唔係全量日均。  
5. **時間／語言／平台規則**隨年改變，跨屆可比性有限。

→ 與民調 Extension、ESC PROXY 同類：**報告可寫「另開一層」**；主表 MAE 不變。

---

## 倫理與合規

- 只用**公開**搜尋 API（經官方 MCP），聚合統計為主；raw 含公開帖文文字作可重現。  
- **唔**針對私人帳、**唔**做 doxx、**唔**為操盤／水軍用途設計。  
- Feature 權重 ≠ 因果；報告須標 **PROXY／試點**。  
- 重跑需 X 讀取配額（約 30／分、1000／日）；本試點遠低於上限。

---

## 可重現性

1. Raw JSON 頁：`data/x_sentiment_extension/raw/*_p*.json.txt`（MCP 回傳）。  
2. `python scripts/build_x_sentiment_extension.py` → 重算 CSV／summary。  
3. 若要刷新貼文：用同一 query／視窗再呼叫 `search_posts_all`（可 `pagination_token`），覆蓋 raw 後重跑腳本。  
4. `data/x_sentiment_extension/PROVENANCE.json`。

---

## 同其他缺口嘅關係

見 `docs/剩餘缺口公開替代方案.md` 新增列 **G. X 情緒／聲量**；`docs/美台資料對等表.md` 資料層表已加一行。  
**冇**破壞既有 `公開替代可分析範圍.md`（該檔本 repo **唔存在**，無需改）。
