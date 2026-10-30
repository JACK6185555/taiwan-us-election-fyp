# 新聞／政策議題顯著度特徵 — Extension 說明

**日期：** 2026-10-30（Asia/Hong_Kong, UTC+8）  
**狀態：** **Extension 完成**；**唔改 Family A MAE／鎖定特徵集**  
**問題來源：** 用戶認為新聞或政策議題容易影響中間選民投票意向，要求加入特徵。

---

## 一句結論

**已加入作 Extension；可解釋中間選民議題氣候，但估唔到個人轉向。**

特徵＝選年×議題嘅 **聲量（volume）＋語氣（tone）PROXY**，描述競選期「邊啲議題喺公開討論入面較顯眼／較偏正負」。  
**唔係**個體 MIP（most important problem）、**唔係**縣市特徵、**唔係**因果識別。

---

## 點解係 Extension（唔入 Family A）

| 理由 | 說明 |
|------|------|
| 粒度 | 全國×選年（或選前月窗）；公開 X 搜尋**無**穩定縣／鄉鎮 geo |
| 觀測對象 | 睇到嘅係帖文／政綱標籤嘅議題注意力，**睇唔到**中間選民個人有冇轉向 |
| 樣本偏誤 | X 活躍用戶 ≠ 選民；海外中文／極化帳號偏多；新聞媒體帳重複發佈 |
| 理論角色 | 文獻上議題顯著度／議題所有權／Zaller RAS／MIP 同「弱黨性／中間選民可被說服」有關——呢度只提供 **氣候 PROXY**，唔做因果 ID |
| 權重 | `results/feature_weights/extension_issue_news_national_associations.csv` 入面嘅數字＝描述性 volume／tone，**≠** Family A 迴歸權重、**≠**因果 |

→ 同 `docs/X情緒特徵Extension說明.md`、民調／政綱 Extension 同一層：報告可寫「另開一層」；主表 MAE 不變（Prev **0.0793**）。

---

## 中間選民動機（簡）

- **Zaller（RAS）**：政治意識較低嘅人較易受近期訊息流影響。  
- **Issue ownership / salience**：邊個議題喺競選期「浮上嚟」，會改變比較維度。  
- **MIP**：調查問「最重要問題」先直接對到個人；我哋**冇**縣／個體 MIP 面板 → 用公開討論聲量作粗 PROXY。

**誠實限制：** 聲量高唔等於中間選民真係改票；反向同選擇偏誤都可能。

---

## 資料來源（只公開）

### A) X（官方 MCP，唔 HTML 爬牆）

- 工具：`search_posts_all`（全檔案庫）  
- 視窗（UTC）：TW2024 `2023-12-01～2024-01-12`；TW2020 `2019-12-01～2020-01-10`；US2024 `2024-10-01～2024-11-04`；US2020 `2020-10-01～2020-11-02`  
- TW 議題：兩岸、居住、能源、經濟；（2020 另加）防疫  
- US 議題：economy、immigration、abortion、healthcare、democracy  
- 每議題×視窗 1 頁、`max_results=25`；本 Extension **19** 次有效讀取（另有少量重複存檔）  
- 評分：TW SnowNLP＋政治詞 nudge；US VADER compound  

### B) GDELT

- DOC API 探測全部 **HTTP 429** → **唔用**（見 `data/extension_issue_news/gdelt/GDELT_STATUS.md`）。

### C) 政綱議題標籤（已有）

- TW `data/extension_poll_platform/taiwan_platforms_long.csv` → `issue_category` 年內佔比  
- US `data/us_parity_extension/us_platforms_long.csv` → 主題標籤佔比  
- 輸出：`results/extension_issue_news/platform_issue_salience_*.csv`

---

## 主數字（試點）

路徑：`data/extension_issue_news/`、`results/extension_issue_news/`  
約 **404** 帖、**19** 個議題×年格。

### 各屆「聲量最高」議題（X 抽樣內）

| 國 | 年 | Top volume | n | mean tone | 全國結果（參考） |
|----|----|------------|---|-----------|------------------|
| TW | 2020 | 經濟 | 23 | −0.79 | DPP 0.571 |
| TW | 2024 | 兩岸 | 24 | −0.51 | DPP 0.401 |
| US | 2020 | abortion／democracy（並列最高檔） | 25 | +0.16／+0.54 | Dem TP 0.523 |
| US | 2024 | abortion／democracy／immigration | 23 | +0.27／+0.24／−0.34 | Dem TP 0.492 |

完整表：`issue_salience_tone_by_election.csv`、`top_issue_signals_by_election.csv`。

### 同結果嘅關聯

- 每國每議題多數只有 **2** 屆 → **唔報 Pearson**（`n<3_not_reported`）。  
- 跨國混合「經濟語氣 vs 得票」亦只作敏感度，見 `issue_outcome_associations.csv`。  
- **全國多特徵 Ridge：n_years≤2 → 明確唔擬合**（見 `national_extension_regression_NOTE.csv`），避免過擬合扮預測力。

**解讀：** 可以描述「呢屆討論較集中邊啲議題／語氣偏負定偏正」；**唔可以**話「議題 X 令中間選民改投」。

---

## 可重現

```bash
# raw 已在 data/extension_issue_news/raw/
.venv/bin/python scripts/build_issue_news_extension.py
```

刷新貼文需再呼叫 X MCP（配額約 30／分、1000／日）。

---

## 相關文件

- `docs/X情緒特徵Extension說明.md`（實體／人物情緒；本檔係**議題**軸）  
- `docs/美台資料對等表.md`（已加「議題顯著度」列）  
- `docs/必備數據對照清單.md`（預測清單對照）  
- `docs/權重重算說明.md`（權重≠因果）
