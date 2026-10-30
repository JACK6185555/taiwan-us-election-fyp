# 資料缺口與申請步驟（Taiwan FYP）

更新：2026-10-30（Asia/Hong_Kong）

## 已於本輪補上（公開、無需登入）

| 項目 | 位置 |
|------|------|
| ESC 認同年序列 CSV | `data/esc_proxies/esc_identity_annual.csv` |
| ESC 認同半年點 | `data/esc_proxies/esc_identity_semi_annual_wiki.csv` |
| ESC 統獨年序列 CSV | `data/esc_proxies/esc_tongdu_annual.csv` |
| ESC 政黨偏好公開點（稀疏） | `data/esc_proxies/esc_party_preference_published_points.csv` |
| 選舉年全國 PROXY 面板 | `data/esc_proxies/esc_national_election_year_proxy.csv` |
| 縣市失業率（主計總處） | `raw/economy/county_unemployment_*.csv` + XML |
| 不分區政黨票全國占比 | `cleaned/`／github `data/partylist_national_from_elretks.csv`（既有） |
| 民調可得性說明 | `raw/polls/README.md`（結論：無鄉鎮歷史面板） |

## 仍需申請／無法公開取得

### 1. TEDS 個體微資料（最高優先）

- **不要**爬 SRDA／TEDS 登入後檔案。  
- 步驟見：`raw/teds-notes/TEDS_ACCESS.md`  
- 入口：https://teds.nccu.edu.tw/ 、https://srda.sinica.edu.tw/

### 2. ESC 官方趨勢 Excel（完整政黨偏好序列）

- 簽名申請表 → esc@nccu.edu.tw  
- 說明：`raw/esc-nccu/SOURCES_AND_ACCESS.md`  
- 目前政黨偏好 CSV 僅有新聞公開點，2016 年黨性 PROXY 多為空。

### 3. 鄉鎮失業率

- 公開一致全國鄉鎮序列不存在於 data.gov.tw dataset 6640（僅縣市）。  
- SE GIS 零星產品未納入，避免半套 join。

### 4. 鄉鎮／縣市歷史選舉民調面板

- 見 `raw/polls/README.md`：無可用公開機器可讀面板；`poll_index` 維持缺值。

### 5. 方法延伸（非「缺檔」而是未做實驗）

- US 縣級對照、GCN／GAT、domain adaptation、校準穩健的預測區間。

## 建議申請順序（給老師／自己）

1. ESC 趨勢 Excel（最快補齊黨性全國序列）  
2. TEDS 2020／2024 面訪或電訪微資料（個體黨性／統獨）  
3. 再考慮人工民調彙整或 US 對照資料


## 2026-10-30 文獻脈絡路線輪

| 項目 | 位置／結論 |
|------|------------|
| 設計文件 | `docs/文獻脈絡路線.md`（預測線 × 解釋線；文獻連結；公開 vs 申請） |
| 報告可貼限制段 | `docs/申請資料未取得時之報告說明.md` |
| 老師報告 §五-B | `docs/老師說明報告.md` |
| 客家開放 CSV 探查 | `raw/ethnicity/hakka_DTST20171200004.csv`（data.gov.tw 73163）—**僅全國彙總，無縣市欄，未入模** |
| 外省／本省縣市占比 | **仍缺**可穩定公開機器可讀長面板（不捏造） |
| 新特徵入 A／A+ | **無**（本輪不重跑 MAE；家族 A Prev 0.0793／A+ Ridge 0.1425 維持） |

仍需申請：TEDS、ESC 官方 Excel、鄉鎮／縣市民調面板。**本輪未代申請、未爬登入牆。**


## 2026-10-30 論文方法與資料來源對照輪

| 項目 | 位置／結論 |
|------|------------|
| 方法／資料對照表 | `docs/論文方法與資料來源對照.md`（文獻做法 × 本專案對應 × 申請唔到時） |
| 文獻脈絡指針 | `docs/文獻脈絡路線.md` 已加連到上表 |
| 老師報告 §五-C | `docs/老師說明報告.md` |
| 新特徵／新 MAE | **無**（本輪只寫文獻對照；家族 A Prev 0.0793／A+ Ridge 0.1425 維持） |

仍需申請：TEDS、ESC 官方 Excel、鄉鎮／縣市民調面板。**本輪未代申請、未爬登入牆。**
