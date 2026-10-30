# data/hongkong_factor — 香港因素相關公開數字

本目錄提供**可引用的 CSV**，對應老師報告「香港因素」一節與 `docs/香港因素.md`。  
**未**把這些欄位 join 進鄉鎮 ML panel（無港裔開票分拆；學生／居留 ≠ 選舉人）。

## 檔案

| 檔案 | 內容 |
|------|------|
| `mac_hk_residence_settlement_annual.csv` | 陸委會香港居民來臺**居留／定居許可**年統計（流量） |
| `mac_hk_residence_settlement_permits.csv` | 同上來源之年＋月明細 |
| `mac_hk_residence_settlement_raw_download.csv` | 下載原文（保留） |
| `moe_hk_tertiary_enrollment.csv` | 教育部大專校院**香港生在學人數**（表 A2-1b） |
| `hk_voting_rights_scale_estimates.csv` | 具投票權港／澳移民**數量級估計**（AP 次級來源）vs 選舉人數量級 |
| `tw_voter_hk_issue_self_report_2020.csv` | 2020 選後民調：自報因反送中改變支持 **26.7%** 等（中央社） |

## 來源 URL

- 陸委會 CSV：`https://www.mac.gov.tw/big5/data/112_香港居民來臺居留、定居統計.csv`
- 教育部僑港澳生統計：`https://stats.moe.gov.tw/files/ebook/oversea/`（109–114）
- 中央社 2020-01-16：`https://www.cna.com.tw/news/aipl/202001160141.aspx`
- AP（投票權估計）：`https://apnews.com/article/hong-kong-taiwan-election-vote-immigrants-china-acd858e0f58c9a7d30f0ce95e95e2e09`
- 選罷法／報導者（資格脈絡）：見 `docs/香港因素.md`

## 解讀限制（必讀）

1. 居留／定居許可是**年流量**，不是在台港人總存量，更不是選民數。  
2. 大專香港生多數**未必**具投票權。  
3. 「約 1 萬具投票權」來自**媒體引述**，不是中選會港裔分拆清冊。  
4. 26.7% 是**臺灣選民自報**受香港事件影響改態度，屬間接議題效應；**2020 強於 2024**（敘事），且 TEDS／黨性文獻仍指政黨認同為主因。  
5. 本目錄數字**不構成**「港人票讓某黨 +/- X pp」的因果估計。

整理日期：2026-10-29（Asia/Hong_Kong）
