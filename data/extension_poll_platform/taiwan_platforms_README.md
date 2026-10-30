# taiwan_platforms_long.csv — 說明（更新 2026-10-27 午後）

**欄位**：year, election, candidate, party, issue_category, stance_text, source_url, published_date, pre_election_flag, verified（留空）, section。

## 來源優先序
1. 中選會選舉公報 PDF：`inputs/cec_gazettes/`／box `gazette/`。**注意**：自 1996 年民選總統以來，總統選舉公報**不刊登**候選人政見欄（風傳媒 2020-01-09 說明）；故 2000/2020 無法從公報 OCR 取得政見正文。2024 公報 OCR 仍可用（該年有政見欄）。
2. 選前新聞整理／公報外政見：風傳媒、中央社、華視、遠見、Taipei Times 等（見各列 `source_url`）。
3. Wikipedia（二手）：其餘年份。

## 議題分類
兩岸/國防、經濟/產業、能源/環境、社福/長照/年金、居住/土地、教育/文化、司法/政治改革、公共衛生/防疫、交通/建設、勞工、族群/認同、競選口號、其他。

## 本版筆數（約）
| year | 重點 |
|---|---|
| 2000 | 已補（遠見／華視／Taipei Times／學術引述）；覆蓋陳水扁、連戰、宋楚瑜、許信良、李敖 |
| 2020 | 已補風傳媒政見整理＋中央社長照政見；覆蓋蔡英文、韓國瑜、宋楚瑜（取代舊口號列） |
| 2024 | CEC OCR（賴／柯／侯） |
| 其他 | 多為維基 |

## 仍缺／限制
- 2000「連十條」全文、各候選人官方白皮書原文未取得（中時 Wayback Big5 頁損壞）
- 許信良／李敖僅少數二次引述列
- verified 欄仍待人工覆核
- 2004–2016 公報政見欄若存在可再 OCR；否則同樣需新聞／白皮書

## 驗證狀態（2026-10-30）
見 `docs/extension_poll_platform_VERIFICATION.md`。CSV `verified` 欄仍全空＝未全面手核。
