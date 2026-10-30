# taiwan_polls QA spot-check (2026-10-27)

Total rows **5169**; distinct polls **1608**; by year polls {'2000': 23, '2004': 8, '2008': 73, '2012': 295, '2016': 201, '2020': 699, '2024': 309}

## Automated checks
- OK: no post-election rows; values in 0–100

## Manual sample (2 rows/year)
| year | poll_id | pollster | field_end | target | value | flags |
|---|---|---|---|---|---|---|
| 2000 | 2000-zh10-004 | 中央警察大學 | 2000-02-19 | 2000|國民 連戰 | 25.2 |  |
| 2000 | 2000-zh10-005 | 三立電視/蓋洛普 | 2000-02-29 | 2000|國民 連戰 | 15.0 |  |
| 2004 | 2004-apple-blackout | 蘋果日報 | 2004-03-09 | 2004|連戰 宋楚瑜 | 42.6 | QA:secondary_cite |
| 2004 | 2004-udn-voa-0303 | 聯合報 | 2004-03-02 | 2004|陳水扁 呂秀蓮 | 35.0 | QA:secondary_cite |
| 2008 | 2008-en6-003 | TVBS | 2008-03-18 | 2008|undecided | 19.0 |  |
| 2008 | 2008-en6-044 | Global Views (遠見雜誌)† | 2007-12-19 | 2008|Hsieh-Su (DPP) | 39.2 |  |
| 2012 | 2012-en1-074 | China Times | 2011-11-02 | 2012|undecided | 12.8 |  |
| 2012 | 2012-en0-011 | Apple Daily | 2011-05-13 | 2012|Tsai Ing-wen | 31.97 |  |
| 2016 | 2016-zh2-017 | 聯合報 | 2015-08-06 | 2016|undecided | 23.0 |  |
| 2016 | 2016-zh2-047 | TVBS | 2015-10-07 | 2016|undecided | 17.0 |  |
| 2020 | 2020-zh25-040 | 新台灣國策智庫 | 2018-12-20 | 2020|民進賴清德 | 30.5 | date_col=published_date_only |
| 2020 | 2020-zh12-019 | 蘋果日報 | 2019-05-27 | 2020|民進蔡英文 | 33.7 | date_col=published_date_only |
| 2024 | 2024-zh7-030 | 關鍵調查 | 2023-09-12 | 2024|民眾 柯文哲 | 16.7 |  |
| 2024 | 2024-zh6-072 | 聯合報 | 2023-05-18 | 2024|國民 侯友宜 | 24.0 |  |

## Watch
- 2004 mostly secondary cites except TVBS-0301 (n=1052).
- Some 2024 dates = published_date_only.
- National only; exit polls excluded.


## 2026-10-30 補充

- `region`：long 表 **5165／5165 = national**（確認無縣市列）。
- `field_start` 無法解析：**7** 個 poll_id（2004-tvbs-pre228、udn-voa-0303、各 blackout 二次引用等）——保留 NaN，以 `field_end`／note 為準；**不回填虛構開始日**。
- 完整驗證狀態表：`docs/extension_poll_platform_VERIFICATION.md`；縣市缺口：`docs/縣市民調缺口說明.md`。
