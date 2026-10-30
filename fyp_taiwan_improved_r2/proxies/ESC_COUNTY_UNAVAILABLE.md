# ESC / 政大選研 — 縣市級黨性

**結論：公開管道仍然攞唔到縣市（或鄉鎮）級政黨認同／統獨時間序列。**

- 政大選研中心網站公開嘅係**全國**趨勢圖；官方 Excel 要申請，亦主要係全國序列。
- TEDS 微資料可估計縣市，但要正式申請，本專案未納入。
- 因此 Round2 繼續只用：
  1. 全國 ESC PROXY（`proxies/esc_nccu_national_proxy.csv`）當選舉年「氣候／體制」指標；
  2. 地理 dummy + 同縣空間滯後／縣平均滯後；
  3. Ridge 顯式 **year fixed effects**（測試年 FE=0，靠 ESC 帶 regime）。

唔會捏造縣市級黨性數字。
