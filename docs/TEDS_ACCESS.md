# TEDS（台灣選舉與民主化調查）微資料取得說明

**更新日期：** 2026-10-30（Asia/Hong_Kong, UTC+8）  
**狀態：** 本交付包**沒有** TEDS 原始微資料；代理人**不會**繞過登入牆爬取。

## 為何沒有原始檔

TEDS 個體問卷須註冊／申請。未取得前，解釋線只引用文獻＋全國 ESC PROXY，**不**估計個人層效應（見 `申請資料未取得時之報告說明.md`、`待申請資料清單.md`）。

## 途徑 A：TEDS 官網（checklist）

| 步驟 | 動作 | 完成？ |
|-----:|------|:------:|
| 1 | 開 https://teds.nccu.edu.tw/ →「會員註冊」，等管理員 email 核准 | ☐ |
| 2 | 讀「資料申請」規範：https://teds.nccu.edu.tw/data/super_pages.php?ID=data1&Sn=7 | ☐ |
| 3 | 登入填「資料申請表」（用途：FYP／選情預測與投票行為；單位／指導老師） | ☐ |
| 4 | 核准後 **5 天內**至「下載資料」下載（逾期重申請） | ☐ |
| 5 | FAQ：https://teds.nccu.edu.tw/qa/super_pages.php?ID=qa1 | ☐ |
| 6 | 檔案放**私有**路徑；更新 provenance；**勿** commit 公開 repo | ☐ |

## 途徑 B：SRDA checklist

| 步驟 | 動作 | 完成？ |
|-----:|------|:------:|
| 1 | 註冊 https://srda.sinica.edu.tw/（必要時升一般會員＋在學證明） | ☐ |
| 2 | 流程：https://srda.sinica.edu.tw/news/edf16712-af3f-47f7-bf60-b7057ef4000e | ☐ |
| 3 | 查「台灣選舉調查資料」／TEDS：https://srda.sinica.edu.tw/plan/?idx=SRDA.AS006 | ☐ |
| 4 | MySRDA 依「直接下載／申請下載／限制使用」送用途說明 | ☐ |
| 5 | 下載說明：https://srda.sinica.edu.tw/news/217acd6a-a36d-4846-aa77-90de8ef3edd5 ；聯絡 srda@gate.sinica.edu.tw | ☐ |

## 取得後優先變項

政黨認同、統獨立場、國族認同、候選人評價、投票回憶、人口統計 → 個人／多層；可選 MRP。**勿**覆蓋家族 A 主表 MAE。

## 已公開替代（本包已用）

CEC 開票、滯後社經、ESC 公開趨勢 PROXY（`data/esc_proxies/`）。
