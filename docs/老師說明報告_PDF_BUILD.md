# 老師說明報告 PDF 產生紀錄

**日期：** 2026-10-30（Asia/Hong_Kong, UTC+8）

## 作法（最終）

1. 來源正文：`docs/老師說明報告.md`（含本輪「交付補強」段）。  
2. 以 Python 轉成簡易 HTML（`/tmp/老師說明報告.html`）。  
3. LibreOffice headless：  
   `libreoffice --headless --convert-to pdf …/老師說明報告.html`  
4. 輸出：`docs/老師說明報告.pdf`（已同步至 github 交付包與 data 工作樹）。  
5. `docs/老師說明報告.docx`：以 python-docx 於原檔末追加同一「本輪交付補強」段（HTML→docx filter 不可用）。

## 較早嘗試

- 僅自舊 docx 轉 PDF（內容略舊於當日 md 追加前）— 已由上述 HTML 路徑覆蓋。

## 環境

- LibreOffice／soffice 可用；weasyprint／pandoc／reportlab 未裝。


## 2026-10-30 老師回饋後重產

來源：重組後 `docs/老師說明報告.md` → HTML → LibreOffice PDF；docx 由 python-docx 重生。因素表：`results/factor_importance_edu_econ_media/`。
