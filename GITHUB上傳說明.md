# 上載 GitHub 說明

本資料夾已整理成可直接用 **GitHub Desktop** 開嘅 repo 結構。

## 建議步驟

1. 開 [GitHub Desktop](https://desktop.github.com/)
2. **Add** → **Add existing repository** → 揀呢個資料夾 `taiwan-us-election-fyp`
3. 若未初始化：喺資料夾開 Git Bash／終端執行 `git init`，再 Add
4. Publish repository（可 private）到你嘅 GitHub 帳號

## 資料夾結構

| 路徑 | 內容 |
|------|------|
| `README.md` | 專案說明同主結果（Family A） |
| `data/` | 美台公開面板同 Extension 特徵 |
| `docs/` | 報告、對等表、老師說明、儀表板 HTML |
| `results/` | 回測／權重／因素重要性輸出 |
| `scripts/` | 可重跑腳本 |
| `analysis/` | 分析說明同重現筆記 |
| `requirements.txt` | Python 依賴 |
| `.gitignore` | 已排除 raw zip／venv |

## 注意

- **未包含** 巨型 `raw/` 原始下載（可另存；見 `docs/DATA_GAPS.md`）
- 主結果數字以 `README.md` 同 `docs/SUBMISSION_AUDIT.md` 為準
- 儀表板：`美台選舉預測_儀表板_zh.html`（根目錄同 `docs/` 各一份）
