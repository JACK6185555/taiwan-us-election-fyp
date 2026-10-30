# 臺灣選舉預測 FYP（臺灣部分）— 主結果＝核心設定（Core specification）

**English abstract.** Taiwan half of a US–Taiwan comparative election-forecasting FYP. **Primary result (**Core specification**, formerly “Family A” / RQ1):** county×year DPP presidential valid-vote share, expanding-window out-of-time tests on **2012／2016／2020／2024** (**88** region-years). Previous-election baseline beats Ridge／RF／HGB. Township multiparty, spatial, US transfer, HK／platforms／swing-voter notes are **Extensions only** — do not mix MAEs with Core specification. Ecological features — **no TEDS**. Methods exercise, not advocacy.

**提交數字稽核：** [`docs/SUBMISSION_AUDIT.md`](docs/SUBMISSION_AUDIT.md)（I-F／I-A／I-C）

---

## 主結果：核心設定（Core specification；舊稱「家族 A」）（縣市 DPP 總統・嚴格時間外）

| 項目 | 內容 |
|------|------|
| 資料 | `data/county_panel_features_2000_2024.csv`（154 列） |
| 目標 | 縣市 DPP 總統有效票份額 |
| 切分 | 擴展窗；**訓練年 < 測試年**；測試屆 2012／2016／2020／2024 |
| 樣本 | 每折 22 縣市；共 **88** region-year（同屆縣市**非獨立**，勿當獨立 N） |
| 輸出 | `results/family_a_county_presidential/`（鏡像 `analysis/taiwan_core/`） |
| 腳本 | `python scripts/taiwan_core_backtest.py` |

### 四屆平均 MAE（2026-10-30 乾淨重跑・定稿）

| 模型 | MAE | RMSE |
|------|----:|-----:|
| **Previous-election baseline** | **0.0793** | 0.0814 |
| Random Forest | 0.1041 | 0.1165 |
| HistGradientBoosting | 0.1189 | 0.1268 |
| Ridge | 0.1786 | 0.1855 |

相對舊 tracker（RF 曾寫 0.1040）：本環境 sklearn 1.9.1 重跑 RF＝**0.1041**（Δ +0.0001）；其餘 4 位不變。結論不變：**沒有訓練模型贏過「照抄上屆」**。詳見 `analysis/taiwan_core/REPRODUCIBILITY.md`。

**口試（I-A）：** 這是 4 測試屆的**歷史方法比較**，不是未來準確率保證，也**不**發布勝選機率。

**I-C：** permutation importance／feature ablation 只描述 **model dependence in OOT**（模型依賴／移除後 MAE 變化），**不是因果**、不是「勝選因素」。圖見 `results/family_a_county_presidential/charts/`。

---

## 補充：資料缺口補齊（2026-10-29）

本輪**只補公開、免登入**資料；不與第二輪模型腳本搶寫 `results/`。

| 新增 | 路徑 |
|------|------|
| ESC 認同年／半年序列 | `data/esc_proxies/esc_identity_*.csv` |
| ESC 統獨年序列 | `data/esc_proxies/esc_tongdu_annual.csv` |
| ESC 政黨偏好公開點（稀疏） | `data/esc_proxies/esc_party_preference_published_points.csv` |
| 選舉年全國 PROXY | `data/esc_proxies/esc_national_election_year_proxy.csv` |
| 縣市失業率（主計總處） | `data/economy/county_unemployment_*.csv` |
| 不分區全國占比 | `data/partylist_national_from_elretks.csv`（既有，保留） |
| 缺口說明／TEDS 申請 | `docs/DATA_GAPS.md`、`docs/TEDS_ACCESS.md` |
| 民調可得性 | `docs/POLLS_AVAILABILITY.md`（**無**鄉鎮歷史民調面板） |
| **香港因素**（敘事；未入模） | `docs/香港因素.md`；`data/hongkong_factor/`（MAC／教育部／民調 CSV） |

**仍需申請：** TEDS 微資料（SRDA／TEDS 會員）；ESC **官方**趨勢 Excel（完整政黨偏好）。見 `docs/DATA_GAPS.md`。

**文獻脈絡（預測線 × 解釋線）：** [`docs/文獻脈絡路線.md`](docs/文獻脈絡路線.md)；申請資料未取得可貼 [`docs/申請資料未取得時之報告說明.md`](docs/申請資料未取得時之報告說明.md)。

## Guide 完成度與 Extension（2026-10-30）

- **總表：** [`docs/GUIDE_COMPLETION_STATUS.md`](docs/GUIDE_COMPLETION_STATUS.md)（I-A…I-K、Demo 1–7）
- **追蹤／問答：** [`docs/GUIDE_ALIGNED_ISSUE_TRACKER.md`](docs/GUIDE_ALIGNED_ISSUE_TRACKER.md)、[`docs/GUIDE_ANSWERS.md`](docs/GUIDE_ANSWERS.md)
- **RQ4 Conformal：** 有實作與 improved protocols（`results/conformal_rq4_improved/`；美國 Core 見 `results/family_a_us_county_presidential/conformal_*`）——**標探索／未當不確定性產品**，不進主貢獻敘事。
- **Extension 民調／政綱 ablation（非核心設定）：** [`results/extension_poll_platform/`](results/extension_poll_platform/)（例：UniformPollSwing 四屆平均 MAE **0.0266**；History+Poll Ridge **0.0611**）——全國 poll、來源未全面覆核；**勿與 0.0793 主表混報**。
- **Demo 1／7 合成 US/TW：** `scripts/make_demo_us_tw_synthetic.py`、`scripts/run_demo_joint_synthetic.py`；真實 RQ3：`docs/美台對照_RQ3.md`。
- **I-C：** importance／ablation／權重 = **model dependence in OOT**，**不是因果**、不是勝選因素。




## 補充：這是什麼？（鄉鎮三黨範圍）

對應 FYP guide 的**臺灣部分**，現已擴成**三大政黨**：

| 政黨 | 代碼 | 總統選舉涵蓋 |
|------|------|--------------|
| 民進黨 DPP | 16 | 2012–2024（建模年；2008 作 lag） |
| 國民黨 KMT | 1 | 同上 |
| 台灣民眾黨 TPP | 350 | **僅 2024**（2019 創黨；2020 總統無民眾黨候選人） |

另將親民黨（PFP，2016／2020 宋楚瑜）得票作 **third-party 地理代理**（`prev_third_share`），**不是**獨立建模目標。

評估主軸：**out-of-time**（訓練年 < 測試年）。隨機切分／縣市交叉驗證只作補充，並標明 **NOT temporal**。

## 補充：目錄

```
taiwan-election-github/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── harmonized_tw_parties.csv         # 總統三黨 panel
│   ├── harmonized_tw_legislative.csv     # 區域立委三黨 panel
│   ├── partylist_national_from_elretks.csv
│   ├── harmonized_tw_regions.csv
│   ├── township_multiparty_votes.csv
│   ├── national_share_verification.csv
│   └── DATA_PROVENANCE.md
├── results/
│   ├── family_a_county_presidential/  # ★ 主結果（核心設定）
│   ├── parties/ / legislative/        # 補充：鄉鎮三黨
│   ├── improved/ / improved_r2/       # 補充
│   └── spatial_* / us_taiwan_transfer # 補充 RQ2／RQ3
│   └── us_census_adjacency_pilot/     # US Census 鄰接＋SGC／MA GCN-lite
├── analysis/taiwan_core/              # 核心設定 鏡像
├── scripts/
│   ├── taiwan_core_backtest.py        # ★ 核心設定
│   └── run_taiwan_*.py                # 補充線
└── docs/
    ├── SUBMISSION_AUDIT.md
    ├── 老師說明報告.md / .docx / .pdf
    └── …
```

## 補充：全國得票核對（中選會加總，非捏造）

| 年 | DPP | KMT | TPP | PFP |
|----|-----|-----|-----|-----|
| 2016 | 56.12% 蔡 | 31.04% 朱 | — | 12.84% 宋 |
| 2020 | 57.13% 蔡 | 38.61% 韓 | — | 4.26% 宋 |
| 2024 | **40.05% 賴** | **33.49% 侯** | **26.46% 柯** | — |

鄉鎮加總與 `data/cec_president_national_totals.csv` 差異 < 0.01pp（見 `national_share_verification.csv`）。

## 補充：鄉鎮三黨結果（勿與核心設定 混報）

### 2024 Level MAE（鄉鎮；越小越好）

| Party | Model | MAE | 評估類型 |
|-------|-------|-----|----------|
| DPP | PreviousShare | **0.149** | out-of-time |
| DPP | RandomForest | 0.210 | out-of-time |
| KMT | PreviousShare | **0.060** | out-of-time |
| KMT | RandomForest | 0.084 | out-of-time |
| TPP | ZeroTPP | 0.247 | no TPP history |
| TPP | PrevThirdPartyProxy (2020 PFP geo) | 0.209 | no TPP history |
| TPP | TransferPFP→RF + **national-mean cal** | 0.043 | proxy；校準用到 2024 全國均值（半 oracle） |
| TPP | RF leave-one-county-out | 0.021 | **NOT temporal** |

**解讀：** 2024 三腳督下，KMT 得票率比 DPP 更接近「照抄上屆」；DPP 全國大跌令 PreviousShare 對綠營誤差更大。TPP **沒有**上屆總統得票可抄；時間外推本質上做不到真正的「上屆 TPP」。縣市交叉驗證看起來準，但**不是**未來選舉預測。

### 補充：RF 生態 model dependence（鄉鎮；非因果、非個體黨性）

| Party | Top features（約） |
|-------|-------------------|
| DPP | `prev_dpp_share` ≈ 0.89；其餘城市化／上屆藍／第三方皆 < 0.03 |
| KMT | `prev_dpp_share` ≈ 0.71；`prev_kmt_swing` ≈ 0.10；`prev_third_share` ≈ 0.07 |
| TPP | `prev_third_share` ≈ 0.67（描述性、in-sample 2024）；教育／投票率等次之 |

詳見下方「文獻 vs 本模型」與 `docs/老師說明報告.md`。



## 補充：立委三黨（區域立委）

- **單位**：鄉鎮市區（與總統對齊）；票源為中選會**區域立委**候選人得票依政黨加總（join key = 縣市+選區+號次，因號次非全國唯一）。
- **年份**：2016／2020／2024（本 pack 無 2012 區域立委 → 2016 無 lag；主 OOT 為 2020、2024）。
- **TPP**：2016 無；2020、2024 有區域提名；**2024 可用 2020 區域 TPP 作 PreviousShare**（比總統條件好）。
- **不分區**：僅全國 `elretks`（2016 DPP 44.06%／KMT 26.91%；2020 DPP 33.98%／KMT 33.36%／TPP 11.22%；2024 DPP 36.16%／KMT 34.58%／TPP 22.07%）。無鄉鎮票檔 → **不入模型**。

### 區域立委全國加總（鄉鎮加總，CEC）

| 年 | DPP | KMT | TPP（提名範圍有限） |
|----|-----|-----|---------------------|
| 2016 | 45.08% | 38.71% | — |
| 2020 | 45.60% | 40.57% | 1.90%（81／368 鄉鎮所屬選區有提名） |
| 2024 | 45.17% | 40.42% | 2.97%（53／368） |

區域 TPP 遠低於不分區／總統，因為許多選區未提名——模型特徵含 `tpp_contested`。

### 2024 Level MAE（區域立委・out-of-time）

| Party | PreviousShare | RandomForest | Ridge | HistGB |
|-------|---------------|--------------|-------|--------|
| DPP | **0.105** | 0.164 | 0.106 | 0.142 |
| KMT | **0.133** | 0.139 | 0.144 | 0.139 |
| TPP | 0.039 | **0.021** | 0.026 | 0.023 |

TPP 立委：RF 贏過 PreviousShare，但 impurity 上 **`tpp_contested` ≈ 0.80**——主要學到「有沒有提名」而非穩定選民結構。DPP／KMT 仍多半 PreviousShare 最好或接近最好。

### 總統 vs 立委（2024 MAE 速覽）

| Party | 總統 PreviousShare | 立委 PreviousShare | 備註 |
|-------|-------------------|-------------------|------|
| DPP | 0.149 | 0.105 | 立委 DPP 地理較穩 |
| KMT | **0.060** | 0.133 | 總統藍較「黏」；立委藍較難抄上屆 |
| TPP | 無上屆（proxy 0.21） | **有 2020 lag**（RF 0.021） | 立委才有真正 TPP 時間外推 |

詳見 `results/legislative/` 與老師報告「立委三黨」節。

## 如何重跑（核心設定 優先）

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/taiwan_core_backtest.py     # 核心設定（主結果）
python scripts/run_taiwan_parties.py       # 補充：總統三黨鄉鎮
python scripts/run_taiwan_legislative.py   # 補充：區域立委三黨
# 或只讀 results/family_a_county_presidential/，不必重跑
```

精簡 GitHub pack 以 `results/` 為準；從 raw 重建需完整 `/workspace/taiwan-election-data/raw/`。

## 補充：文獻個體權重 vs 生態 model dependence（非因果）

本模型**沒有**放入 TEDS 政黨認同／統獨個體變數。文獻常見排序（黨性 ≫ 統獨 ≫ 人口／經濟）與本鄉鎮 RF（幾乎被上屆得票壟斷）**不可直接等同**。對照表見 `docs/老師說明報告.md` §文獻對照。

## 刻意不做／不做假

- 不捏造 2016／2020 的 TPP 總統得票  
- 不把 unemployment／township poll 填假值  
- 不把 LOCO 或 mean-calibrated transfer 寫成「已可預測下一屆總統」
- 不把區域立委 TPP 全國低得票誤解成民眾黨支持度等同總統／不分區（提名範圍不同）
- 不把 `tpp_contested` 主導的 RF 重要性說成「黨性」

## 補充：本輪改進（`results/improved/`）

對應建議嘅 priority improvements，已實作（唔止討論）：

1. **SWING + DEMEANED** 目標（DPP／KMT 總統＋區域立委；TPP 有滯後先做）
2. **ESC 全國 PROXY**（政黨偏好／統獨 lean；標明 PROXY，非 TEDS；無縣市公開序列）
3. **2024 三黨分頭** 維持
4. **Conformal 打喺 SWING** + leave-one-election residual quantile
5. **同縣空間滯後**（上屆得票 leave-one-out mean）+ ablation

### 2024 前後對照（摘要）

| 賽事 | 黨 | 目標 | 基準 MAE | 改進 MAE | 贏？ |
|------|----|------|----------|----------|------|
| 總統 | DPP | swing | ZeroSwing **0.149** | Ridge **0.042** | ✅ |
| 總統 | DPP | level | PreviousShare 0.149 | Ridge **0.067** | ✅ |
| 總統 | DPP | demeaned | PrevDemeaned **0.036** | RF 0.040 | ❌ |
| 總統 | KMT | swing | ZeroSwing **0.060** | RF 0.073 | ❌ |
| 總統 | KMT | level | PreviousShare 0.060 | RF **0.053** | ✅ 微 |
| 總統 | TPP | level | PrevThird 0.209 | ESC_TPP PROXY **0.104** | ✅（全國常數） |

**重點解讀**：Ridge 贏 DPP swing／level，主要靠 ESC 全國氣候當「選舉年體制」；**唔等於**學識排鄉鎮。相對版圖（demeaned）仍然 PreviousDemeaned 最穩。空間滯後有細增益；naive conformal 喺 2024 仍然幾乎失效。詳見 `results/improved/README.md`、`before_after_2024.csv`。

重跑：`python scripts/run_taiwan_improved.py`（需本機有完整 raw／cleaned 面板路徑時，或改 script 內 ROOT）。


## 補充：第二輪改進（`results/improved_r2/`）

在第一輪（swing／demeaned／ESC PROXY／同縣 LOO）之上：

1. **更強空間**：`spatial_lag_*`（同縣 LOO）+ `county_mean_*`（縣平均）；ablation NoSpatial／LOO／CountyMean／Both。
2. **一致 pipeline**：總統＋區域立委 DPP／KMT（及有滯後嘅 TPP）都跑 swing／demeaned／level。
3. **Demeaned→Level**：預測相對全國殘差後加返 `prev_nat`／`ESC_nat`／`oracle_nat`（oracle 標明洩漏）。
4. **Conformal 改打 demeaned**：2024 總統 DPP 實際覆蓋約 **85%**（名義 90%），遠好過第一輪喺 swing 上嘅 ~0.3%。
5. **ESC 縣市級仍不可得**（見 `results/improved_r2/proxies/ESC_COUNTY_UNAVAILABLE.md`）；Ridge 加 year FE（測試年 FE=0）。

### 2024 關鍵誠實數字

| 項目 | 結果 |
|------|------|
| 總統 DPP swing | Ridge 0.040 vs ZeroSwing 0.149 ✅ |
| 總統 DPP level（直接） | Ridge 0.071 vs PreviousShare 0.149 ✅ |
| 總統 DPP demeaned→level（Ridge+ESC_nat） | **0.063** vs PreviousShare 0.149 ✅ |
| 總統 KMT demeaned→level（RF+prev_nat） | **0.026** vs PreviousShare 0.060 ✅ |
| 總統 DPP demeaned | PreviousDemeaned 0.036 仍贏 RF 0.041 ❌ |
| 空間 RF（DPP swing） | NoSpatial 0.171 → Both 0.166（細增益） |
| 空間 RF（KMT swing） | Both 有時更差；CountyMean_only 較穩 |

## 補充：空間圖模型（`results/spatial_gnn/`／共邊）

- 無 PyTorch：numpy **SGC 風格**圖擴散 `H←ÂH`，`Â=D^{-1/2}(A+I)D^{-1/2}`，再接 Ridge／RF。
- 鄰接＝同年**同縣完全圖＋self-loop**（資料集無鄉鎮 shapefile；唔造假共邊）。
- 2024 OOT：Ridge 多數情況下 `Raw+GCN_L2` 略贏 NonSpatial（例如 KMT swing Ridge ΔMAE≈+0.021）；RF 有時無增益。

```bash
.venv/bin/python scripts/run_taiwan_improved_r2.py
.venv/bin/python scripts/run_taiwan_spatial_gnn.py
```



## 補充：政綱／口號因素

- 資料：`data/platforms/`（口號編碼、codebook、文獻對照、全國得票合併）
- 說明：`docs/政綱口號因素.md`
- 結果：`results/platforms/`（描述性關聯；**非因果**）

## 補充：中間／獨立選民（文獻；無新係數）

- 說明：`docs/中間選民因素.md`
- 資料：`data/swing_voters/`（排序、公開數字、類型、feature mapping）
- **不**把文獻係數寫進鄉鎮 ML；無 TEDS 子樣本估計


## 補充：US–Taiwan transfer（RQ3／Demo 1+7）

Methodological country-only vs joint vs transfer OOT on synthetic + real MEDSL/CEC county panels.

- Module: `data/fyp_us_taiwan_transfer/`
- Teacher note (ZH): `docs/美台對照_RQ3.md`
- Schema: `docs/shared_region_year_schema.md`
- Metrics: `results/us_taiwan_transfer/`
- **Status:** pipeline tested; real US county panel landed; ACS socio blocked; no DA yet. Not political transfer claims (Guide §19).

## 美台資料對等

TW／US 須同一資料層級。對等表：`docs/美台資料對等表.md`。美國 Extension 鏡像：`data/us_parity_extension/`（民調 PROXY、黨綱 metadata、MEDSL partisan lean）。**勿將 Extension MAE 併入 Core specification。**

