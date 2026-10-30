# US–Taiwan transfer track (RQ3 / Demo 1 + Demo 7 + I-K)

**Date:** 2026-10-30 (UTC+8)  
**Scope:** Methodological research — out-of-time regional vote-share MAE, country-only vs joint vs transfer, CORAL/MMD/adversarial DA, low-data curves.  
**Not:** advocacy, win probabilities, or claims that Dem ≡ DPP.

## What landed

| Deliverable | Path | Status |
|---|---|---|
| Shared schema | `schema/shared_region_year_schema.md` | Done |
| Demo 1 synthetic + Demo 7 harness | `cleaned/synthetic_*`, `results/synthetic/` | Done |
| Real US county panel (MEDSL) | `cleaned/us_county_presidential_panel.csv` | Done + **socio enriched** |
| TW county joint panel | `cleaned/tw_county_presidential_panel_joint.csv` | Done + income/density proxies |
| Joint shared panel (history+socio) | `cleaned/joint_real_shared_panel.csv` | Done |
| Shared feature list | `cleaned/joint_shared_feature_list.csv` | Done |
| History-only OOT MAE | `results/real/real_joint_mae_*.csv` | Done (prior run) |
| **ACS/ERS socio OOT MAE** | `results/real/real_joint_mae_*_acs.csv` | **Done** |
| **CORAL / MMD / adv DANN** | `results/real/da_transfer_mae_*.csv` | **Done** |
| **Low-data transfer curves** | `results/real/da_lowdata_mae_*.csv`, `results/charts/da_lowdata_curves.*` | **Done** |
| Teacher note (ZH) | `docs/美台對照_RQ3.md` | Updated |

## Shared features (documented)

**Joint socio set (both countries non-null):**
`prev_share_A`, `log_total_votes`, `z_income`, `z_log_density`

- `z_income` / `z_log_density` = within-country z-score of log income / log density  
- **US:** ERS median HH income 2022 (fallback ACS2020), ACS2020 density, ERS bachelor waves + yearly unemployment (US-only extras)  
- **TW:** tax `mean_income` and pop density as **static county proxies** (≠ US constructs; Guide §19)

## Key MAE (computed — do not invent)

### History-only (prior): see `real_joint_mae_mean.csv`

### With ACS/ERS socio (`real_joint_mae_mean_acs.csv`, mean OOT 2012–2024)

| protocol | best fitted / baseline | mean MAE |
|---|---|---|
| country_only_TW | Ridge **0.0787** ≈ PrevShare 0.0793 | |
| country_only_US | PrevShare **0.0345** (Ridge 0.0356) | |
| joint_pooled | PrevShare **0.0348** | US-N dominated |
| transfer_US→TW | HGB **0.0690** / Ridge 0.0705 (PrevShare 0.0793) | exploratory |
| transfer_TW→US | Ridge 0.0461 (worse than PrevShare 0.0345) | |

### DA transfer mean OOT (`da_transfer_mae_mean.csv`)

| direction | no_da | mmd | coral | adv_dann |
|---|---|---|---|---|
| US→TW | **0.0705** | **0.0700** | 0.0784 | 0.1178 |
| TW→US | **0.0461** | 0.0632 | 0.0955 | 0.1060 |

MMD ≈ no_da on US→TW; CORAL/DANN did **not** beat no_da here (feature-thin + construct mismatch).

### Low-data (US→TW / TW→US, no_da): adding k labeled target regions

See `da_lowdata_mae_mean.csv`. TW→US no_da improves 0.046→**0.039** as k→22; US→TW stays ~0.070–0.072 (source already large).

## How to re-run

```bash
PY=/workspace/taiwan-election-data/.venv/bin/python
cd /workspace/taiwan-election-data/fyp_us_taiwan_transfer
$PY scripts/build_us_county_panel.py
$PY scripts/build_tw_county_for_joint.py
$PY scripts/enrich_joint_panel_with_socio.py
$PY scripts/run_real_country_joint_oot.py          # history-only
$PY scripts/run_real_country_joint_oot_acs.py      # + socio
$PY scripts/run_da_lowdata_transfer.py             # CORAL/MMD/DANN + low-data
```

## RQ3 status

| Question | Status |
|---|---|
| Pipeline Demo 1+7? | Yes |
| Empirical country-only / joint / transfer? | Yes (history + socio) |
| ACS/ERS socio merged? | **Yes** |
| CORAL / MMD / adversarial DA + low-data? | **Yes (I-K)** |
| Identical constructs US/TW? | **No — do not claim** |
