# Attempt to beat locked Prev (residual-on-Prev / poll swing)

**Date:** 2026-10-31（Asia/Hong_Kong UTC+8）  
**Status:** Extension experiment — **NOT** Core specification  
**Script:** `scripts/run_beat_prev_residual.py`  
**Outputs:** `results/beat_prev_residual/`

## Locked Core anchors (unchanged)

| Side | Locked Prev MAE |
|------|----------------:|
| Taiwan Core | **0.0793** |
| US Core | **0.0345** |

Same OOT test years as Core: **2012 / 2016 / 2020 / 2024** (expanding window).

## What we tried

1. **UniformPollSwing** — `pred = county_prev + (national_poll − national_prev)` (no ML training).
2. **Residual-on-Prev** — train Ridge / RF / HGB on `y − prev` only; `pred = prev + residual̂`.
3. Feature families (public / in-repo only): national poll swing, fundamentals (unrate, CPI, UMCSENT, turnout, polarization, incumbency), county socio (edu / income / unemployment / density), history Δ, light US **spatial lag of previous share** (adjacency; no current-target leakage).
4. **Not** deeper GCN (per brief).

## Headline results (mean OOT MAE)

| Country | Best method | Model | MAE | vs locked Prev |
|---------|-------------|-------|----:|---------------:|
| **TW** | UniformPollSwing (ablation poll def.) | prev+national_poll_swing | **0.0233** | **beats** (−0.056) |
| TW | Residual-on-Prev × poll_swing | Ridge | **0.0342** | **beats** (−0.045) |
| TW | Prev (recomputed) | baseline | 0.0793 | ≈ locked |
| TW | Residual × socio / shocks (no poll) | best ~HGB shocks | ≥0.0849 | **loses** |
| **US** | UniformPollSwing | prev+national_poll_swing | **0.0333** | **beats** (−0.0012) |
| US | Prev (recomputed) | baseline | 0.0345 | ≈ locked |
| US | Residual × shocks+socio+spatialLagPrev | RandomForest | 0.0364 | **loses** |
| US | Residual / Level ML (all other) | — | ≥0.0367 | **loses** |

### TW UniformPollSwing by year

| Year | MAE | national swing used |
|-----:|----:|---------------------|
| 2012 | 0.0290 | +0.0113 |
| 2016 | 0.0202 | +0.1169 |
| 2020 | 0.0197 | +0.0231 |
| 2024 | 0.0244 | −0.1636 |

### US UniformPollSwing by year

| Year | MAE | national swing used |
|-----:|----:|---------------------|
| 2012 | 0.0246 | −0.0332 |
| 2016 | 0.0643 | −0.0023 |
| 2020 | 0.0302 | +0.0331 |
| 2024 | 0.0142 | −0.0071 |

US edge is **thin** and **year-fragile** (2016 UniformPollSwing **worse** than Prev that year).

## What helped

- **National pre-election poll swing** (especially TW ablation definition: DPP share among non-undecided poll targets, 60-day window — same construction as prior poll Extension).
- Learned residual-on-Prev with **only** `poll_swing` (Ridge shrinks the swing) also beats Prev on TW.

## What did **not** help (still lose to Prev)

- County socio alone (edu / income / unemployment / density).
- National fundamentals without a good poll swing (TW shocks-only ≥0.0849).
- US residual ML + spatial lag of **previous** neighbor share — closer than deep GCN pilots, still **above** 0.0345.
- Deeper trees / more history Δ — tended to overfit the residual.

## Honest framing (required for report / viva)

1. **Do not replace Core Prev 0.0793 / 0.0345** with these numbers in the Core results table.
2. UniformPollSwing / poll residual is **national information painted onto every county** — it is **not** county-level poll skill (TEDS / county polls still missing).
3. Beating Prev here means: *given a usable national poll average, a uniform (or shrunk) swing beats pure inertia* — a known fundamentals/polls result, not evidence that socio GNN/ML “solved” local prediction.
4. US improvement is ~**0.001** MAE; treat as fragile Extension, not a product claim.
5. Poll sources: TW `data/extension_poll_platform/taiwan_polls.csv` + must-have averages; US FiveThirtyEight / Wikipedia aggregates in `data/extension_must_have/us_national_poll_averages_extension.csv` — secondary sources, QA caveats remain.

## Files

| Path | Content |
|------|---------|
| `results/beat_prev_residual/mean_mae_summary.csv` | All mean MAE |
| `results/beat_prev_residual/metrics_by_year.csv` | Per-year |
| `results/beat_prev_residual/models_that_beat_locked_prev.csv` | Beaters only |
| `results/beat_prev_residual/best_per_country.csv` | Best each side |
| `results/beat_prev_residual/best_by_year.csv` | Best method year detail |
| `results/beat_prev_residual/run_summary.json` | Machine summary |

## Reproduce

```bash
python3 scripts/run_beat_prev_residual.py
```
