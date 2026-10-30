# RQ2 US county spatial vs non-spatial

## Design
- Panel MEDSL + socio; target Dem two-party `party_share_A`
- Features: ['prev_share_A', 'log_total_votes', 'unemployment', 'education_rate', 'median_income', 'log_density']
- OOT years [2012, 2016, 2020, 2024]; SEED=42; n=3111 counties
- **Primary adjacency: Queen contiguity** (plotly geojson-counties-fips; Census TIGER ZIP HTTP 403 here)
- Sensitivity: k-NN k=5; distance-decay σ=80.0 km
- Queen edges=9072, avg degree=5.83

## Trainable GCN
**Deferred** on full US (~3111 nodes) for runtime. Partial 2012 MAE in `gcn_partial_note.json`.
Headline spatial comparison uses **SGC diffusion + classical heads**.

## Mean OOT MAE (2012–2024) — from `rq2_us_county_mean_mae.csv`

| model | adjacency | mean MAE |
|---|---|---:|
| PreviousShare | none | 0.0345 |
| NonSpatial RF | none | 0.0419 |
| Best non-baseline | Ridge / none | 0.0388 |

**Does spatial help vs PreviousShare?** See CSV. US county voting has strong inertia.

## Feature weights
See `feature_weights_*.csv` / `feature_importance_*.csv` (NonSpatial arm).

## Reproduce
```bash
.venv/bin/python fyp_us_taiwan_transfer/scripts/run_rq2_us_county_sgc_only.py
.venv/bin/python fyp_us_taiwan_transfer/scripts/export_rq2_us_feature_weights.py
```
