# RQ2 county-level spatial vs non-spatial (Family A window)

Headline RQ2 table for the Taiwan FYP Guide. Same panel, features, and expanding OOT years as Family A (`taiwan_core_backtest.py`). Township experiments remain supplements — do not mix township MAE into Family A.

## Run

```bash
cd /workspace/taiwan-election-data
.venv/bin/python fyp_taiwan_spatial_gnn/scripts/run_rq2_county_spatial.py
```

## Key outputs

- `rq2_county_backtest.csv` — all folds / models
- `rq2_county_mean_mae_dpp.csv` / `rq2_county_mean_mae_kmt.csv` — mean OOT
- `rq2_county_year_comparison_dpp.csv` — year bars data
- `graph_meta_county.json` — Queen / kNN / distance-decay stats
- `rq2_county_protocol.json` — hyperparams + protocol
- `charts/` — figures

## Honest conclusion

PreviousShare still best on mean MAE. SGC+RF with distance-decay edges marginally beats NonSpatial RF (~0.0015 MAE). Trainable GCN/GAT/ResidualGCN unstable across years. See `docs/空間共邊GCN.md` § RQ2 結論.
