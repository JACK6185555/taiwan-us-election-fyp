#!/usr/bin/env python3
"""NonSpatial feature weights for US RQ2 (Ridge coef + RF impurity + OOT permutation)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/workspace/taiwan-election-data")
PANEL = ROOT / "fyp_us_taiwan_transfer/cleaned/us_county_presidential_panel.csv"
OUT = ROOT / "fyp_us_taiwan_transfer/rq2_us_county"
SEED = 42
FEATURES = ["prev_share_A","log_total_votes","unemployment","education_rate","median_income","log_density"]
YCOL = "party_share_A"
TEST_YEARS = [2012, 2016, 2020, 2024]
OUT.mkdir(parents=True, exist_ok=True)

def main():
    np.random.seed(SEED)
    df = pd.read_csv(PANEL, dtype={"region_id": str})
    df["region_id"] = df["region_id"].str.zfill(5)
    frame = df.dropna(subset=[YCOL, *FEATURES]).copy()
    rows_c, rows_i, rows_p = [], [], []
    for ty in TEST_YEARS:
        tr = frame[frame.year < ty]
        te = frame[frame.year == ty]
        Xtr, ytr = tr[FEATURES].to_numpy(), tr[YCOL].to_numpy()
        Xte, yte = te[FEATURES].to_numpy(), te[YCOL].to_numpy()
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        pipe.fit(Xtr, ytr)
        for f, c in zip(FEATURES, pipe.named_steps["ridge"].coef_):
            rows_c.append({"test_year": ty, "model": "Ridge", "family": "NonSpatial",
                           "feature": f, "coefficient": float(c), "abs_coefficient": float(abs(c))})
        rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_depth=16,
                                   random_state=SEED, n_jobs=4)
        rf.fit(Xtr, ytr)
        for f, imp in zip(FEATURES, rf.feature_importances_):
            rows_i.append({"test_year": ty, "model": "RandomForest", "family": "NonSpatial",
                           "feature": f, "importance": float(imp), "kind": "impurity"})
        for name, mdl in [("Ridge", pipe), ("RandomForest", rf)]:
            r = permutation_importance(mdl, Xte, yte, n_repeats=8, random_state=SEED,
                                       scoring="neg_mean_absolute_error", n_jobs=4)
            for f, m, s in zip(FEATURES, r.importances_mean, r.importances_std):
                rows_p.append({"test_year": ty, "model": name, "family": "NonSpatial",
                               "feature": f, "perm_importance_mae": float(m),
                               "perm_importance_std": float(s),
                               "note": "OOT test-year; higher => more MAE increase when shuffled"})
        print(f"weights year={ty} done", flush=True)
    pd.DataFrame(rows_c).to_csv(OUT / "feature_weights_nonspatial_ridge.csv", index=False)
    pd.DataFrame(rows_i).to_csv(OUT / "feature_importance_nonspatial_rf.csv", index=False)
    pd.DataFrame(rows_p).to_csv(OUT / "feature_perm_importance_nonspatial_oot.csv", index=False)
    coef = pd.DataFrame(rows_c)
    coef.groupby("feature", as_index=False).agg(
        mean_abs_coef=("abs_coefficient", "mean"), mean_coef=("coefficient", "mean")
    ).sort_values("mean_abs_coef", ascending=False).to_csv(OUT / "feature_weights_ridge_summary.csv", index=False)
    pd.DataFrame(rows_i).groupby("feature", as_index=False).agg(
        mean_importance=("importance", "mean")
    ).sort_values("mean_importance", ascending=False).to_csv(OUT / "feature_importance_rf_summary.csv", index=False)
    (OUT / "feature_weights_README.json").write_text(json.dumps({
        "seed": SEED,
        "arm": "NonSpatial only (US RQ2)",
        "interpretation": "Ridge on StandardScaler features; RF impurity train-only; permutation OOT MAE. Not causal.",
        "files": [
            "feature_weights_nonspatial_ridge.csv",
            "feature_importance_nonspatial_rf.csv",
            "feature_perm_importance_nonspatial_oot.csv",
            "feature_weights_ridge_summary.csv",
            "feature_importance_rf_summary.csv",
        ],
    }, indent=2), encoding="utf-8")
    print("wrote NonSpatial feature weights →", OUT)
    print(pd.read_csv(OUT / "feature_weights_ridge_summary.csv").to_string(index=False))

if __name__ == "__main__":
    main()
