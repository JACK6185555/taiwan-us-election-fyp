#!/usr/bin/env python3
"""Light domain-shift diagnostic: feature means and covariate L2 distance US vs TW.

Not full CORAL/MMD/adversarial DA — report distances only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "cleaned"
RES = ROOT / "results" / "real"
RES.mkdir(parents=True, exist_ok=True)

FEATURES = ["prev_share_A", "log_total_votes", "party_share_A"]


def main() -> None:
    panel = pd.read_csv(CLEAN / "joint_real_shared_panel.csv")
    rows = []
    for y in sorted(panel["year"].unique()):
        us = panel[(panel.country == "US") & (panel.year == y)]
        tw = panel[(panel.country == "TW") & (panel.year == y)]
        if len(us) == 0 or len(tw) == 0:
            continue
        for f in FEATURES:
            um, tm = us[f].mean(), tw[f].mean()
            us_, ts_ = us[f].std(ddof=0), tw[f].std(ddof=0)
            rows.append(
                {
                    "year": y,
                    "feature": f,
                    "us_mean": um,
                    "tw_mean": tm,
                    "mean_diff": um - tm,
                    "us_std": us_,
                    "tw_std": ts_,
                    "n_us": len(us),
                    "n_tw": len(tw),
                }
            )
        # standardized mean distance on available shared predictors
        feats = ["prev_share_A", "log_total_votes"]
        # use rows with prev
        us2 = us.dropna(subset=feats)
        tw2 = tw.dropna(subset=feats)
        if len(us2) and len(tw2):
            mu_u = us2[feats].mean().values
            mu_t = tw2[feats].mean().values
            # pool std
            pool = pd.concat([us2[feats], tw2[feats]])
            sd = pool.std(ddof=0).replace(0, 1).values
            dist = float(np.linalg.norm((mu_u - mu_t) / sd))
            rows.append(
                {
                    "year": y,
                    "feature": "L2_std_mean_prev_logvotes",
                    "us_mean": np.nan,
                    "tw_mean": np.nan,
                    "mean_diff": dist,
                    "us_std": np.nan,
                    "tw_std": np.nan,
                    "n_us": len(us2),
                    "n_tw": len(tw2),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RES / "domain_shift_feature_means.csv", index=False)
    print(out[out.feature == "L2_std_mean_prev_logvotes"].to_string(index=False))


if __name__ == "__main__":
    main()
