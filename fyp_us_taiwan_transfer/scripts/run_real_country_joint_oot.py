#!/usr/bin/env python3
"""Real-data country-only vs joint OOT + transfer protocols.

Uses cleaned/joint_real_shared_panel.csv (MEDSL US + CEC TW).
Shared features only: prev_share_A, log_total_votes.
Never invents MAE — writes whatever the models produce.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "cleaned"
RES = ROOT / "results" / "real"
CHARTS = ROOT / "results" / "charts"
RES.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

FEATURES = ["prev_share_A", "log_total_votes"]
TEST_YEARS = [2012, 2016, 2020, 2024]
SEED = 42


def models():
    return {
        "PreviousShare": None,
        "Ridge": Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=SEED))]
        ),
        "RF": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, random_state=SEED, n_jobs=-1
        ),
        "HGB": HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.08, max_iter=200, random_state=SEED
        ),
    }


def eval_fold(train: pd.DataFrame, test: pd.DataFrame, tag: str) -> list[dict]:
    rows = []
    train = train.dropna(subset=FEATURES + ["party_share_A"])
    test = test.dropna(subset=FEATURES + ["party_share_A"])
    if len(train) < 10 or len(test) < 1:
        return rows
    Xtr, ytr = train[FEATURES], train["party_share_A"].values
    Xte, yte = test[FEATURES], test["party_share_A"].values
    for name, est in models().items():
        if name == "PreviousShare":
            pred = test["prev_share_A"].values
        else:
            est.fit(Xtr, ytr)
            pred = est.predict(Xte)
        mae = mean_absolute_error(yte, pred)
        rows.append(
            {
                "protocol": tag,
                "model": name,
                "test_year": int(test["year"].iloc[0]),
                "n_train": len(train),
                "n_test": len(test),
                "mae": float(mae),
                "feature_set": ",".join(FEATURES),
                "data_kind": "real",
                "eval_country": "pooled",
            }
        )
        te = test.copy()
        te["_pred"] = pred
        te["_y"] = yte
        for c, g in te.groupby("country"):
            rows.append(
                {
                    "protocol": tag,
                    "model": name,
                    "test_year": int(test["year"].iloc[0]),
                    "n_train": len(train),
                    "n_test": len(g),
                    "mae": float(mean_absolute_error(g["_y"], g["_pred"])),
                    "feature_set": ",".join(FEATURES),
                    "data_kind": "real",
                    "eval_country": c,
                }
            )
    return rows


def main() -> None:
    panel = pd.read_csv(CLEAN / "joint_real_shared_panel.csv")
    rows: list[dict] = []

    # Country-only expanding window
    for c in ["US", "TW"]:
        sub = panel[panel["country"] == c]
        for ty in TEST_YEARS:
            train = sub[sub["year"] < ty]
            test = sub[sub["year"] == ty]
            rows += eval_fold(train, test, f"country_only_{c}")

    # Joint pooled (same calendar years — presidential years coincide 2000–2024)
    for ty in TEST_YEARS:
        train = panel[panel["year"] < ty]
        test = panel[panel["year"] == ty]
        rows += eval_fold(train, test, "joint_pooled")

    # Transfer protocols
    for src, tgt in [("US", "TW"), ("TW", "US")]:
        for ty in TEST_YEARS:
            train = panel[(panel["country"] == src) & (panel["year"] < ty)]
            test = panel[(panel["country"] == tgt) & (panel["year"] == ty)]
            rows += eval_fold(train, test, f"transfer_{src}_to_{tgt}")

    detail = pd.DataFrame(rows)
    detail.to_csv(RES / "real_joint_mae_detail.csv", index=False)

    mean_pooled = (
        detail[detail["eval_country"] == "pooled"]
        .groupby(["protocol", "model"], as_index=False)["mae"]
        .mean()
        .rename(columns={"mae": "mean_mae_oot"})
        .sort_values(["protocol", "mean_mae_oot"])
    )
    mean_pooled.to_csv(RES / "real_joint_mae_mean.csv", index=False)

    by_c = (
        detail[detail["eval_country"] != "pooled"]
        .groupby(["protocol", "model", "eval_country"], as_index=False)["mae"]
        .mean()
        .rename(columns={"mae": "mean_mae_oot"})
    )
    by_c.to_csv(RES / "real_joint_mae_by_country.csv", index=False)

    # Chart
    fig, ax = plt.subplots(figsize=(11, 5))
    protocols = list(mean_pooled["protocol"].unique())
    model_names = ["PreviousShare", "Ridge", "RF", "HGB"]
    x = np.arange(len(protocols))
    width = 0.18
    for i, m in enumerate(model_names):
        vals = []
        for p in protocols:
            v = mean_pooled.loc[(mean_pooled.protocol == p) & (mean_pooled.model == m), "mean_mae_oot"]
            vals.append(float(v.iloc[0]) if len(v) else np.nan)
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(protocols, rotation=30, ha="right")
    ax.set_ylabel("Mean OOT MAE (share)")
    ax.set_title("Real US/TW: country-only vs joint vs transfer (history+size features)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "real_joint_mae.png", dpi=140)
    fig.savefig(CHARTS / "real_joint_mae.pdf")
    plt.close()

    print(mean_pooled.to_string(index=False))
    print("Wrote", RES)


if __name__ == "__main__":
    main()
