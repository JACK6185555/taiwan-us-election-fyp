#!/usr/bin/env python3
"""Demo 7 — joint vs country-only OOT on synthetic US/TW panel.

Produces MAE tables and charts. Labels everything synthetic.
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
RES = ROOT / "results" / "synthetic"
CHARTS = ROOT / "results" / "charts"
RES.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

SHARED_FEATURES = [
    "prev_share_A",
    "log_total_votes",
    "median_income",
    "education_rate",
    "urbanization",
    "unemployment",
]
HISTORY_FEATURES = ["prev_share_A", "log_total_votes"]
TEST_YEARS = [2012, 2016, 2020, 2024]
SEED = 42


def models():
    return {
        "PreviousShare": None,
        "Ridge": Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=SEED))]
        ),
        "RF": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=-1
        ),
        "HGB": HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.08, max_iter=200, random_state=SEED
        ),
    }


def eval_fold(train: pd.DataFrame, test: pd.DataFrame, feats: list[str], tag: str) -> list[dict]:
    rows = []
    ytr, yte = train["party_share_A"].values, test["party_share_A"].values
    Xtr = train[feats].astype(float)
    Xte = test[feats].astype(float)
    # drop rows missing required feats
    tr_ok = Xtr.notna().all(axis=1)
    te_ok = Xte.notna().all(axis=1)
    Xtr, ytr = Xtr.loc[tr_ok], ytr[tr_ok.values]
    Xte, yte = Xte.loc[te_ok], yte[te_ok.values]
    if len(Xtr) < 5 or len(Xte) < 1:
        return rows
    for name, est in models().items():
        if name == "PreviousShare":
            pred = test.loc[te_ok, "prev_share_A"].values
            # if prev missing, skip
            if np.isnan(pred).any():
                continue
        else:
            est.fit(Xtr, ytr)
            pred = est.predict(Xte)
        mae = mean_absolute_error(yte, pred)
        rows.append(
            {
                "protocol": tag,
                "model": name,
                "test_year": int(test["year"].iloc[0]),
                "n_train": int(len(Xtr)),
                "n_test": int(len(Xte)),
                "mae": float(mae),
                "feature_set": ",".join(feats),
                "data_kind": "synthetic",
            }
        )
        # per-country breakdown if both present
        if "country" in test.columns:
            te_df = test.loc[te_ok].copy()
            te_df["_pred"] = pred
            te_df["_y"] = yte
            for c, g in te_df.groupby("country"):
                rows.append(
                    {
                        "protocol": tag,
                        "model": name,
                        "test_year": int(test["year"].iloc[0]),
                        "n_train": int(len(Xtr)),
                        "n_test": int(len(g)),
                        "mae": float(mean_absolute_error(g["_y"], g["_pred"])),
                        "feature_set": ",".join(feats),
                        "data_kind": "synthetic",
                        "eval_country": c,
                    }
                )
    return rows


def country_only(panel: pd.DataFrame, country: str) -> list[dict]:
    sub = panel[panel["country"] == country].copy()
    out = []
    for ty in TEST_YEARS:
        train = sub[sub["year"] < ty].dropna(subset=["prev_share_A"])
        test = sub[sub["year"] == ty].dropna(subset=["prev_share_A"])
        out.extend(eval_fold(train, test, SHARED_FEATURES, f"country_only_{country}"))
    return out


def joint_pooled(panel: pd.DataFrame) -> list[dict]:
    out = []
    for ty in TEST_YEARS:
        train = panel[panel["year"] < ty].dropna(subset=["prev_share_A"])
        test = panel[panel["year"] == ty].dropna(subset=["prev_share_A"])
        out.extend(eval_fold(train, test, SHARED_FEATURES, "joint_pooled"))
    return out


def transfer(panel: pd.DataFrame, src: str, tgt: str) -> list[dict]:
    """Train on src all years < ty, test tgt year == ty; shared history feats only."""
    out = []
    for ty in TEST_YEARS:
        train = panel[(panel["country"] == src) & (panel["year"] < ty)].dropna(subset=["prev_share_A"])
        test = panel[(panel["country"] == tgt) & (panel["year"] == ty)].dropna(subset=["prev_share_A"])
        out.extend(eval_fold(train, test, HISTORY_FEATURES, f"transfer_{src}_to_{tgt}"))
    return out


def main() -> None:
    panel = pd.read_csv(CLEAN / "synthetic_us_tw_panel.csv")
    rows: list[dict] = []
    rows += country_only(panel, "US")
    rows += country_only(panel, "TW")
    rows += joint_pooled(panel)
    rows += transfer(panel, "US", "TW")
    rows += transfer(panel, "TW", "US")
    detail = pd.DataFrame(rows)
    # fill eval_country for overall rows
    if "eval_country" not in detail.columns:
        detail["eval_country"] = "pooled"
    detail["eval_country"] = detail["eval_country"].fillna("pooled")
    detail.to_csv(RES / "demo_joint_synthetic_mae_detail.csv", index=False)

    # Mean MAE tables
    mean_overall = (
        detail[detail["eval_country"] == "pooled"]
        .groupby(["protocol", "model"], as_index=False)["mae"]
        .mean()
        .rename(columns={"mae": "mean_mae_oot"})
        .sort_values(["protocol", "mean_mae_oot"])
    )
    mean_overall.to_csv(RES / "demo_joint_synthetic_mae_mean.csv", index=False)

    by_country = (
        detail[detail["eval_country"] != "pooled"]
        .groupby(["protocol", "model", "eval_country"], as_index=False)["mae"]
        .mean()
        .rename(columns={"mae": "mean_mae_oot"})
    )
    by_country.to_csv(RES / "demo_joint_synthetic_mae_by_country.csv", index=False)

    # Chart: mean MAE by protocol/model
    plot_df = mean_overall.copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    protocols = plot_df["protocol"].unique()
    models_u = ["PreviousShare", "Ridge", "RF", "HGB"]
    x = np.arange(len(protocols))
    width = 0.18
    for i, m in enumerate(models_u):
        vals = [
            plot_df.loc[(plot_df["protocol"] == p) & (plot_df["model"] == m), "mean_mae_oot"].values
            for p in protocols
        ]
        vals = [v[0] if len(v) else np.nan for v in vals]
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(protocols, rotation=25, ha="right")
    ax.set_ylabel("Mean OOT MAE (share)")
    ax.set_title("Synthetic Demo 7: country-only vs joint vs transfer (not real elections)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "demo_joint_synthetic_mae.png", dpi=140)
    fig.savefig(CHARTS / "demo_joint_synthetic_mae.pdf")
    plt.close()

    print(mean_overall.to_string(index=False))
    print("Wrote", RES)


if __name__ == "__main__":
    main()
