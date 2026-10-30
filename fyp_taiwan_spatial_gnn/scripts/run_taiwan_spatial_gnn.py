#!/usr/bin/env python3
"""Taiwan FYP — Spatial graph model (lightweight GCN-style).

No torch / torch-geometric: numpy + networkx adjacency + 1–2 layer
renormalized graph convolution, then Ridge/RF on node embeddings.

Adjacency (no township shapefile in repo):
  - same-county complete graph among townships in a year (distance-free)
  - self-loops included (GCN renormalization trick)

Ablation: non-spatial MLP/Ridge/RF vs GraphConv embeddings, out-of-time
presidential DPP/KMT swing and demeaned.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path("/workspace/taiwan-election-data")
PRES = ROOT / "fyp_taiwan_improved" / "harmonized_presidential_improved.csv"
OUT = ROOT / "fyp_taiwan_spatial_gnn"
CHARTS = OUT / "charts"
SEED = 42

FEAT_BASE = [
    "prev_dpp_share", "prev_kmt_share", "prev_turnout",
    "median_income", "urbanization", "education_rate", "log_people",
    "geo_north", "geo_central", "geo_south", "geo_east", "geo_islands",
    "esc_dpp_delta", "esc_kmt_delta", "esc_dpp_pid", "esc_kmt_pid",
]

for p in (OUT, CHARTS, OUT / "scripts"):
    p.mkdir(parents=True, exist_ok=True)


def mets(y, p):
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
    }


def build_county_adjacency(regions: list[str], counties: list[str]) -> np.ndarray:
    """Same-county complete graph + self loops. Order = regions list."""
    n = len(regions)
    A = np.eye(n, dtype=float)
    by_county: dict[str, list[int]] = {}
    for i, c in enumerate(counties):
        by_county.setdefault(c, []).append(i)
    for idxs in by_county.values():
        for i in idxs:
            for j in idxs:
                if i != j:
                    A[i, j] = 1.0
    return A


def gcn_normalize(A: np.ndarray) -> np.ndarray:
    """D^{-1/2} A D^{-1/2} (A already has self-loops)."""
    deg = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D = np.diag(d_inv_sqrt)
    return D @ A @ D


def graph_convolve(X: np.ndarray, A_hat: np.ndarray, n_layers: int = 2) -> np.ndarray:
    """H_{l+1} = A_hat H_l  (W = I; feature diffusion / SGC-style)."""
    H = X.astype(float)
    for _ in range(n_layers):
        H = A_hat @ H
    return H


def impute_matrix(train_df, test_df, cols):
    tr = train_df.copy()
    te = test_df.copy()
    for c in cols:
        if c not in tr.columns:
            tr[c] = 0.0
            te[c] = 0.0
        fill = tr[c].median()
        if pd.isna(fill):
            fill = 0.0
        tr[c] = tr[c].fillna(fill)
        te[c] = te[c].fillna(fill)
    return tr, te


def run_year_panel(df_year: pd.DataFrame, feat_cols: list[str], n_layers: int):
    """Return X_raw and X_gcn for one year block (rows aligned)."""
    regions = df_year["region"].tolist()
    counties = df_year["county_name"].tolist()
    X = df_year[feat_cols].values.astype(float)
    A = build_county_adjacency(regions, counties)
    A_hat = gcn_normalize(A)
    Xg = graph_convolve(X, A_hat, n_layers=n_layers)
    # also concatenate raw + diffused (GraphSAGE-ish skip)
    Xcat = np.hstack([X, Xg])
    return X, Xg, Xcat, A


def eval_oot(data: pd.DataFrame, party: str, target: str, test_year: int = 2024):
    ycol = f"{party}_{target}" if target != "level" else f"{party}_share"
    if target == "swing":
        ycol = f"{party}_swing"
    elif target == "demeaned":
        ycol = f"{party}_demeaned"
    elif target == "level":
        ycol = f"{party}_share"

    feat_cols = [c for c in FEAT_BASE + [f"prev_{party}_share", f"prev_{party}_swing"] if c in data.columns]
    # unique preserve
    seen, feat_cols2 = set(), []
    for c in feat_cols:
        if c not in seen and c in data.columns:
            seen.add(c)
            feat_cols2.append(c)
    feat_cols = feat_cols2

    train = data[data["year"] < test_year].dropna(subset=[ycol]).copy()
    test = data[data["year"] == test_year].dropna(subset=[ycol]).copy()
    if len(train) < 50 or len(test) < 20:
        return []

    train, test = impute_matrix(train, test, feat_cols)

    rows = []
    # Baselines
    if target == "swing":
        rows.append({"party": party.upper(), "target": target, "test_year": test_year,
                     "model": "ZeroSwing", "repr": "baseline", **mets(test[ycol], np.zeros(len(test)))})
    elif target == "demeaned":
        prev = f"prev_{party}_demeaned"
        if prev in test.columns and test[prev].notna().sum() > 20:
            m = test[prev].notna()
            rows.append({"party": party.upper(), "target": target, "test_year": test_year,
                         "model": "PreviousDemeaned", "repr": "baseline",
                         **mets(test.loc[m, ycol], test.loc[m, prev])})

    # Build year-wise features then stack
    def stack_repr(df, kind, n_layers=2):
        Xs, ys, years_out = [], [], []
        for yr, g in df.groupby("year", sort=True):
            g = g.reset_index(drop=True)
            X, Xg, Xcat, A = run_year_panel(g, feat_cols, n_layers)
            if kind == "raw":
                Xs.append(X)
            elif kind == "gcn":
                Xs.append(Xg)
            else:
                Xs.append(Xcat)
            ys.append(g[ycol].values)
            years_out.append(yr)
            # save graph stats once
        return np.vstack(Xs), np.concatenate(ys)

    # graph stats
    g2024 = test.reset_index(drop=True)
    A = build_county_adjacency(g2024["region"].tolist(), g2024["county_name"].tolist())
    n_edges = int((A > 0).sum() - len(A)) // 2  # undirected excl self
    graph_meta = {
        "adjacency": "same-county complete graph + self-loops",
        "shapefile": "NOT used — no township boundary file in dataset; county clique proxy",
        "n_nodes_2024": len(A),
        "n_undirected_edges_excl_self_2024": n_edges,
        "avg_degree_excl_self": float(A.sum(1).mean() - 1),
    }

    for kind, label in [("raw", "NonSpatial"), ("gcn", "GCN_diffuse_L2"), ("cat", "Raw+GCN_L2")]:
        Xtr, ytr = stack_repr(train, kind, n_layers=2)
        Xte, yte = stack_repr(test, kind, n_layers=2)
        for mname, mdl in [
            ("Ridge", Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))])),
            ("RandomForest", RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=SEED, n_jobs=-1)),
        ]:
            mdl.fit(Xtr, ytr)
            pred = mdl.predict(Xte)
            rows.append({
                "party": party.upper(), "target": target, "test_year": test_year,
                "model": mname, "repr": label, "n_train": len(ytr), "n_test": len(yte),
                "n_features": Xtr.shape[1], **mets(yte, pred),
            })

    # also L1 diffusion
    for kind, label in [("gcn", "GCN_diffuse_L1")]:
        Xtr, ytr = stack_repr(train, kind, n_layers=1)
        # need to rebuild with L1 — stack_repr uses n_layers param
        def stack_L(df, n_layers):
            Xs, ys = [], []
            for yr, g in df.groupby("year", sort=True):
                g = g.reset_index(drop=True)
                X, Xg, Xcat, _ = run_year_panel(g, feat_cols, n_layers)
                Xs.append(Xg)
                ys.append(g[ycol].values)
            return np.vstack(Xs), np.concatenate(ys)
        Xtr, ytr = stack_L(train, 1)
        Xte, yte = stack_L(test, 1)
        mdl = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))])
        mdl.fit(Xtr, ytr)
        pred = mdl.predict(Xte)
        rows.append({
            "party": party.upper(), "target": target, "test_year": test_year,
            "model": "Ridge", "repr": label, "n_train": len(ytr), "n_test": len(yte),
            "n_features": Xtr.shape[1], **mets(yte, pred),
        })

    return rows, graph_meta


def main():
    print("Loading presidential panel…")
    df = pd.read_csv(PRES)
    # ensure region id
    if "region" not in df.columns:
        raise SystemExit("need region column")

    all_rows = []
    graph_meta = None
    for party in ["dpp", "kmt"]:
        for target in ["swing", "demeaned"]:
            print(f" GNN OOT {party} {target}…")
            out = eval_oot(df, party, target, 2024)
            if not out:
                continue
            rows, meta = out
            all_rows.extend(rows)
            graph_meta = meta

    res = pd.DataFrame(all_rows)
    res.to_csv(OUT / "gnn_backtest_2024.csv", index=False)

    # ablation pivot: NonSpatial vs Raw+GCN
    piv_rows = []
    for (party, target, model), g in res.groupby(["party", "target", "model"]):
        d = {r.repr: r.MAE for _, r in g.iterrows()}
        if "NonSpatial" in d and "Raw+GCN_L2" in d:
            piv_rows.append({
                "party": party, "target": target, "model": model,
                "MAE_NonSpatial": d["NonSpatial"],
                "MAE_Raw+GCN_L2": d["Raw+GCN_L2"],
                "MAE_GCN_only_L2": d.get("GCN_diffuse_L2"),
                "delta_NonSpatial_minus_RawGCN": d["NonSpatial"] - d["Raw+GCN_L2"],
                "spatial_helps": d["NonSpatial"] > d["Raw+GCN_L2"],
            })
    piv = pd.DataFrame(piv_rows)
    piv.to_csv(OUT / "gnn_spatial_ablation_2024.csv", index=False)

    # networkx example graph for one county (documentation artifact)
    sample = df[df["year"] == 2024]
    county = sample["county_name"].value_counts().idxmax()
    sub = sample[sample["county_name"] == county]
    G = nx.complete_graph(sub["region"].tolist())
    nx.write_gexf(G, OUT / f"example_county_graph_{county}.gexf")
    with open(OUT / "graph_meta.json", "w", encoding="utf-8") as f:
        json.dump({**(graph_meta or {}), "example_county": county, "example_nodes": G.number_of_nodes()}, f, indent=2, ensure_ascii=False)

    # chart
    if not piv.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        labels = piv["party"] + " " + piv["target"] + " " + piv["model"]
        x = np.arange(len(piv))
        w = 0.35
        ax.bar(x - w/2, piv["MAE_NonSpatial"], w, label="NonSpatial")
        ax.bar(x + w/2, piv["MAE_Raw+GCN_L2"], w, label="Raw+GCN_L2")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("MAE")
        ax.set_title("Graph diffusion vs non-spatial — presidential 2024 OOT")
        ax.legend()
        fig.tight_layout()
        fig.savefig(CHARTS / "gnn_vs_nonspatial_2024.png", dpi=140)
        plt.close()

    readme = f"""# Taiwan FYP — Spatial Graph Model（`fyp_taiwan_spatial_gnn`）

## 方法

- **無 PyTorch / PyG**（環境無 torch）：用 **numpy SGC 風格**圖擴散  
  `H ← Â H`，`Â = D^{{-1/2}}(A+I)D^{{-1/2}}`，1–2 層，再接 Ridge／RF。
- **鄰接**：同年內 **同縣市鄉鎮完全圖 + self-loop**（距離無關）。  
  **未使用**行政區邊界 shapefile（資料集內無）；若之後有共邊 polygon，可換成 true contiguity。
- **對照**：`NonSpatial`（原始特徵）vs `GCN_diffuse_L*` vs `Raw+GCN_L2`（拼接）。
- **評估**：總統 DPP／KMT，目標 swing 同 demeaned，**2024 out-of-time**。

## 圖統計（2024）

```json
{json.dumps(graph_meta or {}, indent=2, ensure_ascii=False)}
```

## 結果摘要（見 CSV）

見 `gnn_spatial_ablation_2024.csv`：`delta_NonSpatial_minus_RawGCN > 0` 代表圖特徵有幫手。

## 重跑

```bash
.venv/bin/python fyp_taiwan_spatial_gnn/scripts/run_taiwan_spatial_gnn.py
```

## 限制

- 同縣完全圖 ≠ 真實共邊；大縣會過度連接。
- 擴散權重 W=I（SGC）；唔係可訓練 GAT attention。
- 未做跨國 transfer。
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(piv.to_string(index=False) if not piv.empty else "no piv")
    print("Done", OUT)


if __name__ == "__main__":
    main()
