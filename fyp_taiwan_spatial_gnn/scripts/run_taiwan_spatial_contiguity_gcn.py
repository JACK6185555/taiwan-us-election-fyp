#!/usr/bin/env python3
"""Taiwan FYP — True contiguity vs county-clique spatial GCN-style ablation.

Uses MOI TOWN_MOI_1140318-derived GeoJSON (pdmaptw mirror) for Queen contiguity.
No PyTorch/PyG: numpy SGC diffusion Â = D^{-1/2}(A+I)D^{-1/2}, 1–2 layers → Ridge/RF.

Compares on identical features / 2024 OOT splits:
  NonSpatial | CountyClique GCN | Contiguity GCN | Raw+each
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
GEOJSON = ROOT / "raw" / "geo" / "town-2025-03-18.geojson"
OUT = ROOT / "fyp_taiwan_spatial_gnn"
CHARTS = OUT / "charts"
SEED = 42

# Historical township upgrades present in 2012 CEC but renamed in MOI 1140318
REGION_ALIASES = {
    "彰化縣員林鎮": "彰化縣員林市",
    "苗栗縣頭份鎮": "苗栗縣頭份市",
    # 台/臺 handled by normalize()
}

FEAT_BASE = [
    "prev_dpp_share", "prev_kmt_share", "prev_turnout",
    "median_income", "urbanization", "education_rate", "log_people",
    "geo_north", "geo_central", "geo_south", "geo_east", "geo_islands",
    "esc_dpp_delta", "esc_kmt_delta", "esc_dpp_pid", "esc_kmt_pid",
]

for p in (OUT, CHARTS, OUT / "scripts"):
    p.mkdir(parents=True, exist_ok=True)


def normalize_region(s: str) -> str:
    s = str(s).replace("台", "臺").strip()
    return REGION_ALIASES.get(s, s)


def mets(y, p):
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
    }


def build_county_adjacency(counties: list[str]) -> np.ndarray:
    """Same-county complete graph + self loops."""
    n = len(counties)
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
    """H_{l+1} = A_hat H_l  (W = I; SGC-style feature diffusion)."""
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


def load_geo_queen():
    """Load GeoJSON, build full Queen adjacency keyed by normalized name."""
    import geopandas as gpd
    from libpysal.weights import Queen

    gdf = gpd.read_file(GEOJSON)
    gdf["region_norm"] = gdf["name"].map(normalize_region)
    # Drop duplicate names if any (keep first)
    if gdf["region_norm"].duplicated().any():
        dups = gdf.loc[gdf["region_norm"].duplicated(keep=False), "region_norm"].unique().tolist()
        print("WARNING duplicate geo names:", dups)
        gdf = gdf.drop_duplicates("region_norm", keep="first").reset_index(drop=True)

    W = Queen.from_dataframe(gdf, use_index=False)
    names = gdf["region_norm"].tolist()
    name_to_idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    A_full = np.eye(n, dtype=float)
    for i, neigh in W.neighbors.items():
        for j in neigh:
            A_full[i, j] = 1.0
            A_full[j, i] = 1.0

    island_names = [names[i] for i in W.islands]
    meta = {
        "source_geojson": str(GEOJSON),
        "official_release": "TOWN_MOI_1140318 (via pdmaptw v2.4.0 GeoJSON)",
        "dataset_url": "https://data.gov.tw/dataset/7441",
        "contiguity": "Queen (libpysal)",
        "n_polygons": n,
        "n_undirected_edges_excl_self_full": int((A_full.sum() - n) // 2),
        "avg_degree_excl_self_full": float(A_full.sum(1).mean() - 1),
        "n_islands_full": len(island_names),
        "island_names": island_names,
        "n_components_reported_by_libpysal": getattr(W, "n_components", None),
        "simplification_note": (
            "pdmaptw simplifies rings for web use; contiguity may miss thin shared edges. "
            "Islands listed are mostly true offshore islands."
        ),
    }
    return gdf, A_full, names, name_to_idx, meta, W


def induce_contiguity(A_full: np.ndarray, name_to_idx: dict[str, int], regions: list[str]):
    """Induce year subgraph; unmatched units get self-loop only. Returns A, match log."""
    n = len(regions)
    A = np.eye(n, dtype=float)
    matched = []
    unmatched = []
    idxs = []
    for r in regions:
        rn = normalize_region(r)
        if rn in name_to_idx:
            matched.append(r)
            idxs.append(name_to_idx[rn])
        else:
            unmatched.append(r)
            idxs.append(None)

    for i in range(n):
        if idxs[i] is None:
            continue
        for j in range(i + 1, n):
            if idxs[j] is None:
                continue
            if A_full[idxs[i], idxs[j]] > 0:
                A[i, j] = 1.0
                A[j, i] = 1.0

    log = {
        "n_nodes": n,
        "n_matched": len(matched),
        "n_unmatched": len(unmatched),
        "match_rate": len(matched) / n if n else 0.0,
        "unmatched": unmatched,
    }
    return A, log


def feat_cols_for(data: pd.DataFrame, party: str) -> list[str]:
    feat_cols = [c for c in FEAT_BASE + [f"prev_{party}_share", f"prev_{party}_swing"] if c in data.columns]
    seen, out = set(), []
    for c in feat_cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def ycol_for(party: str, target: str) -> str:
    if target == "swing":
        return f"{party}_swing"
    if target == "demeaned":
        return f"{party}_demeaned"
    return f"{party}_share"


def stack_repr(df, feat_cols, ycol, kind, n_layers, A_full, name_to_idx, adj_mode: str):
    """kind: raw | gcn | cat. adj_mode: county | contiguity."""
    Xs, ys = [], []
    year_logs = []
    for yr, g in df.groupby("year", sort=True):
        g = g.reset_index(drop=True)
        X = g[feat_cols].values.astype(float)
        if adj_mode == "county":
            A = build_county_adjacency(g["county_name"].tolist())
            log = {"year": int(yr), "adj_mode": "county", "n_nodes": len(g)}
        else:
            A, log = induce_contiguity(A_full, name_to_idx, g["region"].tolist())
            log = {"year": int(yr), "adj_mode": "contiguity", **log}
        year_logs.append(log)
        A_hat = gcn_normalize(A)
        Xg = graph_convolve(X, A_hat, n_layers=n_layers)
        if kind == "raw":
            Xs.append(X)
        elif kind == "gcn":
            Xs.append(Xg)
        else:
            Xs.append(np.hstack([X, Xg]))
        ys.append(g[ycol].values)
    return np.vstack(Xs), np.concatenate(ys), year_logs


def eval_oot(data, party, target, test_year, A_full, name_to_idx):
    ycol = ycol_for(party, target)
    feat_cols = feat_cols_for(data, party)
    train = data[data["year"] < test_year].dropna(subset=[ycol]).copy()
    test = data[data["year"] == test_year].dropna(subset=[ycol]).copy()
    if len(train) < 50 or len(test) < 20:
        return [], {}

    train, test = impute_matrix(train, test, feat_cols)
    rows = []

    if target == "swing":
        rows.append({
            "party": party.upper(), "target": target, "test_year": test_year,
            "model": "ZeroSwing", "repr": "baseline", "adjacency": "none",
            **mets(test[ycol], np.zeros(len(test))),
        })
    elif target == "demeaned":
        prev = f"prev_{party}_demeaned"
        if prev in test.columns and test[prev].notna().sum() > 20:
            m = test[prev].notna()
            rows.append({
                "party": party.upper(), "target": target, "test_year": test_year,
                "model": "PreviousDemeaned", "repr": "baseline", "adjacency": "none",
                **mets(test.loc[m, ycol], test.loc[m, prev]),
            })

    # Graph stats on test year
    A_cliq = build_county_adjacency(test["county_name"].tolist())
    A_cont, match_log = induce_contiguity(A_full, name_to_idx, test["region"].tolist())
    graph_stats = {
        "test_year": test_year,
        "county_clique": {
            "n_nodes": len(A_cliq),
            "n_undirected_edges_excl_self": int((A_cliq.sum() - len(A_cliq)) // 2),
            "avg_degree_excl_self": float(A_cliq.sum(1).mean() - 1),
        },
        "contiguity": {
            **match_log,
            "n_undirected_edges_excl_self": int((A_cont.sum() - len(A_cont)) // 2),
            "avg_degree_excl_self": float(A_cont.sum(1).mean() - 1),
            "n_isolates_degree0_excl_self": int(((A_cont.sum(1) - 1) == 0).sum()),
        },
    }

    configs = [
        ("raw", "NonSpatial", "none", 2),
        ("gcn", "GCN_diffuse_L2", "county", 2),
        ("cat", "Raw+GCN_L2", "county", 2),
        ("gcn", "GCN_diffuse_L1", "county", 1),
        ("gcn", "GCN_diffuse_L2", "contiguity", 2),
        ("cat", "Raw+GCN_L2", "contiguity", 2),
        ("gcn", "GCN_diffuse_L1", "contiguity", 1),
    ]

    for kind, label, adj_mode, n_layers in configs:
        mode = "county" if adj_mode == "county" else ("contiguity" if adj_mode == "contiguity" else "none")
        if kind == "raw":
            # NonSpatial: no graph; use zeros adjacency path via raw only
            Xtr, ytr, _ = stack_repr(train, feat_cols, ycol, "raw", 2, A_full, name_to_idx, "county")
            Xte, yte, _ = stack_repr(test, feat_cols, ycol, "raw", 2, A_full, name_to_idx, "county")
            adj_label = "none"
        else:
            Xtr, ytr, _ = stack_repr(train, feat_cols, ycol, kind, n_layers, A_full, name_to_idx, mode)
            Xte, yte, _ = stack_repr(test, feat_cols, ycol, kind, n_layers, A_full, name_to_idx, mode)
            adj_label = mode

        models = [
            ("Ridge", Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))])),
        ]
        if label in ("NonSpatial", "Raw+GCN_L2", "GCN_diffuse_L2"):
            models.append(
                ("RandomForest", RandomForestRegressor(
                    n_estimators=250, min_samples_leaf=3, random_state=SEED, n_jobs=-1
                ))
            )
        # L1 only Ridge (match original script)
        if label == "GCN_diffuse_L1":
            models = [("Ridge", Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))]))]

        for mname, mdl in models:
            mdl.fit(Xtr, ytr)
            pred = mdl.predict(Xte)
            rows.append({
                "party": party.upper(), "target": target, "test_year": test_year,
                "model": mname, "repr": label, "adjacency": adj_label,
                "n_train": len(ytr), "n_test": len(yte), "n_features": Xtr.shape[1],
                **mets(yte, pred),
            })

    return rows, graph_stats


def export_adjacency_artifacts(A_full, names, meta, panel_df, name_to_idx):
    """Save npz + edge list + meta for 2024 induced graph."""
    # Full geo adjacency
    np.savez_compressed(OUT / "adjacency_contiguity_full.npz", A=A_full, regions=np.array(names, dtype=object))

    # 2024 induced
    te = panel_df[panel_df["year"] == 2024].reset_index(drop=True)
    A24, log24 = induce_contiguity(A_full, name_to_idx, te["region"].tolist())
    regions24 = [normalize_region(r) for r in te["region"].tolist()]
    np.savez_compressed(
        OUT / "adjacency_contiguity.npz",
        A=A24,
        regions=np.array(regions24, dtype=object),
        original_regions=np.array(te["region"].tolist(), dtype=object),
    )

    # Edge list CSV (undirected, excl self)
    edges = []
    n = A24.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if A24[i, j] > 0:
                edges.append({"src": regions24[i], "dst": regions24[j], "weight": 1})
    pd.DataFrame(edges).to_csv(OUT / "adjacency_contiguity_edges_2024.csv", index=False)

    # County clique for comparison export
    A_cliq = build_county_adjacency(te["county_name"].tolist())

    graph_meta = {
        **meta,
        "year_2024_induced": {
            **log24,
            "n_undirected_edges_excl_self": int((A24.sum() - n) // 2),
            "avg_degree_excl_self": float(A24.sum(1).mean() - 1),
            "n_isolates_degree0_excl_self": int(((A24.sum(1) - 1) == 0).sum()),
        },
        "year_2024_county_clique": {
            "n_nodes": len(A_cliq),
            "n_undirected_edges_excl_self": int((A_cliq.sum() - len(A_cliq)) // 2),
            "avg_degree_excl_self": float(A_cliq.sum(1).mean() - 1),
        },
        "method": "numpy SGC diffusion (NOT trainable PyTorch GCN / GAT)",
    }
    with open(OUT / "graph_meta_contiguity.json", "w", encoding="utf-8") as f:
        json.dump(graph_meta, f, indent=2, ensure_ascii=False)

    # Match log all years
    match_rows = []
    for yr, g in panel_df.groupby("year"):
        _, log = induce_contiguity(A_full, name_to_idx, g["region"].tolist())
        match_rows.append({
            "year": int(yr),
            "n_nodes": log["n_nodes"],
            "n_matched": log["n_matched"],
            "n_unmatched": log["n_unmatched"],
            "match_rate": log["match_rate"],
            "unmatched": "|".join(log["unmatched"]),
        })
    pd.DataFrame(match_rows).to_csv(OUT / "geo_match_log.csv", index=False)
    return graph_meta


def main():
    print("Loading geo + building Queen contiguity…")
    gdf, A_full, names, name_to_idx, meta, W = load_geo_queen()
    print(f"  polygons={meta['n_polygons']} edges={meta['n_undirected_edges_excl_self_full']} "
          f"avg_deg={meta['avg_degree_excl_self_full']:.3f} islands={meta['n_islands_full']}")

    print("Loading presidential panel…")
    df = pd.read_csv(PRES)
    if "region" not in df.columns:
        raise SystemExit("need region column")

    graph_meta = export_adjacency_artifacts(A_full, names, meta, df, name_to_idx)
    print("2024 match rate:", graph_meta["year_2024_induced"]["match_rate"])

    all_rows = []
    # Primary: 2024 OOT; also prior OOT years for robustness
    test_years = [2024, 2020, 2016]
    for test_year in test_years:
        for party in ["dpp", "kmt"]:
            for target in ["swing", "demeaned"]:
                print(f" Contiguity OOT {test_year} {party} {target}…")
                rows, _ = eval_oot(df, party, target, test_year, A_full, name_to_idx)
                all_rows.extend(rows)

    res = pd.DataFrame(all_rows)
    res.to_csv(OUT / "gnn_contiguity_backtest.csv", index=False)
    res24 = res[res["test_year"] == 2024].copy()
    res24.to_csv(OUT / "gnn_contiguity_ablation_2024.csv", index=False)

    # Ablation pivot: NonSpatial vs County vs Contiguity (Raw+GCN_L2)
    piv_rows = []
    for (party, target, model), g in res24.groupby(["party", "target", "model"]):
        if model in ("ZeroSwing", "PreviousDemeaned"):
            continue
        d = {}
        for _, r in g.iterrows():
            key = f"{r.repr}|{r.adjacency}"
            d[key] = r.MAE
        ns = d.get("NonSpatial|none")
        cq = d.get("Raw+GCN_L2|county")
        ct = d.get("Raw+GCN_L2|contiguity")
        if ns is None:
            continue
        piv_rows.append({
            "party": party, "target": target, "model": model,
            "MAE_NonSpatial": ns,
            "MAE_Raw+GCN_CountyClique": cq,
            "MAE_Raw+GCN_Contiguity": ct,
            "MAE_GCN_only_County_L2": d.get("GCN_diffuse_L2|county"),
            "MAE_GCN_only_Contiguity_L2": d.get("GCN_diffuse_L2|contiguity"),
            "delta_NonSpatial_minus_Contiguity": (ns - ct) if ct is not None else None,
            "delta_County_minus_Contiguity": (cq - ct) if (cq is not None and ct is not None) else None,
            "contiguity_beats_nonspatial": (ct < ns) if ct is not None else None,
            "contiguity_beats_county": (ct < cq) if (ct is not None and cq is not None) else None,
        })
    piv = pd.DataFrame(piv_rows)
    piv.to_csv(OUT / "gnn_contiguity_vs_county_ablation_2024.csv", index=False)

    # Charts
    if not piv.empty:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        labels = piv["party"] + " " + piv["target"] + "\n" + piv["model"]
        x = np.arange(len(piv))
        w = 0.25
        ax.bar(x - w, piv["MAE_NonSpatial"], w, label="NonSpatial")
        ax.bar(x, piv["MAE_Raw+GCN_CountyClique"], w, label="Raw+GCN CountyClique")
        ax.bar(x + w, piv["MAE_Raw+GCN_Contiguity"], w, label="Raw+GCN Contiguity")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=7)
        ax.set_ylabel("MAE")
        ax.set_title("2024 OOT: NonSpatial vs County-clique vs True contiguity (SGC diffusion)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(CHARTS / "contiguity_vs_county_vs_nonspatial_2024.png", dpi=140)
        plt.close()

        # Degree comparison annotation chart
        fig, ax = plt.subplots(figsize=(6, 3.5))
        stats = [
            graph_meta["year_2024_county_clique"]["avg_degree_excl_self"],
            graph_meta["year_2024_induced"]["avg_degree_excl_self"],
        ]
        ax.bar(["County clique", "Queen contiguity"], stats, color=["#4C72B0", "#55A868"])
        ax.set_ylabel("Avg degree (excl. self-loops)")
        ax.set_title("2024 adjacency density")
        for i, v in enumerate(stats):
            ax.text(i, v + 0.3, f"{v:.2f}", ha="center")
        fig.tight_layout()
        fig.savefig(CHARTS / "adjacency_degree_comparison_2024.png", dpi=140)
        plt.close()

    # Update README
    mr = graph_meta["year_2024_induced"]["match_rate"]
    readme = f"""# Taiwan FYP — Spatial Graph Model（`fyp_taiwan_spatial_gnn`）

## 方法 / Method

- **無 PyTorch / PyG**：numpy **SGC 風格**圖擴散 `H ← Â H`，
  `Â = D^{{-1/2}}(A+I)D^{{-1/2}}`，1–2 層，再接 Ridge／RF。
  **唔係**可訓練 GCN／GAT（權重 W=I）。
- **兩種鄰接對照（同一特徵／同一 2024 OOT split）**：
  1. **CountyClique** — 同年同縣市鄉鎮完全圖 + self-loop（舊 proxy）
  2. **Contiguity** — MOI **TOWN_MOI_1140318** 衍生 GeoJSON 嘅 **Queen 共邊** + self-loop
- **評估**：總統 DPP／KMT；目標 swing 同 demeaned；**2024 out-of-time**（另跑 2020／2016 OOT）。

## 幾何來源 / Geometry

見 `raw/geo/README.md`。檔案：`raw/geo/town-2025-03-18.geojson`
（[data.gov.tw #7441](https://data.gov.tw/dataset/7441) → pdmaptw v2.4.0 mirror；TGOS 直連 403）。

授權：政府資料開放授權條款第1版（OGDL-Taiwan-1.0）。

## 圖統計（2024）

```json
{json.dumps({
  "match_rate_2024": mr,
  "contiguity": graph_meta["year_2024_induced"],
  "county_clique": graph_meta["year_2024_county_clique"],
  "islands_full_geo": graph_meta.get("island_names"),
}, indent=2, ensure_ascii=False)}
```

Match rate 2024 = **{mr:.1%}**. 全圖 Queen islands（多數為真離島）：{", ".join(graph_meta.get("island_names") or [])}.

## 結果檔

| File | Content |
|---|---|
| `gnn_contiguity_ablation_2024.csv` | 全模型列（adjacency × repr × model） |
| `gnn_contiguity_vs_county_ablation_2024.csv` | NonSpatial / County / Contiguity MAE 對照 |
| `gnn_contiguity_backtest.csv` | 含 2016／2020／2024 OOT |
| `adjacency_contiguity.npz` | 2024 induced Queen A + regions |
| `adjacency_contiguity_edges_2024.csv` | undirected edge list |
| `graph_meta_contiguity.json` | 圖統計 + provenance |
| `geo_match_log.csv` | 各年 name match／unmatched |
| `charts/contiguity_vs_county_vs_nonspatial_2024.png` | MAE 柱狀圖 |

舊檔 `gnn_spatial_ablation_2024.csv`／`run_taiwan_spatial_gnn.py` 仍係 **僅 county-clique** 結果，保留作對照。

## 重跑

```bash
.venv/bin/python fyp_taiwan_spatial_gnn/scripts/run_taiwan_spatial_contiguity_gcn.py
```

## 限制 / Limitations（誠實）

- pdmaptw GeoJSON **有簡化**；極窄共邊可能漏連。平均度 ≈ 5.3（合理），但唔係地籍級 contiguity。
- 離島／飛地形成 **disconnected components**；self-loop 令 isolate 仍可擴散（只係唔撈鄰居）。
- 2012：`員林鎮`→`員林市`、`頭份鎮`→`頭份市` 用 alias；其餘年 100% match。
- 擴散權重 W=I（SGC）；**未**訓練 GAT attention。
- 未做跨國 transfer（另線 US 對照）。
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    print("\n=== Ablation pivot 2024 ===")
    if not piv.empty:
        print(piv.to_string(index=False))
    print("Done", OUT)


if __name__ == "__main__":
    main()
