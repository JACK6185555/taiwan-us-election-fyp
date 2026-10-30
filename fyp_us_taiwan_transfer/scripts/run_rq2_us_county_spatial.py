#!/usr/bin/env python3
"""RQ2 US side: county presidential OOT spatial vs non-spatial.

Mirror Taiwan Core specification window (test years 2012/2016/2020/2024), expanding
train year < test_year. Target = Dem two-party share (party_share_A).

Adjacency:
  - Queen contiguity from plotly geojson-counties-fips (Census-style FIPS);
    Census TIGER cartographic ZIP returned 403 in this environment.
  - k-NN (k=5) on centroids (EPSG:5070 / NAD83 Conus Albers)
  - Distance-decay (sigma=80 km) as sensitivity

Models: PreviousShare; NonSpatial Ridge/RF; SGC L1/L2 + Ridge/RF;
        Trainable GCN L1 + ResidualGCN L1 on queen & knn (CPU, runtime-capped).

SEED=42. No invented metrics.
Outputs: fyp_us_taiwan_transfer/rq2_us_county/
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GCNConv

warnings.filterwarnings("ignore")

ROOT = Path("/workspace/taiwan-election-data")
PANEL = ROOT / "fyp_us_taiwan_transfer" / "cleaned" / "us_county_presidential_panel.csv"
GEOJSON = ROOT / "raw" / "geo" / "us_county" / "counties.geojson"
OUT = ROOT / "fyp_us_taiwan_transfer" / "rq2_us_county"
CHARTS = OUT / "charts"
SEED = 42
DEVICE = torch.device("cpu")

FEATURES = [
    "prev_share_A",
    "log_total_votes",
    "unemployment",
    "education_rate",
    "median_income",
    "log_density",
]
YCOL = "party_share_A"
ANCHOR = "prev_share_A"
TEST_YEARS = [2012, 2016, 2020, 2024]

HIDDEN = 32
DROPOUT = 0.15
LR = 5e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 40
PATIENCE = 10
KNN_K = 5
DIST_SIGMA_KM = 80.0

for p in (OUT, CHARTS):
    p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def scores(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
    }


def gcn_normalize(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float).copy()
    np.fill_diagonal(A, np.maximum(np.diag(A), 1.0))
    deg = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D = np.diag(d_inv_sqrt)
    return D @ A @ D


def graph_convolve(X: np.ndarray, A_hat: np.ndarray, n_layers: int) -> np.ndarray:
    H = X.astype(float)
    for _ in range(n_layers):
        H = A_hat @ H
    return H


def adj_to_edge_index(A: np.ndarray, weighted: bool = False):
    A = np.asarray(A, dtype=float)
    src, dst = np.nonzero(A > 0)
    ei = torch.from_numpy(np.vstack([src, dst]).astype(np.int64))
    if weighted:
        w = torch.from_numpy(A[src, dst].astype(np.float32))
        return ei, w
    return ei, None


def build_graphs(panel_fips: list[str]):
    """Queen from GeoJSON + kNN / distance-decay on projected centroids."""
    import geopandas as gpd
    from libpysal.weights import Queen

    g = gpd.read_file(GEOJSON)
    g["GEOID"] = g["id"].astype(str).str.zfill(5)
    # Keep CONUS+AK+HI counties that appear in panel; drop territories if unused
    g = g[g["GEOID"].isin(set(panel_fips))].copy()
    # Stable order matching panel_fips order for those present
    present = [f for f in panel_fips if f in set(g["GEOID"])]
    g = g.set_index("GEOID").loc[present].reset_index()
    names = present  # FIPS ids as node labels
    n = len(names)
    name_to_idx = {f: i for i, f in enumerate(names)}

    W = Queen.from_dataframe(g, use_index=False, silence_warnings=True)
    A_queen = np.eye(n, dtype=float)
    for i, neigh in W.neighbors.items():
        for j in neigh:
            A_queen[i, j] = 1.0
            A_queen[j, i] = 1.0

    # Projected centroids (NAD83 / Conus Albers) — AK/HI distorted but ok for knn sensitivity
    gp = g.to_crs(epsg=5070)
    xy = np.column_stack([gp.geometry.centroid.x, gp.geometry.centroid.y])
    # k-NN
    nn = NearestNeighbors(n_neighbors=min(KNN_K + 1, n), algorithm="ball_tree")
    nn.fit(xy)
    dist_m, idx = nn.kneighbors(xy)
    A_knn = np.eye(n, dtype=float)
    for i in range(n):
        for j in idx[i, 1:]:
            A_knn[i, j] = 1.0
            A_knn[j, i] = 1.0

    # Distance-decay: connect kNN radius envelope + queen; weight = exp(-d/sigma)
    # Avoid full NxN for memory: use radius neighbors with generous radius
    radius_m = DIST_SIGMA_KM * 1000.0 * 3.0  # ~3 sigma
    nn_r = NearestNeighbors(radius=radius_m, algorithm="ball_tree")
    nn_r.fit(xy)
    nbrs = nn_r.radius_neighbors(xy, return_distance=True)
    A_dist = np.eye(n, dtype=float)
    for i in range(n):
        for j, d_m in zip(nbrs[1][i], nbrs[0][i]):
            if i == j:
                continue
            w = float(np.exp(-(d_m / 1000.0) / DIST_SIGMA_KM))
            if w < 0.05 and A_queen[i, j] == 0:
                continue
            A_dist[i, j] = max(A_dist[i, j], w if w >= 0.05 else 0.05)
    # ensure queen edges present
    A_dist = np.maximum(A_dist, (A_queen - np.eye(n)) * 0.05)
    np.fill_diagonal(A_dist, 1.0)
    A_dist_bin = (A_dist > 0).astype(float)

    islands = [names[i] for i in range(n) if (A_queen[i] > 0).sum() == 1]
    meta = {
        "n_counties": n,
        "geo_source": str(GEOJSON),
        "geo_note": (
            "plotly/datasets geojson-counties-fips.json (Census FIPS polygons). "
            "Census TIGER cb_*_us_county_5m.zip returned HTTP 403 here."
        ),
        "queen_n_edges": int(((A_queen > 0).sum() - n) // 2),
        "queen_avg_degree": float(((A_queen > 0).sum(axis=1) - 1).mean()),
        "queen_islands_n": len(islands),
        "queen_islands_sample": islands[:20],
        "knn_k": KNN_K,
        "knn_n_edges": int(((A_knn > 0).sum() - n) // 2),
        "knn_avg_degree": float(((A_knn > 0).sum(axis=1) - 1).mean()),
        "dist_sigma_km": DIST_SIGMA_KM,
        "dist_n_edges": int(((A_dist_bin > 0).sum() - n) // 2),
        "dist_avg_degree": float(((A_dist_bin > 0).sum(axis=1) - 1).mean()),
        "crs_projected": "EPSG:5070",
        "panel_fips_requested": len(panel_fips),
        "panel_fips_matched": n,
        "unmatched_fips_n": len(panel_fips) - n,
    }
    graphs = {
        "queen": A_queen,
        "knn": A_knn,
        "dist_decay": A_dist,
        "dist_decay_bin": A_dist_bin,
    }
    return graphs, names, name_to_idx, meta


def classical_model(name: str):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    return RandomForestRegressor(
        n_estimators=100, min_samples_leaf=5, random_state=SEED, n_jobs=4, max_depth=16
    )


class GCNReg(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, n_layers=1, dropout=DROPOUT):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x).squeeze(-1)


class ResidualGCN(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, n_layers=1, dropout=DROPOUT):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin = nn.Linear(hidden + in_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x0 = x
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, edge_weight=edge_weight)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin(torch.cat([h, x0], dim=-1)).squeeze(-1)


def train_gnn(model, train_graphs, val_graphs, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state, best_val, wait = None, float("inf"), 0
    last_epoch = 0
    for epoch in range(1, max_epochs + 1):
        last_epoch = epoch
        model.train()
        for x, y, ei, ew in train_graphs:
            opt.zero_grad()
            pred = model(x, ei, ew)
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            maes = []
            for x, y, ei, ew in (val_graphs or train_graphs[-1:]):
                pred = model(x, ei, ew).cpu().numpy()
                maes.append(mean_absolute_error(y.cpu().numpy(), pred))
            val_mae = float(np.mean(maes))
        if val_mae < best_val - 1e-6:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val, last_epoch


@torch.no_grad()
def predict_gnn(model, x, ei, ew):
    model.eval()
    return model(x, ei, ew).cpu().numpy()


def load_panel():
    df = pd.read_csv(PANEL, dtype={"region_id": str})
    df["region_id"] = df["region_id"].astype(str).str.zfill(5)
    need = [YCOL, ANCHOR, *FEATURES]
    frame = df.dropna(subset=need).copy()
    frame = frame.sort_values(["year", "region_id"]).reset_index(drop=True)
    return frame


def align_year(df_year: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    return df_year.set_index("region_id").reindex(names)


def run_fold(frame, test_year, names, graphs):
    train = frame[frame.year < test_year]
    test = frame[frame.year == test_year]
    train_years = sorted(train.year.unique())
    rows = []

    A_hats = {}
    for k, A in graphs.items():
        A_hats[k] = gcn_normalize(A)

    tr = train.dropna(subset=[YCOL, *FEATURES])
    te = test.dropna(subset=[YCOL, *FEATURES])
    if tr.empty or te.empty:
        return rows

    ytr = tr[YCOL].to_numpy()
    yte = te[YCOL].to_numpy()
    Xtr = tr[FEATURES].to_numpy()
    Xte = te[FEATURES].to_numpy()

    # PreviousShare
    pred = te[ANCHOR].to_numpy()
    rows.append({
        "party": "DEM", "target": "vote_share", "test_year": int(test_year),
        "train_through": int(tr.year.max()), "n_train": len(tr), "n_test": len(te),
        "model": "PreviousShare", "family": "baseline", "adjacency": "none",
        "n_layers": 0, "n_features": 1, **scores(yte, pred),
    })

    for mname in ["Ridge", "RandomForest"]:
        mdl = classical_model(mname)
        mdl.fit(Xtr, ytr)
        pred = mdl.predict(Xte)
        rows.append({
            "party": "DEM", "target": "vote_share", "test_year": int(test_year),
            "train_through": int(tr.year.max()), "n_train": len(tr), "n_test": len(te),
            "model": mname, "family": "NonSpatial", "adjacency": "none",
            "n_layers": 0, "n_features": len(FEATURES), **scores(yte, pred),
        })

    # SGC
    for adj_name, A in graphs.items():
        A_hat = A_hats[adj_name]
        for n_layers in (1, 2):
            Xtr_list, ytr_list = [], []
            for yr in sorted(tr.year.unique()):
                g = align_year(tr[tr.year == yr], names)
                present = g[FEATURES].notna().all(axis=1) & g[YCOL].notna()
                X = g[FEATURES].to_numpy(dtype=float)
                col_med = np.nanmedian(X, axis=0)
                inds = np.where(np.isnan(X))
                if len(inds[0]):
                    X[inds] = np.take(col_med, inds[1])
                Xg = graph_convolve(X, A_hat, n_layers=n_layers)
                Xcat = np.hstack([X, Xg])
                Xtr_list.append(Xcat[present.to_numpy()])
                ytr_list.append(g.loc[present, YCOL].to_numpy())
            Xtr_s = np.vstack(Xtr_list)
            ytr_s = np.concatenate(ytr_list)

            gte = align_year(te, names)
            present_te = gte[FEATURES].notna().all(axis=1) & gte[YCOL].notna()
            Xte_m = gte[FEATURES].to_numpy(dtype=float)
            col_med = np.nanmedian(Xte_m, axis=0)
            inds = np.where(np.isnan(Xte_m))
            if len(inds[0]):
                Xte_m[inds] = np.take(col_med, inds[1])
            Xg = graph_convolve(Xte_m, A_hat, n_layers=n_layers)
            Xte_s = np.hstack([Xte_m, Xg])[present_te.to_numpy()]
            yte_s = gte.loc[present_te, YCOL].to_numpy()

            # RF skipped on dense dist_decay* — Ridge only (runtime); documented in protocol
            sgc_models = ["Ridge"] if adj_name.startswith("dist_decay") else ["Ridge", "RandomForest"]
            for mname in sgc_models:
                print(f"  SGC L{n_layers}+{mname}/{adj_name} year={test_year} …", flush=True)
                mdl = classical_model(mname)
                mdl.fit(Xtr_s, ytr_s)
                pred = mdl.predict(Xte_s)
                rows.append({
                    "party": "DEM", "target": "vote_share", "test_year": int(test_year),
                    "train_through": int(tr.year.max()), "n_train": len(ytr_s),
                    "n_test": len(yte_s),
                    "model": f"SGC_L{n_layers}+{mname}", "family": "SGC_diffusion",
                    "adjacency": adj_name, "n_layers": n_layers,
                    "n_features": Xtr_s.shape[1], **scores(yte_s, pred),
                })
                print(f"  SGC L{n_layers}+{mname}/{adj_name} year={test_year} MAE={rows[-1]['mae']:.4f}", flush=True)

    # Trainable GNN — queen + knn only; L1 GCN + ResidualGCN (runtime)
    scaler = StandardScaler().fit(Xtr)
    y_scaler = StandardScaler().fit(ytr.reshape(-1, 1))
    fill = np.nanmedian(Xtr, axis=0)

    def year_tensors(yr_df, A, weighted=False):
        g = align_year(yr_df, names)
        X = g[FEATURES].to_numpy(dtype=float)
        inds = np.where(np.isnan(X))
        if len(inds[0]):
            X[inds] = np.take(fill, inds[1])
        y = g[YCOL].to_numpy(dtype=float)
        y = np.where(np.isnan(y), float(np.nanmean(ytr)), y)
        Xn = scaler.transform(X).astype(np.float32)
        yn = y_scaler.transform(y.reshape(-1, 1)).ravel().astype(np.float32)
        ei, ew = adj_to_edge_index(A, weighted=weighted)
        xt = torch.from_numpy(Xn).to(DEVICE)
        yt = torch.from_numpy(yn).to(DEVICE)
        ei = ei.to(DEVICE)
        ew = ew.to(DEVICE) if ew is not None else None
        present = (g[FEATURES].notna().all(axis=1) & g[YCOL].notna()).to_numpy()
        return xt, yt, ei, ew, present, y

    gnn_adjs = [
        ("queen", graphs["queen"], False),
        # knn GCN omitted for US runtime; SGC covers knn sensitivity
    ]
    factories = [
        ("TrainableGCN_L1", lambda d: GCNReg(d, n_layers=1), 1),
        ("ResidualGCN_L1", lambda d: ResidualGCN(d, n_layers=1), 1),
    ]

    for adj_name, A, weighted in gnn_adjs:
        fit_graphs = []
        for yr in train_years:
            yr_df = tr[tr.year == yr]
            if yr_df.empty:
                continue
            xt, yt, ei, ew, _, _ = year_tensors(yr_df, A, weighted=weighted)
            fit_graphs.append((xt, yt, ei, ew))
        if not fit_graphs:
            continue
        val_graphs = fit_graphs[-1:]
        x_te, y_te_z, ei_te, ew_te, present_te, y_te_raw = year_tensors(te, A, weighted=weighted)

        for model_name, factory, n_layers in factories:
            t0 = time.time()
            set_seed(SEED)
            model = factory(len(FEATURES)).to(DEVICE)
            best_val, epochs_ran = train_gnn(model, fit_graphs, val_graphs)
            pred_z = predict_gnn(model, x_te, ei_te, ew_te)
            pred = y_scaler.inverse_transform(pred_z.reshape(-1, 1)).ravel()
            rows.append({
                "party": "DEM", "target": "vote_share", "test_year": int(test_year),
                "train_through": int(tr.year.max()),
                "n_train": int(sum(t.shape[0] for t, _, _, _ in fit_graphs)),
                "n_test": int(present_te.sum()),
                "model": model_name, "family": "trainable_gnn",
                "adjacency": adj_name, "n_layers": n_layers,
                "n_features": len(FEATURES),
                "hidden": HIDDEN, "dropout": DROPOUT, "lr": LR,
                "epochs_ran": epochs_ran, "best_val_mae_z": best_val,
                "train_seconds": round(time.time() - t0, 2),
                **scores(y_te_raw[present_te], pred[present_te]),
            })
            print(f"  GNN {model_name}/{adj_name} year={test_year} "
                  f"MAE={rows[-1]['mae']:.4f} epochs={epochs_ran} "
                  f"({rows[-1]['train_seconds']}s)", flush=True)

    return rows


def main():
    set_seed(SEED)
    t_all = time.time()
    frame = load_panel()
    # Prefer FIPS that appear in every test year for a stable graph
    year_sets = [set(frame.loc[frame.year == y, "region_id"]) for y in TEST_YEARS]
    common = set.intersection(*year_sets)
    # also need at least one prior year with features
    prior = set(frame.loc[frame.year < 2012, "region_id"])
    panel_fips = sorted(common & prior) if prior else sorted(common)
    # fallback: union of test-year counties matched later
    if len(panel_fips) < 2500:
        panel_fips = sorted(set.union(*year_sets))

    print(f"Building US county graphs for {len(panel_fips)} FIPS…", flush=True)
    graphs, names, name_to_idx, meta = build_graphs(panel_fips)
    with open(OUT / "graph_meta_us_county.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: meta[k] for k in meta if k != "queen_islands_sample"}, indent=2), flush=True)

    # Restrict frame to matched graph nodes
    frame = frame[frame.region_id.isin(names)].copy()
    unmatched = meta["unmatched_fips_n"]
    print(f"panel rows after geo match: {len(frame)}; unmatched FIPS: {unmatched}", flush=True)

    # Queen edge list sample
    Aq = graphs["queen"]
    edges = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if Aq[i, j] > 0:
                edges.append({"fips_i": names[i], "fips_j": names[j], "kind": "queen"})
    pd.DataFrame(edges).to_csv(OUT / "us_county_queen_edges.csv", index=False)

    protocol = {
        "seed": SEED,
        "panel": str(PANEL),
        "test_years": TEST_YEARS,
        "protocol": "expanding window: train year < test_year",
        "unit": f"US county (n={len(names)})",
        "target": YCOL,
        "features": FEATURES,
        "adjacency": {
            "primary": "queen contiguity (plotly geojson-counties-fips)",
            "sensitivity": ["knn_k=5", "dist_decay_sigma_km=80"],
            "census_tiger_status": "HTTP 403 — documented fallback",
        },
        "gnn": {
            "hidden": HIDDEN, "dropout": DROPOUT, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "note": "GCN_L1 + ResidualGCN_L1 on queen only (knn via SGC); max_epochs=40; SGC RF skip dist_decay*; RF n_est=100",
        },
        "graphs": meta,
        "note": "US RQ2 extension; do not mix MAE into Taiwan Core specification table.",
    }
    with open(OUT / "rq2_us_county_protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2)

    all_rows = []
    for ty in TEST_YEARS:
        print(f"=== fold test_year={ty} ===", flush=True)
        rows = run_fold(frame, ty, names, graphs)
        all_rows.extend(rows)
        # checkpoint
        pd.DataFrame(all_rows).to_csv(OUT / "rq2_us_county_backtest.csv", index=False)

    bt = pd.DataFrame(all_rows)
    bt.to_csv(OUT / "rq2_us_county_backtest.csv", index=False)

    # Mean MAE across years
    mean = (
        bt.groupby(["family", "model", "adjacency"], as_index=False)
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), n_years=("test_year", "nunique"))
        .sort_values("mae")
    )
    mean.to_csv(OUT / "rq2_us_county_mean_mae.csv", index=False)

    # Year comparison for headline models
    headline = bt[
        bt.model.isin([
            "PreviousShare", "Ridge", "RandomForest",
            "SGC_L1+RandomForest", "SGC_L2+RandomForest", "SGC_L1+Ridge", "SGC_L2+Ridge",
            "TrainableGCN_L1", "ResidualGCN_L1",
        ])
    ].copy()
    piv = headline.pivot_table(
        index=["family", "model", "adjacency"], columns="test_year", values="mae", aggfunc="first"
    )
    piv["mean"] = piv.mean(axis=1)
    piv = piv.sort_values("mean")
    piv.to_csv(OUT / "rq2_us_county_year_comparison.csv")

    # 2024 ablation
    bt[bt.test_year == 2024].sort_values("mae").to_csv(
        OUT / "rq2_us_county_ablation_2024.csv", index=False
    )

    # Chart: mean MAE top models
    top = mean.head(15).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [f"{r.model}|{r.adjacency}" for r in top.itertuples()]
    ax.barh(range(len(top)), top["mae"].values[::-1], color="#4C78A8")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel("Mean OOT MAE (2012–2024)")
    ax.set_title("US RQ2: county Dem share — spatial vs non-spatial")
    ax.axvline(mean.loc[mean.model == "PreviousShare", "mae"].values[0], color="red",
               linestyle="--", label="PreviousShare")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "rq2_us_mean_mae.png", dpi=140)
    plt.close()

    # Chart: year comparison Prev / NS RF / best SGC / best ResidualGCN
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, query in [
        ("PreviousShare", (bt.model == "PreviousShare")),
        ("NonSpatial RF", (bt.model == "RandomForest") & (bt.family == "NonSpatial")),
        ("SGC_L2+RF queen", (bt.model == "SGC_L2+RandomForest") & (bt.adjacency == "queen")),
        ("ResidualGCN queen", (bt.model == "ResidualGCN_L1") & (bt.adjacency == "queen")),
    ]:
        sub = bt.loc[query].sort_values("test_year")
        if sub.empty:
            continue
        ax.plot(sub.test_year, sub.mae, marker="o", label=label)
    ax.set_xlabel("Test year")
    ax.set_ylabel("MAE")
    ax.set_title("US RQ2 year-by-year MAE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "rq2_us_year_mae.png", dpi=140)
    plt.close()

    # README
    best_spatial = mean[mean.family != "baseline"].iloc[0]
    prev = mean[mean.model == "PreviousShare"].iloc[0]
    ns_rf = mean[(mean.model == "RandomForest") & (mean.family == "NonSpatial")]
    readme = f"""# RQ2 US county spatial vs non-spatial

## Design
- Panel: `cleaned/us_county_presidential_panel.csv` (MEDSL + socio)
- Target: Dem two-party share `party_share_A`
- Features: {FEATURES}
- Protocol: expanding OOT test years **{TEST_YEARS}** (train year < test)
- Units: **{len(names)}** counties matched to GeoJSON FIPS
- Seed: 42; CPU

## Adjacency
- **Primary: Queen contiguity** from plotly `geojson-counties-fips.json`
  (Census TIGER county cartographic ZIP returned HTTP 403 in this environment)
- Sensitivity: k-NN k={KNN_K}; distance-decay σ={DIST_SIGMA_KM} km (EPSG:5070)
- Queen edges: {meta['queen_n_edges']}; avg degree {meta['queen_avg_degree']:.2f}; islands {meta['queen_islands_n']}

## Headline mean OOT MAE (2012–2024)
See `rq2_us_county_mean_mae.csv` (computed, not invented).

| model | adjacency | mean MAE |
|---|---|---:|
| PreviousShare | none | {prev['mae']:.4f} |
| NonSpatial RandomForest | none | {float(ns_rf['mae'].iloc[0]) if len(ns_rf) else float('nan'):.4f} |
| Best non-baseline | {best_spatial['adjacency']} / {best_spatial['model']} | {best_spatial['mae']:.4f} |

**Does spatial help vs PreviousShare?** Compare rows in the CSV. US county voting has strong temporal inertia — PreviousShare is a hard baseline.

## Outputs
- `rq2_us_county_backtest.csv` — all folds
- `rq2_us_county_mean_mae.csv`
- `rq2_us_county_year_comparison.csv`
- `rq2_us_county_ablation_2024.csv`
- `graph_meta_us_county.json`, `us_county_queen_edges.csv`
- `charts/*.png`

## Reproduce
```bash
cd /workspace/taiwan-election-data
.venv/bin/python fyp_us_taiwan_transfer/scripts/run_rq2_us_county_spatial.py
```

Do **not** mix these MAE into Taiwan Core specification / `rq2_county` tables.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    print("\n=== MEAN MAE (top 12) ===")
    print(mean.head(12).to_string(index=False))
    print(f"\nDone in {time.time()-t_all:.1f}s → {OUT}")


if __name__ == "__main__":
    main()
