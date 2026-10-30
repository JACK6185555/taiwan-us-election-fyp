#!/usr/bin/env python3
"""RQ2 headline: county-level presidential spatial vs non-spatial, Family A window.

Same panel / features / expanding OOT (test years 2012/2016/2020/2024) as
taiwan_core_backtest.py (Family A). Spatial graphs built by dissolving MOI
township polygons to 22 counties.

Comparisons (identical features & splits):
  - Previous-election baseline
  - NonSpatial: Ridge / RandomForest / HistGradientBoosting
  - SGC diffusion (L1/L2) + Ridge/RF on Queen / kNN / distance-decay edges
  - Trainable GCN / GAT / ResidualGCN (shallow 1–2 layer) on Queen / kNN

SEED=42. CPU. Honest metrics only — no invented numbers.
Outputs: fyp_taiwan_spatial_gnn/rq2_county/
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATConv, GCNConv

warnings.filterwarnings("ignore")

ROOT = Path("/workspace/taiwan-election-data")
PANEL = Path("/workspace/taiwan-election-github/data/county_panel_features_2000_2024.csv")
GEOJSON = ROOT / "raw" / "geo" / "town-2025-03-18.geojson"
OUT = ROOT / "fyp_taiwan_spatial_gnn" / "rq2_county"
CHARTS = OUT / "charts"
SEED = 42
DEVICE = torch.device("cpu")

FEATURES = [
    "prev_DPP_vote_share", "prev_KMT_vote_share", "prev_Other_vote_share",
    "prev2_DPP_vote_share", "prev2_KMT_vote_share", "prev2_Other_vote_share",
    "prev_turnout", "DPP_change_1", "KMT_change_1", "Other_change_1",
    "prev_DPP_KMT_margin", "prev_DPP_Other_margin", "DPP_KMT_margin_change_1",
    "previous_party_change_abs_mean", "previous_party_change_std", "turnout_change_1",
]

# GNN hyperparams (documented)
HIDDEN = 32
DROPOUT = 0.15
LR = 5e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 300
PATIENCE = 40
GAT_HEADS = 2
KNN_K = 3
DIST_SIGMA_KM = 80.0  # distance-decay length scale (km, projected)

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


def build_county_geometries():
    import geopandas as gpd
    from shapely import make_valid
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
    from shapely.ops import unary_union
    from libpysal.weights import Queen

    def to_poly(geom):
        geom = make_valid(geom)
        if isinstance(geom, (Polygon, MultiPolygon)):
            return geom
        if isinstance(geom, GeometryCollection):
            polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
            if not polys:
                return geom.buffer(0)
            return unary_union(polys)
        return geom.buffer(0)

    g = gpd.read_file(GEOJSON)
    g["geometry"] = g.geometry.map(to_poly)
    c = g.dissolve(by="county").reset_index()
    c["geometry"] = c.geometry.map(to_poly)
    c = c.sort_values("county").reset_index(drop=True)
    names = c["county"].tolist()
    name_to_idx = {n: i for i, n in enumerate(names)}

    # Queen contiguity
    W = Queen.from_dataframe(c, use_index=False, silence_warnings=True)
    n = len(names)
    A_queen = np.eye(n, dtype=float)
    for i, neigh in W.neighbors.items():
        for j in neigh:
            A_queen[i, j] = 1.0
            A_queen[j, i] = 1.0

    # Projected centroids (TWD97 / TM2 zone 121) for metric distances
    cp = c.to_crs(epsg=3826)
    xy = np.column_stack([cp.geometry.centroid.x, cp.geometry.centroid.y])
    # pairwise km
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)) / 1000.0
    np.fill_diagonal(d, 0.0)

    # k-NN (symmetricize)
    A_knn = np.eye(n, dtype=float)
    for i in range(n):
        order = np.argsort(d[i])
        # skip self (0)
        for j in order[1 : KNN_K + 1]:
            A_knn[i, j] = 1.0
            A_knn[j, i] = 1.0

    # Distance-decay weighted adjacency (threshold: connect if w > 0.05 or Queen edge)
    W_dist = np.exp(-d / DIST_SIGMA_KM)
    np.fill_diagonal(W_dist, 0.0)
    A_dist = W_dist.copy()
    A_dist[A_dist < 0.05] = 0.0
    # keep Queen edges at least
    A_dist = np.maximum(A_dist, (A_queen - np.eye(n)) * 0.05)
    A_dist = A_dist + np.eye(n)

    # For binary ops we also store binary version of dist
    A_dist_bin = (A_dist > 0).astype(float)

    meta = {
        "n_counties": n,
        "counties": names,
        "queen_n_edges": int(((A_queen > 0).sum() - n) // 2),
        "queen_avg_degree": float(((A_queen > 0).sum(axis=1) - 1).mean()),
        "queen_islands": [names[i] for i in range(n) if (A_queen[i] > 0).sum() == 1],
        "knn_k": KNN_K,
        "knn_n_edges": int(((A_knn > 0).sum() - n) // 2),
        "knn_avg_degree": float(((A_knn > 0).sum(axis=1) - 1).mean()),
        "dist_sigma_km": DIST_SIGMA_KM,
        "dist_n_edges": int(((A_dist_bin > 0).sum() - n) // 2),
        "dist_avg_degree": float(((A_dist_bin > 0).sum(axis=1) - 1).mean()),
        "crs_projected": "EPSG:3826",
        "geo_source": str(GEOJSON),
    }
    graphs = {
        "queen": A_queen,
        "knn": A_knn,
        "dist_decay": A_dist,  # weighted
        "dist_decay_bin": A_dist_bin,
    }
    return graphs, names, name_to_idx, meta, d


def gcn_normalize(A: np.ndarray) -> np.ndarray:
    """Symmetric normalize; A may be weighted."""
    A = np.asarray(A, dtype=float)
    # ensure self-loops
    A = A.copy()
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
    """Return edge_index and optional edge_weight for PyG."""
    A = np.asarray(A, dtype=float)
    src, dst = np.nonzero(A > 0)
    ei = torch.from_numpy(np.vstack([src, dst]).astype(np.int64))
    if weighted:
        w = torch.from_numpy(A[src, dst].astype(np.float32))
        return ei, w
    return ei, None


def align_year(df_year: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Reindex year frame to fixed county order; drop if missing."""
    g = df_year.set_index("region_name").reindex(names)
    return g


def classical_model(name: str):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=2
        )
    return HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.06, max_leaf_nodes=7,
        min_samples_leaf=5, l2_regularization=1.0, random_state=SEED,
    )


class GCNReg(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, n_layers=2, dropout=DROPOUT):
        super().__init__()
        self.n_layers = n_layers
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
    """Shallow GCN with input skip (concat) to fight over-smoothing."""
    def __init__(self, in_dim, hidden=HIDDEN, n_layers=2, dropout=DROPOUT):
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


class GATReg(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, heads=GAT_HEADS, dropout=DROPOUT):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, dropout=dropout, concat=False)
        self.lin = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        # GATConv ignores edge_weight in this API path
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x).squeeze(-1)


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
    df = pd.read_csv(PANEL)
    df["target_KMT_vote_share"] = df["KMT_votes"] / df["valid_votes"]
    # Family A completeness for DPP features; KMT target may still be present
    frame = df.dropna(subset=["target_DPP_vote_share", *FEATURES]).copy()
    frame = frame.sort_values(["year", "region_name"]).reset_index(drop=True)
    return frame


def sgc_features_for_years(frame, years, names, feat_cols, A_hat, n_layers, kind):
    Xs, mask_ok = [], []
    for yr in years:
        g = align_year(frame[frame.year == yr], names)
        ok = g[feat_cols].notna().all(axis=1)
        X = g[feat_cols].to_numpy(dtype=float)
        # impute row-wise with column median of available in this year for diffusion stability
        col_med = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_med, inds[1])
        Xg = graph_convolve(X, A_hat, n_layers=n_layers)
        if kind == "gcn":
            Xs.append(Xg)
        elif kind == "raw":
            Xs.append(X)
        else:
            Xs.append(np.hstack([X, Xg]))
        mask_ok.append(ok.to_numpy())
    return np.vstack(Xs), np.concatenate(mask_ok)


def run_fold(frame, test_year, names, graphs, targets):
    """One expanding-window fold."""
    train = frame[frame.year < test_year]
    test = frame[frame.year == test_year]
    train_years = sorted(train.year.unique())
    rows = []

    # Fixed adjacency matrices (county graph does not change by year)
    A_hats = {k: gcn_normalize(A) for k, A in graphs.items()}

    for party, ycol, anchor in targets:
        # dropna for this target
        tr = train.dropna(subset=[ycol, *FEATURES])
        te = test.dropna(subset=[ycol, *FEATURES])
        if tr.empty or te.empty:
            continue
        ytr = tr[ycol].to_numpy()
        yte = te[ycol].to_numpy()
        # NonSpatial uses row order of panel (not graph order) — same as Family A
        Xtr = tr[FEATURES].to_numpy()
        Xte = te[FEATURES].to_numpy()

        # Previous share baseline
        pred = te[anchor].to_numpy()
        rows.append({
            "party": party, "target": "vote_share", "test_year": int(test_year),
            "train_through": int(tr.year.max()), "n_train": len(tr), "n_test": len(te),
            "model": "PreviousShare", "family": "baseline", "adjacency": "none",
            "n_layers": 0, "n_features": 1, **scores(yte, pred),
        })

        for mname in ["Ridge", "RandomForest", "HistGradientBoosting"]:
            mdl = classical_model(mname)
            mdl.fit(Xtr, ytr)
            pred = mdl.predict(Xte)
            rows.append({
                "party": party, "target": "vote_share", "test_year": int(test_year),
                "train_through": int(tr.year.max()), "n_train": len(tr), "n_test": len(te),
                "model": mname, "family": "NonSpatial", "adjacency": "none",
                "n_layers": 0, "n_features": len(FEATURES), **scores(yte, pred),
            })

        # --- SGC on each adjacency (graph-ordered) ---
        # Build stacked features in fixed county order, then filter to rows present in panel
        for adj_name, A in graphs.items():
            if adj_name == "dist_decay":
                A_hat = gcn_normalize(A)  # weighted
            else:
                A_hat = A_hats[adj_name]
            for n_layers in (1, 2):
                # train stack
                Xtr_list, ytr_list = [], []
                for yr in sorted(tr.year.unique()):
                    g = align_year(tr[tr.year == yr], names)
                    present = g[FEATURES].notna().all(axis=1) & g[ycol].notna()
                    X = g[FEATURES].to_numpy(dtype=float)
                    col_med = np.nanmedian(X, axis=0)
                    inds = np.where(np.isnan(X))
                    if len(inds[0]):
                        X[inds] = np.take(col_med, inds[1])
                    Xg = graph_convolve(X, A_hat, n_layers=n_layers)
                    Xcat = np.hstack([X, Xg])
                    Xtr_list.append(Xcat[present.to_numpy()])
                    ytr_list.append(g.loc[present, ycol].to_numpy())
                Xtr_s = np.vstack(Xtr_list)
                ytr_s = np.concatenate(ytr_list)

                gte = align_year(te, names)
                present_te = gte[FEATURES].notna().all(axis=1) & gte[ycol].notna()
                Xte_m = gte[FEATURES].to_numpy(dtype=float)
                col_med = np.nanmedian(Xte_m, axis=0)
                inds = np.where(np.isnan(Xte_m))
                if len(inds[0]):
                    Xte_m[inds] = np.take(col_med, inds[1])
                Xg = graph_convolve(Xte_m, A_hat, n_layers=n_layers)
                Xte_s = np.hstack([Xte_m, Xg])[present_te.to_numpy()]
                yte_s = gte.loc[present_te, ycol].to_numpy()

                for mname in ["Ridge", "RandomForest"]:
                    mdl = classical_model(mname)
                    mdl.fit(Xtr_s, ytr_s)
                    pred = mdl.predict(Xte_s)
                    rows.append({
                        "party": party, "target": "vote_share", "test_year": int(test_year),
                        "train_through": int(tr.year.max()), "n_train": len(ytr_s),
                        "n_test": len(yte_s),
                        "model": f"SGC_L{n_layers}+{mname}", "family": "SGC_diffusion",
                        "adjacency": adj_name, "n_layers": n_layers,
                        "n_features": Xtr_s.shape[1], **scores(yte_s, pred),
                    })

        # --- Trainable GNNs (queen + knn; binary edges) ---
        # Standardize on train pool in panel order, but feed graph-ordered tensors
        scaler = StandardScaler().fit(Xtr)
        y_scaler = StandardScaler().fit(ytr.reshape(-1, 1))

        def year_tensors(yr_df, A, weighted=False):
            g = align_year(yr_df, names)
            # For GNN we need full 22-node graph; impute missing features with train medians
            X = g[FEATURES].to_numpy(dtype=float)
            fill = np.nanmedian(Xtr, axis=0)
            inds = np.where(np.isnan(X))
            if len(inds[0]):
                X[inds] = np.take(fill, inds[1])
            y = g[ycol].to_numpy(dtype=float)
            # if target missing, fill with train mean (should not happen for our panel)
            y = np.where(np.isnan(y), float(np.nanmean(ytr)), y)
            Xn = scaler.transform(X).astype(np.float32)
            yn = y_scaler.transform(y.reshape(-1, 1)).ravel().astype(np.float32)
            ei, ew = adj_to_edge_index(A, weighted=weighted)
            xt = torch.from_numpy(Xn).to(DEVICE)
            yt = torch.from_numpy(yn).to(DEVICE)
            ei = ei.to(DEVICE)
            ew = ew.to(DEVICE) if ew is not None else None
            present = (g[FEATURES].notna().all(axis=1) & g[ycol].notna()).to_numpy()
            return xt, yt, ei, ew, present, y

        gnn_adjs = [
            ("queen", graphs["queen"], False),
            ("knn", graphs["knn"], False),
            ("dist_decay", graphs["dist_decay"], True),
        ]

        factories = [
            ("TrainableGCN_L1", lambda d: GCNReg(d, n_layers=1), 1),
            ("TrainableGCN_L2", lambda d: GCNReg(d, n_layers=2), 2),
            ("ResidualGCN_L2", lambda d: ResidualGCN(d, n_layers=2), 2),
            ("TrainableGAT", lambda d: GATReg(d), 2),
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
                # GAT: skip weighted dist (no edge_weight support used)
                if "GAT" in model_name and weighted:
                    continue
                set_seed(SEED)
                model = factory(len(FEATURES)).to(DEVICE)
                best_val, epochs_ran = train_gnn(model, fit_graphs, val_graphs)
                pred_z = predict_gnn(model, x_te, ei_te, ew_te)
                pred = y_scaler.inverse_transform(pred_z.reshape(-1, 1)).ravel()
                # Evaluate only on counties present in test panel (all 22 normally)
                rows.append({
                    "party": party, "target": "vote_share", "test_year": int(test_year),
                    "train_through": int(tr.year.max()),
                    "n_train": int(sum(len(y) for _, y, _, _ in fit_graphs)),
                    "n_test": int(present_te.sum()),
                    "model": model_name, "family": "trainable_gnn",
                    "adjacency": adj_name, "n_layers": n_layers,
                    "n_features": len(FEATURES),
                    "hidden": HIDDEN, "dropout": DROPOUT, "lr": LR,
                    "epochs_ran": epochs_ran, "best_val_mae_z": best_val,
                    **scores(y_te_raw[present_te], pred[present_te]),
                })

    return rows


def main():
    set_seed(SEED)
    print("Building county graphs…")
    graphs, names, name_to_idx, meta, dist_km = build_county_geometries()
    with open(OUT / "graph_meta_county.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    # edge list for queen
    Aq = graphs["queen"]
    edges = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if Aq[i, j] > 0:
                edges.append({"county_i": names[i], "county_j": names[j], "kind": "queen"})
    pd.DataFrame(edges).to_csv(OUT / "county_queen_edges.csv", index=False, encoding="utf-8-sig")

    frame = load_panel()
    years = sorted(frame.year.unique())
    test_years = [int(y) for y in years if y >= 2012]
    print("test_years", test_years, "n_rows", len(frame))

    # Verify name alignment
    panel_names = sorted(frame.region_name.unique())
    missing = [n for n in panel_names if n not in name_to_idx]
    extra = [n for n in names if n not in set(panel_names)]
    assert not missing and not extra, (missing, extra)
    print("county name alignment OK:", len(names))

    targets = [
        ("DPP", "target_DPP_vote_share", "prev_DPP_vote_share"),
        ("KMT", "target_KMT_vote_share", "prev_KMT_vote_share"),
    ]

    all_rows = []
    for ty in test_years:
        print(f"=== fold test_year={ty} ===")
        rows = run_fold(frame, ty, names, graphs, targets)
        all_rows.extend(rows)
        print(f"  +{len(rows)} rows")

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(OUT / "rq2_county_backtest.csv", index=False, encoding="utf-8-sig")

    # Headline ablation: mean MAE over 2012–2024 for DPP
    dpp = metrics[metrics.party == "DPP"].copy()
    summary = (
        dpp.groupby(["family", "model", "adjacency"], as_index=False)[["mae", "rmse"]]
        .mean()
        .sort_values("mae")
    )
    summary.to_csv(OUT / "rq2_county_mean_mae_dpp.csv", index=False, encoding="utf-8-sig")

    kmt = metrics[metrics.party == "KMT"].copy()
    summary_k = (
        kmt.groupby(["family", "model", "adjacency"], as_index=False)[["mae", "rmse"]]
        .mean()
        .sort_values("mae")
    )
    summary_k.to_csv(OUT / "rq2_county_mean_mae_kmt.csv", index=False, encoding="utf-8-sig")

    # 2024 snapshot table
    snap = metrics[metrics.test_year == 2024].copy()
    snap.to_csv(OUT / "rq2_county_ablation_2024.csv", index=False, encoding="utf-8-sig")

    # Chart: DPP mean MAE by family (best per family)
    best = summary.loc[summary.groupby("family")["mae"].idxmin()].sort_values("mae")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = {"baseline": "#555555", "NonSpatial": "#2878B5", "SGC_diffusion": "#E07A2D",
              "trainable_gnn": "#4B9B70"}
    ax.barh(
        [f"{r.family}|{r.model}|{r.adjacency}" for _, r in best.iterrows()],
        best["mae"],
        color=[colors.get(f, "#888") for f in best["family"]],
    )
    ax.set_xlabel("Mean OOT MAE (DPP share, 2012–2024)")
    ax.set_title("RQ2 county: best model per family (lower better)", loc="left")
    ax.grid(axis="x", color="#e5e5e5")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS / "rq2_county_best_by_family_dpp.png", dpi=160)
    plt.close(fig)

    # Chart: NonSpatial vs best spatial for each year (DPP)
    ns = dpp[(dpp.family == "NonSpatial") & (dpp.model == "RandomForest")][
        ["test_year", "mae"]
    ].rename(columns={"mae": "RF_NonSpatial"})
    # best spatial each year
    spat = dpp[dpp.family.isin(["SGC_diffusion", "trainable_gnn"])]
    best_y = spat.loc[spat.groupby("test_year")["mae"].idxmin()][
        ["test_year", "mae", "model", "adjacency"]
    ].rename(columns={"mae": "best_spatial_mae", "model": "best_spatial_model",
                      "adjacency": "best_spatial_adj"})
    prev = dpp[dpp.model == "PreviousShare"][["test_year", "mae"]].rename(
        columns={"mae": "PreviousShare"}
    )
    cmp_ = prev.merge(ns, on="test_year").merge(best_y, on="test_year")
    cmp_.to_csv(OUT / "rq2_county_year_comparison_dpp.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(cmp_))
    w = 0.25
    ax.bar(x - w, cmp_["PreviousShare"], w, label="PreviousShare", color="#555")
    ax.bar(x, cmp_["RF_NonSpatial"], w, label="RF NonSpatial", color="#2878B5")
    ax.bar(x + w, cmp_["best_spatial_mae"], w, label="Best spatial", color="#E07A2D")
    ax.set_xticks(x)
    ax.set_xticklabels(cmp_["test_year"].astype(int))
    ax.set_ylabel("MAE (DPP share)")
    ax.set_title("RQ2 county OOT: Previous vs RF vs best spatial", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color="#e5e5e5")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS / "rq2_county_year_bars_dpp.png", dpi=160)
    plt.close(fig)

    # Hyperparam / protocol note
    protocol = {
        "seed": SEED,
        "panel": str(PANEL),
        "features": FEATURES,
        "test_years": test_years,
        "protocol": "expanding window: train year < test_year",
        "unit": "county (22)",
        "targets": ["target_DPP_vote_share", "target_KMT_vote_share"],
        "gnn": {
            "hidden": HIDDEN, "dropout": DROPOUT, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE, "gat_heads": GAT_HEADS,
            "residual": "ResidualGCN concatenates diffused hidden with raw input",
        },
        "graphs": meta,
        "note": "Township MAE must not be mixed into Family A / this county RQ2 table.",
    }
    with open(OUT / "rq2_county_protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2)

    print("\n=== DPP mean MAE (top 15) ===")
    print(summary.head(15).to_string(index=False))
    print("\n=== DPP NonSpatial / baseline ===")
    print(summary[summary.family.isin(["baseline", "NonSpatial"])].to_string(index=False))
    print("\nWrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
