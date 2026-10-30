#!/usr/bin/env python3
"""US GCN pilot on New England counties only (~67 FIPS).

Label: us_gcn_pilot — NOT full-US trainable GCN; NOT Core specification.
Compare PreviousShare / NonSpatial / SGC / TrainableGCN / ResidualGCN
on the SAME New England subset only. SEED=42. No invented metrics.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

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

ROOT = Path("/workspace/taiwan-election-data")
PANEL = ROOT / "fyp_us_taiwan_transfer" / "cleaned" / "us_county_presidential_panel.csv"
GEOJSON = ROOT / "raw" / "geo" / "us_county" / "counties.geojson"
OUT = ROOT / "results" / "us_gcn_pilot"
PACK_OUT = Path("/workspace/taiwan-election-github") / "results" / "us_gcn_pilot"

# New England state FIPS
NE_STATES = {"09", "23", "25", "33", "44", "50"}  # CT ME MA NH RI VT
SEED = 42
DEVICE = torch.device("cpu")
FEATURES = [
    "prev_share_A", "log_total_votes", "unemployment",
    "education_rate", "median_income", "log_density",
]
YCOL = "party_share_A"
ANCHOR = "prev_share_A"
TEST_YEARS = [2012, 2016, 2020, 2024]
HIDDEN, DROPOUT, LR, WEIGHT_DECAY = 32, 0.15, 5e-3, 1e-4
MAX_EPOCHS, PATIENCE = 25, 8


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def scores(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
    }


def gcn_normalize(A):
    A = np.asarray(A, float).copy()
    np.fill_diagonal(A, np.maximum(np.diag(A), 1.0))
    deg = A.sum(axis=1)
    d_inv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D = np.diag(d_inv)
    return D @ A @ D


def graph_convolve(X, A_hat, n_layers):
    H = X.astype(float)
    for _ in range(n_layers):
        H = A_hat @ H
    return H


def adj_to_edge_index(A):
    src, dst = np.nonzero(A > 0)
    return torch.from_numpy(np.vstack([src, dst]).astype(np.int64))


class GCNReg(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, n_layers=1, dropout=DROPOUT):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_dim, hidden)])
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x).squeeze(-1)


class ResidualGCN(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, n_layers=1, dropout=DROPOUT):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_dim, hidden)])
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin = nn.Linear(hidden + in_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x0 = x
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_weight=edge_weight))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin(torch.cat([h, x0], dim=-1)).squeeze(-1)


def train_gnn(model, train_graphs, val_graphs, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state, best_val, wait, last_epoch = None, float("inf"), 0, 0
    for epoch in range(1, max_epochs + 1):
        last_epoch = epoch
        model.train()
        for x, y, ei in train_graphs:
            opt.zero_grad()
            loss = F.mse_loss(model(x, ei), y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            maes = []
            for x, y, ei in (val_graphs or train_graphs[-1:]):
                pred = model(x, ei).cpu().numpy()
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
def predict_gnn(model, x, ei):
    model.eval()
    return model(x, ei).cpu().numpy()


def build_queen_knn(names):
    import geopandas as gpd
    from libpysal.weights import Queen

    gdf = gpd.read_file(GEOJSON)
    # plotly geojson uses id = FIPS
    if "id" in gdf.columns:
        gdf["fips"] = gdf["id"].astype(str).str.zfill(5)
    elif "GEOID" in gdf.columns:
        gdf["fips"] = gdf["GEOID"].astype(str).str.zfill(5)
    else:
        raise RuntimeError(f"no fips col: {gdf.columns.tolist()}")
    gdf = gdf[gdf.fips.isin(names)].copy()
    gdf = gdf.set_index("fips").reindex(names)
    # drop missing geometries
    missing = gdf.geometry.isna()
    if missing.any():
        raise RuntimeError(f"missing geometries for {missing.sum()} FIPS: {gdf.index[missing].tolist()[:10]}")
    w = Queen.from_dataframe(gdf, use_index=True, silence_warnings=True)
    n = len(names)
    A = np.zeros((n, n), dtype=float)
    id_to_i = {f: i for i, f in enumerate(names)}
    for i, neigh in w.neighbors.items():
        # i may be fips string or positional depending on libpysal
        fi = names[i] if isinstance(i, (int, np.integer)) else str(i)
        ii = id_to_i[fi]
        for j in neigh:
            fj = names[j] if isinstance(j, (int, np.integer)) else str(j)
            A[ii, id_to_i[fj]] = 1.0
    # centroids for knn
    cent = gdf.to_crs(5070).geometry.centroid
    xy = np.column_stack([cent.x.to_numpy(), cent.y.to_numpy()])
    k = min(5, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    _, idx = nn.kneighbors(xy)
    A_knn = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in idx[i, 1:]:
            A_knn[i, j] = 1.0
            A_knn[j, i] = 1.0
    meta = {
        "n": n,
        "queen_edges": int(A.sum() / 2),
        "knn_edges": int(A_knn.sum() / 2),
        "mean_degree_queen": float(A.sum(axis=1).mean()),
        "subset": "NewEngland",
        "states": sorted(NE_STATES),
    }
    return {"queen": A, "knn": A_knn}, meta


def classical(name):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    return RandomForestRegressor(
        n_estimators=100, min_samples_leaf=5, random_state=SEED, n_jobs=2, max_depth=16
    )


def align_year(df_year, names):
    return df_year.set_index("region_id").reindex(names)


def main():
    set_seed()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    PACK_OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PANEL, dtype={"region_id": str})
    df["region_id"] = df["region_id"].astype(str).str.zfill(5)
    if "median_income" not in df.columns:
        # prefer acs then ers
        if "acs_median_income" in df.columns:
            df["median_income"] = df["acs_median_income"]
        elif "ers_median_income_2022" in df.columns:
            df["median_income"] = df["ers_median_income_2022"]
    need = [YCOL, ANCHOR, *FEATURES]
    frame = df.dropna(subset=need).copy()
    frame = frame[frame.region_id.str[:2].isin(NE_STATES)].copy()
    frame = frame.sort_values(["year", "region_id"]).reset_index(drop=True)

    year_sets = [set(frame.loc[frame.year == y, "region_id"]) for y in TEST_YEARS]
    common = set.intersection(*year_sets)
    prior = set(frame.loc[frame.year < 2012, "region_id"])
    names = sorted(common & prior) if prior else sorted(common)
    if len(names) < 20:
        names = sorted(set.union(*year_sets))
    frame = frame[frame.region_id.isin(names)].copy()
    print(f"NE pilot counties={len(names)} rows={len(frame)}", flush=True)

    graphs, gmeta = build_queen_knn(names)
    with open(OUT / "pilot_graph_meta.json", "w") as f:
        json.dump(gmeta, f, indent=2)

    A_hats = {k: gcn_normalize(A) for k, A in graphs.items()}
    rows = []

    for test_year in TEST_YEARS:
        train = frame[frame.year < test_year]
        test = frame[frame.year == test_year]
        tr = train.dropna(subset=[YCOL, *FEATURES])
        te = test.dropna(subset=[YCOL, *FEATURES])
        if tr.empty or te.empty:
            continue
        ytr, yte = tr[YCOL].to_numpy(), te[YCOL].to_numpy()
        Xtr, Xte = tr[FEATURES].to_numpy(), te[FEATURES].to_numpy()
        train_years = sorted(tr.year.unique())

        pred = te[ANCHOR].to_numpy()
        rows.append({
            "subset": "NewEngland", "test_year": int(test_year),
            "n_train": len(tr), "n_test": len(te),
            "model": "PreviousShare", "family": "baseline", "adjacency": "none",
            **scores(yte, pred),
        })

        for mname in ["Ridge", "RandomForest"]:
            mdl = classical(mname)
            mdl.fit(Xtr, ytr)
            rows.append({
                "subset": "NewEngland", "test_year": int(test_year),
                "n_train": len(tr), "n_test": len(te),
                "model": mname, "family": "NonSpatial", "adjacency": "none",
                **scores(yte, mdl.predict(Xte)),
            })

        # SGC L1 queen + knn Ridge/RF
        for adj_name, A in graphs.items():
            A_hat = A_hats[adj_name]
            Xtr_list, ytr_list = [], []
            for yr in train_years:
                g = align_year(tr[tr.year == yr], names)
                present = g[FEATURES].notna().all(axis=1) & g[YCOL].notna()
                X = g[FEATURES].to_numpy(dtype=float)
                col_med = np.nanmedian(X, axis=0)
                inds = np.where(np.isnan(X))
                if len(inds[0]):
                    X[inds] = np.take(col_med, inds[1])
                Xg = graph_convolve(X, A_hat, 1)
                Xtr_list.append(np.hstack([X, Xg])[present.to_numpy()])
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
            Xte_s = np.hstack([Xte_m, graph_convolve(Xte_m, A_hat, 1)])[present_te.to_numpy()]
            yte_s = gte.loc[present_te, YCOL].to_numpy()
            for mname in ["Ridge", "RandomForest"]:
                mdl = classical(mname)
                mdl.fit(Xtr_s, ytr_s)
                rows.append({
                    "subset": "NewEngland", "test_year": int(test_year),
                    "n_train": len(ytr_s), "n_test": len(yte_s),
                    "model": f"SGC_L1+{mname}", "family": "SGC_diffusion",
                    "adjacency": adj_name, **scores(yte_s, mdl.predict(Xte_s)),
                })

        # Trainable GCN / ResidualGCN on queen
        scaler = StandardScaler().fit(Xtr)
        y_scaler = StandardScaler().fit(ytr.reshape(-1, 1))
        fill = np.nanmedian(Xtr, axis=0)
        A = graphs["queen"]

        def year_tensors(yr_df):
            g = align_year(yr_df, names)
            X = g[FEATURES].to_numpy(dtype=float)
            inds = np.where(np.isnan(X))
            if len(inds[0]):
                X[inds] = np.take(fill, inds[1])
            y = g[YCOL].to_numpy(dtype=float)
            y = np.where(np.isnan(y), float(np.nanmean(ytr)), y)
            Xn = scaler.transform(X).astype(np.float32)
            yn = y_scaler.transform(y.reshape(-1, 1)).ravel().astype(np.float32)
            ei = adj_to_edge_index(A).to(DEVICE)
            present = (g[FEATURES].notna().all(axis=1) & g[YCOL].notna()).to_numpy()
            return (
                torch.from_numpy(Xn).to(DEVICE),
                torch.from_numpy(yn).to(DEVICE),
                ei,
                present,
                y,
            )

        fit_graphs = []
        for yr in train_years:
            yr_df = tr[tr.year == yr]
            if yr_df.empty:
                continue
            xt, yt, ei, _, _ = year_tensors(yr_df)
            fit_graphs.append((xt, yt, ei))
        x_te, _, ei_te, present_te, y_te_raw = year_tensors(te)

        for model_name, factory in [
            ("TrainableGCN_L1", lambda: GCNReg(len(FEATURES), n_layers=1)),
            ("ResidualGCN_L1", lambda: ResidualGCN(len(FEATURES), n_layers=1)),
        ]:
            t1 = time.time()
            set_seed()
            model = factory().to(DEVICE)
            best_val, epochs_ran = train_gnn(model, fit_graphs, fit_graphs[-1:])
            pred_z = predict_gnn(model, x_te, ei_te)
            pred = y_scaler.inverse_transform(pred_z.reshape(-1, 1)).ravel()
            rows.append({
                "subset": "NewEngland", "test_year": int(test_year),
                "n_train": int(sum(t.shape[0] for t, _, _ in fit_graphs)),
                "n_test": int(present_te.sum()),
                "model": model_name, "family": "trainable_gnn", "adjacency": "queen",
                "epochs_ran": epochs_ran, "best_val_mae_z": best_val,
                "train_seconds": round(time.time() - t1, 2),
                **scores(y_te_raw[present_te], pred[present_te]),
            })
            print(
                f"  {model_name} {test_year} MAE={rows[-1]['mae']:.4f} "
                f"epochs={epochs_ran} ({rows[-1]['train_seconds']}s)",
                flush=True,
            )

    bt = pd.DataFrame(rows)
    mean = (
        bt.groupby(["family", "model", "adjacency"], dropna=False)["mae"]
        .mean()
        .reset_index()
        .rename(columns={"mae": "mean_mae"})
        .sort_values("mean_mae")
    )
    protocol = {
        "label": "us_gcn_pilot",
        "subset": "NewEngland",
        "states_fips": sorted(NE_STATES),
        "n_counties": len(names),
        "test_years": TEST_YEARS,
        "seed": SEED,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "note": (
            "Scaled-down trainable GCN pilot on New England only. "
            "NOT comparable to full-US RQ2 MAE (0.0345 Prev). "
            "Compare models only within this subset. "
            "Full-US trainable GCN remains DEFERRED."
        ),
        "wall_seconds": round(time.time() - t0, 2),
        "graph": gmeta,
    }
    for dest in (OUT, PACK_OUT):
        dest.mkdir(parents=True, exist_ok=True)
        bt.to_csv(dest / "us_gcn_pilot_backtest.csv", index=False)
        mean.to_csv(dest / "us_gcn_pilot_mean_mae.csv", index=False)
        with open(dest / "us_gcn_pilot_protocol.json", "w") as f:
            json.dump(protocol, f, indent=2)
        with open(dest / "pilot_graph_meta.json", "w") as f:
            json.dump(gmeta, f, indent=2)

    print("MEAN MAE (NE subset only):", flush=True)
    print(mean.to_string(index=False), flush=True)
    print(f"DONE in {protocol['wall_seconds']}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
