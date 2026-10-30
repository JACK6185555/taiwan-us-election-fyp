#!/usr/bin/env python3
"""Export feature / channel weights for US RQ2 (NonSpatial + SGC + GCN).

Produces:
  - NonSpatial Ridge coefficients (standardized features)
  - NonSpatial RF impurity importances
  - Permutation importance (OOT test year) for NonSpatial RF / Ridge
  - SGC+Ridge coefficients on Raw||Diffused channels (queen L1)
  - Trainable GCN / ResidualGCN: |lin.weight| on hidden (and residual input skip)

SEED=42. Honest metrics only.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GCNConv

warnings.filterwarnings("ignore")

ROOT = Path("/workspace/taiwan-election-data")
PANEL = ROOT / "fyp_us_taiwan_transfer" / "cleaned" / "us_county_presidential_panel.csv"
OUT = ROOT / "fyp_us_taiwan_transfer" / "rq2_us_county"
GEOJSON = ROOT / "raw" / "geo" / "us_county" / "counties.geojson"
SEED = 42
DEVICE = torch.device("cpu")
FEATURES = [
    "prev_share_A", "log_total_votes", "unemployment",
    "education_rate", "median_income", "log_density",
]
YCOL = "party_share_A"
TEST_YEARS = [2012, 2016, 2020, 2024]
HIDDEN = 32
DROPOUT = 0.15
LR = 5e-3
WD = 1e-4
MAX_EPOCHS = 120
PATIENCE = 20


def set_seed(s=SEED):
    np.random.seed(s)
    torch.manual_seed(s)


def gcn_normalize(A):
    A = np.asarray(A, dtype=float).copy()
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


def load_queen_and_names():
    meta = json.loads((OUT / "graph_meta_us_county.json").read_text())
    # Rebuild queen from edges CSV + meta order from protocol / edges
    import geopandas as gpd
    from libpysal.weights import Queen
    from sklearn.neighbors import NearestNeighbors

    df = pd.read_csv(PANEL, dtype={"region_id": str})
    df["region_id"] = df["region_id"].astype(str).str.zfill(5)
    year_sets = [set(df.loc[df.year == y, "region_id"]) for y in TEST_YEARS]
    prior = set(df.loc[df.year < 2012, "region_id"])
    panel_fips = sorted(set.intersection(*year_sets) & prior)

    g = gpd.read_file(GEOJSON)
    g["GEOID"] = g["id"].astype(str).str.zfill(5)
    present = [f for f in panel_fips if f in set(g["GEOID"])]
    g = g.set_index("GEOID").loc[present].reset_index()
    names = present
    n = len(names)
    W = Queen.from_dataframe(g, use_index=False, silence_warnings=True)
    A = np.eye(n, dtype=float)
    for i, neigh in W.neighbors.items():
        for j in neigh:
            A[i, j] = A[j, i] = 1.0
    return A, names, df


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
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
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
            h = conv(h, edge_index, edge_weight=edge_weight)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin(torch.cat([h, x0], dim=-1)).squeeze(-1)


def train_gnn(model, graphs, val, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best, best_val, wait = None, float("inf"), 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y, ei in graphs:
            opt.zero_grad()
            loss = F.mse_loss(model(x, ei), y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            x, y, ei = val[0]
            pred = model(x, ei).cpu().numpy()
            mae = float(np.mean(np.abs(pred - y.cpu().numpy())))
        if mae < best_val - 1e-6:
            best_val = mae
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best:
        model.load_state_dict(best)
    return best_val


def main():
    set_seed()
    A_queen, names, raw = load_queen_and_names()
    A_hat = gcn_normalize(A_queen)
    frame = raw.dropna(subset=[YCOL, *FEATURES]).copy()
    frame = frame[frame.region_id.isin(names)].sort_values(["year", "region_id"])

    rows_coef, rows_imp, rows_perm, rows_sgc, rows_gnn = [], [], [], [], []

    for ty in TEST_YEARS:
        tr = frame[frame.year < ty]
        te = frame[frame.year == ty]
        Xtr, ytr = tr[FEATURES].to_numpy(), tr[YCOL].to_numpy()
        Xte, yte = te[FEATURES].to_numpy(), te[YCOL].to_numpy()

        # --- NonSpatial Ridge coefficients (on standardized X) ---
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        pipe.fit(Xtr, ytr)
        coef = pipe.named_steps["ridge"].coef_
        for f, c in zip(FEATURES, coef):
            rows_coef.append({
                "test_year": ty, "model": "Ridge", "family": "NonSpatial",
                "feature": f, "coefficient": float(c), "abs_coefficient": float(abs(c)),
            })

        # --- NonSpatial RF impurity ---
        rf = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, max_depth=20,
            random_state=SEED, n_jobs=2,
        )
        rf.fit(Xtr, ytr)
        for f, imp in zip(FEATURES, rf.feature_importances_):
            rows_imp.append({
                "test_year": ty, "model": "RandomForest", "family": "NonSpatial",
                "feature": f, "importance": float(imp), "kind": "impurity",
            })

        # permutation on test (OOT)
        for name, mdl in [("Ridge", pipe), ("RandomForest", rf)]:
            r = permutation_importance(
                mdl, Xte, yte, n_repeats=10, random_state=SEED, scoring="neg_mean_absolute_error", n_jobs=2
            )
            for f, m, s in zip(FEATURES, r.importances_mean, r.importances_std):
                rows_perm.append({
                    "test_year": ty, "model": name, "family": "NonSpatial",
                    "feature": f, "perm_importance_mae": float(m),
                    "perm_importance_std": float(s),
                    "note": "decrease in neg-MAE = increase in MAE when shuffled; OOT test year",
                })

        # --- SGC L1 queen: Raw||Diffused → Ridge coef ---
        def align(yr_df):
            return yr_df.set_index("region_id").reindex(names)

        Xtr_list, ytr_list = [], []
        for yr in sorted(tr.year.unique()):
            g = align(tr[tr.year == yr])
            present = g[FEATURES].notna().all(axis=1) & g[YCOL].notna()
            X = g[FEATURES].to_numpy(dtype=float)
            med = np.nanmedian(X, axis=0)
            inds = np.where(np.isnan(X))
            if len(inds[0]):
                X[inds] = np.take(med, inds[1])
            Xg = graph_convolve(X, A_hat, 1)
            Xtr_list.append(np.hstack([X, Xg])[present.to_numpy()])
            ytr_list.append(g.loc[present, YCOL].to_numpy())
        Xtr_s = np.vstack(Xtr_list)
        ytr_s = np.concatenate(ytr_list)
        channel_names = [f"raw:{f}" for f in FEATURES] + [f"sgcL1:{f}" for f in FEATURES]
        sgc_pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        sgc_pipe.fit(Xtr_s, ytr_s)
        for ch, c in zip(channel_names, sgc_pipe.named_steps["ridge"].coef_):
            rows_sgc.append({
                "test_year": ty, "model": "SGC_L1+Ridge", "adjacency": "queen",
                "channel": ch, "coefficient": float(c), "abs_coefficient": float(abs(c)),
            })

        # --- GCN / ResidualGCN lin weights ---
        src, dst = np.nonzero(A_queen > 0)
        ei = torch.from_numpy(np.vstack([src, dst]).astype(np.int64)).to(DEVICE)
        scaler = StandardScaler().fit(Xtr)
        y_scaler = StandardScaler().fit(ytr.reshape(-1, 1))
        fill = np.nanmedian(Xtr, axis=0)

        def year_t(yr_df):
            g = align(yr_df)
            X = g[FEATURES].to_numpy(dtype=float)
            inds = np.where(np.isnan(X))
            if len(inds[0]):
                X[inds] = np.take(fill, inds[1])
            y = g[YCOL].to_numpy(dtype=float)
            y = np.where(np.isnan(y), float(np.nanmean(ytr)), y)
            xt = torch.from_numpy(scaler.transform(X).astype(np.float32)).to(DEVICE)
            yt = torch.from_numpy(
                y_scaler.transform(y.reshape(-1, 1)).ravel().astype(np.float32)
            ).to(DEVICE)
            return xt, yt, ei

        fit_graphs = []
        for yr in sorted(tr.year.unique()):
            yd = tr[tr.year == yr]
            if len(yd):
                fit_graphs.append(year_t(yd))
        val = fit_graphs[-1:]

        for mname, factory in [
            ("TrainableGCN_L1", lambda: GCNReg(len(FEATURES), n_layers=1)),
            ("ResidualGCN_L1", lambda: ResidualGCN(len(FEATURES), n_layers=1)),
        ]:
            set_seed()
            model = factory().to(DEVICE)
            train_gnn(model, fit_graphs, val)
            # First GCNConv weight: shape [hidden, in_dim] (PyG stores as [out, in])
            conv_w = model.convs[0].lin.weight.detach().cpu().numpy()  # [hidden, in]
            # channel sensitivity ≈ L2 norm of columns
            col_norm = np.linalg.norm(conv_w, axis=0)
            for f, nrm in zip(FEATURES, col_norm):
                rows_gnn.append({
                    "test_year": ty, "model": mname, "adjacency": "queen",
                    "part": "GCNConv0_input_col_L2", "feature": f,
                    "weight": float(nrm),
                })
            lin_w = model.lin.weight.detach().cpu().numpy().ravel()
            if mname.startswith("Residual"):
                # lin is [hidden + in_dim]
                for i, w in enumerate(lin_w[:HIDDEN]):
                    rows_gnn.append({
                        "test_year": ty, "model": mname, "adjacency": "queen",
                        "part": "lin_hidden", "feature": f"h{i}", "weight": float(w),
                        "abs_weight": float(abs(w)),
                    })
                for f, w in zip(FEATURES, lin_w[HIDDEN:]):
                    rows_gnn.append({
                        "test_year": ty, "model": mname, "adjacency": "queen",
                        "part": "lin_input_skip", "feature": f, "weight": float(w),
                        "abs_weight": float(abs(w)),
                    })
            else:
                for i, w in enumerate(lin_w):
                    rows_gnn.append({
                        "test_year": ty, "model": mname, "adjacency": "queen",
                        "part": "lin_hidden", "feature": f"h{i}", "weight": float(w),
                        "abs_weight": float(abs(w)),
                    })
            print(f"weights year={ty} {mname} done", flush=True)

    pd.DataFrame(rows_coef).to_csv(OUT / "feature_weights_nonspatial_ridge.csv", index=False)
    pd.DataFrame(rows_imp).to_csv(OUT / "feature_importance_nonspatial_rf.csv", index=False)
    pd.DataFrame(rows_perm).to_csv(OUT / "feature_perm_importance_nonspatial_oot.csv", index=False)
    pd.DataFrame(rows_sgc).to_csv(OUT / "feature_weights_sgc_ridge_queen.csv", index=False)
    pd.DataFrame(rows_gnn).to_csv(OUT / "feature_weights_gnn_queen.csv", index=False)

    # Summaries (mean |coef| / importance across years)
    coef = pd.DataFrame(rows_coef)
    summary = (
        coef.groupby("feature", as_index=False)
        .agg(mean_abs_coef=("abs_coefficient", "mean"), mean_coef=("coefficient", "mean"))
        .sort_values("mean_abs_coef", ascending=False)
    )
    summary.to_csv(OUT / "feature_weights_ridge_summary.csv", index=False)

    imp = pd.DataFrame(rows_imp)
    imp_sum = (
        imp.groupby("feature", as_index=False)
        .agg(mean_importance=("importance", "mean"))
        .sort_values("mean_importance", ascending=False)
    )
    imp_sum.to_csv(OUT / "feature_importance_rf_summary.csv", index=False)

    sgc = pd.DataFrame(rows_sgc)
    sgc_sum = (
        sgc.groupby("channel", as_index=False)
        .agg(mean_abs_coef=("abs_coefficient", "mean"))
        .sort_values("mean_abs_coef", ascending=False)
    )
    sgc_sum.to_csv(OUT / "feature_weights_sgc_summary.csv", index=False)

    # GCN input channel summary
    gnn = pd.DataFrame(rows_gnn)
    gnn_in = gnn[gnn.part == "GCNConv0_input_col_L2"]
    gnn_sum = (
        gnn_in.groupby(["model", "feature"], as_index=False)
        .agg(mean_col_L2=("weight", "mean"))
        .sort_values(["model", "mean_col_L2"], ascending=[True, False])
    )
    gnn_sum.to_csv(OUT / "feature_weights_gnn_input_summary.csv", index=False)

    note = {
        "seed": SEED,
        "interpretation": (
            "Ridge coefficients: on StandardScaler features (comparable magnitudes). "
            "RF impurity: in-sample train split only. "
            "Permutation: OOT test-year MAE degradation. "
            "SGC channels: raw:* vs sgcL1:* after Queen diffusion. "
            "GCNConv0_input_col_L2: column L2 of first graph-conv weight (input feature channel strength). "
            "ResidualGCN lin_input_skip: direct skip weights on raw inputs. "
            "Not causal."
        ),
        "files": [
            "feature_weights_nonspatial_ridge.csv",
            "feature_importance_nonspatial_rf.csv",
            "feature_perm_importance_nonspatial_oot.csv",
            "feature_weights_sgc_ridge_queen.csv",
            "feature_weights_gnn_queen.csv",
            "feature_weights_ridge_summary.csv",
            "feature_importance_rf_summary.csv",
            "feature_weights_sgc_summary.csv",
            "feature_weights_gnn_input_summary.csv",
        ],
    }
    (OUT / "feature_weights_README.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    print("Wrote feature weight tables →", OUT)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
