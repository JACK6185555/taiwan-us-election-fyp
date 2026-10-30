#!/usr/bin/env python3
"""Taiwan FYP — Trainable GCN / GAT on Queen contiguity (vs SGC / NonSpatial).

Uses PyTorch + torch-geometric (installed). OOT: train years < test_year.
SEED=42. Honest metrics only.
"""
from __future__ import annotations

import json
import sys
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATConv, GCNConv

warnings.filterwarnings("ignore")

ROOT = Path("/workspace/taiwan-election-data")
PRES = ROOT / "fyp_taiwan_improved" / "harmonized_presidential_improved.csv"
GEOJSON = ROOT / "raw" / "geo" / "town-2025-03-18.geojson"
OUT = ROOT / "fyp_taiwan_spatial_gnn"
CHARTS = OUT / "charts"
SEED = 42
DEVICE = torch.device("cpu")

# Hyperparams (reported in meta)
HIDDEN = 32
N_LAYERS = 2
DROPOUT = 0.15
LR = 5e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 250
PATIENCE = 35
GAT_HEADS = 2

REGION_ALIASES = {
    "彰化縣員林鎮": "彰化縣員林市",
    "苗栗縣頭份鎮": "苗栗縣頭份市",
}

FEAT_BASE = [
    "prev_dpp_share", "prev_kmt_share", "prev_turnout",
    "median_income", "urbanization", "education_rate", "log_people",
    "geo_north", "geo_central", "geo_south", "geo_east", "geo_islands",
    "esc_dpp_delta", "esc_kmt_delta", "esc_dpp_pid", "esc_kmt_pid",
]

for p in (OUT, CHARTS, OUT / "scripts"):
    p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_region(s: str) -> str:
    s = str(s).replace("台", "臺").strip()
    return REGION_ALIASES.get(s, s)


def mets(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
    }


def build_county_adjacency(counties: list[str]) -> np.ndarray:
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
    deg = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D = np.diag(d_inv_sqrt)
    return D @ A @ D


def graph_convolve(X: np.ndarray, A_hat: np.ndarray, n_layers: int = 2) -> np.ndarray:
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
    import geopandas as gpd
    from libpysal.weights import Queen

    gdf = gpd.read_file(GEOJSON)
    gdf["region_norm"] = gdf["name"].map(normalize_region)
    if gdf["region_norm"].duplicated().any():
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
    return A_full, names, name_to_idx


def induce_contiguity(A_full: np.ndarray, name_to_idx: dict[str, int], regions: list[str]):
    n = len(regions)
    A = np.eye(n, dtype=float)
    unmatched = []
    idxs = []
    for r in regions:
        rn = normalize_region(r)
        if rn in name_to_idx:
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
    return A, {"n_unmatched": len(unmatched), "unmatched": unmatched}


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


def adj_to_edge_index(A: np.ndarray) -> torch.Tensor:
    """Undirected edges from dense A (includes self-loops already)."""
    src, dst = np.nonzero(A > 0)
    ei = np.vstack([src, dst]).astype(np.int64)
    return torch.from_numpy(ei)


def year_graph(df_year: pd.DataFrame, feat_cols, ycol, A_full, name_to_idx, adj_mode: str):
    g = df_year.reset_index(drop=True)
    X = g[feat_cols].values.astype(np.float32)
    y = g[ycol].values.astype(np.float32)
    if adj_mode == "county":
        A = build_county_adjacency(g["county_name"].tolist())
    else:
        A, _ = induce_contiguity(A_full, name_to_idx, g["region"].tolist())
    edge_index = adj_to_edge_index(A)
    return X, y, edge_index, A


class GCNReg(nn.Module):
    def __init__(self, in_dim: int, hidden: int = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x).squeeze(-1)


class GATReg(nn.Module):
    def __init__(self, in_dim: int, hidden: int = HIDDEN, heads: int = GAT_HEADS, dropout: float = DROPOUT):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, dropout=dropout, concat=False)
        self.lin = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x).squeeze(-1)


def train_gnn(model, train_graphs, val_graphs=None, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """train_graphs: list of (x_t, y_t, edge_index) already on DEVICE, x standardized."""
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state = None
    best_val = float("inf")
    wait = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_nodes = 0
        for x, y, ei in train_graphs:
            opt.zero_grad()
            pred = model(x, ei)
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(y)
            n_nodes += len(y)
        train_mse = total_loss / max(n_nodes, 1)

        # validation = mean MAE on val graphs (or last train year if none)
        model.eval()
        with torch.no_grad():
            val_list = val_graphs if val_graphs else train_graphs[-1:]
            maes = []
            for x, y, ei in val_list:
                pred = model(x, ei).cpu().numpy()
                maes.append(mean_absolute_error(y.cpu().numpy(), pred))
            val_mae = float(np.mean(maes))
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mae": val_mae})

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
    return history, best_val, epoch


@torch.no_grad()
def predict_gnn(model, x, edge_index):
    model.eval()
    return model(x, edge_index).cpu().numpy()


def sgc_stack(df, feat_cols, ycol, A_full, name_to_idx, adj_mode, n_layers=2, kind="cat"):
    Xs, ys = [], []
    for _, g in df.groupby("year", sort=True):
        g = g.reset_index(drop=True)
        X = g[feat_cols].values.astype(float)
        if adj_mode == "county":
            A = build_county_adjacency(g["county_name"].tolist())
        else:
            A, _ = induce_contiguity(A_full, name_to_idx, g["region"].tolist())
        A_hat = gcn_normalize(A)
        Xg = graph_convolve(X, A_hat, n_layers=n_layers)
        if kind == "gcn":
            Xs.append(Xg)
        elif kind == "raw":
            Xs.append(X)
        else:
            Xs.append(np.hstack([X, Xg]))
        ys.append(g[ycol].values)
    return np.vstack(Xs), np.concatenate(ys)


def eval_one(data, party, target, test_year, A_full, name_to_idx, run_county_gnn: bool):
    set_seed(SEED)
    ycol = ycol_for(party, target)
    feat_cols = feat_cols_for(data, party)
    train = data[data["year"] < test_year].dropna(subset=[ycol]).copy()
    test = data[data["year"] == test_year].dropna(subset=[ycol]).copy()
    if len(train) < 50 or len(test) < 20:
        return [], None

    train, test = impute_matrix(train, test, feat_cols)
    rows = []
    loss_curves = {}

    # --- NonSpatial classical ---
    Xtr_raw, ytr = sgc_stack(train, feat_cols, ycol, A_full, name_to_idx, "contiguity", kind="raw")
    Xte_raw, yte = sgc_stack(test, feat_cols, ycol, A_full, name_to_idx, "contiguity", kind="raw")

    for mname, mdl in [
        ("Ridge", Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))])),
        ("RandomForest", RandomForestRegressor(
            n_estimators=250, min_samples_leaf=3, random_state=SEED, n_jobs=2
        )),
        ("MLP", Pipeline([
            ("s", StandardScaler()),
            ("m", MLPRegressor(
                hidden_layer_sizes=(HIDDEN, HIDDEN),
                max_iter=400, random_state=SEED, early_stopping=True,
                validation_fraction=0.15, n_iter_no_change=25,
            )),
        ])),
    ]:
        mdl.fit(Xtr_raw, ytr)
        pred = mdl.predict(Xte_raw)
        rows.append({
            "party": party.upper(), "target": target, "test_year": test_year,
            "model": mname, "family": "NonSpatial", "adjacency": "none",
            "n_train": len(ytr), "n_test": len(yte), "n_features": Xtr_raw.shape[1],
            **mets(yte, pred),
        })

    # --- SGC diffusion + Ridge/RF (contiguity) ---
    Xtr_sgc, ytr_s = sgc_stack(train, feat_cols, ycol, A_full, name_to_idx, "contiguity", n_layers=2, kind="cat")
    Xte_sgc, yte_s = sgc_stack(test, feat_cols, ycol, A_full, name_to_idx, "contiguity", n_layers=2, kind="cat")
    for mname, mdl in [
        ("Ridge", Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))])),
        ("RandomForest", RandomForestRegressor(
            n_estimators=250, min_samples_leaf=3, random_state=SEED, n_jobs=2
        )),
    ]:
        mdl.fit(Xtr_sgc, ytr_s)
        pred = mdl.predict(Xte_sgc)
        rows.append({
            "party": party.upper(), "target": target, "test_year": test_year,
            "model": f"SGC_L2+{mname}", "family": "SGC_diffusion", "adjacency": "contiguity",
            "n_train": len(ytr_s), "n_test": len(yte_s), "n_features": Xtr_sgc.shape[1],
            **mets(yte_s, pred),
        })

    # --- Trainable GCN / GAT ---
    # Standardize features + target on train pool (inverse-transform at predict)
    scaler = StandardScaler()
    scaler.fit(Xtr_raw)
    y_scaler = StandardScaler()
    y_scaler.fit(ytr.reshape(-1, 1))

    train_years = sorted(train["year"].unique())
    # Fit on ALL pre-test years; monitor last train year for early stopping
    # (last year remains in the training set — ES only, no test leak).
    fit_years = list(train_years)
    val_year = train_years[-1] if len(train_years) >= 1 else None

    def make_graphs(years, adj_mode):
        graphs = []
        for yr in years:
            if int(yr) == int(test_year):
                gdf = test
            else:
                gdf = train[train["year"] == yr]
            if len(gdf) == 0:
                continue
            X, y, ei, _ = year_graph(gdf, feat_cols, ycol, A_full, name_to_idx, adj_mode)
            Xn = scaler.transform(X).astype(np.float32)
            yn = y_scaler.transform(y.reshape(-1, 1)).ravel().astype(np.float32)
            xt = torch.from_numpy(Xn).to(DEVICE)
            yt = torch.from_numpy(yn).to(DEVICE)
            graphs.append((xt, yt, ei.to(DEVICE)))
        return graphs

    adj_modes = ["contiguity"]
    if run_county_gnn:
        adj_modes.append("county")

    for adj_mode in adj_modes:
        fit_graphs = make_graphs(fit_years, adj_mode)
        val_graphs = make_graphs([val_year], adj_mode) if val_year is not None else None
        test_graphs = make_graphs([test_year], adj_mode)
        x_te, y_te, ei_te = test_graphs[0]

        for model_name, factory in [
            ("TrainableGCN", lambda: GCNReg(in_dim=len(feat_cols))),
            ("TrainableGAT", lambda: GATReg(in_dim=len(feat_cols), heads=GAT_HEADS)),
        ]:
            set_seed(SEED)
            model = factory().to(DEVICE)
            hist, best_val, last_epoch = train_gnn(model, fit_graphs, val_graphs)
            pred_z = predict_gnn(model, x_te, ei_te)
            pred = y_scaler.inverse_transform(pred_z.reshape(-1, 1)).ravel()
            # Raw (unscaled) test targets for honest MAE/RMSE/R2
            y_te_raw = year_graph(test, feat_cols, ycol, A_full, name_to_idx, adj_mode)[1]
            key = f"{party}_{target}_{test_year}_{model_name}_{adj_mode}"
            loss_curves[key] = hist
            rows.append({
                "party": party.upper(), "target": target, "test_year": test_year,
                "model": model_name, "family": "trainable_gnn", "adjacency": adj_mode,
                "n_train": int(sum(len(y) for _, y, _ in fit_graphs)),
                "n_test": int(len(y_te_raw)),
                "n_features": len(feat_cols),
                "hidden": HIDDEN, "n_layers": N_LAYERS, "dropout": DROPOUT,
                "lr": LR, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                "epochs_ran": last_epoch, "best_val_mae_z": best_val,
                "target_standardized": True,
                "gat_heads": GAT_HEADS if "GAT" in model_name else None,
                "backend": "torch_geometric",
                **mets(y_te_raw, pred),
            })

    return rows, loss_curves


def main():
    set_seed(SEED)
    print("Loading Queen contiguity…")
    A_full, names, name_to_idx = load_geo_queen()
    print(f"  full polygons={len(names)}")

    df = pd.read_csv(PRES)
    all_rows = []
    all_curves = {}

    # Primary 2024 with county sensitivity; 2020/2016 contiguity-only for speed
    plans = [
        (2024, True),
        (2020, False),
        (2016, False),
    ]
    for test_year, run_county in plans:
        for party in ["dpp", "kmt"]:
            for target in ["swing", "demeaned"]:
                print(f" Trainable OOT {test_year} {party} {target} (county={run_county})…", flush=True)
                rows, curves = eval_one(df, party, target, test_year, A_full, name_to_idx, run_county)
                all_rows.extend(rows)
                if curves:
                    all_curves.update(curves)

    res = pd.DataFrame(all_rows)
    res.to_csv(OUT / "trainable_gcn_gat_backtest.csv", index=False)
    res24 = res[res["test_year"] == 2024].copy()
    res24.to_csv(OUT / "trainable_gcn_gat_ablation_2024.csv", index=False)

    # Compact pivot for 2024 contiguity
    piv = []
    for (party, target), g in res24.groupby(["party", "target"]):
        def pick(family=None, model=None, adj=None):
            q = g
            if family:
                q = q[q["family"] == family]
            if model:
                q = q[q["model"] == model]
            if adj is not None:
                q = q[q["adjacency"] == adj]
            if len(q) == 0:
                return None
            return float(q.iloc[0]["MAE"])

        piv.append({
            "party": party,
            "target": target,
            "MAE_NonSpatial_Ridge": pick(model="Ridge", family="NonSpatial"),
            "MAE_NonSpatial_RF": pick(model="RandomForest", family="NonSpatial"),
            "MAE_NonSpatial_MLP": pick(model="MLP", family="NonSpatial"),
            "MAE_SGC_Ridge_Contig": pick(model="SGC_L2+Ridge"),
            "MAE_SGC_RF_Contig": pick(model="SGC_L2+RandomForest"),
            "MAE_TrainableGCN_Contig": pick(model="TrainableGCN", adj="contiguity"),
            "MAE_TrainableGAT_Contig": pick(model="TrainableGAT", adj="contiguity"),
            "MAE_TrainableGCN_County": pick(model="TrainableGCN", adj="county"),
            "MAE_TrainableGAT_County": pick(model="TrainableGAT", adj="county"),
        })
    piv_df = pd.DataFrame(piv)
    piv_df.to_csv(OUT / "trainable_vs_sgc_ablation_2024.csv", index=False)

    # Loss curves (optional JSON)
    curves_path = OUT / "trainable_gcn_gat_loss_curves.json"
    # trim to last 5 keys sample + all for size control — save all
    with open(curves_path, "w", encoding="utf-8") as f:
        json.dump(all_curves, f)

    meta = {
        "seed": SEED,
        "device": str(DEVICE),
        "backend": "torch + torch_geometric",
        "torch_version": torch.__version__,
        "hidden": HIDDEN,
        "n_layers": N_LAYERS,
        "dropout": DROPOUT,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "gat_heads": GAT_HEADS,
        "features": FEAT_BASE,
        "note": (
            "Trainable 2-layer GCNConv / GATConv on Queen contiguity (and county clique sensitivity). "
            "SGC L2 Raw∥diffuse + Ridge/RF kept as baseline. NonSpatial MLP matches hidden size. "
            "Fit on all years < test_year; early-stop monitors last pre-test year MAE in z-space (that year stays in train). Features and targets StandardScaler-fit on train; predictions inverse-transformed before MAE. "
            "Hyperparams: hidden=32, layers=2, dropout=0.15, lr=5e-3, wd=1e-4, max_epochs=250, patience=35, gat_heads=2, seed=42, CPU."
        ),
    }
    with open(OUT / "trainable_gcn_gat_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Chart: 2024 MAE by family for DPP swing / demeaned
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, target in zip(axes, ["swing", "demeaned"]):
        sub = piv_df[piv_df["target"] == target]
        if len(sub) == 0:
            continue
        labels = [
            "NS Ridge", "NS RF", "NS MLP",
            "SGC+Ridge", "SGC+RF",
            "GCN contig", "GAT contig",
        ]
        for i, (_, r) in enumerate(sub.iterrows()):
            vals = [
                r["MAE_NonSpatial_Ridge"], r["MAE_NonSpatial_RF"], r["MAE_NonSpatial_MLP"],
                r["MAE_SGC_Ridge_Contig"], r["MAE_SGC_RF_Contig"],
                r["MAE_TrainableGCN_Contig"], r["MAE_TrainableGAT_Contig"],
            ]
            x = np.arange(len(labels)) + i * 0.35
            ax.bar(x, vals, width=0.3, label=r["party"])
        ax.set_xticks(np.arange(len(labels)) + 0.15)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("MAE")
        ax.set_title(f"2024 OOT — {target}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "trainable_gcn_gat_vs_baselines_2024.png", dpi=140)
    plt.close()

    print("\n=== 2024 Contiguity MAE pivot ===")
    print(piv_df.to_string(index=False))
    print(f"\nWrote {OUT / 'trainable_gcn_gat_ablation_2024.csv'}")
    print(f"Wrote {OUT / 'trainable_vs_sgc_ablation_2024.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
