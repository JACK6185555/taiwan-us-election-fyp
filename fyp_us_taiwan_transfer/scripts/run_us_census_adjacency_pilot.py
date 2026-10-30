#!/usr/bin/env python3
"""US RQ2 spatial pilot using official Census County Adjacency 2024.

Unblocks DEFERRED blank: builds FIPS-aligned contiguity from Census, runs
(1) full-panel SGC + Prev/NonSpatial OOT MAE, and
(2) Mid-Atlantic multi-state trainable GCN-lite (PyTorch dense, NO torch-geometric).

Label: us_census_adjacency_pilot
Locked Core US Prev MAE 0.0345 is reported for reference; subset MAEs must not
be mixed with that locked number unless the fold set matches full Core protocol.
SEED=42.
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
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "fyp_us_taiwan_transfer" / "cleaned" / "us_county_presidential_panel.csv"
ADJ_TXT = ROOT / "data" / "us_county_adjacency" / "county_adjacency2024.txt"
OUT = ROOT / "results" / "us_census_adjacency_pilot"
ADJ_OUT = ROOT / "data" / "us_county_adjacency"

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
# Mid-Atlantic (NY/NJ/PA) — expansion suggested in US_GCN_DEFERRED.md
MA_STATES = {"36", "34", "42"}
LOCKED_US_PREV_MAE = 0.0345  # Core specification; do not overwrite

HIDDEN, DROPOUT, LR, WEIGHT_DECAY = 32, 0.15, 5e-3, 1e-4
MAX_EPOCHS, PATIENCE = 30, 8


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def scores(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
    }


def gcn_normalize_dense(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, float).copy()
    np.fill_diagonal(A, np.maximum(np.diag(A), 1.0))
    deg = A.sum(axis=1)
    d_inv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D = np.diag(d_inv)
    return D @ A @ D


def gcn_normalize_sparse(A: sparse.csr_matrix) -> sparse.csr_matrix:
    A = A.tocsr(copy=True).astype(float)
    A = A + sparse.eye(A.shape[0], format="csr")
    # ensure self already present then re-add is ok; clip diag to >=1
    deg = np.asarray(A.sum(axis=1)).ravel()
    d_inv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D = sparse.diags(d_inv)
    return (D @ A @ D).tocsr()


def graph_convolve(X: np.ndarray, A_hat, n_layers: int) -> np.ndarray:
    H = X.astype(float)
    for _ in range(n_layers):
        H = A_hat @ H
    return np.asarray(H)


def classical(name: str):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    return RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=5,
        max_depth=16,
        random_state=SEED,
        n_jobs=4,
    )


def load_census_edges(path: Path) -> tuple[set[str], set[tuple[str, str]], dict]:
    raw = pd.read_csv(path, sep="|", dtype=str)
    raw["County GEOID"] = raw["County GEOID"].str.zfill(5)
    raw["Neighbor GEOID"] = raw["Neighbor GEOID"].str.zfill(5)
    all_fips = set(raw["County GEOID"]) | set(raw["Neighbor GEOID"])
    edges: set[tuple[str, str]] = set()
    for a, b in zip(raw["County GEOID"], raw["Neighbor GEOID"]):
        if a == b:
            continue
        if a < b:
            edges.add((a, b))
        else:
            edges.add((b, a))
    meta = {
        "source_url": "https://www2.census.gov/geo/docs/reference/county_adjacency/county_adjacency2024.txt",
        "source_file": str(path.name),
        "year_label": 2024,
        "n_unique_geoids_in_file": len(all_fips),
        "n_undirected_edges": len(edges),
        "n_raw_rows": int(len(raw)),
    }
    return all_fips, edges, meta


def build_adj_for_names(names: list[str], edges: set[tuple[str, str]], sparse_out: bool):
    idx = {f: i for i, f in enumerate(names)}
    n = len(names)
    rows, cols = [], []
    for a, b in edges:
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            rows.extend([i, j])
            cols.extend([j, i])
    if sparse_out:
        data = np.ones(len(rows), dtype=float)
        A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        # zero diagonal (no self)
        A.setdiag(0)
        A.eliminate_zeros()
        return A
    A = np.zeros((n, n), dtype=float)
    for i, j in zip(rows, cols):
        A[i, j] = 1.0
    np.fill_diagonal(A, 0.0)
    return A


def align_year(df_year: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    return df_year.set_index("region_id").reindex(names)


class DenseGCN(nn.Module):
    """1-layer GCN without torch-geometric: y = W2 @ ReLU(A_hat @ X @ W1)."""

    def __init__(self, in_dim: int, hidden: int = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden, bias=True)
        self.W2 = nn.Linear(hidden, 1, bias=True)
        self.dropout = dropout

    def forward(self, x, A_hat):
        h = torch.mm(A_hat, x)
        h = F.relu(self.W1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.mm(A_hat, h)
        return self.W2(h).squeeze(-1)


class ResidualDenseGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden, bias=True)
        self.W2 = nn.Linear(hidden + in_dim, 1, bias=True)
        self.dropout = dropout

    def forward(self, x, A_hat):
        h = torch.mm(A_hat, x)
        h = F.relu(self.W1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.mm(A_hat, h)
        return self.W2(torch.cat([h, x], dim=-1)).squeeze(-1)


def train_gnn(model, A_hat_t, train_xy, val_xy, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state, best_val, wait, last_epoch = None, float("inf"), 0, 0
    for epoch in range(1, max_epochs + 1):
        last_epoch = epoch
        model.train()
        for x, y in train_xy:
            opt.zero_grad()
            loss = F.mse_loss(model(x, A_hat_t), y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vx, vy = val_xy
            pred = model(vx, A_hat_t).cpu().numpy()
            val_mae = float(mean_absolute_error(vy.cpu().numpy(), pred))
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
def predict_gnn(model, x, A_hat_t):
    model.eval()
    return model(x, A_hat_t).cpu().numpy()


def prepare_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL, dtype={"region_id": str})
    df["region_id"] = df["region_id"].astype(str).str.zfill(5)
    if "median_income" not in df.columns or df["median_income"].isna().all():
        if "acs_median_income" in df.columns:
            df["median_income"] = df["acs_median_income"]
        elif "ers_median_income_2022" in df.columns:
            df["median_income"] = df["ers_median_income_2022"]
    if "log_density" not in df.columns and "acs_log_density" in df.columns:
        df["log_density"] = df["acs_log_density"]
    return df


def panel_common_names(df: pd.DataFrame) -> list[str]:
    sets = [set(df.loc[df.year == y, "region_id"]) for y in TEST_YEARS]
    prior = set(df.loc[df.year < 2012, "region_id"])
    common = set.intersection(*sets) & prior
    # require features present at least once
    need = df.dropna(subset=[YCOL, *FEATURES])
    common &= set(need.region_id)
    return sorted(common)


def run_baselines_and_sgc(frame, names, A, A_hat, subset_label, adj_label, rows):
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

        rows.append(
            {
                "subset": subset_label,
                "test_year": int(test_year),
                "n_train": len(tr),
                "n_test": len(te),
                "model": "PreviousShare",
                "family": "baseline",
                "adjacency": "none",
                **scores(yte, te[ANCHOR].to_numpy()),
            }
        )
        for mname in ["Ridge", "RandomForest"]:
            mdl = classical(mname)
            mdl.fit(Xtr, ytr)
            rows.append(
                {
                    "subset": subset_label,
                    "test_year": int(test_year),
                    "n_train": len(tr),
                    "n_test": len(te),
                    "model": mname,
                    "family": "NonSpatial",
                    "adjacency": "none",
                    **scores(yte, mdl.predict(Xte)),
                }
            )

        # SGC L1
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
            rows.append(
                {
                    "subset": subset_label,
                    "test_year": int(test_year),
                    "n_train": len(ytr_s),
                    "n_test": len(yte_s),
                    "model": f"SGC_L1+{mname}",
                    "family": "SGC_diffusion",
                    "adjacency": adj_label,
                    **scores(yte_s, mdl.predict(Xte_s)),
                }
            )
    return rows


def run_trainable_gcn(frame, names, A_dense, subset_label, adj_label, rows):
    A_hat = gcn_normalize_dense(A_dense)
    A_hat_t = torch.from_numpy(A_hat.astype(np.float32)).to(DEVICE)
    for test_year in TEST_YEARS:
        train = frame[frame.year < test_year]
        test = frame[frame.year == test_year]
        tr = train.dropna(subset=[YCOL, *FEATURES])
        te = test.dropna(subset=[YCOL, *FEATURES])
        if tr.empty or te.empty:
            continue
        ytr = tr[YCOL].to_numpy()
        Xtr = tr[FEATURES].to_numpy()
        train_years = sorted(tr.year.unique())
        scaler = StandardScaler().fit(Xtr)
        y_scaler = StandardScaler().fit(ytr.reshape(-1, 1))
        fill = np.nanmedian(Xtr, axis=0)

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
            present = (g[FEATURES].notna().all(axis=1) & g[YCOL].notna()).to_numpy()
            return (
                torch.from_numpy(Xn).to(DEVICE),
                torch.from_numpy(yn).to(DEVICE),
                present,
                y,
            )

        fit_xy = []
        for yr in train_years:
            yr_df = tr[tr.year == yr]
            if yr_df.empty:
                continue
            xt, yt, _, _ = year_tensors(yr_df)
            fit_xy.append((xt, yt))
        x_te, _, present_te, y_te_raw = year_tensors(te)
        val_xy = fit_xy[-1]

        for model_name, factory in [
            ("DenseGCN_L1", lambda: DenseGCN(len(FEATURES))),
            ("ResidualDenseGCN_L1", lambda: ResidualDenseGCN(len(FEATURES))),
        ]:
            t1 = time.time()
            set_seed()
            model = factory().to(DEVICE)
            best_val, epochs_ran = train_gnn(model, A_hat_t, fit_xy, val_xy)
            pred_z = predict_gnn(model, x_te, A_hat_t)
            pred = y_scaler.inverse_transform(pred_z.reshape(-1, 1)).ravel()
            rows.append(
                {
                    "subset": subset_label,
                    "test_year": int(test_year),
                    "n_train": int(sum(t.shape[0] for t, _ in fit_xy)),
                    "n_test": int(present_te.sum()),
                    "model": model_name,
                    "family": "trainable_gnn_lite",
                    "adjacency": adj_label,
                    "epochs_ran": epochs_ran,
                    "best_val_mae_z": best_val,
                    "train_seconds": round(time.time() - t1, 2),
                    **scores(y_te_raw[present_te], pred[present_te]),
                }
            )
            print(
                f"  [{subset_label}] {model_name} {test_year} "
                f"MAE={rows[-1]['mae']:.4f} epochs={epochs_ran} "
                f"({rows[-1]['train_seconds']}s)",
                flush=True,
            )
    return rows


def main():
    set_seed()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ADJ_OUT.mkdir(parents=True, exist_ok=True)

    adj_fips, edges, adj_meta = load_census_edges(ADJ_TXT)
    df = prepare_panel()
    all_panel_fips = set(df.region_id.astype(str).str.zfill(5))
    names_full = panel_common_names(df)

    in_graph = [f for f in names_full if f in adj_fips]
    missing = [f for f in names_full if f not in adj_fips]
    # also: graph nodes that have at least one neighbor inside panel subset
    A_full = build_adj_for_names(in_graph, edges, sparse_out=True)
    deg = np.asarray(A_full.sum(axis=1)).ravel()
    islands = int((deg == 0).sum())
    match = {
        "panel_common_oot_fips": len(names_full),
        "census_file_geoids": adj_meta["n_unique_geoids_in_file"],
        "panel_intersect_census": len(in_graph),
        "match_rate": round(len(in_graph) / max(len(names_full), 1), 6),
        "panel_missing_from_census": missing[:50],
        "n_panel_missing_from_census": len(missing),
        "n_undirected_edges_in_subgraph": int(A_full.nnz // 2),
        "mean_degree": float(deg.mean()) if len(deg) else 0.0,
        "n_islands_degree0": islands,
        "adjacency_meta": adj_meta,
        "locked_us_core_prev_mae_reference": LOCKED_US_PREV_MAE,
    }
    print(
        f"FIPS match: panel_common={len(names_full)} ∩ census={len(in_graph)} "
        f"rate={match['match_rate']:.4f} edges={match['n_undirected_edges_in_subgraph']} "
        f"islands={islands}",
        flush=True,
    )
    (OUT / "fips_match_report.json").write_text(json.dumps(match, indent=2), encoding="utf-8")
    (ADJ_OUT / "fips_match_report.json").write_text(json.dumps(match, indent=2), encoding="utf-8")

    # persist sparse adjacency for reuse
    sparse.save_npz(ADJ_OUT / "us_county_adjacency_panel.npz", A_full)
    pd.Series(in_graph, name="fips").to_csv(ADJ_OUT / "panel_fips_ordered.csv", index=False)

    rows: list[dict] = []
    frame_full = df.dropna(subset=[YCOL, *FEATURES])
    frame_full = frame_full[frame_full.region_id.isin(in_graph)].copy()
    A_hat_full = gcn_normalize_sparse(A_full)

    print("=== Full-US SGC (Census adjacency) ===", flush=True)
    rows = run_baselines_and_sgc(
        frame_full, in_graph, A_full, A_hat_full, "FullUS", "census2024_queen", rows
    )

    # Mid-Atlantic trainable + SGC
    ma_names = [f for f in in_graph if f[:2] in MA_STATES]
    print(f"=== MidAtlantic pilot counties={len(ma_names)} ===", flush=True)
    if len(ma_names) >= 20:
        frame_ma = frame_full[frame_full.region_id.isin(ma_names)].copy()
        A_ma = build_adj_for_names(ma_names, edges, sparse_out=False)
        A_hat_ma = gcn_normalize_dense(A_ma)
        rows = run_baselines_and_sgc(
            frame_ma, ma_names, A_ma, A_hat_ma, "MidAtlantic", "census2024_queen", rows
        )
        rows = run_trainable_gcn(
            frame_ma, ma_names, A_ma, "MidAtlantic", "census2024_queen", rows
        )
        ma_meta = {
            "states_fips": sorted(MA_STATES),
            "n_counties": len(ma_names),
            "n_edges": int(A_ma.sum() / 2),
            "mean_degree": float(A_ma.sum(axis=1).mean()),
        }
    else:
        ma_meta = {"error": "too few counties", "n": len(ma_names)}

    bt = pd.DataFrame(rows)
    mean = (
        bt.groupby(["subset", "family", "model", "adjacency"], dropna=False)["mae"]
        .mean()
        .reset_index()
        .rename(columns={"mae": "mean_mae"})
        .sort_values(["subset", "mean_mae"])
    )

    # FullUS Prev mean for comparison to locked 0.0345
    full_prev = mean[(mean.subset == "FullUS") & (mean.model == "PreviousShare")]
    full_prev_mae = float(full_prev["mean_mae"].iloc[0]) if len(full_prev) else None

    protocol = {
        "label": "us_census_adjacency_pilot",
        "adjacency_source": adj_meta,
        "dependency": {
            "torch_geometric": False,
            "gcn_implementation": "DenseGCN / ResidualDenseGCN (PyTorch mm + A_hat; no PyG)",
            "sgc": "scipy.sparse / numpy graph convolution + Ridge/RF",
            "torch": torch.__version__,
        },
        "test_years": TEST_YEARS,
        "seed": SEED,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "fips_match": {
            k: match[k]
            for k in (
                "panel_common_oot_fips",
                "panel_intersect_census",
                "match_rate",
                "n_undirected_edges_in_subgraph",
                "mean_degree",
                "n_islands_degree0",
                "n_panel_missing_from_census",
            )
        },
        "mid_atlantic": ma_meta,
        "locked_us_core_prev_mae": LOCKED_US_PREV_MAE,
        "this_run_fullus_prev_mean_mae": full_prev_mae,
        "note": (
            "Census 2024 adjacency unblocks official-graph RQ2 spatial. "
            "Full-US trainable GCN still heavy; Mid-Atlantic GCN-lite is the "
            "trainable pilot. FullUS SGC comparable to locked Prev 0.0345 on "
            "same OOT years; MidAtlantic numbers are subset-only."
        ),
        "wall_seconds": round(time.time() - t0, 2),
    }

    bt.to_csv(OUT / "us_census_adj_backtest.csv", index=False)
    mean.to_csv(OUT / "us_census_adj_mean_mae.csv", index=False)
    (OUT / "us_census_adj_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    print("\nMEAN MAE:", flush=True)
    print(mean.to_string(index=False), flush=True)
    print(
        f"\nFullUS Prev mean MAE={full_prev_mae} vs locked Core {LOCKED_US_PREV_MAE}",
        flush=True,
    )
    print(f"DONE in {protocol['wall_seconds']}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
