#!/usr/bin/env python3
"""CORAL + MMD + adversarial DA + low-data transfer curves (I-K).

Protocols (OOT test years 2012–2024):
  - no_da: Ridge/HGB on source only
  - coral: CORAL-align source→target feature cov, then Ridge
  - mmd: reweight source samples toward target (kernel MMD-ish RBF median heuristic via
         simple Gaussian kernel mean matching / KMM-lite closed form ridge on features)
  - adv_dann: small adversarial MLP (torch) if available
  - low_data: train on all source + k labeled target regions (prior years), k in {0,1,2,5,10,22}

Writes results/real/da_*.csv and charts/da_lowdata_curves.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "cleaned"
RES = ROOT / "results" / "real"
CHARTS = ROOT / "results" / "charts"
RES.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

FEATURES = ["prev_share_A", "log_total_votes", "z_income", "z_log_density"]
TEST_YEARS = [2012, 2016, 2020, 2024]
SEED = 42
RNG = np.random.default_rng(SEED)

try:
    import torch
    import torch.nn as nn

    TORCH_OK = True
except Exception:
    TORCH_OK = False


def coral_transform(Xs: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    """Align source covariance to target (CORAL)."""
    xs = Xs - Xs.mean(axis=0, keepdims=True)
    xt = Xt - Xt.mean(axis=0, keepdims=True)
    d = Xs.shape[1]
    cs = (xs.T @ xs) / max(len(xs) - 1, 1) + np.eye(d) * 1e-6
    ct = (xt.T @ xt) / max(len(xt) - 1, 1) + np.eye(d) * 1e-6
    # whitening source then recolor with target
    evals_s, evecs_s = np.linalg.eigh(cs)
    evals_t, evecs_t = np.linalg.eigh(ct)
    evals_s = np.clip(evals_s, 1e-6, None)
    evals_t = np.clip(evals_t, 1e-6, None)
    whitening = evecs_s @ np.diag(1.0 / np.sqrt(evals_s)) @ evecs_s.T
    coloring = evecs_t @ np.diag(np.sqrt(evals_t)) @ evecs_t.T
    A = whitening @ coloring
    return (Xs - Xs.mean(axis=0)) @ A + Xt.mean(axis=0)


def mmd_rbf(Xs: np.ndarray, Xt: np.ndarray) -> float:
    """Unbiased squared MMD with RBF kernel (median heuristic)."""
    Z = np.vstack([Xs, Xt])
    # pairwise sq distances subsample for bandwidth
    n = min(len(Z), 400)
    idx = RNG.choice(len(Z), size=n, replace=False)
    D = np.sum((Z[idx, None, :] - Z[None, idx, :]) ** 2, axis=2)
    med = np.median(D[D > 0]) if np.any(D > 0) else 1.0
    gamma = 1.0 / (2.0 * med) if med > 0 else 1.0

    def k(a, b):
        d = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        return np.exp(-gamma * d)

    Kss = k(Xs, Xs)
    Ktt = k(Xt, Xt)
    Kst = k(Xs, Xt)
    m, n_ = len(Xs), len(Xt)
    # unbiased
    np.fill_diagonal(Kss, 0.0)
    np.fill_diagonal(Ktt, 0.0)
    mmd2 = Kss.sum() / (m * (m - 1)) + Ktt.sum() / (n_ * (n_ - 1)) - 2 * Kst.mean()
    return float(max(mmd2, 0.0))


def kernel_mean_matching_weights(Xs: np.ndarray, Xt: np.ndarray, B: float = 10.0) -> np.ndarray:
    """Simple KMM-style weights (quadratic program surrogate via ridge on kernel)."""
    # Use closed-form approx: w ∝ solve (K+λI) w = κ  clipped
    n = len(Xs)
    Z = np.vstack([Xs, Xt])
    nsub = min(len(Z), 300)
    idx = RNG.choice(len(Z), size=nsub, replace=False)
    D = np.sum((Z[idx, None, :] - Z[None, idx, :]) ** 2, axis=2)
    med = np.median(D[D > 0]) if np.any(D > 0) else 1.0
    gamma = 1.0 / (2.0 * med) if med > 0 else 1.0
    K = np.exp(-gamma * np.sum((Xs[:, None, :] - Xs[None, :, :]) ** 2, axis=2))
    kappa = np.exp(-gamma * np.sum((Xs[:, None, :] - Xt[None, :, :]) ** 2, axis=2)).mean(axis=1) * (
        n / max(len(Xt), 1)
    )
    lam = 0.1
    w = np.linalg.solve(K + lam * np.eye(n), kappa)
    w = np.clip(w, 0.0, B)
    if w.sum() <= 0:
        return np.ones(n)
    w = w * (n / w.sum())
    return w


class _DANN(nn.Module if TORCH_OK else object):
    def __init__(self, d_in: int):
        if not TORCH_OK:
            return
        super().__init__()
        self.feat = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.reg = nn.Linear(16, 1)
        self.dom = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, alpha: float = 1.0):
        h = self.feat(x)
        y = self.reg(h).squeeze(-1)
        h_grl = _GradientReversal.apply(h, alpha)
        d = self.dom(h_grl).squeeze(-1)
        return y, d


if TORCH_OK:

    class _GradientReversal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha):
            ctx.alpha = alpha
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.alpha * grad_output, None


def fit_dann(Xs, ys, Xt, steps=400, lr=1e-2):
    if not TORCH_OK:
        return None
    torch.manual_seed(SEED)
    device = torch.device("cpu")
    model = _DANN(Xs.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xs_t = torch.tensor(Xs, dtype=torch.float32)
    ys_t = torch.tensor(ys, dtype=torch.float32)
    Xt_t = torch.tensor(Xt, dtype=torch.float32)
    # domain labels: 0 source, 1 target
    for step in range(steps):
        alpha = 2 / (1 + np.exp(-10 * step / steps)) - 1
        model.train()
        opt.zero_grad()
        # sample minibatches
        bs = min(128, len(Xs), len(Xt))
        si = RNG.choice(len(Xs), size=bs, replace=False)
        ti = RNG.choice(len(Xt), size=bs, replace=False)
        y_hat, d_s = model(Xs_t[si], alpha)
        _, d_t = model(Xt_t[ti], alpha)
        loss_y = torch.mean((y_hat - ys_t[si]) ** 2)
        # domain BCE
        loss_d = (
            torch.nn.functional.binary_cross_entropy_with_logits(d_s, torch.zeros_like(d_s))
            + torch.nn.functional.binary_cross_entropy_with_logits(d_t, torch.ones_like(d_t))
        ) / 2
        loss = loss_y + loss_d
        loss.backward()
        opt.step()
    model.eval()

    def predict(X):
        with torch.no_grad():
            y, _ = model(torch.tensor(X, dtype=torch.float32), alpha=0.0)
            return y.numpy()

    return predict


def prep(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=FEATURES + ["party_share_A"]).copy()


def eval_transfer(src, tgt, method: str, k_target: int = 0) -> dict | None:
    """Train on src (+ optional k labeled tgt rows from same calendar past), test on tgt year fold already sliced."""
    src = prep(src)
    tgt = prep(tgt)
    if len(src) < 20 or len(tgt) < 1:
        return None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(src[FEATURES].values)
    ys = src["party_share_A"].values
    Xt = scaler.transform(tgt[FEATURES].values)
    yt = tgt["party_share_A"].values

    # optional few-shot target labels from *training* years of target country —
    # caller passes them via src already concatenated; k_target only for tagging

    if method == "no_da":
        model = Ridge(alpha=1.0, random_state=SEED)
        model.fit(Xs, ys)
        pred = model.predict(Xt)
    elif method == "coral":
        Xs_a = coral_transform(Xs, Xt)
        model = Ridge(alpha=1.0, random_state=SEED)
        model.fit(Xs_a, ys)
        pred = model.predict(Xt)
    elif method == "mmd":
        w = kernel_mean_matching_weights(Xs, Xt)
        model = Ridge(alpha=1.0, random_state=SEED)
        model.fit(Xs, ys, sample_weight=w)
        pred = model.predict(Xt)
        mmd_val = mmd_rbf(Xs, Xt)
    elif method == "adv_dann":
        if not TORCH_OK:
            return None
        predict = fit_dann(Xs, ys, Xt)
        if predict is None:
            return None
        pred = predict(Xt)
    else:
        raise ValueError(method)

    mae = float(mean_absolute_error(yt, pred))
    out = {
        "method": method,
        "mae": mae,
        "n_train": len(src),
        "n_test": len(tgt),
        "k_target_labeled": k_target,
        "torch": TORCH_OK,
    }
    if method == "mmd":
        out["mmd2"] = mmd_val if method == "mmd" else np.nan
    # also PreviousShare baseline on same test
    out["mae_prevshare"] = float(mean_absolute_error(yt, tgt["prev_share_A"].values))
    return out


def main() -> None:
    panel = pd.read_csv(CLEAN / "joint_real_shared_panel.csv")
    da_rows = []
    mmd_diag = []

    for src_c, tgt_c in [("US", "TW"), ("TW", "US")]:
        for ty in TEST_YEARS:
            src = panel[(panel["country"] == src_c) & (panel["year"] < ty)]
            tgt = panel[(panel["country"] == tgt_c) & (panel["year"] == ty)]
            # MMD diagnostic on scaled features
            s1, t1 = prep(src), prep(tgt)
            if len(s1) >= 20 and len(t1) >= 1:
                sc = StandardScaler()
                Xs = sc.fit_transform(s1[FEATURES].values)
                Xt = sc.transform(t1[FEATURES].values)
                mmd_diag.append(
                    {
                        "direction": f"{src_c}->{tgt_c}",
                        "test_year": ty,
                        "mmd2": mmd_rbf(Xs, Xt),
                        "n_src": len(s1),
                        "n_tgt": len(t1),
                    }
                )
            for method in ["no_da", "coral", "mmd"] + (["adv_dann"] if TORCH_OK else []):
                r = eval_transfer(src, tgt, method, k_target=0)
                if r:
                    r.update(
                        {
                            "direction": f"{src_c}->{tgt_c}",
                            "test_year": ty,
                            "feature_set": ",".join(FEATURES),
                            "data_kind": "real",
                        }
                    )
                    da_rows.append(r)

    da = pd.DataFrame(da_rows)
    da.to_csv(RES / "da_transfer_mae_detail.csv", index=False)
    da_mean = (
        da.groupby(["direction", "method"], as_index=False)["mae"]
        .mean()
        .rename(columns={"mae": "mean_mae_oot"})
        .sort_values(["direction", "mean_mae_oot"])
    )
    da_mean.to_csv(RES / "da_transfer_mae_mean.csv", index=False)
    pd.DataFrame(mmd_diag).to_csv(RES / "da_mmd_diagnostic.csv", index=False)

    # Low-data curves: source all past + k labeled target regions from past years
    low_rows = []
    ks = [0, 1, 2, 5, 10, 22]
    for src_c, tgt_c in [("US", "TW"), ("TW", "US")]:
        for ty in TEST_YEARS:
            src = panel[(panel["country"] == src_c) & (panel["year"] < ty)]
            tgt_train_pool = panel[(panel["country"] == tgt_c) & (panel["year"] < ty)]
            tgt_test = panel[(panel["country"] == tgt_c) & (panel["year"] == ty)]
            tgt_train_pool = prep(tgt_train_pool)
            # unique regions in latest available target year before ty
            if len(tgt_train_pool) == 0:
                continue
            last_y = tgt_train_pool["year"].max()
            regions = tgt_train_pool[tgt_train_pool["year"] == last_y]["region_id"].unique()
            RNG.shuffle(regions)
            for k in ks:
                if k > len(regions):
                    continue
                chosen = regions[:k]
                extra = tgt_train_pool[tgt_train_pool["region_id"].isin(chosen)]
                train = pd.concat([src, extra], ignore_index=True)
                for method in ["no_da", "coral"]:
                    r = eval_transfer(train, tgt_test, method, k_target=k)
                    if r:
                        r.update(
                            {
                                "direction": f"{src_c}->{tgt_c}",
                                "test_year": ty,
                                "feature_set": ",".join(FEATURES),
                                "data_kind": "real",
                                "protocol": "low_data",
                            }
                        )
                        low_rows.append(r)

    low = pd.DataFrame(low_rows)
    low.to_csv(RES / "da_lowdata_mae_detail.csv", index=False)
    low_mean = (
        low.groupby(["direction", "method", "k_target_labeled"], as_index=False)["mae"]
        .mean()
        .rename(columns={"mae": "mean_mae_oot"})
    )
    low_mean.to_csv(RES / "da_lowdata_mae_mean.csv", index=False)

    # Charts
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, direction in zip(axes, ["US->TW", "TW->US"]):
        sub = da_mean[da_mean["direction"] == direction]
        if len(sub) == 0:
            continue
        ax.bar(sub["method"], sub["mean_mae_oot"])
        ax.set_title(f"DA transfer mean OOT MAE ({direction})")
        ax.set_ylabel("MAE")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(CHARTS / "da_transfer_mae.png", dpi=140)
    fig.savefig(CHARTS / "da_transfer_mae.pdf")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, direction in zip(axes, ["US->TW", "TW->US"]):
        sub = low_mean[low_mean["direction"] == direction]
        for method, g in sub.groupby("method"):
            g = g.sort_values("k_target_labeled")
            ax.plot(g["k_target_labeled"], g["mean_mae_oot"], marker="o", label=method)
        ax.set_title(f"Low-data transfer ({direction})")
        ax.set_xlabel("k labeled target regions (past years)")
        ax.set_ylabel("Mean OOT MAE")
        ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "da_lowdata_curves.png", dpi=140)
    fig.savefig(CHARTS / "da_lowdata_curves.pdf")
    plt.close()

    print("TORCH_OK", TORCH_OK)
    print(da_mean.to_string(index=False))
    print(low_mean.to_string(index=False))
    print("Wrote DA + low-data results")


if __name__ == "__main__":
    main()
