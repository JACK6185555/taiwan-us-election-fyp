#!/usr/bin/env python3
"""Beat-Prev attempt: residual-on-Prev + national poll swing (public data only).

Extension — NOT Core specification. Same OOT folds as Core (2012/2016/2020/2024).
Locked Prev (TW 0.0793 / US 0.0345) are comparison anchors, not rewritten.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "beat_prev_residual"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
TEST_YEARS = [2012, 2016, 2020, 2024]
LOCKED = {"TW": 0.0793, "US": 0.0345}

TW_ED = {
    2000: "2000-03-18",
    2004: "2004-03-20",
    2008: "2008-03-22",
    2012: "2012-01-14",
    2016: "2016-01-16",
    2020: "2020-01-11",
    2024: "2024-01-13",
}
TW_DPP = {
    2000: "陳水扁",
    2004: "陳水扁",
    2008: "Hsieh|謝長廷",
    2012: "Tsai|蔡英文",
    2016: "蔡英文",
    2020: "蔡英文",
    2024: "賴清德|Lai",
}


def scores(y, p):
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
    }


def fit_predict(name: str, Xtr, ytr, Xte):
    if name == "Ridge":
        m = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    elif name == "RandomForest":
        m = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, random_state=SEED, n_jobs=-1
        )
    else:
        m = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=15,
            l2_regularization=1.0,
            random_state=SEED,
        )
    m.fit(Xtr, ytr)
    return np.asarray(m.predict(Xte), dtype=float)


def usable_features(df: pd.DataFrame, feats: list[str], min_frac: float = 0.8) -> list[str]:
    """Keep features with enough non-null coverage on years that appear in OOT train+test."""
    years = df["year"].isin([2008, 2012, 2016, 2020, 2024])
    ok = []
    for f in feats:
        if f not in df.columns:
            continue
        if df.loc[years, f].notna().mean() >= min_frac:
            ok.append(f)
    return ok


def oot_residual(df, *, target, prev, feature_cols, experiment, country, mode="residual"):
    rows = []
    use = usable_features(df, feature_cols)
    if not use:
        return rows
    for ty in TEST_YEARS:
        need = [target, prev] + use
        train = df[df["year"] < ty].dropna(subset=need)
        test = df[df["year"] == ty].dropna(subset=need)
        if len(train) < 10 or test.empty:
            continue
        Xtr = train[use].to_numpy(dtype=float)
        Xte = test[use].to_numpy(dtype=float)
        anchor = test[prev].to_numpy(dtype=float)
        yte = test[target].to_numpy(dtype=float)
        if mode == "level":
            ytr = train[target].to_numpy(dtype=float)
        else:
            ytr = (train[target] - train[prev]).to_numpy(dtype=float)
        for model_name in ["Ridge", "RandomForest", "HistGradientBoosting"]:
            raw = fit_predict(model_name, Xtr, ytr, Xte)
            pred = np.clip(raw if mode == "level" else anchor + raw, 0, 1)
            sc = scores(yte, pred)
            rows.append(
                {
                    "country": country,
                    "experiment": experiment,
                    "mode": mode,
                    "test_year": ty,
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_features": len(use),
                    "model": model_name,
                    "mae": sc["mae"],
                    "rmse": sc["rmse"],
                    "features": "|".join(use),
                }
            )
    return rows


def prev_baseline(df, target, prev, country):
    rows = []
    for ty in TEST_YEARS:
        test = df[df["year"] == ty].dropna(subset=[target, prev])
        if test.empty:
            continue
        sc = scores(test[target], test[prev])
        rows.append(
            {
                "country": country,
                "experiment": "Prev_baseline",
                "mode": "prev",
                "test_year": ty,
                "n_train": int((df["year"] < ty).sum()),
                "n_test": len(test),
                "n_features": 0,
                "model": "Previous-election baseline",
                "mae": sc["mae"],
                "rmse": sc["rmse"],
                "features": "",
            }
        )
    return rows


def national_share(df, country_kind: str) -> dict[int, float]:
    out = {}
    for y, g in df.groupby("year"):
        if country_kind == "TW":
            out[int(y)] = float(g["DPP_votes"].sum() / g["valid_votes"].sum())
        else:
            out[int(y)] = float(g["dem_votes"].sum() / g["two_party_votes"].sum())
    return out


def uniform_poll_swing(df, target, prev, poll_by_year, country, experiment="UniformPollSwing"):
    nat = national_share(df, country)
    years = sorted(nat)
    prev_nat = {y: (nat[years[i - 1]] if i else np.nan) for i, y in enumerate(years)}
    rows = []
    for ty in TEST_YEARS:
        test = df[df["year"] == ty].dropna(subset=[target, prev])
        poll = poll_by_year.get(ty, np.nan)
        pn = prev_nat.get(ty, np.nan)
        if test.empty or pd.isna(poll) or pd.isna(pn):
            continue
        swing = float(poll) - float(pn)
        pred = np.clip(test[prev].to_numpy() + swing, 0, 1)
        sc = scores(test[target].to_numpy(), pred)
        rows.append(
            {
                "country": country,
                "experiment": experiment,
                "mode": "prev_plus_poll_swing",
                "test_year": ty,
                "n_train": 0,
                "n_test": len(test),
                "n_features": 1,
                "model": "prev+national_poll_swing",
                "mae": sc["mae"],
                "rmse": sc["rmse"],
                "features": f"poll={poll:.4f}|prev_nat={pn:.4f}|swing={swing:.4f}",
            }
        )
    return rows


def tw_poll_shares_from_csv() -> dict[int, float]:
    """Match prior ablation: DPP share among non-undecided targets, 60d pre-election."""
    path = ROOT / "data/extension_poll_platform/taiwan_polls.csv"
    polls = pd.read_csv(path)
    polls["y"] = polls["poll_id"].astype(str).str[:4].astype(int)
    feat = {}
    for y, ed in TW_ED.items():
        start = str((pd.Timestamp(ed) - pd.Timedelta(days=60)).date())
        q = polls[(polls["y"] == y) & (polls["field_end"] >= start) & (polls["field_end"] < ed)]
        shares = []
        pat = re.compile(TW_DPP[y], re.I)
        for _, g in q.groupby("poll_id"):
            g2 = g[~g["target"].astype(str).str.contains("undecided|未決定|棄保", case=False, na=False)]
            d = g2[g2["target"].astype(str).apply(lambda t: bool(pat.search(str(t))))]
            if len(d) != 1 or len(g2) < 2:
                continue
            total = g2["value"].astype(float).sum()
            if total <= 0:
                continue
            shares.append(float(d["value"].iloc[0]) / total)
        feat[y] = float(np.mean(shares)) if shares else float("nan")
    return feat


def load_tw():
    panel = pd.read_csv(ROOT / "cleaned/county_panel_features_enriched_2000_2024.csv")
    edu = pd.read_csv(ROOT / "data/extension_education/county_panel_with_lag_education_rate.csv")
    if "lag_education_rate" in edu.columns:
        panel = panel.merge(edu[["region_id", "year", "lag_education_rate"]], on=["region_id", "year"], how="left")

    fund = pd.read_csv(
        ROOT / "data/extension_fundamentals_proxy/national_fundamentals_enriched_extension.csv"
    )
    fund = fund[fund["country"] == "TW"].copy()
    fund["incumbent_is_dpp"] = (fund["incumbent_party"] == "DPP").astype(float)
    fcols = [
        "year",
        "unrate",
        "cpi_yoy_pct",
        "incumbent_running",
        "incumbent_is_dpp",
        "turnout_rate",
        "undecided_share",
        "polarization_proxy",
    ]
    panel = panel.merge(fund[fcols], on="year", how="left")

    # ESC / identity national if present
    esc_path = ROOT / "data/esc_proxies/esc_national_election_year_proxy.csv"
    if esc_path.exists():
        esc = pd.read_csv(esc_path)
        if "year" in esc.columns:
            num = ["year"]
            for c in esc.columns:
                if c == "year":
                    continue
                if pd.api.types.is_numeric_dtype(esc[c]) or pd.to_numeric(esc[c], errors="coerce").notna().mean() > 0.5:
                    esc[c] = pd.to_numeric(esc[c], errors="coerce")
                    num.append(c)
            panel = panel.merge(esc[num], on="year", how="left", suffixes=("", "_esc"))

    poll_ablation = tw_poll_shares_from_csv()
    polls_ext = pd.read_csv(ROOT / "data/extension_must_have/tw_national_poll_averages_extension.csv")
    poll_pct = {int(r.election_year): float(r.dpp_pct_mean) / 100.0 for _, r in polls_ext.iterrows()}
    poll_2p = {int(r.election_year): float(r.dpp_two_party_share_poll) for _, r in polls_ext.iterrows()}

    nat = national_share(panel, "TW")
    years = sorted(nat)
    prev_nat = {y: (nat[years[i - 1]] if i else np.nan) for i, y in enumerate(years)}
    panel["prev_nat_dpp"] = panel["year"].map(prev_nat)
    panel["poll_ablation"] = panel["year"].map(poll_ablation)
    panel["poll_pct"] = panel["year"].map(poll_pct)
    panel["poll_2p"] = panel["year"].map(poll_2p)
    panel["poll_swing_ablation"] = panel["poll_ablation"] - panel["prev_nat_dpp"]
    panel["poll_swing_pct"] = panel["poll_pct"] - panel["prev_nat_dpp"]
    panel["poll_swing_2p"] = panel["poll_2p"] - panel["prev_nat_dpp"]
    return panel, poll_ablation, poll_pct, poll_2p


def load_us():
    panel = pd.read_csv(ROOT / "fyp_us_core/us_county_panel_features_family_a.csv")
    if "_dropped_incomplete" in panel.columns:
        panel = panel[panel["_dropped_incomplete"] != 1].copy()
    panel["region_id"] = panel["region_id"].astype(str)
    raw = pd.read_csv(ROOT / "fyp_us_taiwan_transfer/cleaned/us_county_presidential_panel.csv")
    raw["region_id"] = raw["region_id"].astype(str)
    panel = panel.merge(
        raw[["region_id", "year", "dem_votes", "rep_votes", "two_party_votes", "total_votes"]],
        on=["region_id", "year"],
        how="left",
    )
    polls = pd.read_csv(ROOT / "data/extension_must_have/us_national_poll_averages_extension.csv")
    poll_2p = {int(r.election_year): float(r.dem_two_party_share_poll) for _, r in polls.iterrows()}
    fund = pd.read_csv(
        ROOT / "data/extension_fundamentals_proxy/national_fundamentals_enriched_extension.csv"
    )
    fund = fund[fund["country"] == "US"].copy()
    fund["incumbent_is_dem"] = (fund["incumbent_party"] == "DEM").astype(float)
    fcols = [
        "year",
        "unrate",
        "umcsent",
        "cpi_yoy_pct",
        "incumbent_running",
        "incumbent_is_dem",
        "turnout_rate",
        "undecided_share",
        "polarization_proxy",
    ]
    fund = fund.rename(columns={"unrate": "nat_unrate"})
    fcols = [c if c != "unrate" else "nat_unrate" for c in fcols]
    panel = panel.merge(fund[fcols], on="year", how="left")

    nat = national_share(panel, "US")
    years = sorted(nat)
    prev_nat = {y: (nat[years[i - 1]] if i else np.nan) for i, y in enumerate(years)}
    panel["prev_nat_dem"] = panel["year"].map(prev_nat)
    panel["poll_dem_2p"] = panel["year"].map(poll_2p)
    panel["poll_swing_2p"] = panel["poll_dem_2p"] - panel["prev_nat_dem"]
    return panel, poll_2p


def us_spatial_lag_prev(panel: pd.DataFrame) -> pd.DataFrame:
    npz_path = ROOT / "data/us_county_adjacency/us_county_adjacency_panel.npz"
    order_path = ROOT / "data/us_county_adjacency/panel_fips_ordered.csv"
    if not npz_path.exists():
        return panel
    A = sparse.load_npz(npz_path).tocsr().astype(float)
    fips = pd.read_csv(order_path)["fips"].astype(str).str.zfill(5).tolist()
    if A.shape[0] != len(fips):
        return panel
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    An = sparse.diags(1.0 / deg) @ A
    idx = {f: i for i, f in enumerate(fips)}
    work = panel.copy()
    work["rid5"] = work["region_id"].astype(str).str.zfill(5)
    rows = []
    for y, g in work.groupby("year"):
        vec = np.full(len(fips), np.nan)
        for _, r in g.iterrows():
            i = idx.get(r["rid5"])
            if i is not None and pd.notna(r["prev_dem_two_party_share"]):
                vec[i] = float(r["prev_dem_two_party_share"])
        fill = np.nanmean(vec)
        neigh = An @ np.nan_to_num(vec, nan=fill)
        for f, i in idx.items():
            rows.append({"year": int(y), "rid5": f, "spatial_lag_prev": float(neigh[i])})
    spat = pd.DataFrame(rows)
    return work.merge(spat, on=["year", "rid5"], how="left")


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    g = (
        metrics.groupby(["country", "experiment", "mode", "model"], as_index=False)[["mae", "rmse"]]
        .mean()
        .sort_values(["country", "mae"])
    )
    g["mae_4dp"] = g["mae"].round(4)
    g["rmse_4dp"] = g["rmse"].round(4)
    g["locked_prev"] = g["country"].map(LOCKED)
    g["delta_vs_locked_prev"] = g["mae"] - g["locked_prev"]
    g["beats_locked_prev"] = g["mae"] < g["locked_prev"] - 1e-4
    return g


def main():
    all_rows = []

    # ---- TW ----
    tw, poll_abl, poll_pct, poll_2p = load_tw()
    T, P = "target_DPP_vote_share", "prev_DPP_vote_share"
    all_rows += prev_baseline(tw, T, P, "TW")
    all_rows += uniform_poll_swing(tw, T, P, poll_abl, "TW", "UniformPollSwing_ablation_def")
    all_rows += uniform_poll_swing(tw, T, P, poll_pct, "TW", "UniformPollSwing_pct")
    all_rows += uniform_poll_swing(tw, T, P, poll_2p, "TW", "UniformPollSwing_2p")

    socio = usable_features(
        tw, ["lag_unemployment_pct", "lag_log_mean_income", "lag_log_pop_density", "lag_education_rate"]
    )
    shocks = usable_features(
        tw,
        [
            "poll_swing_ablation",
            "unrate",
            "cpi_yoy_pct",
            "incumbent_running",
            "incumbent_is_dpp",
            "turnout_rate",
            "undecided_share",
            "polarization_proxy",
        ],
    )
    hist = usable_features(
        tw,
        [
            "DPP_change_1",
            "KMT_change_1",
            "turnout_change_1",
            "DPP_KMT_margin_change_1",
            "previous_party_change_abs_mean",
        ],
    )
    identity = usable_features(
        tw,
        [
            c
            for c in tw.columns
            if c.startswith("lag_taiwanese")
            or c.startswith("lag_chinese")
            or c.startswith("lag_indep")
            or c.startswith("lag_unif")
            or c.startswith("lag_both")
            or c.startswith("lag_status")
        ],
    )
    tw_sets = {
        "Residual_on_Prev__poll_swing_ablation": ["poll_swing_ablation"],
        "Residual_on_Prev__poll_swing_pct": ["poll_swing_pct"],
        "Residual_on_Prev__national_shocks": shocks,
        "Residual_on_Prev__socio": socio,
        "Residual_on_Prev__shocks_socio": shocks + socio,
        "Residual_on_Prev__shocks_socio_histDelta": shocks + socio + hist,
        "Residual_on_Prev__shocks_socio_identity": shocks + socio + identity,
        "Level__prev_plus_shocks_socio": [P, "prev_KMT_vote_share", "prev_turnout"] + shocks + socio + hist,
    }
    for name, feats in tw_sets.items():
        mode = "level" if name.startswith("Level") else "residual"
        all_rows += oot_residual(
            tw, target=T, prev=P, feature_cols=feats, experiment=name, country="TW", mode=mode
        )

    # ---- US ----
    us, us_poll = load_us()
    UT, UP = "target_dem_two_party_share", "prev_dem_two_party_share"
    all_rows += prev_baseline(us, UT, UP, "US")
    all_rows += uniform_poll_swing(us, UT, UP, us_poll, "US", "UniformPollSwing")

    us_socio = usable_features(us, ["unemployment", "education_rate", "log_median_income", "log_density"])
    us_shocks = usable_features(
        us,
        [
            "poll_swing_2p",
            "nat_unrate",
            "umcsent",
            "cpi_yoy_pct",
            "incumbent_running",
            "incumbent_is_dem",
            "turnout_rate",
            "undecided_share",
            "polarization_proxy",
        ],
    )
    us_hist = usable_features(
        us,
        [
            "dem_change_1",
            "gop_change_1",
            "dem_gop_margin_change_1",
            "abs_dem_change_1",
            "log_votes_change_1",
            "prev2_dem_two_party_share",
        ],
    )
    us = us_spatial_lag_prev(us)
    us_sets = {
        "Residual_on_Prev__poll_swing_2p": ["poll_swing_2p"],
        "Residual_on_Prev__national_shocks": us_shocks,
        "Residual_on_Prev__socio": us_socio,
        "Residual_on_Prev__shocks_socio": us_shocks + us_socio,
        "Residual_on_Prev__shocks_socio_histDelta": us_shocks + us_socio + us_hist,
        "Residual_on_Prev__shocks_socio_spatialLagPrev": us_shocks
        + us_socio
        + (["spatial_lag_prev"] if "spatial_lag_prev" in us.columns else []),
        "Level__core_plus_poll_shocks": [UP, "prev_gop_two_party_share", "prev_log_total_votes"]
        + us_shocks
        + us_socio
        + us_hist,
    }
    for name, feats in us_sets.items():
        mode = "level" if name.startswith("Level") else "residual"
        all_rows += oot_residual(
            us, target=UT, prev=UP, feature_cols=feats, experiment=name, country="US", mode=mode
        )

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(OUT / "metrics_by_year.csv", index=False)
    mean = summarize(metrics)
    mean.to_csv(OUT / "mean_mae_summary.csv", index=False)
    mean[mean["beats_locked_prev"]].to_csv(OUT / "models_that_beat_locked_prev.csv", index=False)
    best = mean.sort_values("mae").groupby("country", as_index=False).first()
    best.to_csv(OUT / "best_per_country.csv", index=False)

    # year detail for winners
    winners = []
    for c in ["TW", "US"]:
        b = best[best.country == c].iloc[0]
        sub = metrics[
            (metrics.country == c)
            & (metrics.experiment == b.experiment)
            & (metrics.model == b.model)
        ][["country", "experiment", "model", "test_year", "mae", "rmse", "features"]]
        winners.append(sub)
    pd.concat(winners).to_csv(OUT / "best_by_year.csv", index=False)

    summary = {
        "locked_prev": LOCKED,
        "test_years": TEST_YEARS,
        "note": (
            "Extension attempt to beat locked Core Prev. UniformPollSwing = county_prev + "
            "(national_poll - national_prev). Residual-on-Prev trains only (y - prev). "
            "National poll is NOT county-level skill; honesty required in report."
        ),
        "best": best.to_dict(orient="records"),
        "n_mean_rows_beating_locked": int(mean["beats_locked_prev"].sum()),
        "tw_poll_ablation_shares": {str(k): (None if pd.isna(v) else round(v, 4)) for k, v in poll_abl.items()},
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(mean[["country", "experiment", "model", "mae_4dp", "delta_vs_locked_prev", "beats_locked_prev"]].to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
