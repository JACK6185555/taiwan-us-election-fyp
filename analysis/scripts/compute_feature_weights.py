#!/usr/bin/env python3
"""Township-level Taiwan presidential vote-share feature weights.

Builds a panel from CEC elctks + income + MOI density/edu, then fits
LASSO / Random Forest / OLS and reports feature importance.

THIS IS AN AGGREGATE ECOLOGICAL MODEL — not TEDS individual coefficients.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path("/workspace/taiwan-election-data")
CLEAN = ROOT / "cleaned"
RAW_CEC = ROOT / "raw" / "cec" / "extracted"
RAW_INC = ROOT / "raw" / "income"
OUT = ROOT / "analysis"
CHARTS = OUT / "charts"
OUT.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

SEED = 42
DPP_PARTY = {16}  # 民主進步黨
KMT_PARTY = {1}   # 中國國民黨

# Region mapping by county prefix
NORTH = {"臺北市", "新北市", "基隆市", "桃園市", "新竹市", "新竹縣", "宜蘭縣"}
CENTRAL = {"臺中市", "苗栗縣", "彰化縣", "南投縣", "雲林縣"}
SOUTH = {"高雄市", "臺南市", "嘉義市", "嘉義縣", "屏東縣", "澎湖縣"}
EAST = {"花蓮縣", "臺東縣"}
ISLANDS = {"金門縣", "連江縣"}

ELECTION_DIRS = {
    2012: RAW_CEC / "2012總統立委" / "總統",
    2016: RAW_CEC / "2016總統立委" / "總統",
    2020: RAW_CEC / "2020總統立委" / "總統",
    2024: RAW_CEC / "2024總統立委" / "總統",
}

# Covariate CE year mapping (use available year closest at/before election)
INCOME_ROC_FOR_ELECTION = {2012: 105, 2016: 105, 2020: 109, 2024: 112}  # 112=2023
POP_ROC_FOR_ELECTION = {2012: 108, 2016: 108, 2020: 109, 2024: 112}


def _clean_cell(x) -> str:
    s = str(x).strip().strip('"').strip("'").strip()
    # 2012 style leading apostrophe digits
    if s.startswith("'"):
        s = s[1:]
    return s


def read_cec_csv(path: Path) -> pd.DataFrame:
    """Read CEC CSV with no header; handle 2012 quoted/apostrophe style."""
    # Try utf-8 then cp950
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp950", "big5"):
        try:
            df = pd.read_csv(path, header=None, dtype=str, encoding=enc)
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise last_err
    return df.map(_clean_cell) if hasattr(df, "map") else df.applymap(_clean_cell)


def load_elbase(path: Path) -> pd.DataFrame:
    df = read_cec_csv(path)
    df = df.iloc[:, :6]
    df.columns = ["prv", "city", "dept", "twn", "vil", "name"]
    return df


def load_elcand(path: Path) -> pd.DataFrame:
    df = read_cec_csv(path)
    # typical: prv,city,dept,twn,vil, ticket, name, party, ...
    # Use columns 5 (ticket), 6 (name), 7 (party)
    out = pd.DataFrame(
        {
            "ticket": df.iloc[:, 5].astype(str).str.lstrip("0").replace("", "0"),
            "cand_name": df.iloc[:, 6],
            "party_code": pd.to_numeric(df.iloc[:, 7], errors="coerce").fillna(-1).astype(int),
        }
    )
    # Keep first row per ticket (president, not VP) — VP often has same ticket
    # Heuristic: first occurrence per ticket
    out = out.drop_duplicates("ticket", keep="first")
    return out


def load_elctks(path: Path) -> pd.DataFrame:
    df = read_cec_csv(path)
    df = df.iloc[:, :10]
    df.columns = [
        "prv",
        "city",
        "dept",
        "twn",
        "vil",
        "station",
        "ticket",
        "votes",
        "share",
        "elected",
    ]
    df["ticket"] = df["ticket"].astype(str).str.lstrip("0").replace("", "0")
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)
    return df


def county_name_from_elbase(elbase: pd.DataFrame, prv: str, city: str) -> str:
    # Direct municipality: city==000, twn==000, vil==0000
    row = elbase[(elbase.prv == prv) & (elbase.city == city) & (elbase.twn == "000") & (elbase.vil == "0000")]
    if len(row):
        name = row.iloc[0]["name"]
        if name not in ("臺灣省", "福建省", "全國"):
            return name
    # Province-level county: city != 000
    row = elbase[(elbase.prv == prv) & (elbase.city == city) & (elbase.twn == "000") & (elbase.vil == "0000")]
    if len(row):
        return row.iloc[0]["name"]
    return f"{prv}-{city}"


def township_votes(year: int) -> pd.DataFrame:
    d = ELECTION_DIRS[year]
    # 2016 uses el*_P1.csv naming
    if year == 2016:
        elbase = load_elbase(d / "elbase_P1.csv")
        elcand = load_elcand(d / "elcand_P1.csv")
        elctks = load_elctks(d / "elctks_P1.csv")
    else:
        elbase = load_elbase(d / "elbase.csv")
        elcand = load_elcand(d / "elcand.csv")
        elctks = load_elctks(d / "elctks.csv")

    # Township names: twn != 000, vil == 0000
    twn_names = elbase[(elbase.twn != "000") & (elbase.vil == "0000")][
        ["prv", "city", "twn", "name"]
    ].rename(columns={"name": "twn_name"})

    # County names
    counties = []
    for (prv, city), g in elbase.groupby(["prv", "city"]):
        nm = county_name_from_elbase(elbase, prv, city)
        if nm in ("臺灣省", "福建省", "全國"):
            continue
        counties.append({"prv": prv, "city": city, "county_name": nm})
    county_df = pd.DataFrame(counties).drop_duplicates(["prv", "city"])

    # Polling-station rows only (exclude rollups: station==0000 often county/town totals)
    # Also exclude vil==0000 rollups if present as totals
    poll = elctks[elctks["station"] != "0000"].copy()
    # Some years also have village-level totals with station 0000; we already exclude those

    agg = (
        poll.groupby(["prv", "city", "twn", "ticket"], as_index=False)["votes"]
        .sum()
    )
    agg = agg.merge(elcand, on="ticket", how="left")
    agg = agg.merge(twn_names, on=["prv", "city", "twn"], how="left")
    agg = agg.merge(county_df, on=["prv", "city"], how="left")

    # Drop unknown townships / province rollups
    agg = agg.dropna(subset=["twn_name", "county_name"])
    agg = agg[agg["county_name"].isin(NORTH | CENTRAL | SOUTH | EAST | ISLANDS)]

    # Build township key matching income: county + township name
    # For municipalities county_name already includes 市 and twn_name is 區
    agg["township"] = agg["county_name"] + agg["twn_name"]

    # Party shares
    total = agg.groupby("township", as_index=False)["votes"].sum().rename(columns={"votes": "valid_votes"})
    dpp = (
        agg[agg["party_code"].isin(DPP_PARTY)]
        .groupby("township", as_index=False)["votes"]
        .sum()
        .rename(columns={"votes": "dpp_votes"})
    )
    kmt = (
        agg[agg["party_code"].isin(KMT_PARTY)]
        .groupby("township", as_index=False)["votes"]
        .sum()
        .rename(columns={"votes": "kmt_votes"})
    )
    out = total.merge(dpp, on="township", how="left").merge(kmt, on="township", how="left")
    out["dpp_votes"] = out["dpp_votes"].fillna(0)
    out["kmt_votes"] = out["kmt_votes"].fillna(0)
    out["dpp_share"] = out["dpp_votes"] / out["valid_votes"]
    out["kmt_share"] = out["kmt_votes"] / out["valid_votes"]
    out["election_year"] = year
    # attach county
    meta = agg[["township", "county_name"]].drop_duplicates()
    out = out.merge(meta, on="township", how="left")
    return out[
        [
            "election_year",
            "township",
            "county_name",
            "valid_votes",
            "dpp_votes",
            "kmt_votes",
            "dpp_share",
            "kmt_share",
        ]
    ]


def rebuild_income_panel() -> pd.DataFrame:
    rows = []
    for y in range(105, 113):
        path = RAW_INC / f"{y}_township_village_income.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = [c.replace("\ufeff", "") for c in df.columns]
        # unify township col
        tcol = None
        for c in df.columns:
            if c in ("鄉鎮市區", "縣市別"):
                tcol = c
                break
        if tcol is None:
            raise ValueError(f"no township col in {path}: {df.columns}")
        df = df.rename(columns={tcol: "township"})
        for c in ["納稅單位(戶)", "綜合所得總額", "平均數"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # exclude 其他 / blank villages totals carefully: aggregate villages
        g = (
            df.dropna(subset=["township"])
            .groupby("township", as_index=False)
            .agg(
                n_villages=("村里", "count"),
                tax_units=("納稅單位(戶)", "sum"),
                income_total=("綜合所得總額", "sum"),
            )
        )
        g["mean_income"] = g["income_total"] / g["tax_units"].replace(0, np.nan)
        g["roc_year"] = y
        g["ce_year"] = y + 1911
        # drop 其他 buckets
        g = g[~g["township"].str.contains("其他", na=False)]
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def region_of(county: str) -> str:
    if county in NORTH:
        return "north"
    if county in CENTRAL:
        return "central"
    if county in SOUTH:
        return "south"
    if county in EAST:
        return "east"
    if county in ISLANDS:
        return "islands"
    return "other"


def build_panel() -> pd.DataFrame:
    vote_parts = []
    for y in (2012, 2016, 2020, 2024):
        print(f"aggregating presidential votes {y}...")
        v = township_votes(y)
        print(f"  townships={len(v)} national dpp={v['dpp_votes'].sum()/v['valid_votes'].sum():.4f}")
        vote_parts.append(v)
    votes = pd.concat(vote_parts, ignore_index=True)

    print("rebuilding income panel...")
    income = rebuild_income_panel()
    income.to_csv(CLEAN / "income_township_panel_full.csv", index=False)

    pop = pd.read_csv(CLEAN / "moi_township_pop_density_panel.csv")
    edu = pd.read_csv(CLEAN / "moi_edu_township_pop15plus_105.csv")

    # Use 2016/2020/2024 as modeling years; 2012 only for lag
    model_years = [2016, 2020, 2024]
    frames = []
    for y in model_years:
        vy = votes[votes.election_year == y].copy()
        # lag
        prev = votes[votes.election_year == y - 4][
            ["township", "dpp_share", "kmt_share"]
        ].rename(columns={"dpp_share": "lag_dpp_share", "kmt_share": "lag_kmt_share"})
        vy = vy.merge(prev, on="township", how="left")

        # income
        iroc = INCOME_ROC_FOR_ELECTION[y]
        inc = income[income.roc_year == iroc][
            ["township", "tax_units", "income_total", "mean_income"]
        ].copy()
        inc = inc.rename(
            columns={
                "tax_units": "tax_units",
                "mean_income": "mean_income",
                "income_total": "income_total",
            }
        )
        vy = vy.merge(inc, on="township", how="left")

        # density
        proc = POP_ROC_FOR_ELECTION[y]
        # pick closest available
        avail = sorted(pop.roc_year.unique())
        if proc not in avail:
            # nearest
            proc = min(avail, key=lambda a: abs(a - proc))
        dens = pop[pop.roc_year == proc][
            ["township", "people_total", "area_km2", "population_density"]
        ]
        vy = vy.merge(dens, on="township", how="left")

        # edu (static 105)
        ed = edu[["township", "pop15plus_total"]].copy()
        vy = vy.merge(ed, on="township", how="left")

        vy["region"] = vy["county_name"].map(region_of)
        vy["income_roc_used"] = iroc
        vy["pop_roc_used"] = proc
        frames.append(vy)

    panel = pd.concat(frames, ignore_index=True)
    # derived features
    panel["mean_income"] = pd.to_numeric(panel["mean_income"], errors="coerce")
    panel["population_density"] = pd.to_numeric(panel["population_density"], errors="coerce")
    panel["people_total"] = pd.to_numeric(panel["people_total"], errors="coerce")
    panel["tax_units"] = pd.to_numeric(panel["tax_units"], errors="coerce")
    panel["pop15plus_total"] = pd.to_numeric(panel["pop15plus_total"], errors="coerce")
    panel["log_mean_income"] = np.log1p(panel["mean_income"])
    panel["log_pop_density"] = np.log1p(panel["population_density"])
    panel["log_people"] = np.log1p(panel["people_total"])
    panel["log_tax_units"] = np.log1p(panel["tax_units"])
    panel["log_pop15"] = np.log1p(panel["pop15plus_total"])
    panel["dpp_kmt_margin_lag"] = panel["lag_dpp_share"] - panel["lag_kmt_share"]

    panel.to_csv(OUT / "township_panel_2016_2024.csv", index=False)
    print("panel rows", len(panel), "complete lag+income+dens", panel.dropna(subset=["lag_dpp_share", "mean_income", "population_density"]).shape[0])
    return panel


FEATURE_COLS = [
    "lag_dpp_share",
    "lag_kmt_share",
    "dpp_kmt_margin_lag",
    "log_mean_income",
    "log_pop_density",
    "log_people",
    "log_tax_units",
    "log_pop15",
    "region_central",
    "region_south",
    "region_east",
    "region_islands",
    # north is reference
]


def prepare_xy(panel: pd.DataFrame):
    df = panel.copy()
    # region dummies
    for r in ["central", "south", "east", "islands", "north", "other"]:
        df[f"region_{r}"] = (df["region"] == r).astype(int)
    need = ["dpp_share"] + FEATURE_COLS
    df = df.dropna(subset=need).copy()
    X = df[FEATURE_COLS]
    y = df["dpp_share"]
    return df, X, y


def run_models(panel: pd.DataFrame) -> dict:
    df, X, y = prepare_xy(panel)
    results = {
        "n_rows": int(len(df)),
        "n_townships": int(df["township"].nunique()),
        "years": sorted(df["election_year"].unique().tolist()),
        "features": FEATURE_COLS,
    }
    print(f"modeling n={len(df)} townships={df.township.nunique()} years={results['years']}")

    # --- Temporal holdout: train 2016+2020, test 2024 ---
    train_mask = df["election_year"].isin([2016, 2020])
    test_mask = df["election_year"] == 2024
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]
    results["n_train"] = int(train_mask.sum())
    results["n_test"] = int(test_mask.sum())

    # LASSO with CV on train
    lasso_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LassoCV(cv=5, random_state=SEED, max_iter=20000)),
        ]
    )
    lasso_pipe.fit(Xtr, ytr)
    lasso = lasso_pipe.named_steps["model"]
    scaler = lasso_pipe.named_steps["scaler"]
    # coefficients on standardized features
    lasso_coef = pd.Series(lasso.coef_, index=FEATURE_COLS)
    yhat_lasso = lasso_pipe.predict(Xte)
    results["lasso_alpha"] = float(lasso.alpha_)
    results["lasso_test_mae"] = float(mean_absolute_error(yte, yhat_lasso))
    results["lasso_test_r2"] = float(r2_score(yte, yhat_lasso))

    # OLS on standardized features (for comparison)
    ols_pipe = Pipeline(
        [("scaler", StandardScaler()), ("model", LinearRegression())]
    )
    ols_pipe.fit(Xtr, ytr)
    ols_coef = pd.Series(ols_pipe.named_steps["model"].coef_, index=FEATURE_COLS)
    yhat_ols = ols_pipe.predict(Xte)
    results["ols_test_mae"] = float(mean_absolute_error(yte, yhat_ols))
    results["ols_test_r2"] = float(r2_score(yte, yhat_ols))

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=5,
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(Xtr, ytr)
    yhat_rf = rf.predict(Xte)
    results["rf_test_mae"] = float(mean_absolute_error(yte, yhat_rf))
    results["rf_test_r2"] = float(r2_score(yte, yhat_rf))
    rf_impurity = pd.Series(rf.feature_importances_, index=FEATURE_COLS)

    # Permutation importance on TEST set
    perm = permutation_importance(
        rf, Xte, yte, n_repeats=30, random_state=SEED, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    # importance = increase in MAE (negate because scoring is neg MAE)
    rf_perm = pd.Series(perm.importances_mean, index=FEATURE_COLS)
    # With neg_mean_absolute_error, higher (less negative) is better; permutation importance
    # reports decrease in score, so positive = MAE worsened when shuffled. Good.

    # Also leave-one-election-out mean perm importance for robustness
    loeo_rows = []
    for hold in [2016, 2020, 2024]:
        tr = df["election_year"] != hold
        te = df["election_year"] == hold
        if tr.sum() < 50 or te.sum() < 20:
            continue
        rf_h = RandomForestRegressor(
            n_estimators=400, max_depth=8, min_samples_leaf=5, random_state=SEED, n_jobs=-1
        )
        rf_h.fit(X[tr], y[tr])
        p = permutation_importance(
            rf_h,
            X[te],
            y[te],
            n_repeats=20,
            random_state=SEED,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        for i, f in enumerate(FEATURE_COLS):
            loeo_rows.append(
                {
                    "holdout_year": hold,
                    "feature": f,
                    "perm_importance_mae": float(p.importances_mean[i]),
                    "test_mae": float(mean_absolute_error(y[te], rf_h.predict(X[te]))),
                    "test_r2": float(r2_score(y[te], rf_h.predict(X[te]))),
                }
            )
    loeo = pd.DataFrame(loeo_rows)
    loeo.to_csv(OUT / "loeo_permutation_by_year.csv", index=False)
    loeo_mean = (
        loeo.groupby("feature", as_index=False)["perm_importance_mae"]
        .mean()
        .rename(columns={"perm_importance_mae": "loeo_mean_perm_importance"})
    )

    # Assemble weights table
    weights = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "lasso_coef_standardized": lasso_coef.values,
            "ols_coef_standardized": ols_coef.values,
            "rf_impurity_importance": rf_impurity.values,
            "rf_perm_importance_holdout2024": rf_perm.values,
        }
    )
    weights = weights.merge(loeo_mean, on="feature", how="left")
    weights["lasso_abs"] = weights["lasso_coef_standardized"].abs()
    weights = weights.sort_values("rf_perm_importance_holdout2024", ascending=False)
    weights.to_csv(OUT / "feature_weights.csv", index=False)

    # Predictions file for 2024
    pred = df[test_mask][
        ["election_year", "township", "county_name", "dpp_share", "lag_dpp_share"]
    ].copy()
    pred["pred_lasso"] = yhat_lasso
    pred["pred_ols"] = yhat_ols
    pred["pred_rf"] = yhat_rf
    pred.to_csv(OUT / "predictions_holdout_2024.csv", index=False)

    # Charts
    plot_importance(weights)
    plot_lasso(weights)

    # Baseline: predict with lag only
    baseline_mae = float(mean_absolute_error(yte, Xte["lag_dpp_share"]))
    results["baseline_lag_only_test_mae"] = baseline_mae

    # County-level check (aggregate)
    county_check = (
        df.groupby(["election_year", "county_name"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "dpp_share_w": np.average(g["dpp_share"], weights=g["valid_votes"]),
                    "valid_votes": g["valid_votes"].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    # pandas groupby.apply quirks — redo simply
    crows = []
    for (ey, cn), g in df.groupby(["election_year", "county_name"]):
        crows.append(
            {
                "election_year": ey,
                "county_name": cn,
                "dpp_share_w": np.average(g["dpp_share"], weights=g["valid_votes"]),
                "valid_votes": g["valid_votes"].sum(),
            }
        )
    county_agg = pd.DataFrame(crows)
    county_agg.to_csv(OUT / "county_weighted_from_township.csv", index=False)

    results["weights_path"] = str(OUT / "feature_weights.csv")
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results, weights, df


def plot_importance(weights: pd.DataFrame):
    w = weights.sort_values("rf_perm_importance_holdout2024", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(w["feature"], w["rf_perm_importance_holdout2024"], color="#2c7fb8")
    ax.set_xlabel("Permutation importance (Δ MAE on 2024 holdout)")
    ax.set_title("RF feature importance — township DPP share (train 2016+2020 → 2024)")
    fig.tight_layout()
    fig.savefig(CHARTS / "rf_perm_importance_2024.png", dpi=140)
    plt.close(fig)

    w2 = weights.sort_values("loeo_mean_perm_importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(w2["feature"], w2["loeo_mean_perm_importance"], color="#41b6c4")
    ax.set_xlabel("Mean permutation importance (LOEO)")
    ax.set_title("RF permutation importance — leave-one-election-out mean")
    fig.tight_layout()
    fig.savefig(CHARTS / "rf_perm_importance_loeo.png", dpi=140)
    plt.close(fig)


def plot_lasso(weights: pd.DataFrame):
    w = weights.sort_values("lasso_coef_standardized")
    colors = ["#d73027" if v < 0 else "#1a9850" for v in w["lasso_coef_standardized"]]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(w["feature"], w["lasso_coef_standardized"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("LASSO coefficient (standardized features)")
    ax.set_title("LASSO coefficients — township DPP share (train 2016+2020)")
    fig.tight_layout()
    fig.savefig(CHARTS / "lasso_coefficients.png", dpi=140)
    plt.close(fig)


def main():
    panel = build_panel()
    results, weights, df = run_models(panel)
    print("\n=== SUMMARY ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("\n=== FEATURE WEIGHTS (sorted by RF perm 2024) ===")
    print(weights.to_string(index=False))
    print(f"\nWrote outputs under {OUT}")


if __name__ == "__main__":
    main()
