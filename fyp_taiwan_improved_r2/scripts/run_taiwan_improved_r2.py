#!/usr/bin/env python3
"""Taiwan FYP — IMPROVED ROUND 2.

Builds on fyp_taiwan_improved/ harmonized panels.
1) Stronger spatial: LOO same-county lag + county-mean lag; ablation swing+demeaned
2) Consistent demeaned/swing for KMT + legislative DPP/KMT (+TPP if lag)
3) Demeaned residual → add back national mean for level (honest + oracle-labeled)
4) Split-conformal / year-specific residual quantiles on DEMEANED
5) Document ESC county unavailable; Ridge with explicit year FE (+ ESC)
Honest metrics only.
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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path("/workspace/taiwan-election-data")
PREV = ROOT / "fyp_taiwan_improved"
OUT = ROOT / "fyp_taiwan_improved_r2"
CHARTS = OUT / "charts"
PRES = PREV / "harmonized_presidential_improved.csv"
LEG = PREV / "harmonized_legislative_district_improved.csv"
SEED = 42
ALPHA = 0.10

SOCIO = ["median_income", "urbanization", "education_rate", "log_people"]
GEO = ["geo_north", "geo_central", "geo_south", "geo_east", "geo_islands"]
ESC = [
    "esc_dpp_pid", "esc_kmt_pid", "esc_tpp_pid", "esc_ind_pid",
    "esc_indep_lean", "esc_unif_lean", "esc_dpp_delta", "esc_kmt_delta",
]

for p in (OUT, CHARTS, OUT / "scripts"):
    p.mkdir(parents=True, exist_ok=True)


def mae(y, p):
    return float(mean_absolute_error(y, p))


def mets(y, p):
    return {
        "MAE": mae(y, p),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
    }


def impute(tr, te, feats):
    tr, te = tr.copy(), te.copy()
    for c in feats:
        if c not in tr.columns:
            tr[c] = 0.0
            te[c] = 0.0
        fill = tr[c].median()
        if pd.isna(fill):
            fill = 0.0
        tr[c] = tr[c].fillna(fill)
        te[c] = te[c].fillna(fill)
    return tr, te


def add_county_mean_lag(df: pd.DataFrame, lag_cols: list[str]) -> pd.DataFrame:
    """Distance-free county mean of lagged share (includes self — labeled county_mean_lag)."""
    d = df.copy()
    for lag_col in lag_cols:
        if lag_col not in d.columns:
            continue
        name = f"county_mean_{lag_col}"
        d[name] = d.groupby(["year", "county_name"])[lag_col].transform("mean")
    return d


def hist_feats(party: str) -> list[str]:
    base = [f"prev_{party}_share", "prev_dpp_share", "prev_kmt_share", "prev_turnout"]
    if party == "dpp":
        base += ["prev_dpp_swing", "dpp_kmt_margin_prev"]
    elif party == "kmt":
        base += ["prev_kmt_swing", "dpp_kmt_margin_prev"]
    elif party == "tpp":
        base += ["prev_third_share", "dpp_kmt_margin_prev"]
    out, seen = [], set()
    for c in base:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def spatial_feats(party: str) -> list[str]:
    if party == "tpp":
        return [
            "spatial_lag_prev_third_share",
            "spatial_lag_prev_dpp_share",
            "spatial_lag_prev_kmt_share",
            "county_mean_prev_third_share",
            "county_mean_prev_dpp_share",
            "county_mean_prev_kmt_share",
        ]
    return [
        f"spatial_lag_prev_{party}_share",
        f"county_mean_prev_{party}_share",
        "spatial_lag_prev_dpp_share",
        "spatial_lag_prev_kmt_share",
        "county_mean_prev_dpp_share",
        "county_mean_prev_kmt_share",
    ]


def all_feats(party: str, spatial=True, esc=True, year_fe_cols=None) -> list[str]:
    f = hist_feats(party) + SOCIO + GEO
    if spatial:
        f += spatial_feats(party)
    if esc:
        f += ESC
    if year_fe_cols:
        f += year_fe_cols
    out, seen = [], set()
    for c in f:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def fit_predict(model_name, Xtr, ytr, Xte):
    if model_name == "Ridge":
        mdl = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    elif model_name == "RidgeYearFE":
        # year FE already in X; same Ridge
        mdl = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    elif model_name == "RandomForest":
        mdl = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1
        )
    else:
        mdl = HistGradientBoostingRegressor(
            max_depth=5, learning_rate=0.06, random_state=SEED
        )
    mdl.fit(Xtr, ytr)
    return mdl.predict(Xte), mdl


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    lag_cols = [c for c in d.columns if c.startswith("prev_") and c.endswith("_share")]
    d = add_county_mean_lag(d, lag_cols)
    return d


def run_backtest_one(data, party, target_col, target_label, feats, race, model_names=None):
    model_names = model_names or ["Ridge", "RandomForest", "HistGB"]
    years = sorted(data["year"].unique())
    rows = []
    share = f"{party}_share"
    prev_share = f"prev_{party}_share"
    prev_dem = f"prev_{party}_demeaned"
    prev_sw = f"prev_{party}_swing"

    for test_year in years:
        train = data[data["year"] < test_year]
        test = data[data["year"] == test_year]
        if target_col not in test.columns:
            continue
        train_ok = train.dropna(subset=[target_col])
        test_ok = test.dropna(subset=[target_col])
        if len(test_ok) < 20:
            continue
        y = test_ok[target_col]

        # baselines
        if target_label == "swing":
            rows.append({
                "race": race, "party": party.upper(), "target": target_label,
                "test_year": test_year, "model": "ZeroSwing", "feature_set": "baseline",
                "n_train": len(train_ok), "n_test": len(test_ok), **mets(y, np.zeros(len(y))),
            })
            if prev_sw in test_ok.columns and test_ok[prev_sw].notna().sum() >= 20:
                m = test_ok[prev_sw].notna()
                rows.append({
                    "race": race, "party": party.upper(), "target": target_label,
                    "test_year": test_year, "model": "PreviousSwing", "feature_set": "baseline",
                    "n_train": len(train_ok), "n_test": int(m.sum()),
                    **mets(y[m], test_ok.loc[m, prev_sw]),
                })
        elif target_label == "demeaned":
            rows.append({
                "race": race, "party": party.upper(), "target": target_label,
                "test_year": test_year, "model": "ZeroDemeaned", "feature_set": "baseline",
                "n_train": len(train_ok), "n_test": len(test_ok), **mets(y, np.zeros(len(y))),
            })
            if prev_dem in test_ok.columns and test_ok[prev_dem].notna().sum() >= 20:
                m = test_ok[prev_dem].notna()
                rows.append({
                    "race": race, "party": party.upper(), "target": target_label,
                    "test_year": test_year, "model": "PreviousDemeaned", "feature_set": "baseline",
                    "n_train": len(train_ok), "n_test": int(m.sum()),
                    **mets(y[m], test_ok.loc[m, prev_dem]),
                })
        elif target_label == "level":
            if prev_share in test_ok.columns and test_ok[prev_share].notna().sum() >= 20:
                m = test_ok[prev_share].notna()
                rows.append({
                    "race": race, "party": party.upper(), "target": target_label,
                    "test_year": test_year, "model": "PreviousShare", "feature_set": "baseline",
                    "n_train": len(train_ok), "n_test": int(m.sum()),
                    **mets(y[m], test_ok.loc[m, prev_share]),
                })

        if len(train_ok) < 50:
            continue
        use = [c for c in feats if c in data.columns]
        tr, te = impute(train_ok, test_ok, use)
        tr = tr.dropna(subset=[target_col])
        te = te.dropna(subset=[target_col])
        for mname in model_names:
            try:
                pred, _ = fit_predict(mname, tr[use].values, tr[target_col].values, te[use].values)
                rows.append({
                    "race": race, "party": party.upper(), "target": target_label,
                    "test_year": test_year, "model": mname, "feature_set": "r2_all",
                    "n_train": len(tr), "n_test": len(te), **mets(te[target_col].values, pred),
                })
            except Exception as e:
                print(f" fail {race} {party} {target_label} {test_year} {mname}: {e}")
    return rows


def spatial_ablation(data, party, target_col, target_label, race, test_year=2024):
    """With/without spatial (LOO + county mean) for RF and Ridge."""
    rows = []
    train = data[data["year"] < test_year].dropna(subset=[target_col])
    test = data[data["year"] == test_year].dropna(subset=[target_col])
    if len(test) < 20 or len(train) < 50:
        return rows
    configs = {
        "NoSpatial": all_feats(party, spatial=False, esc=True),
        "SpatialLOO_only": hist_feats(party) + SOCIO + GEO + ESC + [
            c for c in spatial_feats(party) if c.startswith("spatial_lag_")
        ],
        "CountyMean_only": hist_feats(party) + SOCIO + GEO + ESC + [
            c for c in spatial_feats(party) if c.startswith("county_mean_")
        ],
        "SpatialBoth": all_feats(party, spatial=True, esc=True),
        "NoSpatialNoESC": all_feats(party, spatial=False, esc=False),
    }
    for fs_name, feats in configs.items():
        use = [c for c in feats if c in data.columns]
        tr, te = impute(train, test, use)
        for mname in ["Ridge", "RandomForest"]:
            pred, _ = fit_predict(mname, tr[use].values, tr[target_col].values, te[use].values)
            rows.append({
                "race": race, "party": party.upper(), "target": target_label,
                "test_year": test_year, "feature_set": fs_name, "model": mname,
                "n_features": len(use), **mets(te[target_col].values, pred),
            })
    return rows


def demeaned_to_level(data, party, race, test_year=2024):
    """Predict demeaned then add back national mean variants; compare to PreviousShare."""
    share = f"{party}_share"
    dem = f"{party}_demeaned"
    prev_share = f"{party}_share".replace(f"{party}_share", f"prev_{party}_share")
    prev_share = f"prev_{party}_share"
    nat = f"{party}_nat"
    prev_nat = f"prev_{party}_nat"

    train = data[data["year"] < test_year].dropna(subset=[dem, share])
    test = data[data["year"] == test_year].dropna(subset=[dem, share, prev_share])
    if len(test) < 20 or len(train) < 50:
        return []

    feats = [c for c in all_feats(party, True, True) if c in data.columns]
    # drop ESC national constants? keep — help year regime for demeaned less, but ok
    tr, te = impute(train, test, feats)
    rows = []
    y_level = te[share].values
    prev = te[prev_share].values
    true_nat = te[nat].values if nat in te.columns else None
    prev_nat_v = te[prev_nat].values if prev_nat in te.columns else None

    # ESC-implied national: prev_nat + esc party delta (for dpp/kmt)
    esc_delta_col = f"esc_{party}_delta" if party in ("dpp", "kmt") else None
    if esc_delta_col and esc_delta_col in te.columns and prev_nat_v is not None:
        esc_nat = prev_nat_v + te[esc_delta_col].values
    else:
        esc_nat = None

    rows.append({
        "race": race, "party": party.upper(), "test_year": test_year,
        "method": "PreviousShare", "MAE_level": mae(y_level, prev),
        "note": "baseline",
    })

    for mname in ["Ridge", "RandomForest", "HistGB"]:
        pred_dem, _ = fit_predict(mname, tr[feats].values, tr[dem].values, te[feats].values)
        # A: add prev national (honest)
        if prev_nat_v is not None:
            pred_a = pred_dem + prev_nat_v
            rows.append({
                "race": race, "party": party.upper(), "test_year": test_year,
                "method": f"{mname}_demeaned+prev_nat", "MAE_level": mae(y_level, pred_a),
                "MAE_demeaned": mae(te[dem].values, pred_dem),
                "note": "honest: add previous election national mean",
            })
        # B: add ESC-shifted national
        if esc_nat is not None:
            pred_b = pred_dem + esc_nat
            rows.append({
                "race": race, "party": party.upper(), "test_year": test_year,
                "method": f"{mname}_demeaned+esc_nat", "MAE_level": mae(y_level, pred_b),
                "MAE_demeaned": mae(te[dem].values, pred_dem),
                "note": "honest PROXY: prev_nat + ESC pid delta",
            })
        # C: oracle true test national (LEAKAGE — labeled)
        if true_nat is not None:
            pred_c = pred_dem + true_nat
            rows.append({
                "race": race, "party": party.upper(), "test_year": test_year,
                "method": f"{mname}_demeaned+oracle_nat", "MAE_level": mae(y_level, pred_c),
                "MAE_demeaned": mae(te[dem].values, pred_dem),
                "note": "ORACLE leakage: true test-year national — ranking skill only",
            })
    return rows


def year_fe_ridge(data, party, target_col, target_label, race, test_year=2024):
    """Ridge with explicit year dummies (train years); test year FE = 0 + ESC for regime."""
    train = data[data["year"] < test_year].dropna(subset=[target_col])
    test = data[data["year"] == test_year].dropna(subset=[target_col])
    if len(test) < 20 or len(train) < 50:
        return []
    train_years = sorted(train["year"].unique())
    d = data.copy()
    yfe = []
    for y in train_years:
        col = f"year_fe_{y}"
        d[col] = (d["year"] == y).astype(float)
        yfe.append(col)
    # test rows: all year FE 0
    tr = d[d["year"] < test_year].dropna(subset=[target_col])
    te = d[d["year"] == test_year].dropna(subset=[target_col])
    rows = []
    for label, feats in [
        ("Ridge_noYearFE", all_feats(party, True, True)),
        ("Ridge_YearFE_noESC", all_feats(party, True, False, yfe)),
        ("Ridge_YearFE+ESC", all_feats(party, True, True, yfe)),
        ("Ridge_ESC_only_yearRegime", hist_feats(party) + SOCIO + GEO + ESC + spatial_feats(party)),
    ]:
        use = [c for c in feats if c in d.columns]
        tr2, te2 = impute(tr, te, use)
        pred, _ = fit_predict("Ridge", tr2[use].values, tr2[target_col].values, te2[use].values)
        rows.append({
            "race": race, "party": party.upper(), "target": target_label,
            "test_year": test_year, "model": label, **mets(te2[target_col].values, pred),
            "n_features": len(use),
            "note": "unseen test year: year FE columns are 0; ESC carries regime if present",
        })
    return rows


def conformal_demeaned(data, party, race, test_year=2024, alpha=ALPHA):
    """Split conformal on demeaned: fit on years < cal, cal = last train year, test = test_year.
    Also year-specific residual quantiles from each leave-one-election train fold.
    """
    dem = f"{party}_demeaned"
    years = sorted(data["year"].unique())
    if test_year not in years:
        return [], None
    prior = [y for y in years if y < test_year]
    if len(prior) < 2:
        return [], None
    cal_year = prior[-1]
    fit_years = [y for y in prior if y < cal_year]
    if not fit_years:
        return [], None

    feats = [c for c in all_feats(party, True, True) if c in data.columns]
    fit = data[data["year"].isin(fit_years)].dropna(subset=[dem])
    cal = data[data["year"] == cal_year].dropna(subset=[dem])
    test = data[data["year"] == test_year].dropna(subset=[dem])
    if min(len(fit), len(cal), len(test)) < 20:
        return [], None

    tr, ca = impute(fit, cal, feats)
    _, te = impute(fit, test, feats)
    pred_cal, mdl = fit_predict("RandomForest", tr[feats].values, tr[dem].values, ca[feats].values)
    # re-fit same hyperparams on fit for test preds
    pred_te, _ = fit_predict("RandomForest", tr[feats].values, tr[dem].values, te[feats].values)
    resid_cal = np.abs(ca[dem].values - pred_cal)
    q = float(np.quantile(resid_cal, 1 - alpha))
    cover = float(np.mean(np.abs(te[dem].values - pred_te) <= q))

    # year-specific residual quantiles (leave-one-election on prior years)
    year_qs = []
    for hold in prior:
        tr_y = data[data["year"] < hold].dropna(subset=[dem])
        te_y = data[data["year"] == hold].dropna(subset=[dem])
        if len(tr_y) < 50 or len(te_y) < 20:
            continue
        tr2, te2 = impute(tr_y, te_y, feats)
        pred_y, _ = fit_predict("RandomForest", tr2[feats].values, tr2[dem].values, te2[feats].values)
        r = np.abs(te2[dem].values - pred_y)
        year_qs.append({
            "holdout_year": int(hold),
            "resid_q50": float(np.quantile(r, 0.50)),
            "resid_q90": float(np.quantile(r, 0.90)),
            "MAE": mae(te2[dem].values, pred_y),
        })
    # apply max prior q90 as conservative half-width
    if year_qs:
        cons_q = max(x["resid_q90"] for x in year_qs)
        cover_cons = float(np.mean(np.abs(te[dem].values - pred_te) <= cons_q))
    else:
        cons_q, cover_cons = None, None

    meta = {
        "race": race, "party": party.upper(), "target": "demeaned",
        "test_year": int(test_year), "cal_year": int(cal_year), "fit_years": [int(x) for x in fit_years],
        "alpha": alpha, "nominal_coverage": 1 - alpha,
        "split_conformal_halfwidth_q": q,
        "split_empirical_coverage": cover,
        "year_specific_max_q90_halfwidth": cons_q,
        "year_specific_empirical_coverage": cover_cons,
        "MAE_point": mae(te[dem].values, pred_te),
        "n_fit": len(tr), "n_cal": len(ca), "n_test": len(te),
        "year_specific_quantiles": year_qs,
        "note": "split conformal on DEMEANED; year-specific = max leave-one-election |resid| q90",
    }
    intervals = te[["region", "year", "county_name"]].copy() if "region" in te.columns else te[["year", "county_name"]].copy()
    intervals["y_true"] = te[dem].values
    intervals["y_pred"] = pred_te
    intervals["lo"] = pred_te - q
    intervals["hi"] = pred_te + q
    intervals["covered"] = (np.abs(te[dem].values - pred_te) <= q)
    if cons_q is not None:
        intervals["lo_yearq"] = pred_te - cons_q
        intervals["hi_yearq"] = pred_te + cons_q
        intervals["covered_yearq"] = (np.abs(te[dem].values - pred_te) <= cons_q)
    return [meta], intervals


def before_after_table(bt: pd.DataFrame, test_year=2024) -> pd.DataFrame:
    rows = []
    sub = bt[bt["test_year"] == test_year]
    for (race, party, target), g in sub.groupby(["race", "party", "target"]):
        baselines = g[g["feature_set"] == "baseline"] if "feature_set" in g.columns else g[g["model"].str.contains("Previous|Zero")]
        improved = g[g["feature_set"] == "r2_all"] if "feature_set" in g.columns else g[~g["model"].str.contains("Previous|Zero")]
        if baselines.empty:
            bas = g[g["model"].isin(["ZeroSwing", "PreviousSwing", "PreviousDemeaned", "ZeroDemeaned", "PreviousShare"])]
        else:
            bas = baselines
        if improved.empty:
            imp = g[g["model"].isin(["Ridge", "RandomForest", "HistGB", "RidgeYearFE"])]
        else:
            imp = improved
        if bas.empty:
            continue
        bbest = bas.loc[bas["MAE"].idxmin()]
        if imp.empty:
            rows.append({
                "race": race, "party": party, "target": target,
                "best_baseline": bbest["model"], "baseline_MAE": bbest["MAE"],
                "best_r2_model": None, "r2_MAE": None, "delta_MAE": None, "r2_wins": None,
            })
            continue
        ibest = imp.loc[imp["MAE"].idxmin()]
        delta = float(bbest["MAE"] - ibest["MAE"])
        rows.append({
            "race": race, "party": party, "target": target,
            "best_baseline": bbest["model"], "baseline_MAE": float(bbest["MAE"]),
            "best_r2_model": ibest["model"], "r2_MAE": float(ibest["MAE"]),
            "delta_MAE_baseline_minus_r2": delta, "r2_wins": delta > 0,
        })
    return pd.DataFrame(rows)


def main():
    print("Loading panels…")
    pres = prepare(pd.read_csv(PRES))
    leg = prepare(pd.read_csv(LEG))
    # save enriched
    pres.to_csv(OUT / "harmonized_presidential_r2.csv", index=False)
    leg.to_csv(OUT / "harmonized_legislative_r2.csv", index=False)

    all_bt = []
    ablations = []
    level_rows = []
    yearfe_rows = []
    conf_metas = []

    jobs = [
        (pres, "presidential", ["dpp", "kmt"]),
        (leg, "legislative_district", ["dpp", "kmt", "tpp"]),
    ]
    # TPP presidential: only demeaned/level if share exists 2024; swing needs lag — skip swing if no prev
    for data, race, parties in jobs:
        for party in parties:
            share = f"{party}_share"
            dem = f"{party}_demeaned"
            swing = f"{party}_swing"
            feats = [c for c in all_feats(party, True, True) if c in data.columns]
            print(f"== {race} {party.upper()} ==")
            if swing in data.columns and data[swing].notna().sum() > 100:
                all_bt += run_backtest_one(data, party, swing, "swing", feats, race)
            if dem in data.columns and data[dem].notna().sum() > 100:
                all_bt += run_backtest_one(data, party, dem, "demeaned", feats, race)
            if share in data.columns:
                all_bt += run_backtest_one(data, party, share, "level", feats, race)

            # spatial ablation presidential DPP/KMT swing+demeaned; also legislative for consistency
            if party in ("dpp", "kmt"):
                if swing in data.columns:
                    ablations += spatial_ablation(data, party, swing, "swing", race)
                if dem in data.columns:
                    ablations += spatial_ablation(data, party, dem, "demeaned", race)
                level_rows += demeaned_to_level(data, party, race)
                if dem in data.columns:
                    yearfe_rows += year_fe_ridge(data, party, dem, "demeaned", race)
                    yearfe_rows += year_fe_ridge(data, party, swing, "swing", race) if swing in data.columns else []
                    meta, intervals = conformal_demeaned(data, party, race)
                    if meta:
                        conf_metas += meta
                        if intervals is not None and race == "presidential":
                            intervals.to_csv(
                                OUT / f"conformal_demeaned_intervals_{party}_{race}_2024.csv",
                                index=False,
                            )

    # TPP presidential level/demeaned (no swing lag)
    print("== presidential TPP ==")
    party = "tpp"
    if "tpp_share" in pres.columns and pres["tpp_share"].notna().sum() > 50:
        feats = [c for c in all_feats(party, True, True) if c in pres.columns]
        all_bt += run_backtest_one(pres, party, "tpp_demeaned", "demeaned", feats, "presidential")
        all_bt += run_backtest_one(pres, party, "tpp_share", "level", feats, "presidential")

    bt = pd.DataFrame(all_bt)
    bt.to_csv(OUT / "backtest_results_r2.csv", index=False)
    abl = pd.DataFrame(ablations)
    abl.to_csv(OUT / "ablation_spatial_r2.csv", index=False)
    lev = pd.DataFrame(level_rows)
    lev.to_csv(OUT / "demeaned_to_level_2024.csv", index=False)
    yfe = pd.DataFrame(yearfe_rows)
    yfe.to_csv(OUT / "ridge_year_fe_2024.csv", index=False)
    ba = before_after_table(bt, 2024)
    ba.to_csv(OUT / "before_after_2024_r2.csv", index=False)

    conf_df = pd.json_normalize(conf_metas)
    # flatten year_specific for csv meta without nested
    conf_meta_flat = []
    for m in conf_metas:
        flat = {k: v for k, v in m.items() if k != "year_specific_quantiles"}
        flat["year_specific_quantiles_json"] = json.dumps(m.get("year_specific_quantiles", []), default=lambda o: int(o) if hasattr(o, "item") else float(o) if isinstance(o, float) else str(o))
        conf_meta_flat.append(flat)
    pd.DataFrame(conf_meta_flat).to_csv(OUT / "conformal_demeaned_meta.csv", index=False)
    with open(OUT / "conformal_demeaned_year_quantiles.json", "w", encoding="utf-8") as f:
        json.dump({f"{m['race']}_{m['party']}": m.get("year_specific_quantiles") for m in conf_metas}, f, indent=2, default=lambda o: int(o) if hasattr(o, "item") else (float(o) if hasattr(o, "__float__") else str(o)))

    # Charts: spatial ablation MAE for presidential DPP/KMT
    if not abl.empty:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=False)
        plot_jobs = [
            ("presidential", "DPP", "swing"),
            ("presidential", "DPP", "demeaned"),
            ("presidential", "KMT", "swing"),
            ("presidential", "KMT", "demeaned"),
        ]
        for ax, (race, party, tgt) in zip(axes.ravel(), plot_jobs):
            s = abl[(abl.race == race) & (abl.party == party) & (abl.target == tgt) & (abl.model == "RandomForest")]
            if s.empty:
                ax.set_visible(False)
                continue
            order = ["NoSpatialNoESC", "NoSpatial", "SpatialLOO_only", "CountyMean_only", "SpatialBoth"]
            s = s.set_index("feature_set").reindex([x for x in order if x in set(s.feature_set)])
            ax.bar(range(len(s)), s["MAE"].values, color="#4C78A8")
            ax.set_xticks(range(len(s)))
            ax.set_xticklabels(s.index, rotation=30, ha="right", fontsize=8)
            ax.set_title(f"{party} {tgt} (RF)")
            ax.set_ylabel("MAE")
        fig.suptitle("Round2 spatial ablation — presidential 2024")
        fig.tight_layout()
        fig.savefig(CHARTS / "spatial_ablation_pres_2024.png", dpi=140)
        plt.close()

    # demeaned→level chart
    if not lev.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sub = lev[lev.race == "presidential"]
        # pick key methods
        keep = sub[sub.method.str.contains("PreviousShare|RandomForest_demeaned\\+prev_nat|RandomForest_demeaned\\+esc_nat|RandomForest_demeaned\\+oracle|Ridge_demeaned\\+esc_nat|Ridge_demeaned\\+prev_nat")]
        if not keep.empty:
            labels = keep["party"] + ":" + keep["method"].str.replace("_demeaned", "", regex=False)
            ax.barh(range(len(keep)), keep["MAE_level"].values)
            ax.set_yticks(range(len(keep)))
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel("Level MAE 2024")
            ax.set_title("Demeaned→level reconstruction vs PreviousShare")
            fig.tight_layout()
            fig.savefig(CHARTS / "demeaned_to_level_2024.png", dpi=140)
            plt.close()

    # ESC county documentation
    esc_doc = """# ESC / 政大選研 — 縣市級黨性

**結論：公開管道仍然攞唔到縣市（或鄉鎮）級政黨認同／統獨時間序列。**

- 政大選研中心網站公開嘅係**全國**趨勢圖；官方 Excel 要申請，亦主要係全國序列。
- TEDS 微資料可估計縣市，但要正式申請，本專案未納入。
- 因此 Round2 繼續只用：
  1. 全國 ESC PROXY（`proxies/esc_nccu_national_proxy.csv`）當選舉年「氣候／體制」指標；
  2. 地理 dummy + 同縣空間滯後／縣平均滯後；
  3. Ridge 顯式 **year fixed effects**（測試年 FE=0，靠 ESC 帶 regime）。

唔會捏造縣市級黨性數字。
"""
    (OUT / "proxies").mkdir(exist_ok=True)
    (OUT / "proxies" / "ESC_COUNTY_UNAVAILABLE.md").write_text(esc_doc, encoding="utf-8")
    # copy proxy csv
    src = PREV / "proxies" / "esc_nccu_national_proxy.csv"
    if src.exists():
        (OUT / "proxies" / "esc_nccu_national_proxy.csv").write_bytes(src.read_bytes())

    # Compare to round1 before_after
    r1 = PREV / "before_after_2024.csv"
    compare = None
    if r1.exists() and not ba.empty:
        r1df = pd.read_csv(r1)
        m = ba.merge(
            r1df,
            on=["race", "party", "target"],
            how="left",
            suffixes=("_r2", "_r1"),
        )
        compare = m
        m.to_csv(OUT / "compare_r1_vs_r2_2024.csv", index=False)

    summary = {
        "round": 2,
        "improvements": [
            "county_mean_lag + spatial_lag LOO; ablation NoSpatial / LOO / CountyMean / Both",
            "consistent swing+demeaned+level for presidential & legislative DPP/KMT (+TPP where possible)",
            "demeaned→level: prev_nat / ESC_nat / oracle_nat (oracle labeled leakage)",
            "split conformal + year-specific residual q90 on DEMEANED",
            "Ridge year FE (test year FE=0) + ESC regime; ESC county still unavailable documented",
        ],
        "before_after_2024": ba.to_dict(orient="records") if not ba.empty else [],
        "spatial_ablation_pres_RF_2024": abl[
            (abl.race == "presidential") & (abl.model == "RandomForest") & (abl.test_year == 2024)
        ].to_dict(orient="records") if not abl.empty else [],
        "demeaned_to_level_highlights": lev.to_dict(orient="records") if not lev.empty else [],
        "conformal_demeaned": conf_meta_flat,
        "esc_county": "unavailable — see proxies/ESC_COUNTY_UNAVAILABLE.md",
    }
    with open(OUT / "run_summary_r2.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=lambda o: int(o) if hasattr(o, "item") and isinstance(o.item(), int) else (float(o) if hasattr(o, "item") else str(o)))

    # README
    def fmt_ba(df):
        if df.empty:
            return "(empty)"
        lines = ["| race | party | target | baseline | base MAE | r2 model | r2 MAE | Δ | win? |",
                 "|------|-------|--------|----------|----------|----------|--------|---|------|"]
        for _, r in df.iterrows():
            lines.append(
                f"| {r.race} | {r.party} | {r.target} | {r.best_baseline} | {r.baseline_MAE:.4f} | "
                f"{r.best_r2_model} | {r.r2_MAE:.4f} | {r.delta_MAE_baseline_minus_r2:+.4f} | {r.r2_wins} |"
            )
        return "\n".join(lines)

    # key spatial deltas
    spat_lines = []
    if not abl.empty:
        for party in ["DPP", "KMT"]:
            for tgt in ["swing", "demeaned"]:
                s = abl[(abl.race == "presidential") & (abl.party == party) & (abl.target == tgt) & (abl.model == "RandomForest")]
                if s.empty:
                    continue
                d = dict(zip(s.feature_set, s.MAE))
                spat_lines.append(
                    f"- Pres {party} {tgt} RF: NoSpatial={d.get('NoSpatial', float('nan')):.4f}, "
                    f"Both={d.get('SpatialBoth', float('nan')):.4f}, "
                    f"Δ(NoSpat-Both)={d.get('NoSpatial', 0)-d.get('SpatialBoth', 0):+.4f}"
                )

    level_lines = []
    if not lev.empty:
        for _, r in lev[(lev.race == "presidential") & (lev.method.isin([
            "PreviousShare",
            "Ridge_demeaned+prev_nat", "Ridge_demeaned+esc_nat", "Ridge_demeaned+oracle_nat",
            "RandomForest_demeaned+prev_nat", "RandomForest_demeaned+esc_nat", "RandomForest_demeaned+oracle_nat",
        ]))].iterrows():
            level_lines.append(f"- {r.party} {r.method}: MAE_level={r.MAE_level:.4f} ({r.note})")

    conf_lines = []
    for m in conf_metas:
        conf_lines.append(
            f"- {m['race']} {m['party']} demeaned: split cover={m['split_empirical_coverage']:.3f} "
            f"(halfwidth={m['split_conformal_halfwidth_q']:.4f}); "
            f"yearQ cover={m.get('year_specific_empirical_coverage')} "
            f"(halfwidth={m.get('year_specific_max_q90_halfwidth')})"
        )

    readme = f"""# Taiwan FYP — 第二輪改進（fyp_taiwan_improved_r2）

建基於 `fyp_taiwan_improved/`。誠實指標；唔造假縣市黨性。

## 本輪做了什麼

1. **更強空間特徵**：同縣 LOO `spatial_lag_*` + 距離無關 `county_mean_*`；ablation：NoSpatial / LOO / CountyMean / Both（總統 DPP／KMT 嘅 swing **同** demeaned）。
2. **一致 pipeline**：總統＋區域立委 DPP／KMT（及有滯後嘅 TPP）都跑 swing／demeaned／level。
3. **Demeaned → Level**：預測相對全國殘差後加返 `prev_nat`／`ESC_nat`／`oracle_nat`（oracle 標明洩漏）。
4. **不確定性**：喺 **demeaned** 上做 split-conformal + leave-one-election 殘差 q90。
5. **ESC 縣市級**：仍不可得 → `proxies/ESC_COUNTY_UNAVAILABLE.md`；Ridge 加顯式 year FE（測試年 FE=0）。

## 2024 before/after

{fmt_ba(ba)}

## 空間 ablation（總統 RF）

{chr(10).join(spat_lines) if spat_lines else "(none)"}

## Demeaned→Level vs PreviousShare

{chr(10).join(level_lines) if level_lines else "(none)"}

## Conformal（demeaned）

{chr(10).join(conf_lines) if conf_lines else "(none)"}

## 重跑

```bash
cd /workspace/taiwan-election-data
.venv/bin/python fyp_taiwan_improved_r2/scripts/run_taiwan_improved_r2.py
```

## 仍然欠

- TEDS 微資料／縣市黨性
- 真實共邊 shapefile GCN（見 `fyp_taiwan_spatial_gnn/` 用同縣圖近似）
- 跨國 US transfer
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print("Done. Outputs in", OUT)
    print(ba.to_string(index=False) if not ba.empty else "no ba")
    if spat_lines:
        print("\n".join(spat_lines))


if __name__ == "__main__":
    main()
