#!/usr/bin/env python3
"""Taiwan FYP — IMPROVED round.

Priority improvements (real CEC panels; no invented TEDS microdata):
1) Targets: SWING (ΔV) and DEMEANED (share − national mean) for DPP/KMT
   (presidential + legislative district where lags exist). Compare vs ZeroSwing / PreviousShare.
2) ESC NCCU national PROXY for party ID / 統獨 (pre-election window) — labeled PROXY;
   plus geo_region dummies. No county ESC public series exists.
3) Separate party heads DPP / KMT / TPP for 2024 three-candidate structure.
4) Split conformal on SWING (+ leave-one-election residual quantile note).
5) Spatial lag: same-county neighbor mean of lagged party share + ablation.
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

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path("/workspace/taiwan-election-data")
OUT = ROOT / "fyp_taiwan_improved"
CHARTS = OUT / "charts"
LEG_OUT = OUT / "legislative"
PROXY_PATH = OUT / "proxies" / "esc_nccu_national_proxy.csv"
PRES_PATH = ROOT / "fyp_taiwan_parties" / "harmonized_tw_parties.csv"
LEG_PATH = ROOT / "fyp_taiwan_legislative" / "harmonized_tw_legislative.csv"
BASELINE_SWING = ROOT / "fyp_taiwan_swing" / "backtest_results.csv"
BASELINE_PRES = ROOT / "fyp_taiwan_parties" / "compare_2024_mae_by_party.csv"
BASELINE_LEG = ROOT / "fyp_taiwan_legislative" / "compare_pres_vs_leg_2024.csv"

for p in (OUT, CHARTS, LEG_OUT, OUT / "scripts"):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
ALPHA = 0.10

SOCIO = ["median_income", "urbanization", "education_rate", "log_people"]
GEO_COLS = ["geo_north", "geo_central", "geo_south", "geo_east", "geo_islands"]
ESC_COLS = [
    "esc_dpp_pid",
    "esc_kmt_pid",
    "esc_tpp_pid",
    "esc_ind_pid",
    "esc_indep_lean",
    "esc_unif_lean",
    "esc_dpp_delta",
    "esc_kmt_delta",
]


def get_models():
    return {
        "Ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1
        ),
        "HistGB": HistGradientBoostingRegressor(
            max_depth=5, learning_rate=0.06, random_state=SEED
        ),
    }


def metrics(y, pred):
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)) if len(np.unique(y)) > 1 else float("nan"),
    }


def impute(train, test, feats):
    tr, te = train.copy(), test.copy()
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


def load_esc_proxy() -> pd.DataFrame:
    esc = pd.read_csv(PROXY_PATH)
    esc = esc.rename(
        columns={
            "election_year": "year",
            "dpp_pid_pct": "esc_dpp_pid",
            "kmt_pid_pct": "esc_kmt_pid",
            "tpp_pid_pct": "esc_tpp_pid",
            "ind_pid_pct": "esc_ind_pid",
            "indep_lean_pct": "esc_indep_lean",
            "unif_lean_pct": "esc_unif_lean",
        }
    )
    for c in [
        "esc_dpp_pid",
        "esc_kmt_pid",
        "esc_tpp_pid",
        "esc_ind_pid",
        "esc_indep_lean",
        "esc_unif_lean",
    ]:
        esc[c] = esc[c] / 100.0
    esc = esc.sort_values("year")
    esc["esc_dpp_delta"] = esc["esc_dpp_pid"].diff()
    esc["esc_kmt_delta"] = esc["esc_kmt_pid"].diff()
    # first year: 0 delta
    esc["esc_dpp_delta"] = esc["esc_dpp_delta"].fillna(0.0)
    esc["esc_kmt_delta"] = esc["esc_kmt_delta"].fillna(0.0)
    return esc[
        [
            "year",
            "proxy_window",
            "proxy_quality",
            "source_note",
            *ESC_COLS,
        ]
    ]


def add_geo_dummies(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    g = d["geo_region"].fillna("other")
    d["geo_north"] = (g == "north").astype(float)
    d["geo_central"] = (g == "central").astype(float)
    d["geo_south"] = (g == "south").astype(float)
    d["geo_east"] = (g == "east").astype(float)
    d["geo_islands"] = (g == "islands").astype(float)
    return d


def add_spatial_lags(df: pd.DataFrame, share_cols: list[str]) -> pd.DataFrame:
    """Same-county leave-one-out mean of lagged shares (spatial lag of history)."""
    d = df.copy()
    for col in share_cols:
        lag_col = f"prev_{col}" if not col.startswith("prev_") else col
        if lag_col not in d.columns:
            continue
        out_name = f"spatial_lag_{lag_col}"
        # group by year + county: leave-one-out mean
        vals = []
        for (yr, county), grp in d.groupby(["year", "county_name"], sort=False):
            s = grp[lag_col]
            n = len(s)
            if n <= 1:
                loo = pd.Series([np.nan] * n, index=grp.index)
            else:
                total = s.sum(skipna=True)
                cnt = s.notna().sum()
                loo = s.apply(
                    lambda x: (total - (x if pd.notna(x) else 0))
                    / max(cnt - (1 if pd.notna(x) else 0), 1)
                    if cnt > (1 if pd.notna(x) else 0)
                    else np.nan
                )
            vals.append(loo)
        d[out_name] = pd.concat(vals).reindex(d.index)
    return d


def national_means(df: pd.DataFrame, party: str) -> pd.Series:
    """Valid-vote-weighted national share by year."""
    share = f"{party}_share"
    w = df.groupby("year").apply(
        lambda g: (g[share] * g["valid_votes"]).sum() / g["valid_votes"].sum(),
        include_groups=False,
    )
    return w


def enrich_panel(df: pd.DataFrame, parties=("dpp", "kmt", "tpp")) -> pd.DataFrame:
    d = add_geo_dummies(df)
    esc = load_esc_proxy()
    d = d.merge(
        esc.drop(columns=["proxy_window", "proxy_quality", "source_note"], errors="ignore"),
        on="year",
        how="left",
    )
    prev_cols = [f"prev_{p}_share" for p in parties if f"prev_{p}_share" in d.columns]
    # also third for TPP geography
    if "prev_third_share" in d.columns:
        prev_cols.append("prev_third_share")
    d = add_spatial_lags(d, [c.replace("prev_", "") for c in prev_cols])

    for p in parties:
        share = f"{p}_share"
        if share not in d.columns:
            continue
        nat = national_means(d.dropna(subset=[share]), p)
        d[f"{p}_nat"] = d["year"].map(nat)
        d[f"{p}_demeaned"] = d[share] - d[f"{p}_nat"]
        if f"prev_{p}_share" in d.columns:
            # previous demeaned: prev share minus previous year's national
            prev_nat = nat.shift(1)  # index is year — remap carefully
            year_to_prev_nat = {
                y: nat.get(y_prev)
                for y, y_prev in zip(sorted(nat.index), [None] + list(sorted(nat.index))[:-1])
            }
            # better: map year -> national of previous election present in data
            years = sorted(d["year"].unique())
            prev_year = {years[i]: years[i - 1] for i in range(1, len(years))}
            d[f"prev_{p}_nat"] = d["year"].map(lambda y: nat.get(prev_year.get(y)))
            d[f"prev_{p}_demeaned"] = d[f"prev_{p}_share"] - d[f"prev_{p}_nat"]
            # swing already may exist
            if f"{p}_swing" not in d.columns:
                d[f"{p}_swing"] = d[share] - d[f"prev_{p}_share"]
            # demeaned swing = change in local relative position
            d[f"{p}_demeaned_swing"] = d[f"{p}_demeaned"] - d[f"prev_{p}_demeaned"]
    return d


def feature_sets_for(party: str, include_spatial=True, include_esc=True) -> dict:
    base_hist = [
        f"prev_{party}_share",
        f"prev_dpp_share",
        f"prev_kmt_share",
        "prev_turnout",
    ]
    # party-specific swing history if available
    extra = []
    if party == "dpp":
        extra = ["prev_dpp_swing", "prev_kmt_share", "dpp_kmt_margin_prev"]
    elif party == "kmt":
        extra = ["prev_kmt_swing", "prev_dpp_share", "dpp_kmt_margin_prev"]
    elif party == "tpp":
        extra = ["prev_third_share", "prev_dpp_share", "prev_kmt_share", "dpp_kmt_margin_prev"]

    hist = []
    for c in base_hist + extra:
        if c not in hist:
            hist.append(c)

    spatial = [f"spatial_lag_prev_{party}_share"]
    if party == "tpp":
        spatial = ["spatial_lag_prev_third_share", "spatial_lag_prev_dpp_share", "spatial_lag_prev_kmt_share"]

    sets = {
        "History": [c for c in hist],
        "History+Socio": hist + SOCIO,
        "History+Socio+Geo": hist + SOCIO + GEO_COLS,
    }
    if include_spatial:
        sets["History+Socio+Geo+Spatial"] = hist + SOCIO + GEO_COLS + spatial
    if include_esc:
        sets["History+Socio+Geo+Spatial+ESC"] = (
            hist + SOCIO + GEO_COLS + spatial + ESC_COLS
        )
        sets["All_improved"] = hist + SOCIO + GEO_COLS + spatial + ESC_COLS
    else:
        sets["All_improved"] = hist + SOCIO + GEO_COLS + (spatial if include_spatial else [])
    # dedupe while preserving order
    for k, v in list(sets.items()):
        seen = set()
        out = []
        for c in v:
            if c not in seen:
                seen.add(c)
                out.append(c)
        sets[k] = out
    return sets


def backtest_target(
    data: pd.DataFrame,
    party: str,
    target: str,
    feats: list[str],
    race: str = "presidential",
) -> list[dict]:
    d = data.copy()
    years = sorted(d["year"].unique())
    models = get_models()
    results = []

    share_col = f"{party}_share"
    prev_share = f"prev_{party}_share"
    demeaned = f"{party}_demeaned"
    prev_demeaned = f"prev_{party}_demeaned"
    swing_col = f"{party}_swing"

    for test_year in years:
        train = d[d["year"] < test_year]
        test = d[d["year"] == test_year]
        # need target defined
        if target not in test.columns:
            continue
        need_cols = [target] + [c for c in feats if c in d.columns]
        # for TPP presidential only 2024 has share; skip early years without target
        train_ok = train.dropna(subset=[target]).copy()
        test_ok = test.dropna(subset=[target]).copy()
        if len(test_ok) < 20:
            continue

        # Baselines depend on target type
        y_test = test_ok[target]

        if target.endswith("_swing") or target == swing_col:
            # ZeroSwing
            pred0 = np.zeros(len(test_ok))
            results.append(
                {
                    "race": race,
                    "party": party.upper(),
                    "target": "swing",
                    "test_year": test_year,
                    "model": "ZeroSwing",
                    "feature_set": "baseline",
                    "n_train": len(train_ok),
                    "n_test": len(test_ok),
                    **metrics(y_test, pred0),
                }
            )
            # PreviousSwing if available
            prev_sw = f"prev_{party}_swing"
            if prev_sw in test_ok.columns and test_ok[prev_sw].notna().sum() > 20:
                mask = test_ok[prev_sw].notna()
                results.append(
                    {
                        "race": race,
                        "party": party.upper(),
                        "target": "swing",
                        "test_year": test_year,
                        "model": "PreviousSwing",
                        "feature_set": "baseline",
                        "n_train": len(train_ok),
                        "n_test": int(mask.sum()),
                        **metrics(y_test[mask], test_ok.loc[mask, prev_sw]),
                    }
                )
        elif target.endswith("_demeaned") or target == demeaned:
            # PreviousDemeaned
            if prev_demeaned in test_ok.columns:
                mask = test_ok[prev_demeaned].notna()
                if mask.sum() >= 20:
                    results.append(
                        {
                            "race": race,
                            "party": party.upper(),
                            "target": "demeaned",
                            "test_year": test_year,
                            "model": "PreviousDemeaned",
                            "feature_set": "baseline",
                            "n_train": len(train_ok),
                            "n_test": int(mask.sum()),
                            **metrics(
                                y_test[mask], test_ok.loc[mask, prev_demeaned]
                            ),
                        }
                    )
            # ZeroDemeaned (predict at national mean)
            results.append(
                {
                    "race": race,
                    "party": party.upper(),
                    "target": "demeaned",
                    "test_year": test_year,
                    "model": "ZeroDemeaned",
                    "feature_set": "baseline",
                    "n_train": len(train_ok),
                    "n_test": len(test_ok),
                    **metrics(y_test, np.zeros(len(test_ok))),
                }
            )
        elif target == share_col or target == "level":
            if prev_share in test_ok.columns:
                mask = test_ok[prev_share].notna()
                if mask.sum() >= 20:
                    results.append(
                        {
                            "race": race,
                            "party": party.upper(),
                            "target": "level",
                            "test_year": test_year,
                            "model": "PreviousShare",
                            "feature_set": "baseline",
                            "n_train": len(train_ok),
                            "n_test": int(mask.sum()),
                            **metrics(y_test[mask], test_ok.loc[mask, prev_share]),
                        }
                    )

        if len(train_ok) < 50:
            continue

        use_feats = [c for c in feats if c in train_ok.columns or c in test_ok.columns]
        tr, te = impute(train_ok, test_ok, use_feats)
        tr = tr.dropna(subset=[target])
        te = te.dropna(subset=[target])
        # drop rows still missing required feats after impute (shouldn't)
        for mname, model in get_models().items():
            try:
                Xtr = tr[use_feats].values
                ytr = tr[target].values
                Xte = te[use_feats].values
                mdl = model
                # re-init each time
                if mname == "Ridge":
                    mdl = Pipeline(
                        [("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]
                    )
                elif mname == "RandomForest":
                    mdl = RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=3,
                        random_state=SEED,
                        n_jobs=-1,
                    )
                else:
                    mdl = HistGradientBoostingRegressor(
                        max_depth=5, learning_rate=0.06, random_state=SEED
                    )
                mdl.fit(Xtr, ytr)
                pred = mdl.predict(Xte)
                tgt_label = (
                    "swing"
                    if "swing" in target
                    else ("demeaned" if "demeaned" in target else "level")
                )
                results.append(
                    {
                        "race": race,
                        "party": party.upper(),
                        "target": tgt_label,
                        "test_year": test_year,
                        "model": mname,
                        "feature_set": "All_improved",
                        "n_train": len(tr),
                        "n_test": len(te),
                        **metrics(te[target].values, pred),
                    }
                )
            except Exception as e:
                print(f"  fail {party} {target} {test_year} {mname}: {e}")
    return results


def ablation_spatial_esc(
    data: pd.DataFrame, party: str, target: str, race: str, test_year: int = 2024
) -> list[dict]:
    sets = feature_sets_for(party, include_spatial=True, include_esc=True)
    # also versions without spatial / without esc
    sets["NoSpatial"] = [
        c for c in sets["All_improved"] if not c.startswith("spatial_lag_")
    ]
    sets["NoESC"] = [c for c in sets["All_improved"] if c not in ESC_COLS]
    sets["NoSpatialNoESC"] = [
        c
        for c in sets["All_improved"]
        if (not c.startswith("spatial_lag_")) and c not in ESC_COLS
    ]

    train = data[data["year"] < test_year].dropna(subset=[target])
    test = data[data["year"] == test_year].dropna(subset=[target])
    if len(train) < 50 or len(test) < 20:
        return []
    rows = []
    for sname, feats in sets.items():
        use = [c for c in feats if c in data.columns]
        if not use:
            continue
        tr, te = impute(train, test, use)
        Xtr, ytr = tr[use].values, tr[target].values
        Xte, yte = te[use].values, te[target].values
        rf = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1
        )
        rf.fit(Xtr, ytr)
        pred = rf.predict(Xte)
        rows.append(
            {
                "race": race,
                "party": party.upper(),
                "target": "swing" if "swing" in target else "demeaned",
                "test_year": test_year,
                "feature_set": sname,
                "model": "RandomForest",
                "n_features": len(use),
                **metrics(yte, pred),
            }
        )
    return rows


def importance_rf(data, party, target, feats, race, test_year=2024):
    train = data[data["year"] < test_year].dropna(subset=[target])
    test = data[data["year"] == test_year].dropna(subset=[target])
    use = [c for c in feats if c in data.columns]
    if len(train) < 50 or len(test) < 20 or not use:
        return []
    tr, te = impute(train, test, use)
    rf = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1
    )
    rf.fit(tr[use], tr[target])
    # permutation on test
    base = mean_absolute_error(te[target], rf.predict(te[use]))
    rows = []
    rng = np.random.default_rng(SEED)
    for i, col in enumerate(use):
        Xp = te[use].copy()
        Xp[col] = rng.permutation(Xp[col].values)
        mae = mean_absolute_error(te[target], rf.predict(Xp))
        rows.append(
            {
                "race": race,
                "party": party.upper(),
                "target": "swing" if "swing" in target else "demeaned",
                "feature": col,
                "rf_impurity_importance": float(rf.feature_importances_[i]),
                "perm_delta_MAE": float(mae - base),
                "base_MAE": float(base),
            }
        )
    return rows


def conformal_swing(
    data: pd.DataFrame, party: str, feats: list[str], race: str, test_year=2024, alpha=0.10
):
    target = f"{party}_swing"
    years = sorted(data["year"].unique())
    if test_year not in years:
        return None, None
    # calibration = last train year before test
    train_years = [y for y in years if y < test_year]
    if len(train_years) < 2:
        return None, None
    cal_year = train_years[-1]
    fit_years = [y for y in train_years if y < cal_year]
    if not fit_years:
        return None, None

    use = [c for c in feats if c in data.columns]
    fit = data[data["year"].isin(fit_years)].dropna(subset=[target])
    cal = data[data["year"] == cal_year].dropna(subset=[target])
    te = data[data["year"] == test_year].dropna(subset=[target])
    if min(len(fit), len(cal), len(te)) < 30:
        return None, None

    fit_i, cal_i = impute(fit, cal, use)
    _, te_i = impute(fit, te, use)
    rf = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1
    )
    rf.fit(fit_i[use], fit_i[target])
    cal_resid = np.abs(cal_i[target].values - rf.predict(cal_i[use]))
    q = float(np.quantile(cal_resid, 1 - alpha))
    pred = rf.predict(te_i[use])
    lo, hi = pred - q, pred + q
    y = te_i[target].values
    cover = float(np.mean((y >= lo) & (y <= hi)))
    meta = {
        "race": race,
        "party": party.upper(),
        "target": "swing",
        "test_year": test_year,
        "cal_year": cal_year,
        "fit_years": fit_years,
        "alpha": alpha,
        "nominal_coverage": 1 - alpha,
        "empirical_coverage": cover,
        "interval_halfwidth_q": q,
        "n_fit": len(fit_i),
        "n_cal": len(cal_i),
        "n_test": len(te_i),
        "MAE_point": float(mean_absolute_error(y, pred)),
        "note": "naive split conformal on SWING; fails if distribution shift between cal and test",
    }
    intervals = pd.DataFrame(
        {
            "region": te_i["region"].values,
            "year": test_year,
            "party": party.upper(),
            "y_true_swing": y,
            "y_pred_swing": pred,
            "lo": lo,
            "hi": hi,
            "covered": (y >= lo) & (y <= hi),
        }
    )
    # leave-one-election residual quantile (pooled absolute residuals from each OOT fold)
    oot_abs = []
    for ty in years:
        trn = data[data["year"] < ty].dropna(subset=[target])
        tst = data[data["year"] == ty].dropna(subset=[target])
        if len(trn) < 50 or len(tst) < 20:
            continue
        trn_i, tst_i = impute(trn, tst, use)
        m = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, random_state=SEED, n_jobs=-1
        )
        m.fit(trn_i[use], trn_i[target])
        oot_abs.append(np.abs(tst_i[target].values - m.predict(tst_i[use])))
    if oot_abs:
        all_r = np.concatenate(oot_abs)
        meta["leave_one_election_resid_q90"] = float(np.quantile(all_r, 0.90))
        meta["leave_one_election_resid_q50"] = float(np.quantile(all_r, 0.50))
    return meta, intervals


def reconstruct_level_from_swing_mae(data, party, feats, race, test_year=2024):
    """Predict swing, add to prev share → level MAE vs PreviousShare."""
    target = f"{party}_swing"
    share = f"{party}_share"
    prev = f"prev_{party}_share"
    train = data[data["year"] < test_year].dropna(subset=[target, prev])
    test = data[data["year"] == test_year].dropna(subset=[target, prev, share])
    if len(train) < 50 or len(test) < 20:
        return None
    use = [c for c in feats if c in data.columns]
    tr, te = impute(train, test, use)
    rf = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1
    )
    rf.fit(tr[use], tr[target])
    pred_sw = rf.predict(te[use])
    pred_level = te[prev].values + pred_sw
    zero_level = te[prev].values  # ZeroSwing → PreviousShare
    return {
        "race": race,
        "party": party.upper(),
        "test_year": test_year,
        "MAE_level_PreviousShare": float(
            mean_absolute_error(te[share], zero_level)
        ),
        "MAE_level_from_RF_swing": float(
            mean_absolute_error(te[share], pred_level)
        ),
        "MAE_swing_ZeroSwing": float(
            mean_absolute_error(te[target], np.zeros(len(te)))
        ),
        "MAE_swing_RF": float(mean_absolute_error(te[target], pred_sw)),
        "improved_vs_PreviousShare": float(
            mean_absolute_error(te[share], zero_level)
            - mean_absolute_error(te[share], pred_level)
        ),
    }


def run_race(path: Path, race: str):
    print(f"\n=== {race} ===")
    raw = pd.read_csv(path)
    d = enrich_panel(raw)
    d.to_csv(OUT / f"harmonized_{race}_improved.csv", index=False)

    all_results = []
    all_ablation = []
    all_imp = []
    recon = []
    conformal_metas = []
    conformal_frames = []

    parties_swing = ["dpp", "kmt"]
    # TPP swing only if prev exists (legislative 2024 has prev from 2020)
    if d["tpp_share"].notna().any() and d.get("prev_tpp_share", pd.Series(dtype=float)).notna().any():
        # check any non-null prev_tpp
        if "prev_tpp_share" in d.columns and d["prev_tpp_share"].notna().sum() > 50:
            parties_swing.append("tpp")

    for party in ["dpp", "kmt", "tpp"]:
        if f"{party}_share" not in d.columns:
            continue
        if d[f"{party}_share"].notna().sum() < 50 and party == "tpp":
            # still run 2024-only baselines later
            pass

        sets = feature_sets_for(party)
        feats = sets["All_improved"]
        # ensure columns exist
        for c in feats:
            if c not in d.columns:
                d[c] = np.nan

        # SWING
        if f"{party}_swing" in d.columns and d[f"{party}_swing"].notna().sum() > 50:
            print(f"  {party} swing...")
            all_results.extend(
                backtest_target(d, party, f"{party}_swing", feats, race=race)
            )
            all_ablation.extend(
                ablation_spatial_esc(d, party, f"{party}_swing", race)
            )
            all_imp.extend(
                importance_rf(d, party, f"{party}_swing", feats, race)
            )
            meta, iv = conformal_swing(d, party, feats, race)
            if meta:
                conformal_metas.append(meta)
                conformal_frames.append(iv)
            r = reconstruct_level_from_swing_mae(d, party, feats, race)
            if r:
                recon.append(r)

        # DEMEANED
        if f"{party}_demeaned" in d.columns and d[f"{party}_demeaned"].notna().sum() > 50:
            # need prev for meaningful OOT with history features
            if f"prev_{party}_share" in d.columns:
                print(f"  {party} demeaned...")
                all_results.extend(
                    backtest_target(
                        d, party, f"{party}_demeaned", feats, race=race
                    )
                )
                all_ablation.extend(
                    ablation_spatial_esc(d, party, f"{party}_demeaned", race)
                )

        # LEVEL (for comparison table)
        if f"{party}_share" in d.columns and f"prev_{party}_share" in d.columns:
            sub = d.dropna(subset=[f"{party}_share", f"prev_{party}_share"])
            if len(sub) > 50:
                print(f"  {party} level...")
                all_results.extend(
                    backtest_target(d, party, f"{party}_share", feats, race=race)
                )

    # TPP presidential special: no swing history — report 2024 demeaned/level with proxies
    if race == "presidential" and "tpp_share" in d.columns:
        te = d[d["year"] == 2024].dropna(subset=["tpp_share"])
        if len(te):
            # baseline PrevThirdPartyProxy
            if "prev_third_share" in te.columns:
                m = metrics(te["tpp_share"], te["prev_third_share"].fillna(0))
                all_results.append(
                    {
                        "race": race,
                        "party": "TPP",
                        "target": "level",
                        "test_year": 2024,
                        "model": "PrevThirdPartyProxy",
                        "feature_set": "baseline",
                        "n_train": 0,
                        "n_test": len(te),
                        **m,
                    }
                )
            all_results.append(
                {
                    "race": race,
                    "party": "TPP",
                    "target": "level",
                    "test_year": 2024,
                    "model": "ZeroTPP",
                    "feature_set": "baseline",
                    "n_train": 0,
                    "n_test": len(te),
                    **metrics(te["tpp_share"], np.zeros(len(te))),
                }
            )
            # RF using socio+geo+spatial+ESC+prev structure (train on nothing temporal —
            # use 2020 as X with tpp target unavailable; instead LOCO already done elsewhere.
            # Here: transfer from prev_third as target proxy trained on 2020 then predict 2024)
            feats_tpp = feature_sets_for("tpp")["All_improved"]
            use = [c for c in feats_tpp if c in d.columns]
            tr = d[d["year"] == 2020].copy()
            # train to predict prev_third / others as stand-in — skip fake training
            # Report ESC-informed constant: national ESC TPP pid as township prediction
            if "esc_tpp_pid" in te.columns:
                all_results.append(
                    {
                        "race": race,
                        "party": "TPP",
                        "target": "level",
                        "test_year": 2024,
                        "model": "ESC_national_TPP_pid_PROXY",
                        "feature_set": "ESC_only",
                        "n_train": 0,
                        "n_test": len(te),
                        **metrics(te["tpp_share"], te["esc_tpp_pid"].fillna(0)),
                        "note": "national constant PROXY — cannot capture township variation",
                    }
                )

    return {
        "results": all_results,
        "ablation": all_ablation,
        "importance": all_imp,
        "recon": recon,
        "conformal_metas": conformal_metas,
        "conformal_frames": conformal_frames,
        "panel": d,
    }


def make_charts(results_df, ablation_df, recon_df):
    # 2024 swing MAE by party (presidential)
    sub = results_df[
        (results_df["race"] == "presidential")
        & (results_df["target"] == "swing")
        & (results_df["test_year"] == 2024)
    ]
    if len(sub):
        fig, ax = plt.subplots(figsize=(9, 5))
        parties = ["DPP", "KMT"]
        models = ["ZeroSwing", "PreviousSwing", "Ridge", "RandomForest", "HistGB"]
        x = np.arange(len(parties))
        width = 0.15
        for i, m in enumerate(models):
            vals = []
            for p in parties:
                r = sub[(sub["party"] == p) & (sub["model"] == m)]
                vals.append(r["MAE"].values[0] if len(r) else np.nan)
            ax.bar(x + i * width, vals, width, label=m)
        ax.set_xticks(x + 2 * width)
        ax.set_xticklabels(parties)
        ax.set_ylabel("MAE (swing)")
        ax.set_title("2024 Presidential SWING MAE — improved features")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(CHARTS / "pres_2024_swing_mae.png", dpi=140)
        plt.close()

    # ablation
    if len(ablation_df):
        a = ablation_df[
            (ablation_df["race"] == "presidential")
            & (ablation_df["target"] == "swing")
            & (ablation_df["party"] == "DPP")
        ]
        if len(a):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(a["feature_set"], a["MAE"])
            ax.set_xlabel("MAE")
            ax.set_title("DPP presidential swing ablation (holdout 2024)")
            fig.tight_layout()
            fig.savefig(CHARTS / "ablation_dpp_swing_2024.png", dpi=140)
            plt.close()

    # reconstruct comparison
    if len(recon_df):
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [f"{r.party}" for r in recon_df.itertuples()]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, recon_df["MAE_level_PreviousShare"], 0.4, label="PreviousShare")
        ax.bar(
            x + 0.2,
            recon_df["MAE_level_from_RF_swing"],
            0.4,
            label="RF swing→level",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Level MAE")
        ax.set_title("2024 Level MAE: baseline vs improved swing reconstruction")
        ax.legend()
        fig.tight_layout()
        fig.savefig(CHARTS / "level_from_swing_2024.png", dpi=140)
        plt.close()


def before_after_table(results_df):
    """Compare ZeroSwing / PreviousShare vs best improved model for 2024."""
    rows = []
    for race in results_df["race"].unique():
        for party in ["DPP", "KMT", "TPP"]:
            for target in ["swing", "demeaned", "level"]:
                sub = results_df[
                    (results_df["race"] == race)
                    & (results_df["party"] == party)
                    & (results_df["target"] == target)
                    & (results_df["test_year"] == 2024)
                ]
                if not len(sub):
                    continue
                base_names = {
                    "swing": ["ZeroSwing", "PreviousSwing"],
                    "demeaned": ["PreviousDemeaned", "ZeroDemeaned"],
                    "level": ["PreviousShare", "PrevThirdPartyProxy", "ZeroTPP"],
                }
                baselines = sub[sub["model"].isin(base_names[target])]
                models = sub[~sub["model"].isin(base_names[target])]
                best_base = (
                    baselines.loc[baselines["MAE"].idxmin()]
                    if len(baselines)
                    else None
                )
                best_ml = (
                    models.loc[models["MAE"].idxmin()] if len(models) else None
                )
                rows.append(
                    {
                        "race": race,
                        "party": party,
                        "target": target,
                        "best_baseline": best_base["model"] if best_base is not None else None,
                        "baseline_MAE": best_base["MAE"] if best_base is not None else None,
                        "best_improved_model": best_ml["model"]
                        if best_ml is not None
                        else None,
                        "improved_MAE": best_ml["MAE"] if best_ml is not None else None,
                        "delta_MAE_baseline_minus_improved": (
                            (best_base["MAE"] - best_ml["MAE"])
                            if best_base is not None and best_ml is not None
                            else None
                        ),
                        "improved_wins": (
                            bool(best_ml["MAE"] < best_base["MAE"])
                            if best_base is not None and best_ml is not None
                            else None
                        ),
                    }
                )
    return pd.DataFrame(rows)


def main():
    pres = run_race(PRES_PATH, "presidential")
    leg = run_race(LEG_PATH, "legislative_district")

    results = pd.DataFrame(pres["results"] + leg["results"])
    ablation = pd.DataFrame(pres["ablation"] + leg["ablation"])
    importance = pd.DataFrame(pres["importance"] + leg["importance"])
    recon = pd.DataFrame(pres["recon"] + leg["recon"])
    conf_meta = pd.DataFrame(pres["conformal_metas"] + leg["conformal_metas"])

    results.to_csv(OUT / "backtest_results_improved.csv", index=False)
    ablation.to_csv(OUT / "ablation_spatial_esc.csv", index=False)
    importance.to_csv(OUT / "feature_importance_improved.csv", index=False)
    recon.to_csv(OUT / "level_from_swing_comparison.csv", index=False)
    conf_meta.to_csv(OUT / "conformal_swing_meta.csv", index=False)

    if pres["conformal_frames"]:
        pd.concat(pres["conformal_frames"]).to_csv(
            OUT / "conformal_swing_intervals_pres_2024.csv", index=False
        )
    if leg["conformal_frames"]:
        pd.concat(leg["conformal_frames"]).to_csv(
            OUT / "conformal_swing_intervals_leg_2024.csv", index=False
        )

    ba = before_after_table(results)
    ba.to_csv(OUT / "before_after_2024.csv", index=False)

    # copy legislative-specific
    results[results["race"] == "legislative_district"].to_csv(
        LEG_OUT / "backtest_results.csv", index=False
    )

    make_charts(results, ablation, recon)

    # summary JSON
    summary = {
        "improvements": [
            "swing and demeaned targets",
            "ESC NCCU national PROXY (not TEDS, not county)",
            "geo_region dummies",
            "same-county spatial lag of lagged shares",
            "separate DPP/KMT/TPP heads",
            "conformal on swing",
        ],
        "esc_proxy_path": str(PROXY_PATH),
        "esc_limitation": (
            "ESC public pages give national trend charts only; Excel needs application; "
            "no county-level party ID. PROXY values are approximate / news-cited; "
            "NEVER treat as TEDS microdata."
        ),
        "n_backtest_rows": len(results),
        "conformal": conf_meta.to_dict(orient="records"),
        "before_after_2024_head": ba.head(20).to_dict(orient="records"),
    }
    with open(OUT / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== BEFORE/AFTER 2024 ===")
    print(ba.to_string(index=False))
    print("\n=== CONFORMAL SWING ===")
    print(conf_meta.to_string(index=False) if len(conf_meta) else "none")
    print("\nDone. Outputs in", OUT)


if __name__ == "__main__":
    main()
