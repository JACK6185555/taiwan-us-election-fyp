#!/usr/bin/env python3
"""Taiwan FYP Core — real-data out-of-time regional vote-share forecasting.

Follows US–Taiwan Election Forecasting FYP guide (sections 4.1, 6, 10–16, Demo 6).
Unit: township/district × election year. Target: DPP presidential share.
Uses ONLY real CEC / MOI / fiscal data under /workspace/taiwan-election-data.
Does NOT invent unemployment or poll_index values.
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
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path("/workspace/taiwan-election-data")
CLEAN = ROOT / "cleaned"
RAW_CEC = ROOT / "raw" / "cec" / "extracted"
RAW_MOI = ROOT / "raw" / "moi"
OUT = ROOT / "fyp_taiwan"
CHARTS = OUT / "charts"
OUT.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

SEED = 42
DPP_PARTY = {16}
KMT_PARTY = {1}

NORTH = {"臺北市", "新北市", "基隆市", "桃園市", "新竹市", "新竹縣", "宜蘭縣"}
CENTRAL = {"臺中市", "苗栗縣", "彰化縣", "南投縣", "雲林縣"}
SOUTH = {"高雄市", "臺南市", "嘉義市", "嘉義縣", "屏東縣", "澎湖縣"}
EAST = {"花蓮縣", "臺東縣"}
ISLANDS = {"金門縣", "連江縣"}

ELECTION_DIRS = {
    2008: RAW_CEC / "2008總統",
    2012: RAW_CEC / "2012總統立委" / "總統",
    2016: RAW_CEC / "2016總統立委" / "總統",
    2020: RAW_CEC / "2020總統立委" / "總統",
    2024: RAW_CEC / "2024總統立委" / "總統",
}

# Covariate ROC years (closest available at/before election)
INCOME_ROC = {2012: 105, 2016: 105, 2020: 109, 2024: 112}
POP_ROC = {2012: 108, 2016: 108, 2020: 109, 2024: 112}
EDU_FILE_ROC = {2012: 108, 2016: 108, 2020: 112, 2024: 112}

# Harmonized feature columns used in models (exclude NaN-only)
MODEL_FEATURES = [
    "prev_vote_share",
    "prev_turnout",
    "prev_swing",
    "median_income",
    "urbanization",
    "education_rate",
    "log_people",
]

FEATURE_SETS = {
    "History": ["prev_vote_share", "prev_turnout", "prev_swing"],
    "History+Economy": ["prev_vote_share", "prev_turnout", "prev_swing", "median_income"],
    "History+Demographics": [
        "prev_vote_share",
        "prev_turnout",
        "prev_swing",
        "urbanization",
        "education_rate",
        "log_people",
    ],
    "History+Economy+Demo": [
        "prev_vote_share",
        "prev_turnout",
        "prev_swing",
        "median_income",
        "urbanization",
        "education_rate",
        "log_people",
    ],
    "All_available": MODEL_FEATURES,  # no unemployment / poll_index
}


def _clean_cell(x) -> str:
    s = str(x).strip().strip('"').strip("'").strip()
    if s.startswith("'"):
        s = s[1:]
    return s


def read_cec_csv(path: Path) -> pd.DataFrame:
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
    out = pd.DataFrame(
        {
            "ticket": df.iloc[:, 5].astype(str).str.lstrip("0").replace("", "0"),
            "cand_name": df.iloc[:, 6],
            "party_code": pd.to_numeric(df.iloc[:, 7], errors="coerce").fillna(-1).astype(int),
        }
    )
    return out.drop_duplicates("ticket", keep="first")


def load_elctks(path: Path) -> pd.DataFrame:
    df = read_cec_csv(path)
    df = df.iloc[:, :10]
    df.columns = [
        "prv", "city", "dept", "twn", "vil", "station", "ticket", "votes", "share", "elected"
    ]
    df["ticket"] = df["ticket"].astype(str).str.lstrip("0").replace("", "0")
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)
    return df


def load_elprof(path: Path) -> pd.DataFrame:
    """Polling-station electorate profile: ballots (col8), eligible (col9), turnout% (col18)."""
    df = read_cec_csv(path)
    out = pd.DataFrame(
        {
            "prv": df.iloc[:, 0],
            "city": df.iloc[:, 1],
            "dept": df.iloc[:, 2],
            "twn": df.iloc[:, 3],
            "vil": df.iloc[:, 4],
            "station": df.iloc[:, 5],
            "ballots": pd.to_numeric(df.iloc[:, 8], errors="coerce"),
            "eligible": pd.to_numeric(df.iloc[:, 9], errors="coerce"),
        }
    )
    return out


def county_name_from_elbase(elbase: pd.DataFrame, prv: str, city: str) -> str:
    row = elbase[
        (elbase.prv == prv) & (elbase.city == city) & (elbase.twn == "000") & (elbase.vil == "0000")
    ]
    if len(row):
        name = row.iloc[0]["name"]
        if name not in ("臺灣省", "福建省", "全國"):
            return name
    return f"{prv}-{city}"


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


def _paths_for_year(year: int):
    d = ELECTION_DIRS[year]
    if year == 2016:
        return d / "elbase_P1.csv", d / "elcand_P1.csv", d / "elctks_P1.csv", d / "elprof_P1.csv"
    return d / "elbase.csv", d / "elcand.csv", d / "elctks.csv", d / "elprof.csv"


def township_votes_and_turnout(year: int) -> pd.DataFrame:
    base_p, cand_p, ctks_p, prof_p = _paths_for_year(year)
    elbase = load_elbase(base_p)
    elcand = load_elcand(cand_p)
    elctks = load_elctks(ctks_p)
    elprof = load_elprof(prof_p)

    twn_names = elbase[(elbase.twn != "000") & (elbase.vil == "0000")][
        ["prv", "city", "twn", "name"]
    ].rename(columns={"name": "twn_name"})

    counties = []
    for (prv, city), _ in elbase.groupby(["prv", "city"]):
        nm = county_name_from_elbase(elbase, prv, city)
        if nm in ("臺灣省", "福建省", "全國"):
            continue
        counties.append({"prv": prv, "city": city, "county_name": nm})
    county_df = pd.DataFrame(counties).drop_duplicates(["prv", "city"])

    poll = elctks[elctks["station"] != "0000"].copy()
    agg = poll.groupby(["prv", "city", "twn", "ticket"], as_index=False)["votes"].sum()
    agg = agg.merge(elcand, on="ticket", how="left")
    agg = agg.merge(twn_names, on=["prv", "city", "twn"], how="left")
    agg = agg.merge(county_df, on=["prv", "city"], how="left")
    agg = agg.dropna(subset=["twn_name", "county_name"])
    agg = agg[agg["county_name"].isin(NORTH | CENTRAL | SOUTH | EAST | ISLANDS)]
    agg["township"] = agg["county_name"] + agg["twn_name"]

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
    meta = agg[["township", "county_name"]].drop_duplicates()
    out = out.merge(meta, on="township", how="left")

    # Turnout from elprof station rows
    prof = elprof[(elprof["station"] != "0000") & (elprof["vil"] != "0000")].copy()
    # Some years station is numeric without leading zeros; already string
    prof = prof.merge(twn_names, on=["prv", "city", "twn"], how="left")
    prof = prof.merge(county_df, on=["prv", "city"], how="left")
    prof = prof.dropna(subset=["twn_name", "county_name"])
    prof = prof[prof["county_name"].isin(NORTH | CENTRAL | SOUTH | EAST | ISLANDS)]
    prof["township"] = prof["county_name"] + prof["twn_name"]
    turn = (
        prof.groupby("township", as_index=False)
        .agg(ballots=("ballots", "sum"), eligible=("eligible", "sum"))
    )
    turn["turnout"] = turn["ballots"] / turn["eligible"].replace(0, np.nan)
    out = out.merge(turn[["township", "turnout", "eligible", "ballots"]], on="township", how="left")

    print(
        f"  {year}: townships={len(out)} national_dpp="
        f"{out['dpp_votes'].sum()/out['valid_votes'].sum():.4f} "
        f"nat_turnout={(turn['ballots'].sum()/turn['eligible'].sum()):.4f}"
    )
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
            "turnout",
            "eligible",
            "ballots",
        ]
    ]


def load_education_rate(roc_year: int) -> pd.DataFrame:
    path = RAW_MOI / f"ODRP024_village_edu_head_{roc_year}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    higher = {"博畢", "碩畢", "大畢", "專畢"}
    df["is_high"] = df["edu"].isin(higher)
    g = (
        df.groupby("site_id")
        .apply(
            lambda x: pd.Series(
                {
                    "edu_heads_total": x["headhousehold_count"].sum(),
                    "edu_heads_college_plus": x.loc[x["is_high"], "headhousehold_count"].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .rename(columns={"site_id": "township"})
    )
    g["education_rate"] = g["edu_heads_college_plus"] / g["edu_heads_total"].replace(0, np.nan)
    return g[["township", "education_rate"]]


def build_harmonized() -> pd.DataFrame:
    print("Aggregating CEC presidential votes + turnout...")
    vote_parts = []
    for y in (2008, 2012, 2016, 2020, 2024):
        vote_parts.append(township_votes_and_turnout(y))
    votes = pd.concat(vote_parts, ignore_index=True)

    income = pd.read_csv(CLEAN / "income_township_panel_full.csv")
    pop = pd.read_csv(CLEAN / "moi_township_pop_density_panel.csv")

    edu_cache = {}
    for ry in sorted(set(EDU_FILE_ROC.values())):
        edu_cache[ry] = load_education_rate(ry)

    # Modeling years: 2012+ (2008 only for lags)
    model_years = [2012, 2016, 2020, 2024]
    frames = []
    for y in model_years:
        vy = votes[votes.election_year == y].copy()
        prev_y = y - 4
        prev = votes[votes.election_year == prev_y][
            ["township", "dpp_share", "kmt_share", "turnout"]
        ].rename(
            columns={
                "dpp_share": "prev_vote_share",
                "kmt_share": "prev_kmt_share",
                "turnout": "prev_turnout",
            }
        )
        vy = vy.merge(prev, on="township", how="left")

        # prev_swing = V(t-1) - V(t-2)
        prev2_y = y - 8
        if prev2_y in votes.election_year.values:
            p2 = votes[votes.election_year == prev2_y][["township", "dpp_share"]].rename(
                columns={"dpp_share": "prev2_vote_share"}
            )
            vy = vy.merge(p2, on="township", how="left")
            vy["prev_swing"] = vy["prev_vote_share"] - vy["prev2_vote_share"]
        else:
            vy["prev_swing"] = np.nan

        # income
        iroc = INCOME_ROC[y]
        avail_i = sorted(income.roc_year.unique())
        if iroc not in avail_i:
            iroc = min(avail_i, key=lambda a: abs(a - iroc))
        inc = income[income.roc_year == iroc][["township", "mean_income", "tax_units"]].copy()
        # mean taxable income (千元) — used as median_income proxy; documented in README
        inc = inc.rename(columns={"mean_income": "median_income"})
        vy = vy.merge(inc, on="township", how="left")

        # density / urbanization
        proc = POP_ROC[y]
        avail_p = sorted(pop.roc_year.unique())
        if proc not in avail_p:
            proc = min(avail_p, key=lambda a: abs(a - proc))
        dens = pop[pop.roc_year == proc][
            ["township", "people_total", "area_km2", "population_density"]
        ]
        vy = vy.merge(dens, on="township", how="left")

        # education
        eroc = EDU_FILE_ROC[y]
        vy = vy.merge(edu_cache[eroc], on="township", how="left")

        vy["region"] = vy["county_name"].map(region_of)
        vy["income_roc_used"] = iroc
        vy["pop_roc_used"] = proc
        vy["edu_roc_used"] = eroc
        frames.append(vy)

    panel = pd.concat(frames, ignore_index=True)

    # Urbanization proxy: min-max of log1p(density) across panel (0–1)
    dens = pd.to_numeric(panel["population_density"], errors="coerce")
    logd = np.log1p(dens)
    panel["urbanization"] = (logd - logd.min()) / (logd.max() - logd.min())

    panel["median_income"] = pd.to_numeric(panel["median_income"], errors="coerce")
    panel["people_total"] = pd.to_numeric(panel["people_total"], errors="coerce")
    panel["log_people"] = np.log1p(panel["people_total"])
    panel["education_rate"] = pd.to_numeric(panel["education_rate"], errors="coerce")
    panel["prev_vote_share"] = pd.to_numeric(panel["prev_vote_share"], errors="coerce")
    panel["prev_turnout"] = pd.to_numeric(panel["prev_turnout"], errors="coerce")
    panel["prev_swing"] = pd.to_numeric(panel["prev_swing"], errors="coerce")

    # Targets
    panel["target_vote_share"] = panel["dpp_share"]
    panel["swing"] = panel["dpp_share"] - panel["prev_vote_share"]

    # Schema section 16 + extras
    panel["country"] = "TW"
    panel["region_id"] = panel["township"]  # harmonized "region" name
    panel["year"] = panel["election_year"]
    panel["unemployment"] = np.nan  # not available at township consistently
    panel["poll_index"] = np.nan  # no township-level poll series

    harm_cols = [
        "country",
        "region_id",
        "year",
        "prev_vote_share",
        "prev_turnout",
        "unemployment",
        "median_income",
        "urbanization",
        "education_rate",
        "poll_index",
        "target_vote_share",
        # extras (documented)
        "swing",
        "prev_swing",
        "log_people",
        "population_density",
        "county_name",
        "region",
        "valid_votes",
        "dpp_votes",
        "kmt_votes",
        "kmt_share",
        "turnout",
        "income_roc_used",
        "pop_roc_used",
        "edu_roc_used",
    ]
    # rename region_id -> region for section 16, keep geographic region as geo_region
    harm = panel.rename(columns={"region": "geo_region", "region_id": "region"})
    harm_cols_out = [
        "country",
        "region",
        "year",
        "prev_vote_share",
        "prev_turnout",
        "unemployment",
        "median_income",
        "urbanization",
        "education_rate",
        "poll_index",
        "target_vote_share",
        "swing",
        "prev_swing",
        "log_people",
        "population_density",
        "county_name",
        "geo_region",
        "valid_votes",
        "dpp_votes",
        "kmt_votes",
        "kmt_share",
        "turnout",
        "income_roc_used",
        "pop_roc_used",
        "edu_roc_used",
    ]
    harm = harm[harm_cols_out].copy()

    # Require core fields for modeling years
    complete = harm.dropna(
        subset=["prev_vote_share", "target_vote_share", "median_income", "urbanization"]
    )
    print(
        f"harmonized rows={len(harm)} complete_core={len(complete)} "
        f"years={sorted(harm.year.unique())}"
    )
    for y, g in harm.groupby("year"):
        nat = g["dpp_votes"].sum() / g["valid_votes"].sum()
        print(f"  year {y}: n={len(g)} national_dpp={nat:.4f}")

    harm.to_csv(OUT / "harmonized_tw_regions.csv", index=False)
    return harm


def get_models():
    return {
        "Ridge": Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1
        ),
        "HistGB": HistGradientBoostingRegressor(
            max_depth=5, learning_rate=0.06, random_state=SEED
        ),
    }


def prepare_xy(df: pd.DataFrame, feats: list[str]):
    """Drop rows with NaN in required features/target; fill prev_swing only if in feats with 0 after documenting."""
    use = df.copy()
    # prev_swing missing for earliest year only — median-impute within train later; here drop if NaN when required
    cols = feats + ["target_vote_share"]
    use = use.dropna(subset=[c for c in cols if c != "prev_swing"])
    if "prev_swing" in feats:
        # keep rows; impute later per split
        pass
    return use


def impute_prev_swing(train: pd.DataFrame, test: pd.DataFrame, feats: list[str]):
    tr, te = train.copy(), test.copy()
    if "prev_swing" in feats:
        fill = tr["prev_swing"].median()
        if pd.isna(fill):
            fill = 0.0
        tr["prev_swing"] = tr["prev_swing"].fillna(fill)
        te["prev_swing"] = te["prev_swing"].fillna(fill)
    # education_rate rare miss
    for c in feats:
        if c in ("education_rate", "prev_turnout", "log_people", "median_income", "urbanization"):
            fill = tr[c].median()
            tr[c] = tr[c].fillna(fill)
            te[c] = te[c].fillna(fill)
    return tr, te


def backtest(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    # Need prev_vote_share
    d = d.dropna(subset=["prev_vote_share", "target_vote_share"])
    years = sorted(d["year"].unique())
    models = get_models()
    feats = MODEL_FEATURES
    results = []

    # Out-of-time: every year with at least one earlier training year
    for test_year in years:
        train = d[d["year"] < test_year]
        test = d[d["year"] == test_year]
        if len(train) == 0 or len(test) == 0:
            continue
        tr, te = impute_prev_swing(train, test, feats)
        # drop remaining NaNs in features
        tr = tr.dropna(subset=feats + ["target_vote_share"])
        te = te.dropna(subset=feats + ["target_vote_share"])
        if len(tr) < 50 or len(te) < 20:
            print(f"  skip {test_year}: train={len(tr)} test={len(te)}")
            continue

        y_test = te["target_vote_share"]
        pred_base = te["prev_vote_share"]
        results.append(
            {
                "country": "TW",
                "test_year": test_year,
                "model": "PreviousElection",
                "n_train": len(tr),
                "n_test": len(te),
                "MAE": mean_absolute_error(y_test, pred_base),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, pred_base))),
                "R2": r2_score(y_test, pred_base),
            }
        )

        X_train, y_train = tr[feats], tr["target_vote_share"]
        X_test = te[feats]
        for name, model in models.items():
            m = clone(model)
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            results.append(
                {
                    "country": "TW",
                    "test_year": test_year,
                    "model": name,
                    "n_train": len(tr),
                    "n_test": len(te),
                    "MAE": mean_absolute_error(y_test, pred),
                    "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                    "R2": r2_score(y_test, pred),
                }
            )
        print(f"  backtest test_year={test_year} train_years={sorted(tr.year.unique())} n_test={len(te)}")

    return pd.DataFrame(results)


def ablation(data: pd.DataFrame, test_year: int = 2024) -> pd.DataFrame:
    d = data.dropna(subset=["prev_vote_share", "target_vote_share"])
    train = d[d["year"] < test_year]
    test = d[d["year"] == test_year]
    rows = []
    for name, feats in FEATURE_SETS.items():
        tr, te = impute_prev_swing(train, test, feats)
        tr = tr.dropna(subset=feats + ["target_vote_share"])
        te = te.dropna(subset=feats + ["target_vote_share"])
        model = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1
        )
        model.fit(tr[feats], tr["target_vote_share"])
        pred = model.predict(te[feats])
        mae = mean_absolute_error(te["target_vote_share"], pred)
        rmse = float(np.sqrt(mean_squared_error(te["target_vote_share"], pred)))
        rows.append(
            {
                "country": "TW",
                "test_year": test_year,
                "features": name,
                "n_features": len(feats),
                "feature_list": ",".join(feats),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2_score(te["target_vote_share"], pred),
            }
        )
        print(f"  ablation {name}: MAE={mae:.4f}")
    return pd.DataFrame(rows)


def feature_importance(data: pd.DataFrame, test_year: int = 2024) -> pd.DataFrame:
    d = data.dropna(subset=["prev_vote_share", "target_vote_share"])
    feats = MODEL_FEATURES
    train = d[d["year"] < test_year]
    test = d[d["year"] == test_year]
    tr, te = impute_prev_swing(train, test, feats)
    tr = tr.dropna(subset=feats + ["target_vote_share"])
    te = te.dropna(subset=feats + ["target_vote_share"])
    model = RandomForestRegressor(
        n_estimators=500, min_samples_leaf=3, random_state=SEED, n_jobs=-1
    )
    model.fit(tr[feats], tr["target_vote_share"])
    imp = pd.DataFrame({"feature": feats, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp["test_year"] = test_year
    imp["country"] = "TW"
    return imp


def conformal(data: pd.DataFrame, test_year: int = 2024, alpha: float = 0.10) -> tuple[pd.DataFrame, dict]:
    """Split conformal: train on years < calib, calib = last year before test."""
    d = data.dropna(subset=["prev_vote_share", "target_vote_share"]).copy()
    feats = MODEL_FEATURES
    years_before = sorted(d[d["year"] < test_year]["year"].unique())
    if len(years_before) < 2:
        raise ValueError("Need ≥2 years before test for conformal calibration")
    calibration_year = years_before[-1]
    train = d[d["year"] < calibration_year]
    calibration = d[d["year"] == calibration_year]
    test = d[d["year"] == test_year]

    tr, cal = impute_prev_swing(train, calibration, feats)
    _, te = impute_prev_swing(train, test, feats)
    # also fill using train medians already done
    tr = tr.dropna(subset=feats + ["target_vote_share"])
    cal = cal.dropna(subset=feats + ["target_vote_share"])
    te = te.dropna(subset=feats + ["target_vote_share"])

    model = RandomForestRegressor(
        n_estimators=500, min_samples_leaf=3, random_state=SEED, n_jobs=-1
    )
    model.fit(tr[feats], tr["target_vote_share"])
    cal_pred = model.predict(cal[feats])
    residuals = np.abs(cal["target_vote_share"].values - cal_pred)
    q = float(np.quantile(residuals, 1 - alpha))

    pred = model.predict(te[feats])
    lower, upper = pred - q, pred + q
    # clip to [0,1] for vote shares
    lower = np.clip(lower, 0, 1)
    upper = np.clip(upper, 0, 1)
    actual = te["target_vote_share"].values
    coverage = float(np.mean((actual >= lower) & (actual <= upper)))
    avg_width = float(np.mean(upper - lower))

    meta = {
        "calibration_year": int(calibration_year),
        "test_year": int(test_year),
        "alpha": alpha,
        "quantile": q,
        "coverage": coverage,
        "avg_width": avg_width,
        "n_train": len(tr),
        "n_calib": len(cal),
        "n_test": len(te),
        "target_coverage": 1 - alpha,
    }
    intervals = pd.DataFrame(
        {
            "country": "TW",
            "region": te["region"].values,
            "year": test_year,
            "actual": actual,
            "prediction": pred,
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
            "covered": (actual >= lower) & (actual <= upper),
        }
    )
    print(
        f"  conformal calib={calibration_year} q={q:.4f} "
        f"coverage={coverage:.4f} (target {1-alpha:.2f}) width={avg_width:.4f}"
    )
    return intervals, meta


def make_charts(back: pd.DataFrame, abl: pd.DataFrame, imp: pd.DataFrame):
    # MAE by year
    plt.figure(figsize=(9, 5))
    for model in back["model"].unique():
        m = back[back["model"] == model].sort_values("test_year")
        plt.plot(m["test_year"], m["MAE"], marker="o", label=model)
    plt.xlabel("Test election year")
    plt.ylabel("Vote-share MAE")
    plt.title("TW: Out-of-time forecast error (DPP share)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHARTS / "mae_by_year.png", dpi=150)
    plt.close()

    # Ablation
    abl_s = abl.sort_values("MAE")
    plt.figure(figsize=(8, 5))
    plt.barh(abl_s["features"], abl_s["MAE"], color="steelblue")
    plt.xlabel("MAE (test 2024)")
    plt.title("TW: Feature ablation (Random Forest)")
    plt.tight_layout()
    plt.savefig(CHARTS / "ablation_mae.png", dpi=150)
    plt.close()

    # Importance
    imp_s = imp.sort_values("importance")
    plt.figure(figsize=(8, 5))
    plt.barh(imp_s["feature"], imp_s["importance"], color="darkorange")
    plt.xlabel("RF impurity importance")
    plt.title("TW: Random Forest feature importance (train < 2024)")
    plt.tight_layout()
    plt.savefig(CHARTS / "feature_importance.png", dpi=150)
    plt.close()


def write_readme(back: pd.DataFrame, abl: pd.DataFrame, imp: pd.DataFrame, conf_meta: dict, harm: pd.DataFrame):
    nat = (
        harm.groupby("year")
        .apply(
            lambda g: pd.Series(
                {"national_dpp": g["dpp_votes"].sum() / g["valid_votes"].sum(), "n": len(g)}
            ),
            include_groups=False,
        )
        .reset_index()
    )
    lines = [
        "# Taiwan FYP Core — Regional Vote-Share Forecasting",
        "",
        "Real-data implementation of the FYP guide (sections 4.1, 6, 10–16, Demo 6) for **Taiwan only**.",
        "",
        "## Unit & target",
        "",
        "- **Unit**: township/district × election year (`country=TW`)",
        "- **Primary target**: `target_vote_share` = DPP presidential candidate vote share",
        "- **Secondary**: `swing` = V(t) − V(t−1)",
        "- **Years modeled**: 2012, 2016, 2020, 2024 (2008 used only for lags)",
        "",
        "## Feature mapping (section 16 schema)",
        "",
        "| Schema column | Taiwan source | Notes |",
        "|---------------|---------------|-------|",
        "| country | constant `TW` | |",
        "| region | `縣市名+鄉鎮市區名` | e.g. `臺北市松山區` |",
        "| year | election year | 2012–2024 |",
        "| prev_vote_share | CEC prior presidential DPP share | lag 4 years |",
        "| prev_turnout | CEC `elprof` ballots/eligible | prior election |",
        "| unemployment | **NaN** | no consistent township series — **excluded from models** |",
        "| median_income | 綜所稅鄉鎮 `平均數` (mean taxable income, 千元) | proxy for median; ROC years 105/109/112 |",
        "| urbanization | min-max of log1p(人口密度) | density proxy in [0,1] |",
        "| education_rate | ODRP024 戶長專科以上占比 | ROC 108→2012/2016; 112→2020/2024 |",
        "| poll_index | **NaN** | no township poll series — **excluded**; do not invent |",
        "| target_vote_share | CEC DPP presidential share | |",
        "",
        "Extra columns kept: `prev_swing`, `swing`, `log_people`, `geo_region`, vote counts, ROC year tags.",
        "",
        "## Missing / N/A",
        "",
        "- **Public opinion / poll_index**: omitted (no consistent township-level polls).",
        "- **Unemployment**: omitted.",
        "- **TEDS / party ID / identity**: not in this ecological model.",
        "",
        "## Methods",
        "",
        "- Models: Previous-election baseline, Ridge (+StandardScaler), Random Forest, HistGradientBoosting",
        "- Evaluation: **out-of-time only** (train years < test year); never random 80/20 as main eval",
        "- Ablation: History; History+Economy; History+Demographics; History+Economy+Demo; All_available",
        "- Importance: RF impurity importance, train on years < 2024",
        "- Conformal: split conformal (Demo 6), α=0.10; calibration = last year before test",
        "",
        "## National DPP share check",
        "",
        nat.to_string(index=False),
        "",
        "## Backtest MAE (from this run)",
        "",
        "```",
        back.pivot_table(index="model", columns="test_year", values="MAE").round(4).to_string(),
        "```",
        "",
        "## Ablation ranking (2024 holdout)",
        "",
        abl.sort_values("MAE")[["features", "MAE", "RMSE", "R2"]].round(4).to_string(index=False),
        "",
        "## Top features (RF, train < 2024)",
        "",
        imp.head(10).to_string(index=False),
        "",
        "## Conformal (2024, α=0.10)",
        "",
        json.dumps(conf_meta, indent=2),
        "",
        "## Caveats",
        "",
        "1. **Ecological fallacy**: township correlations ≠ individual voter behavior.",
        "2. **No TEDS party ID / 統獨** — literature often finds those stronger than socioeconomics.",
        "3. **2024 three-candidate race** (DPP/KMT/TPP) breaks share levels vs 2012–2020 two-bloc contests; baseline lag can beat ML on MAE.",
        "4. Income is **mean** taxable income, not true median household income.",
        "5. Urbanization is a **density transform**, not an official urban classification.",
        "6. Education rate is **household-head** college+ share (ODRP024), not population 15+ attainment.",
        "7. Geographic boundary / name changes may leave a few unmatched townships.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "/workspace/taiwan-election-data/.venv/bin/python \\",
        "  /workspace/taiwan-election-data/fyp_taiwan/scripts/run_taiwan_fyp_core.py",
        "```",
        "",
        "## Outputs",
        "",
        "- `harmonized_tw_regions.csv`",
        "- `backtest_results.csv`",
        "- `ablation_results.csv`",
        "- `feature_importance.csv`",
        "- `conformal_intervals_2024.csv`",
        "- `conformal_meta_2024.json`",
        "- `charts/*.png`",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("=== Building harmonized TW panel ===")
    harm = build_harmonized()

    print("=== Out-of-time backtest ===")
    back = backtest(harm)
    back.to_csv(OUT / "backtest_results.csv", index=False)
    print(back.to_string(index=False))

    print("=== Feature ablation (test 2024) ===")
    abl = ablation(harm, test_year=2024)
    # also run ablation for each backtest year and append
    abl_all = [abl]
    for y in sorted(harm.year.unique()):
        if y == 2024:
            continue
        if (harm["year"] < y).any() and (harm["year"] == y).any():
            print(f"=== Feature ablation (test {y}) ===")
            abl_all.append(ablation(harm, test_year=int(y)))
    abl_full = pd.concat(abl_all, ignore_index=True)
    abl_full.to_csv(OUT / "ablation_results.csv", index=False)

    print("=== Feature importance ===")
    imp = feature_importance(harm, test_year=2024)
    imp.to_csv(OUT / "feature_importance.csv", index=False)
    print(imp.to_string(index=False))

    print("=== Conformal intervals 2024 ===")
    intervals, meta = conformal(harm, test_year=2024, alpha=0.10)
    intervals.to_csv(OUT / "conformal_intervals_2024.csv", index=False)
    (OUT / "conformal_meta_2024.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("=== Charts ===")
    make_charts(back, abl, imp)

    print("=== README ===")
    write_readme(back, abl, imp, meta, harm)

    summary = {
        "backtest_mae_pivot": back.pivot_table(
            index="model", columns="test_year", values="MAE"
        )
        .round(6)
        .to_dict(),
        "ablation_2024": abl.sort_values("MAE")[["features", "MAE"]]
        .round(6)
        .to_dict(orient="records"),
        "top_features": imp.head(5).to_dict(orient="records"),
        "conformal": meta,
        "n_harmonized": len(harm),
        "years": sorted(int(y) for y in harm.year.unique()),
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("DONE →", OUT)


if __name__ == "__main__":
    main()
