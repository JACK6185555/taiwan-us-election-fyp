#!/usr/bin/env python3
"""Taiwan FYP — THREE-PARTY expansion (DPP / KMT / TPP).

Township × year presidential shares for:
  - DPP (party 16): 蔡英文/賴清德 across years
  - KMT (party 1): 朱/韓/侯 across years
  - TPP (party 350): 柯文哲 — presidential ticket ONLY in 2024
  - others: residual (incl. PFP 宋楚瑜 when present)

Out-of-time backtests for DPP/KMT level + swing.
TPP: (a) 2024 prediction from pre-2024 socio + prior party structure
     (b) supplementary leave-one-county-out WITHIN 2024 — labeled NOT temporal.
Never invents vote totals. Verifies national shares against known CEC totals.
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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path("/workspace/taiwan-election-data")
CLEAN = ROOT / "cleaned"
RAW_CEC = ROOT / "raw" / "cec" / "extracted"
RAW_MOI = ROOT / "raw" / "moi"
OUT = ROOT / "fyp_taiwan_parties"
CHARTS = OUT / "charts"
OUT.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

SEED = 42
DPP_PARTY = {16}
KMT_PARTY = {1}
TPP_PARTY = {350}
# 親民黨 90; 2012 宋楚瑜 ticket coded 999 in CEC — treat as third/PFP proxy
PFP_PARTY = {90, 999}

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

INCOME_ROC = {2012: 105, 2016: 105, 2020: 109, 2024: 112}
POP_ROC = {2012: 108, 2016: 108, 2020: 109, 2024: 112}
EDU_FILE_ROC = {2012: 108, 2016: 108, 2020: 112, 2024: 112}

# Known national shares for verification (from cleaned CEC national totals)
KNOWN_NATIONAL = {
    2016: {"dpp": 0.561234, "kmt": 0.310409, "tpp": None, "pfp": 0.128357},
    2020: {"dpp": 0.571307, "kmt": 0.386137, "tpp": None, "pfp": 0.042556},
    2024: {"dpp": 0.400503, "kmt": 0.334900, "tpp": 0.264597, "pfp": None},
}

DPP_KMT_FEATURES = [
    "prev_dpp_share",
    "prev_kmt_share",
    "prev_third_share",
    "prev_turnout",
    "prev_dpp_swing",
    "prev_kmt_swing",
    "median_income",
    "urbanization",
    "education_rate",
    "log_people",
]

TPP_FEATURES = [
    "prev_dpp_share",
    "prev_kmt_share",
    "prev_third_share",  # 2020 PFP etc. as third-party geography proxy
    "prev_turnout",
    "median_income",
    "urbanization",
    "education_rate",
    "log_people",
    "dpp_kmt_margin_prev",  # |DPP-KMT| competitiveness proxy
]


def _clean_cell(x) -> str:
    s = str(x).strip().strip('"').strip("'").strip()
    if s.startswith("'"):
        s = s[1:]
    return s


def read_cec_csv(path: Path) -> pd.DataFrame:
    last_err = None
    df = None
    for enc in ("utf-8-sig", "utf-8", "cp950", "big5"):
        try:
            df = pd.read_csv(path, header=None, dtype=str, encoding=enc)
            break
        except Exception as e:
            last_err = e
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
    df = read_cec_csv(path)
    return pd.DataFrame(
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
    if year == 2008:
        return d / "elbase.csv", d / "elcand.csv", d / "elctks.csv", d / "elprof.csv"
    return d / "elbase.csv", d / "elcand.csv", d / "elctks.csv", d / "elprof.csv"


def township_multiparty(year: int) -> pd.DataFrame:
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

    has_tpp = bool(set(elcand["party_code"]) & TPP_PARTY)
    has_pfp = bool(set(elcand["party_code"]) & PFP_PARTY)

    total = agg.groupby("township", as_index=False)["votes"].sum().rename(columns={"votes": "valid_votes"})

    def party_votes(codes, col):
        sub = agg[agg["party_code"].isin(codes)]
        if len(sub) == 0:
            return pd.DataFrame({"township": total["township"], col: 0})
        return (
            sub.groupby("township", as_index=False)["votes"]
            .sum()
            .rename(columns={"votes": col})
        )

    out = total.merge(party_votes(DPP_PARTY, "dpp_votes"), on="township", how="left")
    out = out.merge(party_votes(KMT_PARTY, "kmt_votes"), on="township", how="left")
    out = out.merge(party_votes(TPP_PARTY, "tpp_votes"), on="township", how="left")
    out = out.merge(party_votes(PFP_PARTY, "pfp_votes"), on="township", how="left")
    for c in ("dpp_votes", "kmt_votes", "tpp_votes", "pfp_votes"):
        out[c] = out[c].fillna(0).astype(int)

    out["others_votes"] = (
        out["valid_votes"] - out["dpp_votes"] - out["kmt_votes"] - out["tpp_votes"]
    ).clip(lower=0)

    out["dpp_share"] = out["dpp_votes"] / out["valid_votes"]
    out["kmt_share"] = out["kmt_votes"] / out["valid_votes"]
    out["others_share"] = out["others_votes"] / out["valid_votes"]
    out["pfp_share"] = out["pfp_votes"] / out["valid_votes"]
    # TPP: NaN when no TPP presidential candidate that year
    if has_tpp:
        out["tpp_share"] = out["tpp_votes"] / out["valid_votes"]
    else:
        out["tpp_share"] = np.nan
        out["tpp_votes"] = 0

    out["election_year"] = year
    out["has_tpp_candidate"] = has_tpp
    out["has_pfp_candidate"] = has_pfp
    meta = agg[["township", "county_name"]].drop_duplicates()
    out = out.merge(meta, on="township", how="left")

    # Turnout
    prof = elprof[(elprof["station"] != "0000") & (elprof["vil"] != "0000")].copy()
    prof = prof.merge(twn_names, on=["prv", "city", "twn"], how="left")
    prof = prof.merge(county_df, on=["prv", "city"], how="left")
    prof = prof.dropna(subset=["twn_name", "county_name"])
    prof = prof[prof["county_name"].isin(NORTH | CENTRAL | SOUTH | EAST | ISLANDS)]
    prof["township"] = prof["county_name"] + prof["twn_name"]
    turn = prof.groupby("township", as_index=False).agg(
        ballots=("ballots", "sum"), eligible=("eligible", "sum")
    )
    turn["turnout"] = turn["ballots"] / turn["eligible"].replace(0, np.nan)
    out = out.merge(turn[["township", "turnout", "eligible", "ballots"]], on="township", how="left")

    nat_dpp = out["dpp_votes"].sum() / out["valid_votes"].sum()
    nat_kmt = out["kmt_votes"].sum() / out["valid_votes"].sum()
    nat_tpp = (
        out["tpp_votes"].sum() / out["valid_votes"].sum() if has_tpp else float("nan")
    )
    print(
        f"  {year}: n={len(out)} DPP={nat_dpp:.4f} KMT={nat_kmt:.4f} "
        f"TPP={nat_tpp if has_tpp else 'N/A'} PFP_cand={has_pfp}"
    )
    return out


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


def verify_national(votes: pd.DataFrame) -> list[dict]:
    rows = []
    for y, known in KNOWN_NATIONAL.items():
        g = votes[votes.election_year == y]
        if len(g) == 0:
            continue
        tot = g["valid_votes"].sum()
        got = {
            "year": y,
            "dpp": g["dpp_votes"].sum() / tot,
            "kmt": g["kmt_votes"].sum() / tot,
            "tpp": (g["tpp_votes"].sum() / tot) if g["has_tpp_candidate"].any() else None,
            "pfp": (g["pfp_votes"].sum() / tot) if g["has_pfp_candidate"].any() else None,
        }
        for party in ("dpp", "kmt", "tpp", "pfp"):
            exp = known[party]
            act = got[party]
            if exp is None and act is None:
                ok = True
                diff = None
            elif exp is None or act is None:
                ok = False
                diff = None
            else:
                diff = abs(act - exp)
                ok = diff < 0.005  # 0.5pp tolerance for township aggregation
            rows.append(
                {
                    "year": y,
                    "party": party,
                    "expected": exp,
                    "observed": act,
                    "abs_diff": diff,
                    "ok": ok,
                }
            )
            status = "OK" if ok else "CHECK"
            print(f"  verify {y} {party}: exp={exp} obs={act} diff={diff} [{status}]")
    return rows


def build_panel() -> pd.DataFrame:
    print("Building multi-party township votes...")
    parts = []
    for y in (2008, 2012, 2016, 2020, 2024):
        parts.append(township_multiparty(y))
    votes = pd.concat(parts, ignore_index=True)
    votes.to_csv(OUT / "township_multiparty_votes.csv", index=False)

    print("National share verification vs cleaned CEC totals:")
    ver = verify_national(votes)
    pd.DataFrame(ver).to_csv(OUT / "national_share_verification.csv", index=False)

    income = pd.read_csv(CLEAN / "income_township_panel_full.csv")
    pop = pd.read_csv(CLEAN / "moi_township_pop_density_panel.csv")
    edu_cache = {ry: load_education_rate(ry) for ry in sorted(set(EDU_FILE_ROC.values()))}

    model_years = [2012, 2016, 2020, 2024]
    frames = []
    for y in model_years:
        vy = votes[votes.election_year == y].copy()
        prev_y = y - 4
        prev = votes[votes.election_year == prev_y][
            [
                "township",
                "dpp_share",
                "kmt_share",
                "tpp_share",
                "others_share",
                "pfp_share",
                "turnout",
            ]
        ].rename(
            columns={
                "dpp_share": "prev_dpp_share",
                "kmt_share": "prev_kmt_share",
                "tpp_share": "prev_tpp_share",
                "others_share": "prev_others_share",
                "pfp_share": "prev_pfp_share",
                "turnout": "prev_turnout",
            }
        )
        vy = vy.merge(prev, on="township", how="left")

        # third-party proxy at t-1: prefer PFP share if present, else others
        vy["prev_third_share"] = vy["prev_pfp_share"].fillna(vy["prev_others_share"])

        prev2_y = y - 8
        if prev2_y in votes.election_year.values:
            p2 = votes[votes.election_year == prev2_y][
                ["township", "dpp_share", "kmt_share"]
            ].rename(columns={"dpp_share": "prev2_dpp", "kmt_share": "prev2_kmt"})
            vy = vy.merge(p2, on="township", how="left")
            vy["prev_dpp_swing"] = vy["prev_dpp_share"] - vy["prev2_dpp"]
            vy["prev_kmt_swing"] = vy["prev_kmt_share"] - vy["prev2_kmt"]
        else:
            vy["prev_dpp_swing"] = np.nan
            vy["prev_kmt_swing"] = np.nan

        iroc = INCOME_ROC[y]
        avail_i = sorted(income.roc_year.unique())
        if iroc not in avail_i:
            iroc = min(avail_i, key=lambda a: abs(a - iroc))
        inc = income[income.roc_year == iroc][["township", "mean_income"]].rename(
            columns={"mean_income": "median_income"}
        )
        vy = vy.merge(inc, on="township", how="left")

        proc = POP_ROC[y]
        avail_p = sorted(pop.roc_year.unique())
        if proc not in avail_p:
            proc = min(avail_p, key=lambda a: abs(a - proc))
        dens = pop[pop.roc_year == proc][
            ["township", "people_total", "area_km2", "population_density"]
        ]
        vy = vy.merge(dens, on="township", how="left")
        vy = vy.merge(edu_cache[EDU_FILE_ROC[y]], on="township", how="left")

        vy["geo_region"] = vy["county_name"].map(region_of)
        vy["income_roc_used"] = iroc
        vy["pop_roc_used"] = proc
        vy["edu_roc_used"] = EDU_FILE_ROC[y]
        frames.append(vy)

    panel = pd.concat(frames, ignore_index=True)
    dens = pd.to_numeric(panel["population_density"], errors="coerce")
    logd = np.log1p(dens)
    panel["urbanization"] = (logd - logd.min()) / (logd.max() - logd.min())
    panel["median_income"] = pd.to_numeric(panel["median_income"], errors="coerce")
    panel["people_total"] = pd.to_numeric(panel["people_total"], errors="coerce")
    panel["log_people"] = np.log1p(panel["people_total"])
    panel["education_rate"] = pd.to_numeric(panel["education_rate"], errors="coerce")

    panel["dpp_swing"] = panel["dpp_share"] - panel["prev_dpp_share"]
    panel["kmt_swing"] = panel["kmt_share"] - panel["prev_kmt_share"]
    # TPP swing undefined historically (only one presidential race)
    panel["tpp_swing"] = np.nan
    panel["dpp_kmt_margin_prev"] = (panel["prev_dpp_share"] - panel["prev_kmt_share"]).abs()

    panel["country"] = "TW"
    panel["region"] = panel["township"]
    panel["year"] = panel["election_year"]
    panel["unemployment"] = np.nan
    panel["poll_index"] = np.nan

    out_cols = [
        "country",
        "region",
        "year",
        "county_name",
        "geo_region",
        "valid_votes",
        "dpp_votes",
        "kmt_votes",
        "tpp_votes",
        "pfp_votes",
        "others_votes",
        "dpp_share",
        "kmt_share",
        "tpp_share",
        "pfp_share",
        "others_share",
        "has_tpp_candidate",
        "turnout",
        "prev_dpp_share",
        "prev_kmt_share",
        "prev_tpp_share",
        "prev_third_share",
        "prev_others_share",
        "prev_pfp_share",
        "prev_turnout",
        "prev_dpp_swing",
        "prev_kmt_swing",
        "dpp_swing",
        "kmt_swing",
        "tpp_swing",
        "dpp_kmt_margin_prev",
        "median_income",
        "urbanization",
        "education_rate",
        "log_people",
        "population_density",
        "unemployment",
        "poll_index",
        "income_roc_used",
        "pop_roc_used",
        "edu_roc_used",
    ]
    harm = panel[out_cols].copy()
    harm.to_csv(OUT / "harmonized_tw_parties.csv", index=False)
    print(
        f"harmonized rows={len(harm)} years={sorted(harm.year.unique())} "
        f"tpp_nonnull={harm['tpp_share'].notna().sum()}"
    )
    return harm


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


def impute(train: pd.DataFrame, test: pd.DataFrame, feats: list[str]):
    tr, te = train.copy(), test.copy()
    for c in feats:
        if c not in tr.columns:
            continue
        fill = tr[c].median()
        if pd.isna(fill):
            fill = 0.0
        tr[c] = tr[c].fillna(fill)
        te[c] = te[c].fillna(fill)
    return tr, te


def metrics_row(party, target, test_year, model, y_true, y_pred, n_train, n_test, eval_type):
    return {
        "party": party,
        "target": target,
        "test_year": test_year,
        "model": model,
        "eval_type": eval_type,
        "n_train": n_train,
        "n_test": n_test,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
    }


def backtest_party_level(data: pd.DataFrame, party: str, share_col: str, prev_col: str, feats: list[str]):
    """Out-of-time level prediction for DPP or KMT."""
    d = data.dropna(subset=[share_col, prev_col]).copy()
    results = []
    models = get_models()
    years = sorted(d["year"].unique())
    for test_year in years:
        train = d[d["year"] < test_year]
        test = d[d["year"] == test_year]
        if len(train) == 0 or len(test) == 0:
            continue
        tr, te = impute(train, test, feats)
        need = [c for c in feats if c in tr.columns] + [share_col]
        tr = tr.dropna(subset=need)
        te = te.dropna(subset=need)
        if len(tr) < 50 or len(te) < 20:
            continue
        use_feats = [c for c in feats if c in tr.columns]
        y_test = te[share_col]

        # Baseline: previous share
        results.append(
            metrics_row(
                party, "level", test_year, "PreviousShare", y_test, te[prev_col],
                len(tr), len(te), "out_of_time",
            )
        )
        for name, model in models.items():
            model.fit(tr[use_feats], tr[share_col])
            pred = model.predict(te[use_feats])
            results.append(
                metrics_row(
                    party, "level", test_year, name, y_test, pred,
                    len(tr), len(te), "out_of_time",
                )
            )
    return results


def backtest_party_swing(data: pd.DataFrame, party: str, swing_col: str, share_col: str, prev_col: str, feats: list[str]):
    d = data.dropna(subset=[swing_col, prev_col]).copy()
    results = []
    models = get_models()
    years = sorted(d["year"].unique())
    for test_year in years:
        train = d[d["year"] < test_year]
        test = d[d["year"] == test_year]
        if len(train) == 0 or len(test) == 0:
            continue
        tr, te = impute(train, test, feats)
        need = [c for c in feats if c in tr.columns] + [swing_col]
        tr = tr.dropna(subset=need)
        te = te.dropna(subset=need)
        if len(tr) < 50 or len(te) < 20:
            continue
        use_feats = [c for c in feats if c in tr.columns]
        y_test = te[swing_col]

        results.append(
            metrics_row(
                party, "swing", test_year, "ZeroSwing", y_test, np.zeros(len(te)),
                len(tr), len(te), "out_of_time",
            )
        )
        # PreviousSwing if available
        prev_swing_col = f"prev_{party.lower()}_swing"
        if prev_swing_col in te.columns and te[prev_swing_col].notna().any():
            ps = te[prev_swing_col].fillna(0)
            results.append(
                metrics_row(
                    party, "swing", test_year, "PreviousSwing", y_test, ps,
                    len(tr), len(te), "out_of_time",
                )
            )
        for name, model in models.items():
            model.fit(tr[use_feats], tr[swing_col])
            pred = model.predict(te[use_feats])
            results.append(
                metrics_row(
                    party, "swing", test_year, name, y_test, pred,
                    len(tr), len(te), "out_of_time",
                )
            )
    return results


def tpp_2024_from_history(data: pd.DataFrame) -> list[dict]:
    """Predict 2024 TPP share using only pre-2024 features (no TPP history).

    Train is impossible for true TPP share (no prior presidential TPP).
    We report:
      - Baseline ZeroTPP (predict 0)
      - Baseline MeanThirdProxy (predict prev_third_share, i.e. 2020 PFP geography)
      - Cross-sectional models that CANNOT be trained on prior TPP — instead we
        fit on 2024 with leave-one-county-out (separate function).
      - A 'transfer' attempt: train RF to predict 2020 third-party (PFP) share
        from socio+structure, then apply to 2024 features as a weak proxy for
        TPP geography. Documented as PROXY, not true TPP temporal forecast.
    """
    results = []
    d24 = data[data["year"] == 2024].dropna(subset=["tpp_share"]).copy()
    d20 = data[data["year"] == 2020].copy()
    feats = [c for c in TPP_FEATURES if c in d24.columns]

    # Baselines on 2024 (no train from TPP history)
    y = d24["tpp_share"]
    results.append(
        metrics_row("TPP", "level", 2024, "ZeroTPP", y, np.zeros(len(d24)), 0, len(d24), "out_of_time_no_history")
    )
    # previous third-party geography (2020 PFP)
    prox = d24["prev_third_share"].fillna(d24["prev_third_share"].median())
    results.append(
        metrics_row("TPP", "level", 2024, "PrevThirdPartyProxy", y, prox, 0, len(d24), "out_of_time_no_history")
    )
    # national mean baseline
    results.append(
        metrics_row(
            "TPP", "level", 2024, "NationalMeanTPP", y, np.full(len(d24), y.mean()),
            0, len(d24), "in_sample_mean_oracle",
        )
    )

    # Transfer: train on 2020 PFP share -> apply to 2024
    d20 = d20.dropna(subset=["pfp_share"] + [c for c in feats if c in d20.columns])
    if len(d20) > 50:
        tr, te = impute(d20, d24, feats)
        use = [c for c in feats if c in tr.columns]
        tr = tr.dropna(subset=use + ["pfp_share"])
        te = te.dropna(subset=use + ["tpp_share"])
        for name, model in get_models().items():
            model.fit(tr[use], tr["pfp_share"])
            pred = model.predict(te[use])
            # scale proxy to match 2024 national TPP mean (calibration note)
            scale = te["tpp_share"].mean() / max(pred.mean(), 1e-6)
            pred_cal = pred * scale
            results.append(
                metrics_row(
                    "TPP", "level", 2024, f"TransferPFP2020_{name}",
                    te["tpp_share"], pred, len(tr), len(te), "proxy_transfer_uncalibrated",
                )
            )
            results.append(
                metrics_row(
                    "TPP", "level", 2024, f"TransferPFP2020_{name}_caled",
                    te["tpp_share"], pred_cal, len(tr), len(te), "proxy_transfer_mean_caled",
                )
            )
    return results


def tpp_loco_2024(data: pd.DataFrame) -> list[dict]:
    """Leave-one-county-out within 2024 ONLY. NOT temporal. Supplementary."""
    d = data[data["year"] == 2024].dropna(subset=["tpp_share"]).copy()
    feats = [c for c in TPP_FEATURES if c in d.columns]
    d = d.dropna(subset=feats + ["tpp_share", "county_name"])
    results = []
    counties = sorted(d["county_name"].unique())
    models = get_models()
    for name in models:
        preds = np.zeros(len(d))
        idx_map = {ix: i for i, ix in enumerate(d.index)}
        for county in counties:
            tr = d[d["county_name"] != county]
            te = d[d["county_name"] == county]
            if len(tr) < 30 or len(te) == 0:
                continue
            tr2, te2 = impute(tr, te, feats)
            m = get_models()[name]
            m.fit(tr2[feats], tr2["tpp_share"])
            p = m.predict(te2[feats])
            for ix, val in zip(te2.index, p):
                preds[idx_map[ix]] = val
        y = d["tpp_share"].values
        results.append(
            metrics_row(
                "TPP", "level", 2024, name, y, preds, len(d) - 1, len(d),
                "leave_one_county_out_2024_NOT_temporal",
            )
        )
    # Also DPP/KMT LOCO for 2024 comparison
    for party, col in (("DPP", "dpp_share"), ("KMT", "kmt_share")):
        for name in models:
            preds = np.zeros(len(d))
            idx_map = {ix: i for i, ix in enumerate(d.index)}
            for county in counties:
                tr = d[d["county_name"] != county]
                te = d[d["county_name"] == county]
                if len(tr) < 30 or len(te) == 0:
                    continue
                tr2, te2 = impute(tr, te, feats)
                m = get_models()[name]
                m.fit(tr2[feats], tr2[col])
                p = m.predict(te2[feats])
                for ix, val in zip(te2.index, p):
                    preds[idx_map[ix]] = val
            results.append(
                metrics_row(
                    party, "level", 2024, name, d[col].values, preds, len(d) - 1, len(d),
                    "leave_one_county_out_2024_NOT_temporal",
                )
            )
    return results


def ablation_and_importance(data: pd.DataFrame, party: str, share_col: str, feats: list[str], test_year: int = 2024):
    """RF ablation-style feature sets + impurity importance for holdout year."""
    feature_sets = {
        "History": [c for c in feats if c.startswith("prev_") or c == "dpp_kmt_margin_prev"],
        "History+Economy": [c for c in feats if c.startswith("prev_") or c in ("dpp_kmt_margin_prev", "median_income")],
        "History+Demo": [
            c for c in feats
            if c.startswith("prev_") or c in ("dpp_kmt_margin_prev", "urbanization", "education_rate", "log_people")
        ],
        "All_available": list(feats),
    }
    d = data.dropna(subset=[share_col]).copy()
    train = d[d["year"] < test_year]
    test = d[d["year"] == test_year]
    abl_rows = []
    for set_name, fs in feature_sets.items():
        fs = [c for c in fs if c in d.columns]
        if not fs or len(train) == 0 or len(test) == 0:
            continue
        # For TPP: no prior share target years — skip temporal ablation if train empty of target
        if party == "TPP":
            continue
        tr, te = impute(train, test, fs)
        tr = tr.dropna(subset=fs + [share_col])
        te = te.dropna(subset=fs + [share_col])
        if len(tr) < 50:
            continue
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
        rf.fit(tr[fs], tr[share_col])
        pred = rf.predict(te[fs])
        abl_rows.append(
            {
                "party": party,
                "feature_set": set_name,
                "test_year": test_year,
                "n_features": len(fs),
                "MAE": mean_absolute_error(te[share_col], pred),
                "RMSE": float(np.sqrt(mean_squared_error(te[share_col], pred))),
                "R2": r2_score(te[share_col], pred),
                "eval_type": "out_of_time",
            }
        )

    # Importance: for DPP/KMT use train<2024; for TPP use full 2024 fit (in-sample impurity — labeled)
    imp_rows = []
    if party == "TPP":
        d24 = d[d["year"] == 2024].dropna(subset=feats + [share_col])
        if len(d24) > 50:
            rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
            rf.fit(d24[feats], d24[share_col])
            for f, v in zip(feats, rf.feature_importances_):
                imp_rows.append(
                    {
                        "party": party,
                        "feature": f,
                        "importance": float(v),
                        "fit": "in_sample_2024_impurity",
                        "note": "TPP has only one presidential election; importance is descriptive not temporal",
                    }
                )
    else:
        tr = d[d["year"] < test_year]
        te = d[d["year"] == test_year]
        tr, te = impute(tr, te, feats)
        tr = tr.dropna(subset=feats + [share_col])
        if len(tr) > 50:
            rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
            rf.fit(tr[feats], tr[share_col])
            for f, v in zip(feats, rf.feature_importances_):
                imp_rows.append(
                    {
                        "party": party,
                        "feature": f,
                        "importance": float(v),
                        "fit": f"train_years_lt_{test_year}",
                        "note": "out_of_time RF impurity",
                    }
                )
    return abl_rows, imp_rows


def make_charts(backtest_df: pd.DataFrame, imp_df: pd.DataFrame, loco_df: pd.DataFrame):
    # 2024 three-party MAE comparison (out-of-time / proxy where applicable)
    fig, ax = plt.subplots(figsize=(10, 5))
    sub = backtest_df[
        (backtest_df["test_year"] == 2024)
        & (backtest_df["target"] == "level")
        & (backtest_df["eval_type"].isin(["out_of_time", "out_of_time_no_history", "proxy_transfer_mean_caled"]))
    ].copy()
    # pick best ML + baselines per party for display
    plot_models = {
        "DPP": ["PreviousShare", "RandomForest", "Ridge"],
        "KMT": ["PreviousShare", "RandomForest", "Ridge"],
        "TPP": ["ZeroTPP", "PrevThirdPartyProxy", "TransferPFP2020_RandomForest_caled"],
    }
    xs, heights, colors, labels = [], [], [], []
    color_map = {"DPP": "#1f7a1f", "KMT": "#1f4e79", "TPP": "#c9a227"}
    i = 0
    for party, models in plot_models.items():
        for m in models:
            row = sub[(sub["party"] == party) & (sub["model"] == m)]
            if len(row) == 0:
                continue
            xs.append(i)
            heights.append(row.iloc[0]["MAE"])
            colors.append(color_map[party])
            labels.append(f"{party}\n{m[:18]}")
            i += 1
    ax.bar(xs, heights, color=colors)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("MAE (vote share)")
    ax.set_title("2024 township MAE by party (out-of-time / TPP proxy)")
    fig.tight_layout()
    fig.savefig(CHARTS / "mae_2024_by_party.png", dpi=140)
    plt.close()

    # Feature importance per party
    if len(imp_df):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
        for ax, party in zip(axes, ["DPP", "KMT", "TPP"]):
            s = imp_df[imp_df["party"] == party].sort_values("importance", ascending=True)
            if len(s) == 0:
                ax.set_visible(False)
                continue
            ax.barh(s["feature"], s["importance"], color=color_map[party])
            ax.set_title(f"{party} RF importance")
            ax.set_xlabel("impurity importance")
        fig.suptitle("Ecological township RF feature importance (not individual party-ID)")
        fig.tight_layout()
        fig.savefig(CHARTS / "feature_importance_by_party.png", dpi=140)
        plt.close()

    # LOCO 2024 three-party
    if len(loco_df):
        fig, ax = plt.subplots(figsize=(8, 4))
        sub = loco_df[loco_df["model"] == "RandomForest"]
        parties = ["DPP", "KMT", "TPP"]
        maes = [sub[sub["party"] == p]["MAE"].values[0] if len(sub[sub["party"] == p]) else np.nan for p in parties]
        ax.bar(parties, maes, color=[color_map[p] for p in parties])
        ax.set_ylabel("MAE")
        ax.set_title("2024 Leave-One-County-Out MAE (NOT temporal — supplementary)")
        fig.tight_layout()
        fig.savefig(CHARTS / "loco_2024_three_party.png", dpi=140)
        plt.close()


def main():
    panel = build_panel()

    all_results = []
    print("\n=== DPP/KMT out-of-time LEVEL ===")
    all_results += backtest_party_level(panel, "DPP", "dpp_share", "prev_dpp_share", DPP_KMT_FEATURES)
    all_results += backtest_party_level(panel, "KMT", "kmt_share", "prev_kmt_share", DPP_KMT_FEATURES)

    print("=== DPP/KMT out-of-time SWING ===")
    all_results += backtest_party_swing(panel, "DPP", "dpp_swing", "dpp_share", "prev_dpp_share", DPP_KMT_FEATURES)
    all_results += backtest_party_swing(panel, "KMT", "kmt_swing", "kmt_share", "prev_kmt_share", DPP_KMT_FEATURES)

    print("=== TPP 2024 (no presidential history) ===")
    all_results += tpp_2024_from_history(panel)

    print("=== Leave-one-county-out 2024 (NOT temporal) ===")
    loco = tpp_loco_2024(panel)
    all_results += loco

    bt = pd.DataFrame(all_results)
    bt.to_csv(OUT / "backtest_results_parties.csv", index=False)

    abl_all, imp_all = [], []
    for party, col, feats in (
        ("DPP", "dpp_share", DPP_KMT_FEATURES),
        ("KMT", "kmt_share", DPP_KMT_FEATURES),
        ("TPP", "tpp_share", TPP_FEATURES),
    ):
        abl, imp = ablation_and_importance(panel, party, col, feats, 2024)
        abl_all.extend(abl)
        imp_all.extend(imp)
    abl_df = pd.DataFrame(abl_all)
    imp_df = pd.DataFrame(imp_all)
    abl_df.to_csv(OUT / "ablation_by_party.csv", index=False)
    imp_df.to_csv(OUT / "feature_importance_by_party.csv", index=False)

    make_charts(bt, imp_df, pd.DataFrame(loco))

    # Compact 2024 comparison table
    cmp_2024 = bt[(bt["test_year"] == 2024) & (bt["target"] == "level")].copy()
    cmp_2024.to_csv(OUT / "compare_2024_mae_by_party.csv", index=False)

    summary = {
        "unit": "township x year",
        "parties": {
            "DPP": "party_code 16 — all years",
            "KMT": "party_code 1 — all years",
            "TPP": "party_code 350 — presidential ONLY 2024 (founded 2019; no 2020 presidential ticket)",
            "PFP": "party 90 / 999 — used as third-party geography proxy, not a modeled target",
        },
        "national_verification_2024": {
            "expected": KNOWN_NATIONAL[2024],
            "note": "see national_share_verification.csv",
        },
        "n_rows_harmonized": int(len(panel)),
        "tpp_caveat": (
            "TPP has only one presidential election (2024). True out-of-time TPP share "
            "forecasting from prior TPP presidential results is impossible. We report "
            "ZeroTPP / PrevThirdPartyProxy baselines, PFP→TPP transfer proxies, and "
            "leave-one-county-out within 2024 clearly labeled NOT temporal."
        ),
        "literature_vs_ecological": (
            "Literature individual-level weights (party ID, unification-independence) "
            "are NOT in this ecological township model; TEDS microdata not included. "
            "See docs for contrast."
        ),
        "top_importance": {
            p: (
                imp_df[imp_df.party == p]
                .sort_values("importance", ascending=False)
                .head(5)[["feature", "importance"]]
                .to_dict("records")
                if len(imp_df[imp_df.party == p])
                else []
            )
            for p in ("DPP", "KMT", "TPP")
        },
    }
    with open(OUT / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Print key MAE table
    print("\n=== Key 2024 level MAE ===")
    show = cmp_2024[
        cmp_2024["model"].isin(
            [
                "PreviousShare",
                "RandomForest",
                "Ridge",
                "ZeroTPP",
                "PrevThirdPartyProxy",
                "TransferPFP2020_RandomForest_caled",
            ]
        )
    ][["party", "model", "MAE", "eval_type"]]
    print(show.to_string(index=False))
    print("\nDone. Outputs in", OUT)


if __name__ == "__main__":
    main()
