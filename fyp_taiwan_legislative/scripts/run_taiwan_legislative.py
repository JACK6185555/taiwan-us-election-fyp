#!/usr/bin/env python3
"""Taiwan FYP — LEGISLATIVE (立委) three-party analysis (DPP / KMT / TPP).

Geographic unit: 鄉鎮市區 (township), aggregating 區域立委 candidate votes by party.
Party codes (CEC elpaty): DPP=16, KMT=1, TPP=350.

Why township not 選區:
  - Matches presidential FYP panel for fair comparison
  - Socio features (income/density/education) are township-coded
  - Documented: a township may sit inside one district; votes are summed across
    any district stations in that township

不分區政黨: CEC extract has national elretks only (no township vote file) —
  we record national party-list shares for verification/context, NOT model them.

TPP: founded 2019 → no 2016; district + party-list presence in 2020 and 2024.
No 2012 區域立委 in this extract → 2016 has no legislative lag; main OOT years
are 2020 and 2024 (2024 is first year with prev_tpp_share for district TPP).

Never invents vote totals. Verifies against CEC aggregates / elretks.
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
CLEAN = ROOT / "cleaned"
RAW_CEC = ROOT / "raw" / "cec" / "extracted"
RAW_MOI = ROOT / "raw" / "moi"
OUT = ROOT / "fyp_taiwan_legislative"
CHARTS = OUT / "charts"
OUT.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

SEED = 42
DPP_PARTY = {16}
KMT_PARTY = {1}
TPP_PARTY = {350}

NORTH = {"臺北市", "新北市", "基隆市", "桃園市", "新竹市", "新竹縣", "宜蘭縣"}
CENTRAL = {"臺中市", "苗栗縣", "彰化縣", "南投縣", "雲林縣"}
SOUTH = {"高雄市", "臺南市", "嘉義市", "嘉義縣", "屏東縣", "澎湖縣"}
EAST = {"花蓮縣", "臺東縣"}
ISLANDS = {"金門縣", "連江縣"}
ALL_COUNTIES = NORTH | CENTRAL | SOUTH | EAST | ISLANDS

# 區域立委 directories
DISTRICT_DIRS = {
    2016: RAW_CEC / "2016總統立委" / "區域立委",
    2020: RAW_CEC / "2020總統立委" / "區域立委",
    2024: RAW_CEC / "2024總統立委" / "區域立委",
}
PARTYLIST_DIRS = {
    2016: RAW_CEC / "2016總統立委" / "不分區政黨",
    2020: RAW_CEC / "2020總統立委" / "不分區政黨",
    2024: RAW_CEC / "2024總統立委" / "不分區政黨",
}

INCOME_ROC = {2016: 105, 2020: 109, 2024: 112}
POP_ROC = {2016: 108, 2020: 109, 2024: 112}
EDU_FILE_ROC = {2016: 108, 2020: 112, 2024: 112}

# National 不分區 shares from CEC elretks (percent → fraction). Source of truth for list ballot.
KNOWN_PARTYLIST = {
    2016: {"dpp": 0.440598, "kmt": 0.269148, "tpp": None},
    2020: {"dpp": 0.339774, "kmt": 0.333578, "tpp": 0.112203},
    2024: {"dpp": 0.361629, "kmt": 0.345842, "tpp": 0.220706},
}

DPP_KMT_FEATURES = [
    "prev_dpp_share",
    "prev_kmt_share",
    "prev_tpp_share",
    "prev_others_share",
    "prev_turnout",
    "prev_dpp_swing",
    "prev_kmt_swing",
    "median_income",
    "urbanization",
    "education_rate",
    "log_people",
    "tpp_contested",
]

TPP_FEATURES = [
    "prev_dpp_share",
    "prev_kmt_share",
    "prev_tpp_share",
    "prev_others_share",
    "prev_turnout",
    "median_income",
    "urbanization",
    "education_rate",
    "log_people",
    "tpp_contested",
    "dpp_kmt_margin_prev",
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


def _district_paths(year: int):
    d = DISTRICT_DIRS[year]
    if year == 2016:
        return d / "elbase_T1.csv", d / "elcand_T1.csv", d / "elctks_T1.csv", d / "elprof_T1.csv"
    return d / "elbase.csv", d / "elcand.csv", d / "elctks.csv", d / "elprof.csv"


def _partylist_retks_path(year: int) -> Path:
    d = PARTYLIST_DIRS[year]
    if year == 2016:
        return d / "elretks_T4.csv"
    return d / "elretks.csv"


def load_party_list_national(year: int) -> dict:
    """Parse CEC 不分區 elretks: party_code, vote%, seat%?, n_list, n_elected."""
    path = _partylist_retks_path(year)
    df = read_cec_csv(path)
    df = df.iloc[:, :5]
    df.columns = ["party_code", "vote_pct", "seat_pct", "n_nom", "n_elected"]
    df["party_code"] = pd.to_numeric(df["party_code"], errors="coerce").astype("Int64")
    df["vote_pct"] = pd.to_numeric(df["vote_pct"], errors="coerce")
    out = {"year": year, "source": str(path.name)}
    for code, key in ((16, "dpp"), (1, "kmt"), (350, "tpp")):
        row = df[df["party_code"] == code]
        if len(row):
            out[f"{key}_list_share"] = float(row.iloc[0]["vote_pct"]) / 100.0
        else:
            out[f"{key}_list_share"] = None
    return out


def township_district_party_votes(year: int) -> pd.DataFrame:
    """Aggregate 區域立委 poll-station votes → township party shares."""
    base_p, cand_p, ctks_p, prof_p = _district_paths(year)
    elbase = read_cec_csv(base_p).iloc[:, :6]
    elbase.columns = ["prv", "city", "dept", "twn", "vil", "name"]

    elcand = read_cec_csv(cand_p)
    cand = pd.DataFrame(
        {
            "prv": elcand.iloc[:, 0],
            "city": elcand.iloc[:, 1],
            "dept": elcand.iloc[:, 2],
            "ticket": elcand.iloc[:, 5].astype(str).str.lstrip("0").replace("", "0"),
            "cand_name": elcand.iloc[:, 6],
            "party_code": pd.to_numeric(elcand.iloc[:, 7], errors="coerce").fillna(-1).astype(int),
        }
    ).drop_duplicates(["prv", "city", "dept", "ticket"], keep="first")

    elctks = read_cec_csv(ctks_p).iloc[:, :10]
    elctks.columns = [
        "prv", "city", "dept", "twn", "vil", "station", "ticket", "votes", "share", "elected"
    ]
    elctks["ticket"] = elctks["ticket"].astype(str).str.lstrip("0").replace("", "0")
    elctks["votes"] = pd.to_numeric(elctks["votes"], errors="coerce").fillna(0).astype(int)

    elprof = read_cec_csv(prof_p)
    prof = pd.DataFrame(
        {
            "prv": elprof.iloc[:, 0],
            "city": elprof.iloc[:, 1],
            "dept": elprof.iloc[:, 2],
            "twn": elprof.iloc[:, 3],
            "vil": elprof.iloc[:, 4],
            "station": elprof.iloc[:, 5],
            "ballots": pd.to_numeric(elprof.iloc[:, 8], errors="coerce"),
            "eligible": pd.to_numeric(elprof.iloc[:, 9], errors="coerce"),
        }
    )

    # County names: dept == 00, twn == 000, vil == 0000
    county_rows = elbase[(elbase.dept == "00") & (elbase.twn == "000") & (elbase.vil == "0000")]
    county_df = county_rows[["prv", "city", "name"]].rename(columns={"name": "county_name"})
    county_df = county_df[county_df["county_name"].isin(ALL_COUNTIES)].drop_duplicates(["prv", "city"])

    # Township names (may repeat across districts) — unique on prv,city,twn
    twn_rows = elbase[(elbase.twn != "000") & (elbase.vil == "0000")]
    twn_names = (
        twn_rows.groupby(["prv", "city", "twn"], as_index=False)
        .agg(twn_name=("name", "first"))
    )

    has_tpp = bool(set(cand["party_code"]) & TPP_PARTY)

    poll = elctks[elctks["station"] != "0000"].copy()
    poll = poll.merge(cand, on=["prv", "city", "dept", "ticket"], how="left")
    unmatched = int(poll.loc[poll["party_code"].isna(), "votes"].sum())
    if unmatched > 0:
        raise RuntimeError(f"{year}: unmatched district votes={unmatched}")

    # Flag districts that fielded a TPP candidate
    tpp_dists = cand[cand["party_code"].isin(TPP_PARTY)][["prv", "city", "dept"]].drop_duplicates()
    tpp_dists["tpp_in_district"] = 1

    poll = poll.merge(twn_names, on=["prv", "city", "twn"], how="left")
    poll = poll.merge(county_df, on=["prv", "city"], how="left")
    poll = poll.dropna(subset=["twn_name", "county_name"])
    poll["township"] = poll["county_name"] + poll["twn_name"]
    poll = poll.merge(tpp_dists, on=["prv", "city", "dept"], how="left")
    poll["tpp_in_district"] = poll["tpp_in_district"].fillna(0).astype(int)

    total = poll.groupby("township", as_index=False)["votes"].sum().rename(columns={"votes": "valid_votes"})

    def party_votes(codes, col):
        sub = poll[poll["party_code"].isin(codes)]
        if len(sub) == 0:
            return pd.DataFrame({"township": total["township"], col: 0})
        return sub.groupby("township", as_index=False)["votes"].sum().rename(columns={"votes": col})

    out = total.merge(party_votes(DPP_PARTY, "dpp_votes"), on="township", how="left")
    out = out.merge(party_votes(KMT_PARTY, "kmt_votes"), on="township", how="left")
    out = out.merge(party_votes(TPP_PARTY, "tpp_votes"), on="township", how="left")
    for c in ("dpp_votes", "kmt_votes", "tpp_votes"):
        out[c] = out[c].fillna(0).astype(int)
    out["others_votes"] = (
        out["valid_votes"] - out["dpp_votes"] - out["kmt_votes"] - out["tpp_votes"]
    ).clip(lower=0)

    out["dpp_share"] = out["dpp_votes"] / out["valid_votes"]
    out["kmt_share"] = out["kmt_votes"] / out["valid_votes"]
    out["others_share"] = out["others_votes"] / out["valid_votes"]
    if has_tpp:
        out["tpp_share"] = out["tpp_votes"] / out["valid_votes"]
    else:
        out["tpp_share"] = np.nan
        out["tpp_votes"] = 0

    # Township contested if ANY station in a TPP-fielded district
    contested = (
        poll.groupby("township", as_index=False)["tpp_in_district"]
        .max()
        .rename(columns={"tpp_in_district": "tpp_contested"})
    )
    out = out.merge(contested, on="township", how="left")
    out["tpp_contested"] = out["tpp_contested"].fillna(0).astype(int)

    meta = poll[["township", "county_name"]].drop_duplicates()
    out = out.merge(meta, on="township", how="left")

    # Turnout from elprof stations
    p = prof[(prof["station"] != "0000") & (prof["vil"] != "0000")].copy()
    p = p.merge(twn_names, on=["prv", "city", "twn"], how="left")
    p = p.merge(county_df, on=["prv", "city"], how="left")
    p = p.dropna(subset=["twn_name", "county_name"])
    p["township"] = p["county_name"] + p["twn_name"]
    turn = p.groupby("township", as_index=False).agg(
        ballots=("ballots", "sum"), eligible=("eligible", "sum")
    )
    turn["turnout"] = turn["ballots"] / turn["eligible"].replace(0, np.nan)
    out = out.merge(turn[["township", "turnout", "eligible", "ballots"]], on="township", how="left")

    out["election_year"] = year
    out["has_tpp_candidate"] = has_tpp
    out["race"] = "district_legislator"

    nat_dpp = out["dpp_votes"].sum() / out["valid_votes"].sum()
    nat_kmt = out["kmt_votes"].sum() / out["valid_votes"].sum()
    nat_tpp = out["tpp_votes"].sum() / out["valid_votes"].sum() if has_tpp else float("nan")
    n_cont = int((out["tpp_contested"] == 1).sum())
    print(
        f"  district {year}: n={len(out)} DPP={nat_dpp:.4f} KMT={nat_kmt:.4f} "
        f"TPP={nat_tpp if has_tpp else 'N/A'} tpp_contested_townships={n_cont}/{len(out)}"
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


def verify_district_and_list(votes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    list_rows = []
    for y in (2016, 2020, 2024):
        g = votes[votes.election_year == y]
        if len(g) == 0:
            continue
        tot = g["valid_votes"].sum()
        dist = {
            "dpp": g["dpp_votes"].sum() / tot,
            "kmt": g["kmt_votes"].sum() / tot,
            "tpp": (g["tpp_votes"].sum() / tot) if g["has_tpp_candidate"].any() else None,
        }
        for party, act in dist.items():
            rows.append(
                {
                    "year": y,
                    "ballot": "district_legislator",
                    "party": party,
                    "observed_national_share": act,
                    "note": "sum of township candidate votes by party / total valid",
                }
            )
        pl = load_party_list_national(y)
        list_rows.append(pl)
        known = KNOWN_PARTYLIST[y]
        for party in ("dpp", "kmt", "tpp"):
            exp = known[party]
            act = pl[f"{party}_list_share"]
            if exp is None and act is None:
                ok, diff = True, None
            elif exp is None or act is None:
                ok, diff = False, None
            else:
                diff = abs(act - exp)
                ok = diff < 1e-6  # exact parse of elretks
            rows.append(
                {
                    "year": y,
                    "ballot": "party_list",
                    "party": party,
                    "expected_from_elretks_const": exp,
                    "observed_national_share": act,
                    "abs_diff": diff,
                    "ok": ok,
                    "note": "CEC elretks national %",
                }
            )
            print(f"  list verify {y} {party}: {act} vs {exp} ok={ok}")
    pd.DataFrame(list_rows).to_csv(OUT / "partylist_national_from_elretks.csv", index=False)
    return pd.DataFrame(rows)


def build_panel() -> pd.DataFrame:
    print("Building legislative district township votes...")
    parts = [township_district_party_votes(y) for y in (2016, 2020, 2024)]
    votes = pd.concat(parts, ignore_index=True)
    votes.to_csv(OUT / "township_district_party_votes.csv", index=False)

    print("Verification (district aggregates + party-list elretks):")
    ver = verify_district_and_list(votes)
    ver.to_csv(OUT / "national_share_verification_legislative.csv", index=False)

    income = pd.read_csv(CLEAN / "income_township_panel_full.csv")
    pop = pd.read_csv(CLEAN / "moi_township_pop_density_panel.csv")
    edu_cache = {ry: load_education_rate(ry) for ry in sorted(set(EDU_FILE_ROC.values()))}

    frames = []
    for y in (2016, 2020, 2024):
        vy = votes[votes.election_year == y].copy()
        prev_y = y - 4
        prev_cols = [
            "township",
            "dpp_share",
            "kmt_share",
            "tpp_share",
            "others_share",
            "turnout",
            "tpp_contested",
        ]
        if prev_y in votes.election_year.values:
            prev = votes[votes.election_year == prev_y][prev_cols].rename(
                columns={
                    "dpp_share": "prev_dpp_share",
                    "kmt_share": "prev_kmt_share",
                    "tpp_share": "prev_tpp_share",
                    "others_share": "prev_others_share",
                    "turnout": "prev_turnout",
                    "tpp_contested": "prev_tpp_contested",
                }
            )
            vy = vy.merge(prev, on="township", how="left")
        else:
            for c in (
                "prev_dpp_share",
                "prev_kmt_share",
                "prev_tpp_share",
                "prev_others_share",
                "prev_turnout",
                "prev_tpp_contested",
            ):
                vy[c] = np.nan

        prev2_y = y - 8
        if prev2_y in votes.election_year.values:
            p2 = votes[votes.election_year == prev2_y][["township", "dpp_share", "kmt_share"]].rename(
                columns={"dpp_share": "prev2_dpp", "kmt_share": "prev2_kmt"}
            )
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
    panel["tpp_swing"] = panel["tpp_share"] - panel["prev_tpp_share"]
    panel["dpp_kmt_margin_prev"] = (panel["prev_dpp_share"] - panel["prev_kmt_share"]).abs()

    panel["country"] = "TW"
    panel["region"] = panel["township"]
    panel["year"] = panel["election_year"]
    panel["unemployment"] = np.nan
    panel["poll_index"] = np.nan

    out_cols = [
        "country", "region", "year", "county_name", "geo_region", "race",
        "valid_votes", "dpp_votes", "kmt_votes", "tpp_votes", "others_votes",
        "dpp_share", "kmt_share", "tpp_share", "others_share",
        "has_tpp_candidate", "tpp_contested", "turnout",
        "prev_dpp_share", "prev_kmt_share", "prev_tpp_share", "prev_others_share",
        "prev_turnout", "prev_tpp_contested", "prev_dpp_swing", "prev_kmt_swing",
        "dpp_swing", "kmt_swing", "tpp_swing", "dpp_kmt_margin_prev",
        "median_income", "urbanization", "education_rate", "log_people",
        "population_density", "unemployment", "poll_index",
        "income_roc_used", "pop_roc_used", "edu_roc_used",
    ]
    harm = panel[out_cols].copy()
    harm.to_csv(OUT / "harmonized_tw_legislative.csv", index=False)
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


def backtest_party_level(data, party, share_col, prev_col, feats):
    d = data.dropna(subset=[share_col]).copy()
    results = []
    models = get_models()
    for test_year in sorted(d["year"].unique()):
        train = d[d["year"] < test_year]
        test = d[d["year"] == test_year]
        if len(train) == 0 or len(test) == 0:
            continue
        # PreviousShare only if prev available on test
        te_prev = test.dropna(subset=[prev_col])
        if len(te_prev) >= 20:
            results.append(
                metrics_row(
                    party, "level", test_year, "PreviousShare",
                    te_prev[share_col], te_prev[prev_col],
                    len(train), len(te_prev), "out_of_time",
                )
            )
        use_feats = [c for c in feats if c in train.columns]
        tr, te = impute(train, test, use_feats)
        # Drop features that are entirely NaN in raw train (e.g. all prev_* in 2016)
        usable = []
        for c in use_feats:
            if train[c].notna().sum() >= max(30, int(0.3 * len(train))):
                usable.append(c)
        if not usable:
            continue
        tr = tr.dropna(subset=usable + [share_col])
        te = te.dropna(subset=usable + [share_col])
        if len(tr) < 50 or len(te) < 20:
            continue
        y_test = te[share_col]
        for name, model in models.items():
            model.fit(tr[usable], tr[share_col])
            pred = model.predict(te[usable])
            results.append(
                metrics_row(
                    party, "level", test_year, name, y_test, pred,
                    len(tr), len(te), "out_of_time",
                )
            )
    return results


def backtest_party_swing(data, party, swing_col, prev_col, feats):
    d = data.dropna(subset=[swing_col]).copy()
    results = []
    models = get_models()
    for test_year in sorted(d["year"].unique()):
        train = d[d["year"] < test_year]
        test = d[d["year"] == test_year]
        if len(train) == 0 or len(test) == 0:
            continue
        use_feats = [c for c in feats if c in train.columns]
        tr, te = impute(train, test, use_feats)
        usable = [c for c in use_feats if train[c].notna().sum() >= max(30, int(0.3 * len(train)))]
        tr = tr.dropna(subset=usable + [swing_col]) if usable else tr.iloc[0:0]
        te = te.dropna(subset=usable + [swing_col]) if usable else te.iloc[0:0]
        if len(te) < 20:
            continue
        y_test = te[swing_col] if len(te) else test[swing_col]
        # ZeroSwing baseline on all test rows with defined swing
        te0 = test.dropna(subset=[swing_col])
        results.append(
            metrics_row(
                party, "swing", test_year, "ZeroSwing",
                te0[swing_col], np.zeros(len(te0)),
                len(train), len(te0), "out_of_time",
            )
        )
        prev_swing_col = f"prev_{party.lower()}_swing"
        if prev_swing_col in te0.columns and te0[prev_swing_col].notna().sum() >= 20:
            ps = te0[prev_swing_col].fillna(0)
            results.append(
                metrics_row(
                    party, "swing", test_year, "PreviousSwing",
                    te0[swing_col], ps, len(train), len(te0), "out_of_time",
                )
            )
        if len(tr) < 50 or not usable:
            continue
        for name, model in models.items():
            model.fit(tr[usable], tr[swing_col])
            pred = model.predict(te[usable])
            results.append(
                metrics_row(
                    party, "swing", test_year, name, te[swing_col], pred,
                    len(tr), len(te), "out_of_time",
                )
            )
    return results


def tpp_2020_baselines(data: pd.DataFrame) -> list[dict]:
    """2020 is first TPP district year — no prev_tpp. Baselines + LOCO only."""
    results = []
    d = data[(data["year"] == 2020) & data["tpp_share"].notna()].copy()
    if len(d) < 20:
        return results
    y = d["tpp_share"]
    results.append(
        metrics_row("TPP", "level", 2020, "ZeroTPP", y, np.zeros(len(d)), 0, len(d), "no_history_first_tpp_year")
    )
    # contested-aware: predict 0 where not contested, national mean of contested elsewhere — still descriptive
    results.append(
        metrics_row(
            "TPP", "level", 2020, "NationalMeanTPP", y, np.full(len(d), y.mean()),
            0, len(d), "in_sample_mean_oracle",
        )
    )
    return results


def loco_2024(data: pd.DataFrame) -> list[dict]:
    d = data[data["year"] == 2024].copy()
    results = []
    models = get_models()
    for party, col, feats in (
        ("DPP", "dpp_share", DPP_KMT_FEATURES),
        ("KMT", "kmt_share", DPP_KMT_FEATURES),
        ("TPP", "tpp_share", TPP_FEATURES),
    ):
        dd = d.dropna(subset=[col, "county_name"]).copy()
        feats_u = [c for c in feats if c in dd.columns]
        dd = dd.dropna(subset=feats_u + [col])
        if len(dd) < 50:
            continue
        counties = sorted(dd["county_name"].unique())
        for name in models:
            preds = np.zeros(len(dd))
            idx_map = {ix: i for i, ix in enumerate(dd.index)}
            for county in counties:
                tr = dd[dd["county_name"] != county]
                te = dd[dd["county_name"] == county]
                if len(tr) < 30 or len(te) == 0:
                    continue
                tr2, te2 = impute(tr, te, feats_u)
                m = get_models()[name]
                m.fit(tr2[feats_u], tr2[col])
                p = m.predict(te2[feats_u])
                for ix, val in zip(te2.index, p):
                    preds[idx_map[ix]] = val
            results.append(
                metrics_row(
                    party, "level", 2024, name, dd[col].values, preds,
                    len(dd) - 1, len(dd), "leave_one_county_out_2024_NOT_temporal",
                )
            )
    return results


def ablation_and_importance(data, party, share_col, feats, test_year=2024):
    feature_sets = {
        "History": [c for c in feats if c.startswith("prev_") or c in ("dpp_kmt_margin_prev", "tpp_contested")],
        "History+Economy": [
            c for c in feats
            if c.startswith("prev_") or c in ("dpp_kmt_margin_prev", "tpp_contested", "median_income")
        ],
        "History+Demo": [
            c for c in feats
            if c.startswith("prev_")
            or c in ("dpp_kmt_margin_prev", "tpp_contested", "urbanization", "education_rate", "log_people")
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
        # require some history coverage in train for History sets
        usable = [c for c in fs if train[c].notna().sum() >= max(30, int(0.3 * len(train)))]
        if not usable:
            continue
        tr, te = impute(train, test, usable)
        tr = tr.dropna(subset=usable + [share_col])
        te = te.dropna(subset=usable + [share_col])
        if len(tr) < 50:
            continue
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
        rf.fit(tr[usable], tr[share_col])
        pred = rf.predict(te[usable])
        abl_rows.append(
            {
                "party": party,
                "feature_set": set_name,
                "test_year": test_year,
                "n_features": len(usable),
                "MAE": mean_absolute_error(te[share_col], pred),
                "RMSE": float(np.sqrt(mean_squared_error(te[share_col], pred))),
                "R2": r2_score(te[share_col], pred),
                "eval_type": "out_of_time",
            }
        )

    imp_rows = []
    feats_u = [c for c in feats if c in d.columns]
    if party == "TPP" and d[d["year"] < test_year]["tpp_share"].notna().sum() < 50:
        # Prefer OOT if 2020 TPP exists as train for 2024
        pass
    tr = d[d["year"] < test_year]
    usable = [c for c in feats_u if tr[c].notna().sum() >= max(30, int(0.3 * len(tr)))] if len(tr) else []
    if party == "TPP":
        tr = tr.dropna(subset=[share_col])
    if len(tr) > 50 and usable:
        tr2, _ = impute(tr, tr, usable)
        tr2 = tr2.dropna(subset=usable + [share_col])
        if len(tr2) > 50:
            rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
            rf.fit(tr2[usable], tr2[share_col])
            for f, v in zip(usable, rf.feature_importances_):
                imp_rows.append(
                    {
                        "party": party,
                        "feature": f,
                        "importance": float(v),
                        "fit": f"train_years_lt_{test_year}",
                        "note": "out_of_time RF impurity",
                    }
                )
    elif party == "TPP":
        d24 = d[d["year"] == 2024].dropna(subset=feats_u + [share_col])
        if len(d24) > 50:
            rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
            rf.fit(d24[feats_u], d24[share_col])
            for f, v in zip(feats_u, rf.feature_importances_):
                imp_rows.append(
                    {
                        "party": party,
                        "feature": f,
                        "importance": float(v),
                        "fit": "in_sample_2024_impurity",
                        "note": "fallback descriptive",
                    }
                )
    return abl_rows, imp_rows


def compare_pres_vs_leg():
    """Brief MAE comparison vs presidential three-party results if present."""
    pres_path = ROOT / "fyp_taiwan_parties" / "backtest_results_parties.csv"
    if not pres_path.exists():
        return pd.DataFrame()
    pres = pd.read_csv(pres_path)
    leg = pd.read_csv(OUT / "backtest_results_parties.csv")
    rows = []
    for race, df in (("presidential", pres), ("legislative_district", leg)):
        sub = df[
            (df["target"] == "level")
            & (df["test_year"] == 2024)
            & (df["eval_type"].isin(["out_of_time", "out_of_time_no_history", "no_history_first_tpp_year"]))
        ]
        for party in ("DPP", "KMT", "TPP"):
            for model in ("PreviousShare", "RandomForest", "Ridge", "HistGB", "ZeroTPP", "PrevThirdPartyProxy"):
                r = sub[(sub["party"] == party) & (sub["model"] == model)]
                if len(r) == 0:
                    continue
                rows.append(
                    {
                        "race": race,
                        "party": party,
                        "model": model,
                        "test_year": 2024,
                        "MAE": r.iloc[0]["MAE"],
                        "eval_type": r.iloc[0]["eval_type"],
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "compare_pres_vs_leg_2024.csv", index=False)
    return out


def make_charts(backtest_df, imp_df, loco_df, votes_df):
    color_map = {"DPP": "#1f7a1f", "KMT": "#1f4e79", "TPP": "#c9a227"}

    # National district shares by year
    fig, ax = plt.subplots(figsize=(8, 4))
    years = [2016, 2020, 2024]
    x = np.arange(len(years))
    width = 0.25
    for i, (party, col) in enumerate((("DPP", "dpp_share"), ("KMT", "kmt_share"), ("TPP", "tpp_share"))):
        vals = []
        for y in years:
            g = votes_df[votes_df.election_year == y]
            if col == "tpp_share" and not g["has_tpp_candidate"].any():
                vals.append(0.0)
            else:
                vals.append(g[col.replace("_share", "_votes")].sum() / g["valid_votes"].sum())
        ax.bar(x + (i - 1) * width, vals, width, label=party, color=color_map[party])
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("National district vote share")
    ax.set_title("區域立委 party vote share (township-summed CEC)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "district_national_shares_by_year.png", dpi=140)
    plt.close()

    # 2024 MAE by party
    fig, ax = plt.subplots(figsize=(10, 5))
    sub = backtest_df[
        (backtest_df["test_year"] == 2024)
        & (backtest_df["target"] == "level")
        & (backtest_df["eval_type"] == "out_of_time")
    ]
    plot_models = {
        "DPP": ["PreviousShare", "RandomForest", "Ridge"],
        "KMT": ["PreviousShare", "RandomForest", "Ridge"],
        "TPP": ["PreviousShare", "RandomForest", "Ridge"],
    }
    xs, heights, colors, labels = [], [], [], []
    i = 0
    for party, models in plot_models.items():
        for m in models:
            row = sub[(sub["party"] == party) & (sub["model"] == m)]
            if len(row) == 0:
                continue
            xs.append(i)
            heights.append(row.iloc[0]["MAE"])
            colors.append(color_map[party])
            labels.append(f"{party}\n{m}")
            i += 1
    if xs:
        ax.bar(xs, heights, color=colors)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("MAE (vote share)")
        ax.set_title("2024 區域立委 township MAE by party (out-of-time)")
        fig.tight_layout()
        fig.savefig(CHARTS / "mae_2024_legislative_by_party.png", dpi=140)
    plt.close()

    if len(imp_df):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for ax, party in zip(axes, ["DPP", "KMT", "TPP"]):
            s = imp_df[imp_df["party"] == party].sort_values("importance", ascending=True)
            if len(s) == 0:
                ax.set_visible(False)
                continue
            ax.barh(s["feature"], s["importance"], color=color_map[party])
            ax.set_title(f"{party} RF importance")
        fig.suptitle("Legislative district RF impurity importance (township ecological)")
        fig.tight_layout()
        fig.savefig(CHARTS / "feature_importance_legislative.png", dpi=140)
        plt.close()

    if len(loco_df):
        fig, ax = plt.subplots(figsize=(8, 4))
        sub = loco_df[loco_df["model"] == "RandomForest"]
        parties = ["DPP", "KMT", "TPP"]
        maes = [
            sub[sub["party"] == p]["MAE"].values[0] if len(sub[sub["party"] == p]) else np.nan
            for p in parties
        ]
        ax.bar(parties, maes, color=[color_map[p] for p in parties])
        ax.set_ylabel("MAE")
        ax.set_title("2024 LOCO MAE — legislative district (NOT temporal)")
        fig.tight_layout()
        fig.savefig(CHARTS / "loco_2024_legislative.png", dpi=140)
        plt.close()

    # Pres vs leg chart if available
    cmp_path = OUT / "compare_pres_vs_leg_2024.csv"
    if cmp_path.exists():
        cmp = pd.read_csv(cmp_path)
        fig, ax = plt.subplots(figsize=(9, 4))
        focus = cmp[cmp["model"].isin(["PreviousShare", "RandomForest"])]
        # grouped bars
        parties = ["DPP", "KMT", "TPP"]
        x = np.arange(len(parties))
        w = 0.2
        for j, (race, model) in enumerate(
            (("presidential", "PreviousShare"), ("presidential", "RandomForest"),
             ("legislative_district", "PreviousShare"), ("legislative_district", "RandomForest"))
        ):
            vals = []
            for p in parties:
                r = focus[(focus.race == race) & (focus.party == p) & (focus.model == model)]
                vals.append(r.iloc[0]["MAE"] if len(r) else np.nan)
            ax.bar(x + (j - 1.5) * w, vals, w, label=f"{race[:4]}:{model[:4]}")
        ax.set_xticks(x)
        ax.set_xticklabels(parties)
        ax.set_ylabel("MAE")
        ax.set_title("2024 Pres vs 區域立委 MAE (where comparable)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(CHARTS / "pres_vs_leg_2024_mae.png", dpi=140)
        plt.close()


def main():
    panel = build_panel()
    votes = pd.read_csv(OUT / "township_district_party_votes.csv")

    all_results = []
    print("\n=== DPP/KMT out-of-time LEVEL (legislative district) ===")
    all_results += backtest_party_level(panel, "DPP", "dpp_share", "prev_dpp_share", DPP_KMT_FEATURES)
    all_results += backtest_party_level(panel, "KMT", "kmt_share", "prev_kmt_share", DPP_KMT_FEATURES)

    print("=== TPP level (2020 no history; 2024 has prev from 2020) ===")
    all_results += tpp_2020_baselines(panel)
    all_results += backtest_party_level(panel, "TPP", "tpp_share", "prev_tpp_share", TPP_FEATURES)

    print("=== Swing where lag exists ===")
    all_results += backtest_party_swing(panel, "DPP", "dpp_swing", "prev_dpp_share", DPP_KMT_FEATURES)
    all_results += backtest_party_swing(panel, "KMT", "kmt_swing", "prev_kmt_share", DPP_KMT_FEATURES)
    all_results += backtest_party_swing(panel, "TPP", "tpp_swing", "prev_tpp_share", TPP_FEATURES)

    print("=== LOCO 2024 (NOT temporal) ===")
    loco = loco_2024(panel)
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

    # 2024 compare table
    cmp24 = bt[(bt.test_year == 2024) & (bt.target == "level")].copy()
    cmp24.to_csv(OUT / "compare_2024_mae_by_party.csv", index=False)

    compare_pres_vs_leg()
    make_charts(bt, imp_df, pd.DataFrame(loco), votes)

    summary = {
        "unit": "township (鄉鎮市區)",
        "race": "regional district legislator (區域立委)",
        "years": [2016, 2020, 2024],
        "parties": {"DPP": 16, "KMT": 1, "TPP": 350},
        "tpp_note": "No TPP in 2016; 2020+2024 district presence; party-list national only in elretks",
        "n_rows_harmonized": int(len(panel)),
        "district_national_2024": {
            "dpp": float(
                votes.loc[votes.election_year == 2024, "dpp_votes"].sum()
                / votes.loc[votes.election_year == 2024, "valid_votes"].sum()
            ),
            "kmt": float(
                votes.loc[votes.election_year == 2024, "kmt_votes"].sum()
                / votes.loc[votes.election_year == 2024, "valid_votes"].sum()
            ),
            "tpp": float(
                votes.loc[votes.election_year == 2024, "tpp_votes"].sum()
                / votes.loc[votes.election_year == 2024, "valid_votes"].sum()
            ),
        },
        "partylist_national_elretks": {y: KNOWN_PARTYLIST[y] for y in (2016, 2020, 2024)},
    }
    # add key MAE
    for party in ("DPP", "KMT", "TPP"):
        for model in ("PreviousShare", "RandomForest", "ZeroTPP"):
            r = bt[
                (bt.party == party)
                & (bt.model == model)
                & (bt.test_year == 2024)
                & (bt.target == "level")
            ]
            if len(r):
                summary.setdefault("mae_2024_level", {})[f"{party}_{model}"] = float(r.iloc[0]["MAE"])

    with open(OUT / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # README
    readme = f"""# 臺灣立委三黨分析（區域立委・鄉鎮層）

## 單位與範圍
- **地理單位**：鄉鎮市區（與總統 FYP 對齊，方便比較）
- **選票**：中選會 **區域立委** 候選人得票，依政黨代碼加總（DPP=16、KMT=1、TPP=350）
- **年份**：2016、2020、2024（本資料包無 2012 區域立委）
- **不分區**：僅有全國 `elretks`（見 `partylist_national_from_elretks.csv`），**無**鄉鎮票檔，故不入預測模型

## 全國核對（CEC）
### 區域立委（鄉鎮加總）
見 `run_summary.json` → `district_national_2024`（約 DPP 45.2% / KMT 40.4% / TPP 3.0%——因民眾黨只在部分選區提名，遠低於不分區）。

### 不分區政黨票（elretks）
| 年 | DPP | KMT | TPP |
|----|-----|-----|-----|
| 2016 | 44.06% | 26.91% | — |
| 2020 | 33.98% | 33.36% | 11.22% |
| 2024 | 36.16% | 34.58% | 22.07% |

## 模型
PreviousShare、Ridge、RandomForest、HistGB；swing 有 lag 時加 ZeroSwing。
TPP：2016 無；2020 無上屆；**2024 可用 2020 區域立委 TPP 作 PreviousShare**（比總統 TPP 條件更好）。

## 輸出
- `harmonized_tw_legislative.csv`、`township_district_party_votes.csv`
- `backtest_results_parties.csv`、`feature_importance_by_party.csv`、`ablation_by_party.csv`
- `compare_pres_vs_leg_2024.csv`、`charts/`

## 注意
- 區域得票率 ≠ 不分區得票率（候選人結構／提名範圍不同）
- `tpp_contested`：該鄉鎮所屬選區是否有 TPP 候選人
- LOCO 標明 **NOT temporal**
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print("Done.", json.dumps(summary.get("mae_2024_level", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()
