#!/usr/bin/env python3
"""Fill must-have checklist Extension layer from PUBLIC / in-repo data.
Does NOT modify Family A locked features or MAE.
Asia/Hong_Kong timestamps in notes.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "extension_must_have"
FUND = ROOT / "data" / "extension_fundamentals_proxy"
DOCS = ROOT / "docs"
RESULTS = ROOT / "results" / "must_have_processing_demo"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

HK = timezone(timedelta(hours=8))
NOW = datetime.now(HK).strftime("%Y-%m-%d %H:%M Asia/Hong_Kong UTC+8")

ELECTION_DATES = {
    ("TW", 2000): "2000-03-18",
    ("TW", 2004): "2004-03-20",
    ("TW", 2008): "2008-03-22",
    ("TW", 2012): "2012-01-14",
    ("TW", 2016): "2016-01-16",
    ("TW", 2020): "2020-01-11",
    ("TW", 2024): "2024-01-13",
    ("US", 2000): "2000-11-07",
    ("US", 2004): "2004-11-02",
    ("US", 2008): "2008-11-04",
    ("US", 2012): "2012-11-06",
    ("US", 2016): "2016-11-08",
    ("US", 2020): "2020-11-03",
    ("US", 2024): "2024-11-05",
}

# Coder incumbency (same as prior fundamentals proxy)
INCUMBENCY = {
    ("US", 2000): ("DEM", 0),  # Clinton term-limited; Gore running as VP not sitting president
    ("US", 2004): ("GOP", 1),
    ("US", 2008): ("GOP", 0),
    ("US", 2012): ("DEM", 1),
    ("US", 2016): ("DEM", 0),
    ("US", 2020): ("GOP", 1),
    ("US", 2024): ("DEM", 0),  # Biden withdrew; Harris not sitting president — incumbent_party DEM
    ("TW", 2000): ("KMT", 0),
    ("TW", 2004): ("DPP", 1),
    ("TW", 2008): ("DPP", 0),
    ("TW", 2012): ("KMT", 1),
    ("TW", 2016): ("KMT", 0),
    ("TW", 2020): ("DPP", 1),
    ("TW", 2024): ("DPP", 0),  # Tsai term-limited; Lai not sitting president
}


def _party_from_target(t: str) -> str | None:
    t = str(t)
    if "undecided" in t.lower() or "未表態" in t or "未決定" in t or "未決定／不知道" in t or "不知道" in t or "無意見" in t:
        return "UNDECIDED"
    if re.search(r"民進|DPP|蔡|Tsai|賴清德|陳水扁|謝長廷|Hsieh|Chen Shui", t, re.I):
        return "DPP"
    if re.search(r"國民|KMT|連戰|馬英九|Ma |韓國瑜|朱立倫|侯友宜|洪秀柱", t, re.I):
        return "KMT"
    if re.search(r"親民|宋楚瑜|Soong|PFP", t, re.I):
        return "OTHER"
    if re.search(r"民眾|柯文哲|TPP", t, re.I):
        return "OTHER"
    if re.search(r"無黨|郭台銘", t, re.I):
        return "OTHER"
    return None


def build_tw_poll_averages() -> pd.DataFrame:
    long = pd.read_csv(ROOT / "data/extension_poll_platform/taiwan_polls_long.csv")
    long["field_end"] = pd.to_datetime(long["field_end"], errors="coerce", utc=True).dt.tz_localize(None)
    long["year"] = long["poll_id"].astype(str).str[:4].astype(int)
    long["party"] = long["target"].map(_party_from_target)
    long = long[long["party"].notna()].copy()
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    rows = []
    for year, ed in [(y, ELECTION_DATES[("TW", y)]) for y in [2000, 2004, 2008, 2012, 2016, 2020, 2024]]:
        ed_dt = pd.Timestamp(ed)
        win_start = ed_dt - pd.Timedelta(days=45)
        sub = long[(long["year"] == year) & (long["field_end"].notna())]
        sub = sub[(sub["field_end"] >= win_start) & (sub["field_end"] <= ed_dt)]
        if sub.empty:
            sub = long[long["year"] == year]
            note = "fallback_all_year_polls_thin_window"
        else:
            note = "pre_election_45d_window"
        # simple unweighted mean by party of last poll per pollster-party (dedupe by poll_id)
        # Use mean across polls: group poll_id x party -> mean value then mean across polls
        g = sub.groupby(["poll_id", "party"], as_index=False)["value"].mean()
        means = g.groupby("party")["value"].agg(["mean", "std", "count"])
        # last as-of
        asof = sub["field_end"].max()
        # undecided
        und = means.loc["UNDECIDED", "mean"] if "UNDECIDED" in means.index else np.nan
        dpp = means.loc["DPP", "mean"] if "DPP" in means.index else np.nan
        kmt = means.loc["KMT", "mean"] if "KMT" in means.index else np.nan
        other = means.loc["OTHER", "mean"] if "OTHER" in means.index else np.nan
        # two-party-ish among DPP+KMT
        if pd.notna(dpp) and pd.notna(kmt) and (dpp + kmt) > 0:
            dpp_2p = dpp / (dpp + kmt)
        else:
            dpp_2p = np.nan
        # house effect sketch: pollster mean residual vs overall for DPP in window
        house_note = "sketch_only"
        house_rows_n = 0
        if "DPP" in g["party"].values:
            # merge pollster
            pid_pollster = sub.drop_duplicates("poll_id")[["poll_id", "pollster"]]
            gg = g.merge(pid_pollster, on="poll_id", how="left")
            dpp_g = gg[gg["party"] == "DPP"]
            if len(dpp_g) >= 5 and dpp_g["pollster"].nunique() >= 3:
                overall = dpp_g["value"].mean()
                he = dpp_g.groupby("pollster")["value"].mean() - overall
                house_rows_n = int((he.abs() > 0).sum())
                he.to_csv(OUT / f"tw_house_effects_sketch_{year}.csv", header=["dpp_mean_minus_overall"])
                house_note = f"pollster_fixed_effect_sketch_n={house_rows_n}"
            else:
                house_note = "insufficient_pollsters_in_window"
        rows.append(
            {
                "country": "TW",
                "election_year": year,
                "as_of_date": None if pd.isna(asof) else asof.date().isoformat(),
                "window": note,
                "dpp_pct_mean": None if pd.isna(dpp) else round(float(dpp), 3),
                "kmt_pct_mean": None if pd.isna(kmt) else round(float(kmt), 3),
                "other_pct_mean": None if pd.isna(other) else round(float(other), 3),
                "undecided_pct_mean": None if pd.isna(und) else round(float(und), 3),
                "dpp_two_party_share_poll": None if pd.isna(dpp_2p) else round(float(dpp_2p), 6),
                "dpp_std": None if "DPP" not in means.index or pd.isna(means.loc["DPP", "std"]) else round(float(means.loc["DPP", "std"]), 3),
                "n_poll_party_rows": int(len(g)),
                "house_effects_note": house_note,
                "expectation_polls": "GAP_no_systematic_public_archive",
                "family_a": "NOT_INCLUDED",
                "source": "taiwan_polls_long Extension simple average (not Bayesian)",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tw_national_poll_averages_extension.csv", index=False)
    return df


def build_us_poll_averages() -> pd.DataFrame:
    """Expand US voting-intention averages from existing 538/parity files."""
    year_proxy = pd.read_csv(ROOT / "data/us_parity_extension/us_national_poll_election_year_proxy.csv")
    daily = pd.read_csv(ROOT / "data/us_parity_extension/us_polls_national_daily_538.csv")
    legacy = pd.read_csv(
        ROOT / "data/us_parity_extension/us_polls_national_daily_538_legacy_1968_2016_national.csv"
    )
    rows = []
    for _, r in year_proxy.iterrows():
        y = int(r["election_year"])
        dem, gop = r["dem_pct"], r["gop_pct"]
        # try get sd from daily near end
        sd = np.nan
        n = np.nan
        sub = daily[daily["election_year"] == y]
        if sub.empty and "cycle" in legacy.columns:
            # legacy format differs — skip sd
            pass
        elif not sub.empty:
            # value column may be pct
            vals = pd.to_numeric(sub.get("pct_estimate", sub.get("value")), errors="coerce")
            # crude: take dem side last 14 days std of daily estimate if available
            if "side" in sub.columns:
                dem_s = sub[sub["side"].astype(str).str.lower().isin(["dem", "democratic", "clinton", "biden", "harris", "obama", "gore", "kerry"])]
            else:
                dem_s = sub
            if "field_end" in dem_s.columns:
                dem_s = dem_s.copy()
                dem_s["field_end"] = pd.to_datetime(dem_s["field_end"], errors="coerce")
                dem_s = dem_s.sort_values("field_end")
                tail = dem_s.tail(14)
                v = pd.to_numeric(tail.get("pct_estimate", tail.get("value")), errors="coerce").dropna()
                if len(v) >= 3:
                    sd = float(v.std())
                    n = int(len(v))
        rows.append(
            {
                "country": "US",
                "election_year": y,
                "as_of_date": r.get("as_of_date"),
                "dem_pct": dem,
                "gop_pct": gop,
                "dem_two_party_share_poll": r.get("dem_two_party_share_poll"),
                "poll_sd_proxy_last14": None if pd.isna(sd) else round(sd, 3),
                "n_daily_tail": None if pd.isna(n) else int(n),
                "undecided_pct": None,
                "undecided_note": "538 historical averages / national daily lack stable undecided column",
                "expectation_polls": "GAP_no_systematic_public_archive_in_repo",
                "house_effects": "GAP_full_house_effects_model_not_built; 538 raw averages may embed adjustments",
                "source": r.get("source"),
                "source_url": r.get("source_url"),
                "family_a": "NOT_INCLUDED",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "us_national_poll_averages_extension.csv", index=False)
    return df


def build_partisan_floor_and_turnout() -> tuple[pd.DataFrame, pd.DataFrame]:
    tw = pd.read_csv(ROOT / "cleaned/county_panel_features_enriched_2000_2024.csv")
    # shares
    tw["dpp_share"] = tw["DPP_votes"] / tw["valid_votes"]
    tw["kmt_share"] = tw["KMT_votes"] / tw["valid_votes"]
    # expanding min of past elections (exclude current) = floor as of year
    tw = tw.sort_values(["region_id", "year"])
    floors = []
    for rid, g in tw.groupby("region_id"):
        g = g.sort_values("year")
        past_dpp, past_kmt = [], []
        for _, row in g.iterrows():
            floor_dpp = min(past_dpp) if past_dpp else np.nan
            floor_kmt = min(past_kmt) if past_kmt else np.nan
            floors.append(
                {
                    "country": "TW",
                    "region_id": rid,
                    "region_name": row["region_name"],
                    "year": int(row["year"]),
                    "partisan_floor_dpp": None if pd.isna(floor_dpp) else round(float(floor_dpp), 6),
                    "partisan_floor_kmt": None if pd.isna(floor_kmt) else round(float(floor_kmt), 6),
                    "n_past_elections": len(past_dpp),
                    "turnout": row["turnout"],
                    "prev_turnout": row.get("prev_turnout"),
                    "electorate": row["electorate"],
                    "family_a": "NOT_INCLUDED_extension_derived",
                    "note": "floor=min share over past presidential elections in panel (iron-ballot PROXY)",
                }
            )
            if pd.notna(row["dpp_share"]):
                past_dpp.append(float(row["dpp_share"]))
            if pd.notna(row["kmt_share"]):
                past_kmt.append(float(row["kmt_share"]))
    tw_floor = pd.DataFrame(floors)
    tw_floor.to_csv(OUT / "tw_county_partisan_floor_turnout.csv", index=False)

    # national turnout + electorate change PROXY
    nat = (
        tw.groupby("year")
        .agg(
            electorate=("electorate", "sum"),
            votes_cast=("votes_cast", "sum"),
            valid=("valid_votes", "sum"),
        )
        .reset_index()
    )
    nat["turnout_rate"] = nat["votes_cast"] / nat["electorate"]
    nat["electorate_change_pct"] = nat["electorate"].pct_change()
    nat["new_voters_stock_change_proxy"] = nat["electorate_change_pct"]
    nat["country"] = "TW"
    nat["note"] = "new_voters PROXY = national electorate % change vs prior presidential cycle (not first-time voter registry)"
    nat["family_a"] = "NOT_INCLUDED"
    nat.to_csv(OUT / "tw_national_turnout_electorate_proxy.csv", index=False)

    # US county
    us = pd.read_csv(ROOT / "fyp_us_core/us_county_panel_features_family_a.csv", usecols=[
        "region_id", "region_name", "state_po", "year",
        "target_dem_two_party_share", "prev_dem_two_party_share", "prev_gop_two_party_share",
        "prev2_dem_two_party_share", "prev2_gop_two_party_share",
    ])
    us = us.sort_values(["region_id", "year"])
    # reconstruct current dem share is target; for floor use expanding min of realized shares
    # We only have target as current dem two-party; use that history
    us_floors = []
    for rid, g in us.groupby("region_id"):
        g = g.sort_values("year")
        past_dem, past_gop = [], []
        for _, row in g.iterrows():
            us_floors.append(
                {
                    "country": "US",
                    "region_id": rid,
                    "region_name": row["region_name"],
                    "state_po": row["state_po"],
                    "year": int(row["year"]),
                    "partisan_floor_dem_2p": None if not past_dem else round(min(past_dem), 6),
                    "partisan_floor_gop_2p": None if not past_gop else round(min(past_gop), 6),
                    "n_past_elections": len(past_dem),
                    "family_a": "NOT_INCLUDED_extension_derived",
                    "note": "floor from expanding min of past target_dem_two_party_share / (1-target)",
                }
            )
            d = float(row["target_dem_two_party_share"])
            past_dem.append(d)
            past_gop.append(1.0 - d)
    us_floor = pd.DataFrame(us_floors)
    us_floor.to_csv(OUT / "us_county_partisan_floor.csv", index=False)
    return tw_floor, us_floor


def build_polarization_proxy() -> pd.DataFrame:
    tw = pd.read_csv(ROOT / "cleaned/county_panel_features_enriched_2000_2024.csv")
    tw["dpp_share"] = tw["DPP_votes"] / tw["valid_votes"]
    tw["kmt_share"] = tw["KMT_votes"] / tw["valid_votes"]
    tw["abs_margin"] = (tw["dpp_share"] - tw["kmt_share"]).abs()
    tw_nat = tw.groupby("year").agg(
        mean_abs_margin=("abs_margin", "mean"),
        sd_dpp_share=("dpp_share", "std"),
        mean_dpp=("dpp_share", "mean"),
    ).reset_index()
    tw_nat["country"] = "TW"
    tw_nat["polarization_proxy"] = tw_nat["mean_abs_margin"]
    tw_nat["definition"] = "mean county |DPP-KMT| share margin (cross-sectional)"

    us = pd.read_csv(ROOT / "fyp_us_core/us_county_panel_features_family_a.csv", usecols=[
        "year", "target_dem_two_party_share"
    ])
    us["abs_margin"] = (2 * us["target_dem_two_party_share"] - 1).abs()
    us_nat = us.groupby("year").agg(
        mean_abs_margin=("abs_margin", "mean"),
        sd_dem_share=("target_dem_two_party_share", "std"),
        mean_dem=("target_dem_two_party_share", "mean"),
    ).reset_index()
    us_nat["country"] = "US"
    us_nat["polarization_proxy"] = us_nat["mean_abs_margin"]
    us_nat["definition"] = "mean county |2*dem_2p-1| (two-party margin abs)"

    out = pd.concat([tw_nat, us_nat], ignore_index=True)
    out["family_a"] = "NOT_INCLUDED"
    out["note"] = "NOT same as affective polarization / party ID; geographic sorting PROXY only"
    out.to_csv(OUT / "polarization_proxy_national_by_year.csv", index=False)
    return out


def build_fundamentals_enriched() -> pd.DataFrame:
    """Merge existing fundamentals + TW unemp national + US approval where available."""
    base = pd.read_csv(FUND / "national_fundamentals_election_year_proxy.csv")
    # TW national unemployment from wide
    w = pd.read_csv(ROOT / "raw/economy/county_unemployment_rate_wide.csv")
    w["y"] = pd.to_numeric(w["year_or_ym"], errors="coerce")
    tw_u = w[w["y"].isin([2000, 2004, 2008, 2012, 2016, 2020, 2024])][
        ["y", "taiwan_area_pct"]
    ].rename(columns={"y": "year", "taiwan_area_pct": "tw_unrate_national"})

    # Trump approval from datahub — election years 2020 (as incumbent going into 2020? Trump was incumbent), 2024 not applicable for Biden
    appr_path = FUND / "us_trump_approval_topline_datahub.csv"
    approval_map = {}  # (country, year) -> (approve, note)
    if appr_path.exists():
        ap = pd.read_csv(appr_path)
        ap["modeldate"] = pd.to_datetime(ap["modeldate"], errors="coerce")
        ap = ap[ap["subgroup"].astype(str).str.lower().isin(["voters", "all polls", "all adults", "adults"])]
        # Prefer voters
        for year, ed in [(2020, "2020-11-03")]:
            ed_dt = pd.Timestamp(ed)
            sub = ap[(ap["modeldate"] <= ed_dt) & (ap["modeldate"] >= ed_dt - pd.Timedelta(days=14))]
            if sub.empty:
                sub = ap[ap["modeldate"] <= ed_dt].sort_values("modeldate").tail(5)
            if not sub.empty:
                # voters preferred
                v = sub[sub["subgroup"].str.lower() == "voters"]
                use = v if not v.empty else sub
                approval_map[("US", year)] = (
                    float(use.sort_values("modeldate").iloc[-1]["approve_estimate"]),
                    f"datahub 538 trump approval_topline asof {use.sort_values('modeldate').iloc[-1]['modeldate'].date()} subgroup={use.sort_values('modeldate').iloc[-1]['subgroup']}",
                )

    # Wikipedia career averages as WEAK PROXY only for documentation — do not invent election-eve
    # Leave blank for other years with gap note

    # TW undecided
    und = pd.read_csv(FUND / "tw_undecided_pre_election_proxy.csv")

    # electorate proxy
    tw_nat = pd.read_csv(OUT / "tw_national_turnout_electorate_proxy.csv") if (OUT / "tw_national_turnout_electorate_proxy.csv").exists() else None

    rows = []
    for country in ["US", "TW"]:
        years = [2000, 2004, 2008, 2012, 2016, 2020, 2024]
        for y in years:
            # start from base row if exists
            br = base[(base["country"] == country) & (base["year"] == y)]
            row = br.iloc[0].to_dict() if len(br) else {"country": country, "year": y, "grain": "national"}
            inc = INCUMBENCY.get((country, y), (None, None))
            row["incumbent_party"] = inc[0]
            row["incumbent_running"] = inc[1]
            if country == "TW":
                uu = tw_u[tw_u["year"] == y]
                if len(uu):
                    row["unrate"] = float(uu.iloc[0]["tw_unrate_national"])
                    row["unrate_asof"] = f"{y}-annual"
                    row["unrate_source"] = "DGBAS county_unemployment_rate_wide taiwan_area_pct (data.gov.tw/6640)"
                # CPI TW — gap this session (DGBAS filter API parameter error; MOL download blocked)
                row["cpi_yoy_pct"] = row.get("cpi_yoy_pct") if country == "US" else None
                row["cpi_note"] = "TW CPI YoY: GAP this session — DGBAS SDMX filter returned parameter error; MOL dataset 13228 download timed out"
                row["wages_note"] = "TW avg monthly wage: GAP — same MOL 13228 download blocked"
                row["cci_note"] = "TW consumer confidence: GAP — no free long FRED twin; NDC/CCI not downloaded"
                if tw_nat is not None:
                    tn = tw_nat[tw_nat["year"] == y]
                    if len(tn):
                        row["turnout_rate"] = float(tn.iloc[0]["turnout_rate"])
                        row["turnout_definition"] = "CEC sum votes_cast/electorate county panel"
                        row["new_voters_stock_change_proxy"] = (
                            None
                            if pd.isna(tn.iloc[0]["new_voters_stock_change_proxy"])
                            else float(tn.iloc[0]["new_voters_stock_change_proxy"])
                        )
                ud = und[und["year"] == y] if "year" in und.columns else und[und.get("year") == y]
                # und has year
                ud = und[und["year"] == y]
                if len(ud):
                    row["undecided_share"] = float(ud.iloc[0]["undecided_pct_mean_window"])
                    row["undecided_note"] = "TW polls_long undecided 45d mean"
                row["presidential_approval"] = None
                row["approval_note"] = "TW presidential satisfaction: GAP — scattered in poll misc, not extracted to series this round"
            else:
                # US: keep FRED fields from base; add approval if any
                if (country, y) in approval_map:
                    row["presidential_approval"] = round(approval_map[(country, y)][0], 3)
                    row["approval_note"] = approval_map[(country, y)][1]
                else:
                    row["presidential_approval"] = row.get("presidential_approval") or None
                    row["approval_note"] = (
                        "US approval election-eve series incomplete: 538 projects HTML-blocked; "
                        "datahub has Trump topline (used for 2020 where applicable); "
                        "pre-2017 Gallup election-eve not systematically ingested — GAP"
                    )
                row["wages_note"] = "US AHE (CES0500000003): GAP — FRED graph/API blocked without key this session"
                row["undecided_share"] = None
                row["undecided_note"] = "US historical 538 averages lack stable undecided column"
            row["family_a"] = "NOT_INCLUDED"
            row["built_at"] = NOW
            rows.append(row)
    df = pd.DataFrame(rows)
    # polarization merge
    pol = pd.read_csv(OUT / "polarization_proxy_national_by_year.csv")
    df = df.merge(
        pol[["country", "year", "polarization_proxy", "definition"]].rename(
            columns={"definition": "polarization_definition"}
        ),
        on=["country", "year"],
        how="left",
    )
    df.to_csv(OUT / "national_fundamentals_enriched_extension.csv", index=False)
    # also refresh FUND copy pointer
    df.to_csv(FUND / "national_fundamentals_enriched_extension.csv", index=False)
    return df


def build_age_structure_proxy() -> pd.DataFrame:
    """National age shares from MOI edu_age_marital village files (105/107 ROC = 2016/2018)."""
    files = {
        2016: ROOT / "raw/moi/edu_age_marital_105.csv",
        2018: ROOT / "raw/moi/edu_age_marital_107.csv",
    }
    rows = []
    for cal_year, path in files.items():
        if not path.exists():
            continue
        # large file — use chunks, only need 年齡 and 總計
        chunks = []
        for ch in pd.read_csv(path, usecols=["年齡", "總計"], chunksize=200000, encoding="utf-8"):
            ch["總計"] = pd.to_numeric(ch["總計"], errors="coerce").fillna(0)
            chunks.append(ch.groupby("年齡", as_index=False)["總計"].sum())
        if not chunks:
            continue
        g = pd.concat(chunks).groupby("年齡", as_index=False)["總計"].sum()
        total = g["總計"].sum()
        if total <= 0:
            continue
        g["share"] = g["總計"] / total
        # map age bands to broad groups
        def band(a):
            a = str(a)
            m = re.match(r"(\d+)", a)
            if not m:
                return "other"
            lo = int(m.group(1))
            if lo < 30:
                return "15_29"
            if lo < 45:
                return "30_44"
            if lo < 65:
                return "45_64"
            return "65plus"

        g["band"] = g["年齡"].map(band)
        bg = g.groupby("band")["總計"].sum()
        for band_name, val in bg.items():
            rows.append(
                {
                    "country": "TW",
                    "calendar_year": cal_year,
                    "moi_roc_year": cal_year - 1911,
                    "age_band": band_name,
                    "share": round(float(val / total), 6),
                    "count": int(val),
                    "grain": "national_from_village_sum",
                    "source": path.name,
                    "family_a": "NOT_INCLUDED",
                    "note": "PROXY — not election-year aligned county panel; 105/107 only",
                }
            )
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(OUT / "tw_national_age_structure_proxy_moi.csv", index=False)
    else:
        df = pd.DataFrame(
            [{"country": "TW", "status": "GAP", "note": "MOI age files missing or unreadable"}]
        )
        df.to_csv(OUT / "tw_national_age_structure_proxy_moi.csv", index=False)
    # US ACS age — not in repo as joinable election panel
    gap = pd.DataFrame(
        [
            {
                "country": "US",
                "status": "GAP",
                "note": "ACS age-by-county election-year panel not built this round; would need Census API join",
                "family_a": "NOT_INCLUDED",
            }
        ]
    )
    gap.to_csv(OUT / "us_age_structure_STATUS.csv", index=False)
    return df


def processing_demo(tw_polls: pd.DataFrame, us_polls: pd.DataFrame, fund: pd.DataFrame) -> None:
    """National illustration: simple average, turnout sketch, Monte Carlo DEMO."""
    rng = np.random.default_rng(20261030)
    demo_rows = []
    mc_summary = []

    # TW: compare poll dpp_2p to actual national DPP share
    tw = pd.read_csv(ROOT / "cleaned/county_panel_features_enriched_2000_2024.csv")
    actual = tw.groupby("year").apply(
        lambda g: g["DPP_votes"].sum() / g["valid_votes"].sum(), include_groups=False
    )
    for _, r in tw_polls.iterrows():
        y = int(r["election_year"])
        poll = r["dpp_two_party_share_poll"]
        act = float(actual.get(y, np.nan)) if y in actual.index else np.nan
        # turnout-adjusted sketch: scale nothing for 2p share; report turnout * intention as "effective" toy
        turn = fund[(fund["country"] == "TW") & (fund["year"] == y)]
        tr = float(turn.iloc[0]["turnout_rate"]) if len(turn) and pd.notna(turn.iloc[0].get("turnout_rate")) else np.nan
        sd = r["dpp_std"] if pd.notna(r.get("dpp_std")) else 3.0
        # convert pct sd to 2p roughly: use sd/100 on raw then renormalize crudely
        if pd.notna(poll):
            # Monte Carlo on two-party share with sd proxy
            # if dpp_std is in percentage points of raw support, approximate 2p sd ≈ sd/100
            sigma = float(sd) / 100.0 if pd.notna(sd) else 0.03
            sigma = max(0.01, min(0.08, sigma))
            draws = rng.normal(loc=float(poll), scale=sigma, size=20000)
            draws = np.clip(draws, 0.01, 0.99)
            p_win = float((draws > 0.5).mean())  # DPP plurality among 2p
            mc_summary.append(
                {
                    "country": "TW",
                    "election_year": y,
                    "poll_mean_2p": poll,
                    "sigma_used": round(sigma, 4),
                    "mc_p_dpp_2p_gt_0_5": round(p_win, 4),
                    "actual_dpp_share_all_parties": None if pd.isna(act) else round(act, 6),
                    "abs_error_poll_vs_actual_raw_dpp": None
                    if pd.isna(act) or pd.isna(poll)
                    else round(abs(float(poll) - act), 6),
                    "note": "ILLUSTRATION only — 2p poll vs all-party actual DPP share; NOT 538-grade; independent normal draws",
                }
            )
        demo_rows.append(
            {
                "country": "TW",
                "election_year": y,
                "simple_poll_average_dpp_2p": poll,
                "historical_turnout": tr,
                "turnout_adjusted_sketch": None
                if pd.isna(poll) or pd.isna(tr)
                else round(float(poll) * float(tr), 6),
                "turnout_adjusted_note": "toy: poll_2p * turnout; NOT effective vote model",
                "actual_dpp_valid_share": None if pd.isna(act) else round(act, 6),
            }
        )

    # US
    # actual dem 2p from panel mean weighted? use mean of target as rough national — better: vote-weighted if available
    us = pd.read_csv(ROOT / "fyp_us_core/us_county_panel_features_family_a.csv", usecols=["year", "target_dem_two_party_share"])
    # unweighted county mean as PROXY national (honest limit)
    us_act = us.groupby("year")["target_dem_two_party_share"].mean()
    for _, r in us_polls.iterrows():
        y = int(r["election_year"])
        poll = r["dem_two_party_share_poll"]
        act = float(us_act[y]) if y in us_act.index else np.nan
        sd = r.get("poll_sd_proxy_last14")
        sigma = float(sd) / 100.0 if pd.notna(sd) else 0.02
        sigma = max(0.008, min(0.05, sigma))
        if pd.notna(poll):
            draws = np.clip(rng.normal(float(poll), sigma, 20000), 0.01, 0.99)
            p_win = float((draws > 0.5).mean())
            mc_summary.append(
                {
                    "country": "US",
                    "election_year": y,
                    "poll_mean_2p": poll,
                    "sigma_used": round(sigma, 4),
                    "mc_p_dem_2p_gt_0_5": round(p_win, 4),
                    "actual_dem_2p_unweighted_county_mean": None if pd.isna(act) else round(act, 6),
                    "abs_error_poll_vs_county_mean": None
                    if pd.isna(act) or pd.isna(poll)
                    else round(abs(float(poll) - act), 6),
                    "note": "ILLUSTRATION — national poll vs unweighted county mean actual; no state EV, no correlation; NOT 538-grade",
                }
            )
        turn = fund[(fund["country"] == "US") & (fund["year"] == y)]
        tr = float(turn.iloc[0]["turnout_rate"]) if len(turn) and pd.notna(turn.iloc[0].get("turnout_rate")) else np.nan
        demo_rows.append(
            {
                "country": "US",
                "election_year": y,
                "simple_poll_average_dem_2p": poll,
                "historical_turnout": tr,
                "turnout_adjusted_sketch": None
                if pd.isna(poll) or pd.isna(tr)
                else round(float(poll) * float(tr), 6),
                "turnout_adjusted_note": "toy: poll_2p * turnout; NOT effective vote model",
                "actual_dem_2p_unweighted_county_mean": None if pd.isna(act) else round(act, 6),
            }
        )

    pd.DataFrame(demo_rows).to_csv(RESULTS / "simple_poll_and_turnout_sketch.csv", index=False)
    pd.DataFrame(mc_summary).to_csv(RESULTS / "monte_carlo_winprob_DEMO.csv", index=False)
    readme = RESULTS / "README.md"
    readme.write_text(
        f"""# Must-have processing DEMO (national)

Built: {NOW}

**Not 538-grade.** Independent normal draws around simple poll averages; no Bayesian house effects,
no state/county correlation, no Electoral College, no voter-file turnout model.

Files:
- `simple_poll_and_turnout_sketch.csv` — simple averages + toy poll×turnout
- `monte_carlo_winprob_DEMO.csv` — P(2p share > 0.5) from 20k draws

Family A untouched.
""",
        encoding="utf-8",
    )


def write_docs(status: dict) -> None:
    fill_md = DOCS / "必備數據補齊說明.md"
    fill_md.write_text(
        f"""# 必備數據補齊說明

**日期：** {NOW}  
**規則：** 只用公開／repo 內資料；**唔改 Family A** 鎖定特徵同 MAE。

## 本輪做咗咩

### 1. 民調（Polls）
- **TW 全國投票意向簡單平均**（選前 45 日窗，唔係 Bayesian）：`data/extension_must_have/tw_national_poll_averages_extension.csv`
- **TW undecided**：已併入上表；亦保留 `data/extension_fundamentals_proxy/tw_undecided_pre_election_proxy.csv`
- **TW house effects 草圖**：有足夠 pollster 嘅年份輸出 `tw_house_effects_sketch_YYYY.csv`（pollster 均值 − 總均值，**唔係**正式機構效應模型）
- **US 全國意向**：由既有 538／parity year proxy 擴成 `us_national_poll_averages_extension.csv`；undecided **仍缺**（538 歷史平均無穩定欄）
- **期望民調（誰會贏）**：**仍缺**（無系統公開歷史檔）

### 2. 歷史（Historical）
- **政黨 floor（鐵票 PROXY）**：過去屆次得票率最小值  
  - TW：`tw_county_partisan_floor_turnout.csv`  
  - US：`us_county_partisan_floor.csv`
- **投票率**：TW 縣／全國由 CEC panel；US 沿用既有 VEP 近似（fundamentals）
- **在任旗標**：coder `incumbent_party` / `incumbent_running` 寫入 enriched fundamentals

### 3. 人口（Demographics）
- **TW 年齡結構 PROXY**：MOI 村里 `edu_age_marital_105/107` 加總 → 全國年齡帶比例（**唔係**選年縣面板）：`tw_national_age_structure_proxy_moi.csv`
- **教育**：US Family A 已有；TW A+／Extension 已有（唔入 Family A）
- **新登記選民 PROXY**：全國 `electorate` 較上屆 % 變動（**唔等於**首投族名冊）：`tw_national_turnout_electorate_proxy.csv`
- **US ACS 年齡縣面板／族群**：**仍缺**

### 4. 基本面（Fundamentals）
- 合併檔：`data/extension_must_have/national_fundamentals_enriched_extension.csv`（另存 fundamentals_proxy）
- **TW 全國失業率**：data.gov.tw/6640 `taiwan_area_pct` 選年
- **US 通膨／失業／ICS**：沿用已有 FRED CSV
- **US approval**：datahub 538 Trump `approval_topline` 已入庫；**2020** 寫入選前估計；其餘年份／Biden 動態頁 HTML 擋 → **部分缺**
- **TW CPI／薪資／CCI／總統滿意度**：**仍缺**（DGBAS 篩選 API 參數錯誤；勞動部 13228 下載超時）
- **極化 PROXY**：縣級 |邊際| 全國平均 → `polarization_proxy_national_by_year.csv`（**唔係**情感極化）

### 5. 處理層 DEMO（全國、誠實限制）
- `results/must_have_processing_demo/simple_poll_and_turnout_sketch.csv`
- `results/must_have_processing_demo/monte_carlo_winprob_DEMO.csv` — 2 萬次獨立常態抽樣，**示範勝率**，唔係 538

## 仍然缺／申請級
- 縣市民調歷史面板；期望民調；完整 house-effects＋Bayesian dynamic average
- TEDS／ANES 微資料；新選民名冊；TW 滿意度／CCI／CPI 長序列（本輪下載失敗）
- US 選前 Gallup approval 全歷史；州選舉人團相關結構 Monte Carlo

## Family A
**未改。** Prev MAE 仍以既有鎖定結果為準（唔喺呢輪重跑）。
""",
        encoding="utf-8",
    )

    # Update 對照清單
    checklist = DOCS / "必備數據對照清單.md"
    checklist.write_text(
        f"""# 必備預測數據對照清單

**日期：** {NOW}  
**用途：** 將「預測模型應有嘅數據」對照而家 repo：**Family A 已有／Extension 已有／公開可補／申請或缺**。  
**規則：** 唔改 Family A MAE；公開可補優先入 Extension；唔捏造。  
**補齊說明：** 見 `必備數據補齊說明.md`。

分類代碼：

| 代碼 | 意思 |
|------|------|
| **Family A已有** | 已入縣／鄉鎮鎖定特徵或開票 prev 主線 |
| **Extension已有** | 全國／PROXY／試點層，報告可寫、唔併主表 MAE |
| **公開可補** | 公開來源存在，尚未下載或未入模 |
| **申請或缺** | 要申請、訂閱、或公開亦無穩定歷史面板 |

---

## 1. 民調（Polls）

| 項目 | 狀態 | 路徑／說明 |
|------|------|------------|
| 投票意向（voting intention）全國 | **Extension已有** | TW `extension_must_have/tw_national_poll_averages_extension.csv`；US `us_national_poll_averages_extension.csv`＋既有 538 |
| 縣／鄉鎮投票意向歷史面板 | **申請或缺** | 兩邊對稱缺 |
| 期望民調（expectation／誰會贏） | **申請或缺** | 公開散見、未系統入庫 |
| House effects（民調機構偏差） | **Extension已有（草圖）** | TW `tw_house_effects_sketch_*.csv`；正式模型未建 |
| 未表態／undecided | **Extension已有（TW）**；**US 缺** | TW 已入 averages；US 538 歷史平均無穩定欄 |
| Bayesian poll average | **公開可補**（處理層） | 而家係簡單平均／538；未自建 Bayesian |
| 均勻民調擺動（UniformPollSwing） | **Extension已有** | 民調 ablation；**禁止**寫入 Family A 主表 |

---

## 2. 歷史（Historical）

| 項目 | 狀態 | 路徑／說明 |
|------|------|------------|
| 政黨基本盤／floor | **Extension已有**（由 Family A 票推） | `tw_county_partisan_floor_turnout.csv`；`us_county_partisan_floor.csv` — 過去屆 min share |
| 歷史票 prev／prev2 | **Family A已有** | TW／US county panel |
| 歷史投票率（總統） | **Extension已有** | TW CEC；US VEP 近似（fundamentals） |
| 其他選舉類型投票率長面板 | **公開可補／部分缺** | 地方／中期另表未建 |
| 在任／incumbency | **Extension已有** | enriched fundamentals coder flags |
| 縣級在任立法／地方首長 | **申請或缺** | 未建 |

---

## 3. 人口結構（Demographics）

| 項目 | 狀態 | 路徑／說明 |
|------|------|------------|
| 年齡／世代 | **Extension已有（TW PROXY）**；**US 缺** | TW MOI 105/107 全國年齡帶；US ACS 縣選年面板未建 |
| 性別 | **申請或缺** | 縣選年面板未建 |
| 教育 | **US Family A已有**；**TW Extension／A+** | US `education_rate`；TW `lag_education_rate` 未入 Family A |
| 族群／籍貫／種族 | **申請或缺** | TW 外省／本省縣市長面板無；US race ACS 未逐年入模 |
| 新登記選民 | **Extension PROXY（TW）** | 選民人數 % 變動；**唔等於**首投族名冊；US 缺 |

---

## 4. 基本面（Fundamentals）

| 項目 | 狀態 | 路徑／說明 |
|------|------|------------|
| 總統／執政滿意度 | **Extension部分（US Trump 2020）**；**TW 缺** | datahub 538 Trump topline；Biden／歷史 Gallup 選前序列未齊；TW 滿意度未抽序列 |
| CPI／通膨 | **Extension已有（US）**；**TW 缺** | FRED CPIAUCSL；TW DGBAS／MOL 本輪下載失敗 |
| 失業 | **Family A已有（縣滯後）**＋**Extension 全國** | TW 全國 `taiwan_area_pct`；US FRED UNRATE |
| 薪資 | **申請或缺／公開可補失敗** | US FRED／TW MOL 本輪擋 |
| 消費者信心 CCI／ICS | **Extension已有（US）**；**TW 缺** | FRED UMCSENT |
| 極化（polarization） | **Extension PROXY** | `polarization_proxy_national_by_year.csv` — 地理邊際，概念異於情感極化 |

---

## 5. 新聞／議題／情緒

| 項目 | 狀態 | 路徑／說明 |
|------|------|------------|
| 政策議題顯著度＋語氣 | **Extension已有** | `data/extension_issue_news/` |
| 政黨／人物 X 情緒 | **Extension已有** | `data/x_sentiment_extension/` |
| 政綱議題標籤 | **Extension已有** | platforms_long |
| GDELT | **公開可補**（曾 429） | 見既有 GDELT_STATUS |
| 個體 MIP／中間選民 | **申請或缺** | TEDS／ANES／CES |

---

## 6. 處理層（Processing）

| 項目 | 狀態 | 說明 |
|------|------|------|
| 簡單民調平均 | **Extension DEMO 已有** | `results/must_have_processing_demo/` |
| Bayesian poll average | **公開可補** | 未實作完整版 |
| Turnout modeling | **DEMO 玩具** | poll×turnout sketch only |
| Monte Carlo 勝率 | **DEMO 已有** | 獨立常態；標明非 538 級 |
| 權重解讀 | **已有說明** | `權重重算說明.md` |

---

## 本輪產出路徑（Extension only）

1. `data/extension_must_have/` — 民調平均、floor、極化、年齡 PROXY、enriched fundamentals  
2. `data/extension_fundamentals_proxy/us_trump_approval_topline_datahub.csv`  
3. `results/must_have_processing_demo/` — 簡單平均／turnout sketch／MC DEMO  
4. `docs/必備數據補齊說明.md`

**未改 Family A。**

---

## 誠實總結

- **補到可分析：** 全國意向平均、TW undecided、floor、turnout、在任、極化 PROXY、TW 年齡 PROXY、新選民人數 PROXY、處理層示範。  
- **仍然缺：** 期望民調、正式 house effects／Bayesian、縣市民調、TW CPI／CCI／滿意度、US 全歷史選前 approval、完整 EC Monte Carlo。  
- **一句：** 清單「公開做得到」大部分已落地 Extension；**538 級全管道**仍然做唔到。
""",
        encoding="utf-8",
    )


def main():
    print("Building TW poll averages...")
    tw_p = build_tw_poll_averages()
    print(tw_p[["election_year", "dpp_pct_mean", "kmt_pct_mean", "undecided_pct_mean"]].to_string(index=False))
    print("Building US poll averages...")
    us_p = build_us_poll_averages()
    print("Building partisan floors / turnout...")
    build_partisan_floor_and_turnout()
    print("Polarization...")
    build_polarization_proxy()
    print("Age structure...")
    build_age_structure_proxy()
    print("Fundamentals enriched...")
    fund = build_fundamentals_enriched()
    print("Processing demo...")
    processing_demo(tw_p, us_p, fund)
    write_docs({})
    # provenance
    prov = {
        "built_at": NOW,
        "family_a_modified": False,
        "outputs": [
            "data/extension_must_have/",
            "results/must_have_processing_demo/",
            "docs/必備數據補齊說明.md",
            "docs/必備數據對照清單.md",
        ],
    }
    (OUT / "PROVENANCE.json").write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE", NOW)


if __name__ == "__main__":
    main()
