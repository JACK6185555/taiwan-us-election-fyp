#!/usr/bin/env python3
"""Symmetric TW|US fill for must-have checklist (Extension only; no Family A changes).

For every checklist item: (1) real public data, (2) same-layer PROXY, or (3) explicit 兩邊皆缺.
Asia/Hong_Kong timestamps.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "extension_must_have"
FUND = ROOT / "data" / "extension_fundamentals_proxy"
DOCS = ROOT / "docs"
RESULTS = ROOT / "results" / "must_have_processing_demo"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
FUND.mkdir(parents=True, exist_ok=True)

HK = timezone(timedelta(hours=8))
NOW = datetime.now(HK).strftime("%Y-%m-%d %H:%M Asia/Hong_Kong UTC+8")

ELECTION_YEARS = [2000, 2004, 2008, 2012, 2016, 2020, 2024]

# Pre-election month for TW (presidential usually Jan/Mar)
TW_PRE_MONTH = {2000: 2, 2004: 2, 2008: 2, 2012: 1, 2016: 1, 2020: 1, 2024: 1}
US_PRE_MONTH = {y: 10 for y in ELECTION_YEARS}  # October before Nov election


def fill_tw_cpi() -> dict[int, float]:
    src = FUND / "tw_cpi_yoy_theeconomypulse_dgbas.csv"
    if not src.exists():
        raise FileNotFoundError(src)
    cpi = pd.read_csv(src, parse_dates=["date"])
    out = {}
    rows = []
    for y in ELECTION_YEARS:
        m = TW_PRE_MONTH[y]
        sub = cpi[(cpi["date"].dt.year == y) & (cpi["date"].dt.month == m)]
        if len(sub):
            v = float(sub.iloc[0]["value"])
            out[y] = v
            rows.append(
                {
                    "country": "TW",
                    "election_year": y,
                    "asof": f"{y}-{m:02d}-01",
                    "cpi_yoy_pct": v,
                    "source": "theeconomypulse tw.dgbas.cpi_yoy (DGBAS-based republish)",
                    "source_url": "https://theeconomypulse.com/data/series/tw.dgbas.cpi_yoy.csv",
                    "family_a": "NOT_INCLUDED",
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "tw_cpi_yoy_election_eve_proxy.csv", index=False)
    return out


def fill_tw_cci() -> dict[int, float]:
    """OECD-style CCI from NTPC open data (starts ~2013). Not NCU CCI scale."""
    src = FUND / "tw_cci_oecd_style_ntpc.csv"
    cci = pd.read_csv(src)

    def parse_ym(s):
        m = re.match(r"(\d+)年(\d+)月", str(s))
        if not m:
            return None, None
        return 1911 + int(m.group(1)), int(m.group(2))

    cci["y"], cci["m"] = zip(*cci["field1"].map(parse_ym))
    cci["val"] = pd.to_numeric(cci["item value2"], errors="coerce")
    out = {}
    rows = []
    for y in ELECTION_YEARS:
        m = TW_PRE_MONTH[y]
        sub = cci[(cci["y"] == y) & (cci["m"] == m)]
        if len(sub) and pd.notna(sub.iloc[0]["val"]):
            v = float(sub.iloc[0]["val"])
            out[y] = v
            note = "OECD-style CCI index (~100 long-run); NOT NCU Taiwan CCI (0-200); scale ≠ UMCSENT"
        else:
            v = np.nan
            note = "GAP_pre_2013_or_missing_month — NTPC series starts ROC102/2013"
        rows.append(
            {
                "country": "TW",
                "election_year": y,
                "asof": None if pd.isna(v) else f"{y}-{m:02d}",
                "cci_oecd_style": None if pd.isna(v) else v,
                "note": note,
                "source": "data.ntpc.gov.tw dataset 9d0483b6-513a-46b2-92e3-5f9b315999bb",
                "family_a": "NOT_INCLUDED",
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "tw_cci_oecd_style_election_eve_proxy.csv", index=False)
    return out


def fill_us_age_wb() -> pd.DataFrame:
    """World Bank national age shares for US (TW WB often empty → keep MOI)."""
    bands = {
        "age_0_14": "SP.POP.0014.TO.ZS",
        "age_15_64": "SP.POP.1564.TO.ZS",
        "age_65plus": "SP.POP.65UP.TO.ZS",
    }
    series = {}
    for col, ind in bands.items():
        path = Path(f"/tmp/parity_probe/{ind}.json")
        if not path.exists():
            import urllib.request

            urllib.request.urlretrieve(
                f"https://api.worldbank.org/v2/country/USA/indicator/{ind}?format=json&per_page=300",
                path,
            )
        data = json.loads(path.read_text())
        rows = data[1]
        series[col] = {
            int(r["date"]): float(r["value"])
            for r in rows
            if r.get("value") is not None and r.get("countryiso3code") == "USA"
        }
    out_rows = []
    for y in ELECTION_YEARS:
        # use election year or nearest prior
        def pick(d):
            if y in d:
                return d[y], y
            prior = [k for k in d if k <= y]
            if not prior:
                return np.nan, None
            k = max(prior)
            return d[k], k

        a014, y014 = pick(series["age_0_14"])
        a1564, y1564 = pick(series["age_15_64"])
        a65, y65 = pick(series["age_65plus"])
        for band, share, ys in [
            ("0_14", a014, y014),
            ("15_64", a1564, y1564),
            ("65plus", a65, y65),
        ]:
            out_rows.append(
                {
                    "country": "US",
                    "election_year": y,
                    "age_band": band,
                    "share_pct": None if pd.isna(share) else round(share, 4),
                    "share": None if pd.isna(share) else round(share / 100.0, 6),
                    "wb_year_used": ys,
                    "grain": "national",
                    "source": "World Bank WDI SP.POP.*TO.ZS",
                    "family_a": "NOT_INCLUDED",
                    "note": "PROXY national age structure; NOT ACS county election-year panel; scale=percent of population",
                }
            )
    df = pd.DataFrame(out_rows)
    df.to_csv(OUT / "us_national_age_structure_proxy_worldbank.csv", index=False)
    # status: TW MOI already exists; US now filled at national PROXY layer
    pd.DataFrame(
        [
            {
                "country": "US",
                "status": "EXTENSION_PROXY_NATIONAL",
                "note": "World Bank national age shares by election year; county ACS panel still GAP",
                "family_a": "NOT_INCLUDED",
            },
            {
                "country": "TW",
                "status": "EXTENSION_PROXY_NATIONAL",
                "note": "MOI village sum 105/107 → national age bands; election-year county panel still GAP; WB TWN empty",
                "path": "tw_national_age_structure_proxy_moi.csv",
                "family_a": "NOT_INCLUDED",
            },
        ]
    ).to_csv(OUT / "age_structure_STATUS.csv", index=False)
    # remove old one-sided status
    old = OUT / "us_age_structure_STATUS.csv"
    if old.exists():
        old.unlink()
    return df


def fill_us_undecided_residual() -> pd.DataFrame:
    """US undecided PROXY = 100 - dem - gop from national averages (other+undecided residual)."""
    us = pd.read_csv(OUT / "us_national_poll_averages_extension.csv")
    rows = []
    for _, r in us.iterrows():
        dem, gop = r.get("dem_pct"), r.get("gop_pct")
        if pd.notna(dem) and pd.notna(gop):
            residual = 100.0 - float(dem) - float(gop)
            # clip nonsense
            if residual < 0:
                residual_note = f"residual_negative_clipped_from_{residual:.3f}"
                residual = max(0.0, residual)
            else:
                residual_note = "residual_100_minus_dem_minus_gop"
        else:
            residual = np.nan
            residual_note = "GAP_missing_dem_gop"
        rows.append(
            {
                "country": "US",
                "election_year": int(r["election_year"]),
                "as_of_date": r.get("as_of_date"),
                "dem_pct": dem,
                "gop_pct": gop,
                "undecided_pct_residual_proxy": None if pd.isna(residual) else round(residual, 3),
                "undecided_note": (
                    "PROXY — residual after Dem+GOP from national averages; may include third-party + true undecided; "
                    "NOT a survey undecided item. 538 historical averages lack stable undecided column."
                ),
                "method": residual_note,
                "expectation_polls": "兩邊皆缺_GAP_no_systematic_public_archive",
                "house_effects": "formal_model_兩邊皆缺; TW has pollster sketch from Extension polls; US raw pollster panel blocked this session",
                "family_a": "NOT_INCLUDED",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "us_undecided_residual_proxy.csv", index=False)
    # patch us poll averages
    us = us.copy()
    us["undecided_pct"] = df["undecided_pct_residual_proxy"].values
    us["undecided_note"] = df["undecided_note"].values
    us["expectation_polls"] = "兩邊皆缺_GAP_no_systematic_public_archive"
    us["house_effects"] = (
        "formal_model_兩邊皆缺; TW has Extension pollster sketch; US sketch unavailable (538 averages only / raw polls HTML-blocked)"
    )
    us.to_csv(OUT / "us_national_poll_averages_extension.csv", index=False)
    # patch TW averages expectation flag to same wording
    twp = OUT / "tw_national_poll_averages_extension.csv"
    if twp.exists():
        tw = pd.read_csv(twp)
        tw["expectation_polls"] = "兩邊皆缺_GAP_no_systematic_public_archive"
        tw.to_csv(twp, index=False)
    return df


def fill_us_house_effects_status():
    """Symmetric documentation: formal house effects 兩邊皆缺; TW sketch exists."""
    rows = []
    for y in ELECTION_YEARS:
        tw_path = OUT / f"tw_house_effects_sketch_{y}.csv"
        rows.append(
            {
                "country": "TW",
                "election_year": y,
                "formal_house_effects_model": "兩邊皆缺",
                "sketch_available": tw_path.exists(),
                "sketch_path": str(tw_path.relative_to(ROOT)) if tw_path.exists() else "",
                "note": "pollster mean − overall DPP mean in 45d window; NOT Bayesian house effects",
                "family_a": "NOT_INCLUDED",
            }
        )
        rows.append(
            {
                "country": "US",
                "election_year": y,
                "formal_house_effects_model": "兩邊皆缺",
                "sketch_available": False,
                "sketch_path": "",
                "note": "538 national daily in repo is already averaged (pollster=FiveThirtyEight_poll_average); raw pollster CSV HTML-blocked this session — sketch deferred; layer marked 兩邊皆缺 for formal model",
                "family_a": "NOT_INCLUDED",
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "house_effects_STATUS_symmetric.csv", index=False)
    # empty US sketch marker files for years (documentation)
    pd.DataFrame(
        [
            {
                "status": "GAP",
                "reason": "no_raw_pollster_panel_in_repo_this_session",
                "formal_model": "兩邊皆缺",
                "family_a": "NOT_INCLUDED",
            }
        ]
    ).to_csv(OUT / "us_house_effects_sketch_STATUS.csv", index=False)


def fill_tw_approval_from_wiki() -> dict[int, float]:
    """Tsai election-eve approve mean from Wikipedia tables (2020 from late 2019; 2024 from late 2023)."""
    html = Path("/tmp/parity_probe/tsai_wiki.html")
    if not html.exists():
        import urllib.request

        urllib.request.urlretrieve(
            "https://en.wikipedia.org/wiki/Opinion_polling_on_the_Tsai_Ing-wen_presidency",
            html,
        )
    tables = pd.read_html(html)
    # Heuristic: tables appear chronological by calendar year starting 2016.
    # Table 3 ≈ 2019 (use Dec rows for 2020 election)
    # Table 7 ≈ 2023 (use Dec rows for 2024 election)
    mapping = {2020: 3, 2024: 7}
    out = {}
    rows = []
    for ey, ti in mapping.items():
        if ti >= len(tables):
            continue
        t = tables[ti].copy()
        flat = []
        for c in t.columns:
            if isinstance(c, tuple):
                parts = [str(x) for x in c if "Unnamed" not in str(x)]
                flat.append("_".join(parts) if parts else str(c[-1]))
            else:
                flat.append(str(c))
        t.columns = flat
        # pick columns by substring (first match)
        field_col = next((c for c in t.columns if "field" in c.lower() or "date" in c.lower()), None)
        appr_col = next((c for c in t.columns if c.lower() == "approve" or c.lower().endswith("approve")), None)
        if field_col is None or appr_col is None:
            continue
        mask = t[field_col].astype(str).str.contains("Dec|Nov", case=False, na=False)
        vals = pd.to_numeric(t.loc[mask, appr_col], errors="coerce").dropna()
        if vals.empty:
            continue
        mean_appr = float(vals.mean())
        out[ey] = mean_appr
        rows.append(
            {
                "country": "TW",
                "election_year": ey,
                "presidential_approval": round(mean_appr, 2),
                "n_polls_used": int(len(vals)),
                "window": "Nov-Dec before election (Wikipedia Tsai approval tables)",
                "source": "https://en.wikipedia.org/wiki/Opinion_polling_on_the_Tsai_Ing-wen_presidency",
                "family_a": "NOT_INCLUDED",
                "note": "Sitting president Tsai approve%; 2000-2016 GAP (Chen/Ma pages not systematically ingested)",
            }
        )
    # GAP rows for earlier years
    for y in ELECTION_YEARS:
        if y not in out:
            rows.append(
                {
                    "country": "TW",
                    "election_year": y,
                    "presidential_approval": None,
                    "n_polls_used": 0,
                    "window": None,
                    "source": None,
                    "family_a": "NOT_INCLUDED",
                    "note": "GAP — no systematic public election-eve approval archive ingested for this year",
                }
            )
    pd.DataFrame(rows).sort_values("election_year").to_csv(
        OUT / "tw_presidential_approval_election_eve_proxy.csv", index=False
    )
    return out


def fill_us_approval_extend() -> dict[int, float]:
    """Keep 2020 Trump datahub; mark other years GAP at same layer (no invented Gallup)."""
    out = {}
    rows = []
    # existing 2020 from enriched if present
    enriched = pd.read_csv(OUT / "national_fundamentals_enriched_extension.csv")
    r2020 = enriched[(enriched.country == "US") & (enriched.year == 2020)]
    if len(r2020) and pd.notna(r2020.iloc[0].get("presidential_approval")):
        out[2020] = float(r2020.iloc[0]["presidential_approval"])
    else:
        # recompute from datahub
        ap = FUND / "us_trump_approval_topline_datahub.csv"
        if ap.exists():
            a = pd.read_csv(ap)
            a["modeldate"] = pd.to_datetime(a["modeldate"], errors="coerce")
            ed = pd.Timestamp("2020-11-03")
            sub = a[(a["modeldate"] <= ed) & (a["modeldate"] >= ed - pd.Timedelta(days=14))]
            if sub.empty:
                sub = a[a["modeldate"] <= ed].sort_values("modeldate").tail(5)
            v = sub[sub["subgroup"].astype(str).str.lower() == "voters"]
            use = v if len(v) else sub
            if len(use):
                out[2020] = float(use.sort_values("modeldate").iloc[-1]["approve_estimate"])
    for y in ELECTION_YEARS:
        if y in out:
            rows.append(
                {
                    "country": "US",
                    "election_year": y,
                    "presidential_approval": round(out[y], 2),
                    "source": "datahub 538 trump approval_topline",
                    "note": "election-eve window voters subgroup",
                    "family_a": "NOT_INCLUDED",
                }
            )
        else:
            rows.append(
                {
                    "country": "US",
                    "election_year": y,
                    "presidential_approval": None,
                    "source": None,
                    "note": "GAP — Gallup/538 election-eve historical series not systematically ingested (projects HTML-blocked; UCSB JS-only)",
                    "family_a": "NOT_INCLUDED",
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "us_presidential_approval_election_eve_proxy.csv", index=False)
    return out


def fill_us_electorate_proxy() -> pd.DataFrame:
    """VEP stock-change PROXY from county vote totals / turnout_rate."""
    us = pd.read_csv(ROOT / "fyp_us_core/us_county_panel_features_family_a.csv")
    us["votes"] = np.exp(us["log_total_votes"])
    nat = us.groupby("year", as_index=False)["votes"].sum()
    fund = pd.read_csv(FUND / "national_fundamentals_election_year_proxy.csv")
    fund = fund[fund["country"] == "US"][["year", "turnout_rate"]]
    m = nat.merge(fund, on="year", how="left")
    m["vep_proxy"] = m["votes"] / m["turnout_rate"]
    m["electorate_change_pct"] = m["vep_proxy"].pct_change()
    m["new_voters_stock_change_proxy"] = m["electorate_change_pct"]
    m["country"] = "US"
    m["note"] = (
        "PROXY — VEP≈sum(exp(log_total_votes))/turnout_rate from Family A panel + fundamentals turnout; "
        "NOT first-time voter registry; panel starts 2008 so 2000/2004 GAP"
    )
    m["family_a"] = "NOT_INCLUDED"
    # add empty 2000/2004 for symmetry of years list
    for y in [2000, 2004]:
        if y not in set(m["year"]):
            m = pd.concat(
                [
                    m,
                    pd.DataFrame(
                        [
                            {
                                "year": y,
                                "votes": np.nan,
                                "turnout_rate": np.nan,
                                "vep_proxy": np.nan,
                                "electorate_change_pct": np.nan,
                                "new_voters_stock_change_proxy": np.nan,
                                "country": "US",
                                "note": "GAP — US Family A county panel starts 2008",
                                "family_a": "NOT_INCLUDED",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    m = m.sort_values("year")
    m.to_csv(OUT / "us_national_turnout_electorate_proxy.csv", index=False)
    return m


def patch_fundamentals(tw_cpi, tw_cci, tw_appr, us_appr, us_und, us_elect):
    path = OUT / "national_fundamentals_enriched_extension.csv"
    df = pd.read_csv(path)
    # ensure columns
    for c in [
        "cpi_yoy_pct",
        "cpi_asof",
        "cpi_note",
        "cci_note",
        "umcsent",
        "presidential_approval",
        "approval_note",
        "undecided_share",
        "undecided_note",
        "new_voters_stock_change_proxy",
        "wages_note",
    ]:
        if c not in df.columns:
            df[c] = np.nan

    und_map = {
        int(r.election_year): r.undecided_pct_residual_proxy
        for _, r in us_und.iterrows()
    }
    elect_map = {
        int(r.year): r.new_voters_stock_change_proxy
        for _, r in us_elect.iterrows()
        if pd.notna(r.get("new_voters_stock_change_proxy"))
    }

    rows = []
    for _, r in df.iterrows():
        d = r.to_dict()
        c, y = d["country"], int(d["year"])
        if c == "TW":
            if y in tw_cpi:
                d["cpi_yoy_pct"] = tw_cpi[y]
                d["cpi_asof"] = f"{y}-{TW_PRE_MONTH[y]:02d}-01"
                d["cpi_note"] = "DGBAS CPI YoY via theeconomypulse republish; election-eve month"
            else:
                d["cpi_note"] = "GAP"
            if y in tw_cci:
                # store in umcsent-analog column? Keep separate via cci_note; also write value into a parallel field if exists
                d["cci_note"] = (
                    f"TW OECD-style CCI={tw_cci[y]} (NTPC); scale≠UMCSENT; conceptual consumer-sentiment PROXY"
                )
                # Don't overwrite US umcsent semantics — leave umcsent NaN for TW, document in cci_note
            else:
                d["cci_note"] = "TW CCI GAP for this year (NTPC OECD-style starts 2013); NCU CCI long CSV not openly bulk-downloaded"
            if y in tw_appr:
                d["presidential_approval"] = tw_appr[y]
                d["approval_note"] = "Wikipedia Tsai approve mean Nov-Dec before election"
            else:
                d["approval_note"] = d.get("approval_note") or "TW approval GAP this year"
            # wages both gap
            d["wages_note"] = "兩邊皆缺 this session — TW MOL / US FRED AHE download blocked without key"
            # undecided already from prior for some years
            if pd.isna(d.get("undecided_share")):
                # from tw poll averages
                twp = pd.read_csv(OUT / "tw_national_poll_averages_extension.csv")
                m = twp[twp.election_year == y]
                if len(m) and pd.notna(m.iloc[0].get("undecided_pct_mean")):
                    d["undecided_share"] = float(m.iloc[0]["undecided_pct_mean"])
                    d["undecided_note"] = "TW Extension poll undecided mean (45d)"
        if c == "US":
            if y in us_appr:
                d["presidential_approval"] = us_appr[y]
                d["approval_note"] = "datahub 538 Trump approval topline election-eve"
            else:
                d["approval_note"] = (
                    "US election-eve approval GAP (except 2020 Trump datahub); Gallup historical not ingested"
                )
            if y in und_map and pd.notna(und_map[y]):
                d["undecided_share"] = und_map[y]
                d["undecided_note"] = "US residual PROXY 100-dem-gop (not survey undecided item)"
            if y in elect_map:
                d["new_voters_stock_change_proxy"] = elect_map[y]
            d["wages_note"] = "兩邊皆缺 this session — FRED AHE blocked without key"
            d["cpi_note"] = d.get("cpi_note") or "FRED CPIAUCSL YoY"
            d["cci_note"] = d.get("cci_note") or "FRED UMCSENT (Michigan ICS)"
        d["built_at"] = NOW
        rows.append(d)
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    out.to_csv(FUND / "national_fundamentals_enriched_extension.csv", index=False)
    return out


def rebuild_processing_demo():
    """Ensure MC + turnout sketch cover BOTH TW and US national series."""
    tw_p = pd.read_csv(OUT / "tw_national_poll_averages_extension.csv")
    us_p = pd.read_csv(OUT / "us_national_poll_averages_extension.csv")
    fund = pd.read_csv(OUT / "national_fundamentals_enriched_extension.csv")
    tw_panel = pd.read_csv(ROOT / "cleaned/county_panel_features_enriched_2000_2024.csv")
    tw_panel["dpp_share"] = tw_panel["DPP_votes"] / tw_panel["valid_votes"]
    tw_act = tw_panel.groupby("year")["dpp_share"].mean()

    us_panel = pd.read_csv(
        ROOT / "fyp_us_core/us_county_panel_features_family_a.csv",
        usecols=["year", "target_dem_two_party_share"],
    )
    us_act = us_panel.groupby("year")["target_dem_two_party_share"].mean()

    sketch_rows = []
    mc_rows = []
    rng = np.random.default_rng(42)
    N = 20000

    for _, r in tw_p.iterrows():
        y = int(r["election_year"])
        poll = r["dpp_two_party_share_poll"]
        fr = fund[(fund.country == "TW") & (fund.year == y)]
        turnout = float(fr.iloc[0]["turnout_rate"]) if len(fr) and pd.notna(fr.iloc[0]["turnout_rate"]) else np.nan
        actual = float(tw_act.get(y, np.nan))
        sketch_rows.append(
            {
                "country": "TW",
                "election_year": y,
                "simple_poll_average_dpp_2p": poll,
                "historical_turnout": turnout,
                "turnout_adjusted_sketch": None
                if pd.isna(poll) or pd.isna(turnout)
                else round(float(poll) * float(turnout), 6),
                "turnout_adjusted_note": "toy: poll_2p * turnout; NOT effective vote model",
                "actual_dpp_valid_share": None if pd.isna(actual) else round(actual, 6),
                "simple_poll_average_dem_2p": None,
                "actual_dem_2p_unweighted_county_mean": None,
            }
        )
        if pd.notna(poll):
            sd = r.get("dpp_std")
            sigma = float(sd) / 100.0 if pd.notna(sd) and float(sd) > 0 else 0.02
            # dpp_std is in percentage points of party % — convert roughly
            if pd.notna(sd) and float(sd) > 1:
                sigma = float(sd) / 100.0
            draws = rng.normal(float(poll), sigma, size=N)
            p = float((draws > 0.5).mean())
            mc_rows.append(
                {
                    "country": "TW",
                    "election_year": y,
                    "poll_mean_2p": float(poll),
                    "sigma_used": round(sigma, 4),
                    "mc_p_dpp_2p_gt_0_5": round(p, 4),
                    "actual_dpp_share_all_parties": None if pd.isna(actual) else round(actual, 6),
                    "abs_error_poll_vs_actual_raw_dpp": None
                    if pd.isna(actual)
                    else round(abs(float(poll) - actual), 5),
                    "note": "ILLUSTRATION only — independent normal; NOT 538-grade; TW 2p poll vs all-party DPP share",
                    "mc_p_dem_2p_gt_0_5": None,
                    "actual_dem_2p_unweighted_county_mean": None,
                    "abs_error_poll_vs_county_mean": None,
                }
            )

    for _, r in us_p.iterrows():
        y = int(r["election_year"])
        poll = r["dem_two_party_share_poll"]
        fr = fund[(fund.country == "US") & (fund.year == y)]
        turnout = float(fr.iloc[0]["turnout_rate"]) if len(fr) and pd.notna(fr.iloc[0]["turnout_rate"]) else np.nan
        actual = float(us_act.get(y, np.nan)) if y in us_act.index else np.nan
        sketch_rows.append(
            {
                "country": "US",
                "election_year": y,
                "simple_poll_average_dpp_2p": None,
                "historical_turnout": turnout,
                "turnout_adjusted_sketch": None
                if pd.isna(poll) or pd.isna(turnout)
                else round(float(poll) * float(turnout), 6),
                "turnout_adjusted_note": "toy: poll_2p * turnout; NOT effective vote model",
                "actual_dpp_valid_share": None,
                "simple_poll_average_dem_2p": poll,
                "actual_dem_2p_unweighted_county_mean": None if pd.isna(actual) else round(actual, 6),
            }
        )
        if pd.notna(poll):
            sd = r.get("poll_sd_proxy_last14")
            sigma = float(sd) / 100.0 if pd.notna(sd) and float(sd) > 0 else 0.015
            if pd.notna(sd) and float(sd) > 1:
                sigma = float(sd) / 100.0
            draws = rng.normal(float(poll), max(sigma, 0.008), size=N)
            p = float((draws > 0.5).mean())
            mc_rows.append(
                {
                    "country": "US",
                    "election_year": y,
                    "poll_mean_2p": float(poll),
                    "sigma_used": round(max(sigma, 0.008), 4),
                    "mc_p_dpp_2p_gt_0_5": None,
                    "actual_dpp_share_all_parties": None,
                    "abs_error_poll_vs_actual_raw_dpp": None,
                    "note": "ILLUSTRATION only — independent normal national Dem 2p; NOT EC/state Monte Carlo; NOT 538-grade",
                    "mc_p_dem_2p_gt_0_5": round(p, 4),
                    "actual_dem_2p_unweighted_county_mean": None if pd.isna(actual) else round(actual, 6),
                    "abs_error_poll_vs_county_mean": None
                    if pd.isna(actual)
                    else round(abs(float(poll) - actual), 5),
                }
            )

    pd.DataFrame(sketch_rows).to_csv(RESULTS / "simple_poll_and_turnout_sketch.csv", index=False)
    pd.DataFrame(mc_rows).to_csv(RESULTS / "monte_carlo_winprob_DEMO.csv", index=False)
    (RESULTS / "README.md").write_text(
        f"""# Must-have processing DEMO

**Built:** {NOW}  
**Family A:** not modified.

- `simple_poll_and_turnout_sketch.csv` — TW + US national toy poll×turnout  
- `monte_carlo_winprob_DEMO.csv` — TW + US independent normal win-prob illustration (not 538 / not Electoral College)

Expectation polls + formal Bayesian house-effects: **兩邊皆缺**.
""",
        encoding="utf-8",
    )


def write_symmetry_docs(fund: pd.DataFrame):
    # Short symmetry memo
    (DOCS / "美台必備數據對稱說明.md").write_text(
        f"""# 美台必備數據對稱說明

**日期：** {NOW}  
**規則：** 每一項 = 兩邊都有真實公開資料／同一概念層 PROXY／或明確「兩邊皆缺」。**未改 Family A。**

## 本輪對稱補齊

| 項目 | TW | US | 狀態 |
|------|----|----|------|
| 全國投票意向 | Extension 簡單平均 | Extension 538／Wiki | **對等** |
| 未表態 | 民調 undecided 項 | residual PROXY `100−Dem−GOP` | **層級對等（定義異）** |
| 期望民調 | — | — | **兩邊皆缺** |
| House effects 正式模型 | — | — | **兩邊皆缺**（TW 另有 pollster 草圖） |
| 鐵票 floor／投票率／在任 | 有 | 有 | **對等** |
| 年齡全國 PROXY | MOI 105/107 | World Bank WDI | **層級對等（來源異）**；縣選年面板兩邊仍缺 |
| 新選民人數 PROXY | electorate %Δ | VEP≈votes/turnout %Δ | **層級對等**；US 2000/04 面板缺 |
| CPI YoY | DGBAS via theeconomypulse | FRED CPIAUCSL | **對等** |
| 消費者信心 | OECD-style CCI（2016+） | UMCSENT | **層級對等（尺度異）**；TW 2013 前缺 |
| 執政滿意度選前 | Tsai Wiki 2020／2024 | Trump datahub 2020 | **部分對等**；其餘年兩邊多缺 |
| 薪資 | — | — | **兩邊皆缺**（本輪下載擋） |
| 極化 PROXY | 縣邊際 | 縣邊際 | **對等** |
| MC／turnout DEMO | 有 | 有 | **對等（示範級）** |

詳表：`必備數據對照清單.md`（已改 TW\\|US\\|status 欄）；總對等：`美台資料對等表.md`。
""",
        encoding="utf-8",
    )

    # Rewrite checklist with TW|US|status columns
    (DOCS / "必備數據對照清單.md").write_text(
        f"""# 必備預測數據對照清單（美台對稱）

**日期：** {NOW}  
**用途：** 每項標 **TW｜US｜status**（有資料／PROXY／兩邊皆缺）。  
**規則：** 唔改 Family A；Extension only；唔捏造。  
**對稱說明：** `美台必備數據對稱說明.md` · **補齊：** `必備數據補齊說明.md`

status 代碼：`對等`＝兩邊都有同層；`層級對等（定義異）`＝兩邊都有但建構唔同；`部分對等`＝兩邊都只有部分年；`兩邊皆缺`；`Family A已有`。

---

## 1. 民調（Polls）

| 項目 | TW | US | status |
|------|----|----|--------|
| 投票意向全國 | `tw_national_poll_averages_extension.csv` | `us_national_poll_averages_extension.csv` | **對等**（Extension） |
| 縣／鄉鎮民調面板 | 缺 | 缺 | **兩邊皆缺** |
| 期望民調 | 缺 | 缺 | **兩邊皆缺** |
| House effects 正式模型 | 缺 | 缺 | **兩邊皆缺**（TW 有 sketch；US sketch 本輪無） |
| 未表態 | poll undecided 項 | residual PROXY | **層級對等（定義異）** |
| Bayesian average | 未建 | 未建（用 538） | **兩邊皆缺**（自建）；處理層 DEMO 用簡單平均 |

## 2. 歷史（Historical）

| 項目 | TW | US | status |
|------|----|----|--------|
| prev／prev2 歷史票 | Family A panel | Family A panel | **Family A已有／對等** |
| 鐵票 floor | `tw_county_partisan_floor_turnout.csv` | `us_county_partisan_floor.csv` | **對等**（Extension） |
| 總統投票率 | CEC | VEP 近似 | **對等** |
| 在任旗標 | coder | coder | **對等** |
| 縣級地方首長在任長面板 | 缺 | 缺 | **兩邊皆缺** |

## 3. 人口（Demographics）

| 項目 | TW | US | status |
|------|----|----|--------|
| 年齡全國 PROXY | MOI `tw_national_age_structure_proxy_moi.csv` | WB `us_national_age_structure_proxy_worldbank.csv` | **層級對等（來源異）** |
| 年齡縣選年面板 | 缺 | 缺 | **兩邊皆缺** |
| 教育 | Extension／A+ `lag_education_rate` | Family A `education_rate` | **層級大致對等**（TW 未入 Family A） |
| 性別縣面板 | 缺 | 缺 | **兩邊皆缺** |
| 族群／籍貫／種族縣面板 | 缺 | 缺（ACS 未入） | **兩邊皆缺** |
| 新選民人數 PROXY | `tw_national_turnout_electorate_proxy.csv` | `us_national_turnout_electorate_proxy.csv` | **層級對等**（US 2000/04 GAP） |

## 4. 基本面（Fundamentals）

| 項目 | TW | US | status |
|------|----|----|--------|
| CPI YoY | DGBAS／theeconomypulse 已填 | FRED | **對等** |
| 失業全國 | DGBAS | FRED UNRATE | **對等** |
| 消費者信心 | OECD-style CCI（2016+） | UMCSENT | **層級對等（尺度異）** |
| 執政滿意度選前 | Tsai Wiki 2020／2024 | Trump 2020 datahub | **部分對等** |
| 薪資 | 缺 | 缺 | **兩邊皆缺** |
| 極化 PROXY | 有 | 有 | **對等** |

## 5. 新聞／議題／情緒

| 項目 | TW | US | status |
|------|----|----|--------|
| X 情緒／議題聲量 | Extension | Extension | **對等（試點）** |
| 政綱標籤 | 有 | 有 | **對等** |
| 個體 MIP | 缺 | 缺（ANES 未入模） | **兩邊皆缺／未入模** |

## 6. 處理層 DEMO

| 項目 | TW | US | status |
|------|----|----|--------|
| 簡單民調平均×turnout | 有 | 有 | **對等（DEMO）** |
| Monte Carlo 勝率示範 | 有 | 有 | **對等（DEMO；非 EC）** |
| 正式 Bayesian＋EC MC | 缺 | 缺 | **兩邊皆缺** |

---

## 產物

- `data/extension_must_have/`（含本輪對稱補檔）  
- `data/extension_fundamentals_proxy/tw_cpi_yoy_*.csv`、`tw_cci_oecd_style_*.csv`  
- `results/must_have_processing_demo/`（TW+US）  
- `docs/美台必備數據對稱說明.md`

**未改 Family A。**
""",
        encoding="utf-8",
    )

    (DOCS / "必備數據補齊說明.md").write_text(
        f"""# 必備數據補齊說明（對稱輪）

**日期：** {NOW}  
**規則：** 公開資料；**唔改 Family A**。

## 本輪對稱補咗咩

1. **TW CPI YoY** — theeconomypulse／DGBAS → `tw_cpi_yoy_election_eve_proxy.csv`；寫入 enriched fundamentals  
2. **TW CCI** — NTPC OECD-style（2013+）→ `tw_cci_oecd_style_election_eve_proxy.csv`（尺度 ≠ UMCSENT）  
3. **US 年齡** — World Bank 全國年齡帶 → `us_national_age_structure_proxy_worldbank.csv`（同層對齊 TW MOI PROXY）  
4. **US undecided PROXY** — `100−Dem−GOP` residual → `us_undecided_residual_proxy.csv`  
5. **期望民調／正式 house effects** — 明確 **兩邊皆缺**（`house_effects_STATUS_symmetric.csv`）  
6. **TW 滿意度** — Wikipedia Tsai 2020／2024 選前 Nov–Dec 均值；其餘年 GAP  
7. **US 滿意度** — 保留 2020 Trump datahub；其餘年 GAP（唔捏造 Gallup）  
8. **US 新選民 PROXY** — VEP≈votes/turnout → `us_national_turnout_electorate_proxy.csv`  
9. **薪資** — **兩邊皆缺**  
10. **Monte Carlo／turnout DEMO** — 重跑覆蓋 **TW+US**

## 仍然兩邊缺／部分

- 縣市民調；期望民調；正式 Bayesian house effects  
- 縣選年年齡／性別／族群面板  
- 選前滿意度全歷史（兩邊都只有部分年）  
- TW NCU CCI 長序列；US Gallup 選前全序列  
- 薪資；完整選舉人團 Monte Carlo  

## Family A

**未改。**
""",
        encoding="utf-8",
    )


def patch_parity_table():
    path = DOCS / "美台資料對等表.md"
    text = path.read_text(encoding="utf-8")
    marker = "## 必備清單對稱（本輪）"
    block = f"""
{marker}

**日期：** {NOW} · 詳見 `美台必備數據對稱說明.md`、`必備數據對照清單.md`。

| 層 | TW | US | 對等 |
|----|----|----|------|
| 全國意向民調 | Extension | Extension | 對等 |
| 未表態 | 調查項 | residual PROXY | 層級對等（定義異） |
| 期望民調／正式 house effects | 缺 | 缺 | **兩邊皆缺** |
| CPI／失業 | 有 | 有 | 對等 |
| 消費者信心 | OECD-style CCI（部分年） | UMCSENT | 層級對等（尺度異） |
| 年齡全國 PROXY | MOI | World Bank | 層級對等 |
| 新選民 PROXY | electorate %Δ | VEP %Δ | 層級對等 |
| 滿意度選前 | Tsai 2020/24 | Trump 2020 | 部分對等 |
| 薪資 | 缺 | 缺 | **兩邊皆缺** |
| MC DEMO | 有 | 有 | 對等（示範） |

**未改 Family A。**

"""
    if marker in text:
        # replace from marker to next ## or end
        import re as _re

        text = _re.sub(
            rf"{re.escape(marker)}.*?(?=\n## |\Z)",
            block.strip() + "\n\n",
            text,
            flags=_re.S,
        )
    else:
        text = text.rstrip() + "\n\n" + block
    path.write_text(text, encoding="utf-8")


def main():
    print("TW CPI...")
    tw_cpi = fill_tw_cpi()
    print(tw_cpi)
    print("TW CCI...")
    tw_cci = fill_tw_cci()
    print({k: tw_cci[k] for k in tw_cci})
    print("US age WB...")
    fill_us_age_wb()
    print("US undecided residual...")
    us_und = fill_us_undecided_residual()
    print("House effects status...")
    fill_us_house_effects_status()
    print("TW approval wiki...")
    tw_appr = fill_tw_approval_from_wiki()
    print(tw_appr)
    print("US approval...")
    us_appr = fill_us_approval_extend()
    print(us_appr)
    print("US electorate...")
    us_elect = fill_us_electorate_proxy()
    print("Patch fundamentals...")
    fund = patch_fundamentals(tw_cpi, tw_cci, tw_appr, us_appr, us_und, us_elect)
    print("Rebuild processing demo...")
    rebuild_processing_demo()
    print("Docs...")
    write_symmetry_docs(fund)
    patch_parity_table()
    prov = {
        "built_at": NOW,
        "family_a_modified": False,
        "script": "scripts/fill_must_have_tw_us_symmetry.py",
        "outputs": [
            "data/extension_must_have/",
            "data/extension_fundamentals_proxy/tw_cpi_yoy_theeconomypulse_dgbas.csv",
            "data/extension_fundamentals_proxy/tw_cci_oecd_style_ntpc.csv",
            "results/must_have_processing_demo/",
            "docs/美台必備數據對稱說明.md",
            "docs/必備數據對照清單.md",
            "docs/必備數據補齊說明.md",
            "docs/美台資料對等表.md",
        ],
    }
    (OUT / "PROVENANCE_symmetry.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("DONE", NOW)


if __name__ == "__main__":
    main()
