#!/usr/bin/env python3
"""Build US Extension layers parallel to TW polls / platforms / ESC identity proxies.

Public sources only; no login scrape; no invented poll numbers.
Extension only — do NOT merge into Core specification Core MAE.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace/taiwan-election-data")
GH = Path("/workspace/taiwan-election-github")
RAW_POLLS = ROOT / "raw/us_polls"
OUT_DATA = ROOT / "data/us_parity_extension"
OUT_GH = GH / "data/us_parity_extension"
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_GH.mkdir(parents=True, exist_ok=True)

SEED = 42


def write_both(name: str, df: pd.DataFrame):
    for d in (OUT_DATA, OUT_GH):
        path = d / name
        df.to_csv(path, index=False)
        print(f"wrote {path} rows={len(df)}")


# ---------------------------------------------------------------------------
# 1) National poll averages from 538 GitHub file (2020 + 2024 National)
# ---------------------------------------------------------------------------
avg_path = RAW_POLLS / "presidential_general_averages_2024-09-12_uncorrected.csv"
avg = pd.read_csv(avg_path)
avg["date"] = pd.to_datetime(avg["date"])
nat = avg[avg["state"].astype(str).str.lower() == "national"].copy()

# Map candidate → party side
def side(row):
    c = str(row.get("candidate", "")).lower()
    p = str(row.get("party", "")).lower()
    if "trump" in c or p in ("rep", "republican", "r"):
        return "gop"
    if any(x in c for x in ("biden", "harris", "clinton", "obama", "kerry", "gore")) or p in (
        "dem",
        "democratic",
        "d",
    ):
        return "dem"
    if "kennedy" in c:
        return "other"
    return "other"


nat["side"] = nat.apply(side, axis=1)
# Prefer pct_estimate if present else pct_trend_adjusted
nat["pct"] = nat["pct_estimate"].fillna(nat["pct_trend_adjusted"])

# Last available national average per cycle × side (pre-election snapshot in file)
rows = []
for cycle, g in nat.groupby("cycle"):
    last_date = g["date"].max()
    gg = g[g["date"] == last_date]
    dem = gg.loc[gg["side"] == "dem", "pct"]
    gop = gg.loc[gg["side"] == "gop", "pct"]
    oth = gg.loc[gg["side"] == "other", "pct"]
    dem_v = float(dem.iloc[0]) if len(dem) else np.nan
    gop_v = float(gop.iloc[0]) if len(gop) else np.nan
    oth_v = float(oth.sum()) if len(oth) else np.nan
    # two-party from dem+gop if both present
    if pd.notna(dem_v) and pd.notna(gop_v) and (dem_v + gop_v) > 0:
        dem_2p = dem_v / (dem_v + gop_v)
    else:
        dem_2p = np.nan
    rows.append(
        {
            "election_year": int(cycle),
            "as_of_date": last_date.date().isoformat(),
            "dem_pct": dem_v,
            "gop_pct": gop_v,
            "other_pct": oth_v,
            "dem_two_party_share_poll": dem_2p,
            "source": "fivethirtyeight_github_presidential_general_averages_2024-09-12_uncorrected",
            "source_url": "https://raw.githubusercontent.com/fivethirtyeight/data/master/polls/2024-averages/presidential_general_averages_2024-09-12_uncorrected.csv",
            "geography": "national",
            "note": "Last date in 538 averages file for National; Extension only; NOT county polls",
        }
    )

proxy_538 = pd.DataFrame(rows)

# Daily national long (for Extension archive; parallel to TW poll long at national only)
daily = nat[nat["side"].isin(["dem", "gop"])].copy()
daily_out = daily.rename(
    columns={
        "date": "field_end",
        "candidate": "target",
        "cycle": "election_year",
        "pct": "value",
    }
)[
    [
        "election_year",
        "field_end",
        "target",
        "side",
        "party",
        "value",
        "pct_trend_adjusted",
        "pct_estimate",
        "state",
    ]
]
daily_out["pollster"] = "FiveThirtyEight_poll_average"
daily_out["region"] = "national"
daily_out["question_type"] = "support_average"
daily_out["source_url"] = (
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/polls/2024-averages/presidential_general_averages_2024-09-12_uncorrected.csv"
)
write_both("us_polls_national_daily_538.csv", daily_out.sort_values(["election_year", "field_end", "side"]))

# ---------------------------------------------------------------------------
# 2) Wikipedia final-aggregate RCP/538 rows for older cycles (2008–2016)
#    Extract from already-downloaded wikitext; no invented numbers.
# ---------------------------------------------------------------------------
WIKI = RAW_POLLS / "wiki"
WIKI_PAGES = {
    2016: "Nationwide_opinion_polling_for_the_2016_United_States_presidential_election.json",
    2012: "Nationwide_opinion_polling_for_the_2012_United_States_presidential_election.json",
    2008: "Nationwide_opinion_polling_for_the_2008_United_States_presidential_election.json",
    2020: "Nationwide_opinion_polling_for_the_2020_United_States_presidential_election.json",
    2024: "Nationwide_opinion_polling_for_the_2024_United_States_presidential_election.json",
    2000: "Nationwide_opinion_polling_for_the_2000_United_States_presidential_election.json",
}


def load_wt(year: int) -> str:
    p = WIKI / WIKI_PAGES[year]
    d = json.loads(p.read_text())
    return d["parse"]["wikitext"]["*"]


def extract_pcts(line: str) -> list[float]:
    # match 46.8% or '''46.8%''' or 46.8
    nums = re.findall(r"(\d{1,2}\.\d)\s*%", line)
    return [float(x) for x in nums]


wiki_proxy_rows = []
# For 2008/2012/2016: take first Real Clear Politics / FiveThirtyEight aggregate row with ≥2 pcts
for year in (2008, 2012, 2016):
    wt = load_wt(year)
    lines = wt.splitlines()
    found = None
    for i, line in enumerate(lines):
        if re.search(r"Real\s*Clear\s*Politics|RealClearPolitics|FiveThirtyEight", line, re.I):
            # look ahead a few lines for percentages
            chunk = "\n".join(lines[i : i + 8])
            pcts = extract_pcts(chunk)
            if len(pcts) >= 2:
                found = (line, pcts)
                break
    if not found:
        # fallback: any aggregate table first data row with two pcts after "Aggregate"
        for i, line in enumerate(lines):
            if re.search(r"Polling\s+aggregat|Final\s+polls|poll\s+average", line, re.I):
                for j in range(i, min(i + 40, len(lines))):
                    pcts = extract_pcts(lines[j])
                    if len(pcts) >= 2:
                        found = (lines[j], pcts)
                        break
            if found:
                break
    if found:
        _, pcts = found
        dem_v, gop_v = pcts[0], pcts[1]
        # Heuristic: for these years Dem listed first in wiki tables for 2008 Obama, 2012 Obama, 2016 Clinton
        dem_2p = dem_v / (dem_v + gop_v) if (dem_v + gop_v) > 0 else np.nan
        wiki_proxy_rows.append(
            {
                "election_year": year,
                "as_of_date": f"{year}-11-01",  # approximate pre-election week label; exact date in wiki notes
                "dem_pct": dem_v,
                "gop_pct": gop_v,
                "other_pct": np.nan,
                "dem_two_party_share_poll": dem_2p,
                "source": "wikipedia_nationwide_opinion_polling_aggregate_row",
                "source_url": f"https://en.wikipedia.org/wiki/Nationwide_opinion_polling_for_the_{year}_United_States_presidential_election",
                "geography": "national",
                "note": "First RCP/538 (or nearby aggregate) row parsed from Wikipedia wikitext; as_of_date approximate Nov 1 label; Extension only",
            }
        )
    else:
        wiki_proxy_rows.append(
            {
                "election_year": year,
                "as_of_date": "",
                "dem_pct": np.nan,
                "gop_pct": np.nan,
                "other_pct": np.nan,
                "dem_two_party_share_poll": np.nan,
                "source": "wikipedia_parse_failed",
                "source_url": f"https://en.wikipedia.org/wiki/Nationwide_opinion_polling_for_the_{year}_United_States_presidential_election",
                "geography": "national",
                "note": "Could not parse aggregate; left blank (no invented numbers)",
            }
        )

# 2000: try RCP-like from page; 2004: no dedicated nationwide page → gap
wt2000 = load_wt(2000)
# Look for final Gallup / polls near election with Bush/Gore
bush_gore = []
for line in wt2000.splitlines():
    if re.search(r"Gallup|CBS|NBC|ABC|CNN|Zogby|Pew", line, re.I):
        pcts = extract_pcts("\n".join(wt2000.splitlines()[wt2000.splitlines().index(line) : wt2000.splitlines().index(line) + 6])) if line in wt2000.splitlines() else []
# Simpler: search all lines for two consecutive pct near 'Bush' and 'Gore' in same row block
for i, line in enumerate(wt2000.splitlines()):
    if "Bush" in line and "Gore" in line:
        pcts = extract_pcts(line)
        if len(pcts) >= 2:
            bush_gore.append(pcts[:2])
    elif re.search(r"\|\s*Bush|\|\s*Gore", line):
        chunk = "\n".join(wt2000.splitlines()[max(0, i - 2) : i + 5])
        pcts = extract_pcts(chunk)
        if len(pcts) >= 2:
            bush_gore.append(pcts[:2])

# Prefer last Gallup-like if any table with Final
final_2000 = None
lines = wt2000.splitlines()
for i, line in enumerate(lines):
    if re.search(r"Final|November\s*[0-9]|Nov\.\s*[0-9]", line, re.I):
        chunk = "\n".join(lines[i : i + 15])
        pcts = extract_pcts(chunk)
        if len(pcts) >= 2:
            final_2000 = pcts[:2]
            break
if final_2000 is None and bush_gore:
    final_2000 = bush_gore[-1]

if final_2000:
    # In 2000 tables order often Gore then Bush or Bush then Gore — check page comment
    # Wikipedia 2000 often lists Dem (Gore) then Rep (Bush). Use that convention; document.
    dem_v, gop_v = final_2000[0], final_2000[1]
    wiki_proxy_rows.append(
        {
            "election_year": 2000,
            "as_of_date": "2000-11-01",
            "dem_pct": dem_v,
            "gop_pct": gop_v,
            "other_pct": np.nan,
            "dem_two_party_share_poll": dem_v / (dem_v + gop_v),
            "source": "wikipedia_nationwide_2000_finalish_row",
            "source_url": "https://en.wikipedia.org/wiki/Nationwide_opinion_polling_for_the_2000_United_States_presidential_election",
            "geography": "national",
            "note": "Parsed near-final row; column order assumed Gore(Dem)/Bush(GOP) as common on page; verify before causal use; Extension",
        }
    )
else:
    wiki_proxy_rows.append(
        {
            "election_year": 2000,
            "as_of_date": "",
            "dem_pct": np.nan,
            "gop_pct": np.nan,
            "other_pct": np.nan,
            "dem_two_party_share_poll": np.nan,
            "source": "wikipedia_parse_uncertain",
            "source_url": "https://en.wikipedia.org/wiki/Nationwide_opinion_polling_for_the_2000_United_States_presidential_election",
            "geography": "national",
            "note": "No reliable aggregate parsed; blank",
        }
    )

# 2004 gap
wiki_proxy_rows.append(
    {
        "election_year": 2004,
        "as_of_date": "",
        "dem_pct": np.nan,
        "gop_pct": np.nan,
        "other_pct": np.nan,
        "dem_two_party_share_poll": np.nan,
        "source": "no_dedicated_nationwide_wiki_page",
        "source_url": "https://en.wikipedia.org/wiki/Opinion_polling_for_United_States_presidential_elections",
        "geography": "national",
        "note": "No machine-readable nationwide 2004 page via API; left blank (symmetric documentation of gap)",
    }
)

wiki_proxy = pd.DataFrame(wiki_proxy_rows)

# Merge: prefer 538 for 2020/2024; wiki for earlier
proxy = pd.concat([wiki_proxy, proxy_538], ignore_index=True)
proxy = proxy.sort_values(["election_year", "source"]).drop_duplicates("election_year", keep="last")
# keep_last: 538 rows come after wiki for 2020/2024 if we sort — actually drop_duplicates keep last after sort by source alphabetically is wrong.
# Explicit preference:
pref = []
by_year = {}
for _, r in pd.concat([wiki_proxy, proxy_538], ignore_index=True).iterrows():
    y = int(r["election_year"])
    src = str(r["source"])
    if y not in by_year:
        by_year[y] = r
    else:
        # prefer fivethirtyeight
        if "fivethirtyeight" in src and "fivethirtyeight" not in str(by_year[y]["source"]):
            by_year[y] = r
        elif pd.isna(by_year[y]["dem_pct"]) and pd.notna(r["dem_pct"]):
            by_year[y] = r
proxy = pd.DataFrame(by_year.values()).sort_values("election_year")
write_both("us_national_poll_election_year_proxy.csv", proxy)

# Long poll archive: wiki sources metadata + 538 daily (already written)
poll_long_meta = proxy.copy()
poll_long_meta["layer"] = "national_election_year_proxy"
write_both("us_polls_README_table.csv", poll_long_meta)

# ---------------------------------------------------------------------------
# 3) Partisan climate proxy (parallel to TW ESC identity PROXY)
#    Homebrew county lean from MEDSL votes — NOT Cook PVI (subscription).
# ---------------------------------------------------------------------------
us_panel = pd.read_csv(ROOT / "fyp_us_core/us_county_panel_features_family_a.csv")
# National two-party from county votes weighted? Panel has shares only — use unweighted mean of target for national approx OR recompute from MEDSL.
medsl = pd.read_csv(ROOT / "fyp_us_taiwan_transfer/raw/us/county-president.csv", low_memory=False)
# Expect columns like year, county_fips, party, candidatevotes — inspect
print("MEDSL cols", list(medsl.columns)[:20])

# Flexible column detect
cols = {c.lower(): c for c in medsl.columns}
year_c = cols.get("year")
fips_c = cols.get("county_fips") or cols.get("fips") or cols.get("county_fips")
party_c = cols.get("party")
votes_c = cols.get("candidatevotes") or cols.get("candidate_votes") or cols.get("votes")
# MEDSL often: year, state, county_name, county_fips, office, candidate, party, candidatevotes, totalvotes, version, mode
if year_c and party_c and votes_c:
    m = medsl.copy()
    m[year_c] = pd.to_numeric(m[year_c], errors="coerce")
    m = m[m[year_c].isin([2000, 2004, 2008, 2012, 2016, 2020, 2024])]
    m["_party"] = m[party_c].astype(str).str.upper()
    m["_votes"] = pd.to_numeric(m[votes_c], errors="coerce").fillna(0)
    # DEMOCRAT / REPUBLICAN
    dem = m[m["_party"].str.contains("DEM", na=False)].groupby([year_c, fips_c], as_index=False)["_votes"].sum()
    gop = m[m["_party"].str.contains("REP", na=False)].groupby([year_c, fips_c], as_index=False)["_votes"].sum()
    dem = dem.rename(columns={"_votes": "dem_votes", year_c: "year", fips_c: "region_id"})
    gop = gop.rename(columns={"_votes": "gop_votes", year_c: "year", fips_c: "region_id"})
    cg = dem.merge(gop, on=["year", "region_id"], how="outer").fillna(0)
    cg["two_party"] = cg["dem_votes"] + cg["gop_votes"]
    cg = cg[cg["two_party"] > 0]
    cg["dem_two_party_share"] = cg["dem_votes"] / cg["two_party"]
    # national share
    nat_y = cg.groupby("year", as_index=False).agg(dem_votes=("dem_votes", "sum"), gop_votes=("gop_votes", "sum"))
    nat_y["nat_dem_two_party"] = nat_y["dem_votes"] / (nat_y["dem_votes"] + nat_y["gop_votes"])
    cg = cg.merge(nat_y[["year", "nat_dem_two_party"]], on="year", how="left")
    cg["partisan_lean_dem"] = cg["dem_two_party_share"] - cg["nat_dem_two_party"]
    # lag lean (previous election) to mirror ESC lag_* identity
    cg = cg.sort_values(["region_id", "year"])
    cg["lag_partisan_lean_dem"] = cg.groupby("region_id")["partisan_lean_dem"].shift(1)
    cg["lag_nat_dem_two_party"] = cg.groupby("region_id")["nat_dem_two_party"].shift(1)
    lean = cg[
        [
            "region_id",
            "year",
            "dem_two_party_share",
            "nat_dem_two_party",
            "partisan_lean_dem",
            "lag_partisan_lean_dem",
            "lag_nat_dem_two_party",
        ]
    ].copy()
    lean["proxy_quality"] = "homebrew_from_MEDSL_not_Cook_PVI"
    lean["notes"] = (
        "County Dem two-party share minus national Dem two-party same year; "
        "lag_* = previous presidential election. Parallel to TW ESC national identity PROXY "
        "but geographic (county). NOT official Cook PVI (subscriber)."
    )
    write_both("us_partisan_climate_county_proxy.csv", lean)
else:
    print("MEDSL schema unexpected; skipping lean build")
    lean = None

# National identity-climate analog: national Dem two-party by year (for election-year proxy table)
if lean is not None:
    nat_id = (
        lean.groupby("year", as_index=False)["nat_dem_two_party"]
        .first()
        .rename(columns={"year": "election_year", "nat_dem_two_party": "nat_dem_two_party_share"})
    )
    nat_id["source"] = "MEDSL_county_president_aggregated"
    nat_id["parallel_to_TW"] = "ESC_national_election_year_proxy (party climate side)"
    write_both("us_national_partisan_climate_proxy.csv", nat_id)

# ---------------------------------------------------------------------------
# 4) US party platforms metadata (parallel to TW platforms_long) — NO full text dump
# ---------------------------------------------------------------------------
# Word counts from American Presidency Project index (public page, cited in search results 2026)
platforms = [
    # year, party, nominee, words, url path slug, note
    (2000, "Democratic", "Al Gore", 24220, "https://www.presidency.ucsb.edu/documents/2000-democratic-party-platform", "APP index word count"),
    (2000, "Republican", "George W. Bush", 34555, "https://www.presidency.ucsb.edu/documents/republican-party-platform-2000", "APP index word count; URL may redirect"),
    (2004, "Democratic", "John Kerry", 17751, "https://www.presidency.ucsb.edu/documents/2004-democratic-party-platform", "APP"),
    (2004, "Republican", "George W. Bush", 41156, "https://www.presidency.ucsb.edu/documents/2004-republican-party-platform", "APP"),
    (2008, "Democratic", "Barack Obama", 25997, "https://www.presidency.ucsb.edu/documents/2008-democratic-party-platform", "APP"),
    (2008, "Republican", "John McCain", 23585, "https://www.presidency.ucsb.edu/documents/2008-republican-party-platform", "APP"),
    (2012, "Democratic", "Barack Obama", 26558, "https://www.presidency.ucsb.edu/documents/2012-democratic-party-platform", "APP"),
    (2012, "Republican", "Mitt Romney", 30563, "https://www.presidency.ucsb.edu/documents/2012-republican-party-platform", "APP"),
    (2016, "Democratic", "Hillary Clinton", 26058, "https://www.presidency.ucsb.edu/documents/2016-democratic-party-platform", "APP"),
    (2016, "Republican", "Donald Trump", 35467, "https://www.presidency.ucsb.edu/documents/2016-republican-party-platform", "APP"),
    (2020, "Democratic", "Joe Biden", 42092, "https://www.presidency.ucsb.edu/documents/2020-democratic-party-platform", "APP"),
    (2020, "Republican", "Donald Trump", np.nan, "https://www.presidency.ucsb.edu/documents/resolution-regarding-the-republican-party-platform", "RNC readopted 2016 platform; see 2016 GOP"),
    (2024, "Democratic", "Kamala Harris", 42203, "https://www.presidency.ucsb.edu/documents/2024-democratic-party-platform", "APP"),
    (2024, "Republican", "Donald Trump", 5201, "https://www.presidency.ucsb.edu/documents/2024-republican-party-platform", "APP; shorter 2024 platform"),
]

# Theme tags (short summaries — NOT substantial copyrighted platform text)
# Parallel to TW issue_category coding
theme_map = {
    ("Democratic", 2000): ["economy", "healthcare", "education", "social_security"],
    ("Republican", 2000): ["tax_cuts", "defense", "education", "family"],
    ("Democratic", 2004): ["iraq_war", "economy", "healthcare", "homeland_security"],
    ("Republican", 2004): ["iraq_war", "tax_cuts", "homeland_security", "values"],
    ("Democratic", 2008): ["economy", "iraq_war", "healthcare", "energy"],
    ("Republican", 2008): ["economy", "iraq_war", "tax_cuts", "energy"],
    ("Democratic", 2012): ["economy", "healthcare", "education", "immigration"],
    ("Republican", 2012): ["economy", "debt", "healthcare_repeal", "energy"],
    ("Democratic", 2016): ["economy", "healthcare", "climate", "immigration"],
    ("Republican", 2016): ["immigration", "trade", "tax_cuts", "judiciary"],
    ("Democratic", 2020): ["covid", "economy", "climate", "racial_justice"],
    ("Republican", 2020): ["immigration", "economy", "judiciary", "china"],  # via 2016 + messaging
    ("Democratic", 2024): ["economy", "abortion", "climate", "democracy"],
    ("Republican", 2024): ["border", "deportation", "energy", "tax_cuts"],
}

plat_rows = []
for year, party, nominee, words, url, note in platforms:
    themes = theme_map.get((party, year), ["general"])
    for th in themes:
        plat_rows.append(
            {
                "year": year,
                "election": "US_presidential",
                "candidate": nominee,
                "party": party,
                "issue_category": th,
                "stance_text": "",  # intentionally empty — do not paste copyrighted platform body
                "source_url": url,
                "published_date": "",
                "pre_election_flag": True,
                "verified": "metadata_from_APP_index",
                "section": "party_platform_metadata",
                "word_count_app": words,
                "note": note
                + "; stance_text left empty (copyright); themes are short coder tags for Extension parity with TW platforms_long",
            }
        )

plat_df = pd.DataFrame(plat_rows)
write_both("us_platforms_long.csv", plat_df)

# ---------------------------------------------------------------------------
# 5) Socio feature parity snapshot (document, optional rename mirror)
# ---------------------------------------------------------------------------
socio_cmp = pd.DataFrame(
    [
        {"feature_layer": "lag_unemployment", "TW": "lag_unemployment_pct (county, DGBAS)", "US": "unemployment (ERS yearly in panel)", "parity": "YES_approx"},
        {"feature_layer": "lag_income", "TW": "lag_log_mean_income (tax mean)", "US": "log_median_income / median_income (ERS/ACS wave)", "parity": "YES_approx_diff_def"},
        {"feature_layer": "lag_density", "TW": "lag_log_pop_density (MOI)", "US": "log_density (ACS2020 static)", "parity": "YES_approx_US_static"},
        {"feature_layer": "education", "TW": "MISSING county bachelor panel", "US": "education_rate (ERS)", "parity": "US_only"},
        {"feature_layer": "prev_vote_history", "TW": "prev/prev2 DPP KMT Other + margins", "US": "prev/prev2 dem/gop two-party + margins", "parity": "YES"},
        {"feature_layer": "identity_climate", "TW": "ESC national lag_* id/tongdu PROXY", "US": "county lag_partisan_lean_dem (MEDSL homebrew)", "parity": "YES_diff_construct"},
        {"feature_layer": "national_polls", "TW": "Extension taiwan_polls_long (national)", "US": "538 daily 2020/2024 + wiki year proxy", "parity": "PARTIAL_pre2020_thin"},
        {"feature_layer": "platforms", "TW": "Extension platforms_long (wiki text excerpts)", "US": "APP metadata + theme tags (no full text)", "parity": "YES_metadata_level"},
    ]
)
write_both("us_tw_socio_layer_snapshot.csv", socio_cmp)

# Provenance JSON
prov = {
    "built_at_tz": "Asia/Hong_Kong",
    "seed": SEED,
    "sources": [
        "https://raw.githubusercontent.com/fivethirtyeight/data/master/polls/2024-averages/presidential_general_averages_2024-09-12_uncorrected.csv",
        "Wikipedia Nationwide opinion polling pages (MediaWiki API wikitext)",
        "MEDSL county-president.csv (already in repo)",
        "American Presidency Project party platform index (URLs + published word counts)",
    ],
    "not_used": [
        "Cook PVI subscriber spreadsheets",
        "FiveThirtyEight projects.fivethirtyeight.com live CSVs (blocked/HTML)",
        "ANES microdata (login/application)",
        "Full copyrighted platform body text",
    ],
    "family_a": "NOT modified; Extension only",
}
for d in (OUT_DATA, OUT_GH):
    (d / "PROVENANCE.json").write_text(json.dumps(prov, indent=2))
    (d / "README.md").write_text(
        """# US parity Extension (parallel to TW Extension / ESC PROXY)

Built for FYP **data-layer parity** with Taiwan — not for mixing into Core specification Core MAE.

| File | Parallel to TW |
|------|----------------|
| `us_polls_national_daily_538.csv` | `extension_poll_platform/taiwan_polls_long.csv` (national only) |
| `us_national_poll_election_year_proxy.csv` | national poll index by election year |
| `us_partisan_climate_county_proxy.csv` | `esc_proxies/*` identity climate (different construct) |
| `us_national_partisan_climate_proxy.csv` | `esc_national_election_year_proxy.csv` |
| `us_platforms_long.csv` | `taiwan_platforms_long.csv` (metadata/themes; no full text) |
| `us_tw_socio_layer_snapshot.csv` | quick socio parity checklist |

Rules: public sources only; blanks where parse failed; no invented numbers.
"""
    )

print("DONE")
print(proxy.to_string(index=False))
