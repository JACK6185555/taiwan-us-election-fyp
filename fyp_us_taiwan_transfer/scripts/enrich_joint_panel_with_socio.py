#!/usr/bin/env python3
"""Merge US ERS/ACS socio + TW income/density proxies into joint shared panel.

Shared numeric features (both countries, after within-country z-score):
  prev_share_A, log_total_votes, z_income, z_log_density

US-only extras kept on US panel: education_rate, unemployment (ERS year-matched).
TW constructs: tax mean_income ≠ ACS median HH income — z-score within country
so joint models see relative standing, not dollar levels (Guide §19).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "cleaned"
RAW_US = ROOT / "raw" / "us"
ERS = RAW_US / "ers"
OUT = CLEAN
OUT.mkdir(parents=True, exist_ok=True)

ELECTION_YEARS = [2000, 2004, 2008, 2012, 2016, 2020, 2024]
TW_SOCIO = Path("/workspace/taiwan-election-data/cleaned/county_election_socio_panel_2016_2024.csv")


def _fips5(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)


def load_ers_unemployment() -> pd.DataFrame:
    inc = pd.read_csv(ERS / "unemployment_median_income_2000_23.csv")
    inc["fips"] = _fips5(inc["FIPS_Code"])
    # yearly unemployment rate
    rows = []
    for y in ELECTION_YEARS:
        attr = f"Unemployment_rate_{min(y, 2023)}"
        sub = inc[inc["Attribute"] == attr][["fips", "Value"]].rename(columns={"Value": "unemployment"})
        sub["year"] = y
        rows.append(sub)
    unemp = pd.concat(rows, ignore_index=True)
    unemp["unemployment"] = pd.to_numeric(unemp["unemployment"], errors="coerce")
    # median HH income 2022 (static)
    med = inc[inc["Attribute"] == "Median_Household_Income_2022"][["fips", "Value"]].rename(
        columns={"Value": "ers_median_income_2022"}
    )
    med["ers_median_income_2022"] = pd.to_numeric(med["ers_median_income_2022"], errors="coerce")
    return unemp, med


def load_ers_education() -> pd.DataFrame:
    edu = pd.read_csv(ERS / "education_1970_2023.csv", encoding="latin1")
    edu["fips"] = _fips5(edu["FIPS Code"])
    # Map election year → nearest ERS bachelor-% wave
    wave = {
        2000: "Percent of adults with a bachelor's degree or higher, 2000",
        2004: "Percent of adults with a bachelor's degree or higher, 2000",
        2008: "Percent of adults with a bachelor's degree or higher, 2008-12",
        2012: "Percent of adults with a bachelor's degree or higher, 2008-12",
        2016: "Percent of adults with a bachelor's degree or higher, 2019-23",
        2020: "Percent of adults with a bachelor's degree or higher, 2019-23",
        2024: "Percent of adults with a bachelor's degree or higher, 2019-23",
    }
    rows = []
    for y, attr in wave.items():
        sub = edu[edu["Attribute"] == attr][["fips", "Value"]].rename(columns={"Value": "education_rate"})
        sub["year"] = y
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    out["education_rate"] = pd.to_numeric(out["education_rate"], errors="coerce") / 100.0
    return out


def load_acs_static() -> pd.DataFrame:
    acs = pd.read_csv(RAW_US / "ACS2020_5year_county.csv", low_memory=False)
    acs["fips"] = _fips5(acs["FIPS"])
    pop25 = pd.to_numeric(acs["Population 25 Years and Over:"], errors="coerce")
    bach = pd.to_numeric(acs["Population 25 Years and Over: Bachelor's Degree"], errors="coerce")
    mast = pd.to_numeric(acs["Population 25 Years and Over: Master's Degree"], errors="coerce")
    prof = pd.to_numeric(acs["Population 25 Years and Over: Professional School Degree"], errors="coerce")
    doc = pd.to_numeric(acs["Population 25 Years and Over: Doctorate Degree"], errors="coerce")
    edu = (bach.fillna(0) + mast.fillna(0) + prof.fillna(0) + doc.fillna(0)) / pop25.replace(0, np.nan)
    dens = pd.to_numeric(acs["Population Density (Per Sq. Mile)"], errors="coerce")
    inc = pd.to_numeric(
        acs["Median Household Income (In current year Inflation Adjusted Dollars)"], errors="coerce"
    )
    return pd.DataFrame(
        {
            "fips": acs["fips"],
            "acs_median_income": inc,
            "acs_education_rate": edu,
            "acs_density": dens,
            "acs_log_density": np.log1p(dens),
        }
    )


def enrich_us(panel: pd.DataFrame) -> pd.DataFrame:
    unemp, med = load_ers_unemployment()
    edu = load_ers_education()
    acs = load_acs_static()
    p = panel.copy()
    p["region_id"] = _fips5(p["region_id"])
    p = p.drop(columns=["median_income", "education_rate", "urbanization", "unemployment"], errors="ignore")
    p = p.merge(unemp, left_on=["region_id", "year"], right_on=["fips", "year"], how="left")
    p = p.drop(columns=["fips"], errors="ignore")
    p = p.merge(edu, left_on=["region_id", "year"], right_on=["fips", "year"], how="left")
    p = p.drop(columns=["fips"], errors="ignore")
    p = p.merge(med, left_on="region_id", right_on="fips", how="left").drop(columns=["fips"], errors="ignore")
    p = p.merge(acs, left_on="region_id", right_on="fips", how="left").drop(columns=["fips"], errors="ignore")
    # Prefer ERS median 2022, else ACS; education prefer ERS wave else ACS
    p["median_income"] = p["ers_median_income_2022"].fillna(p["acs_median_income"])
    p["education_rate"] = p["education_rate"].fillna(p["acs_education_rate"])
    p["urbanization"] = p["acs_log_density"]  # log density
    p["log_density"] = p["acs_log_density"]
    p["feature_set"] = "shared_history_size_socio_acs_ers"
    p["split_note"] = (
        "real_medsl+ERS_unemp_year+ERS_bach_wave+ERS_income2022_or_ACS2020; density=ACS2020 log"
    )
    return p


def enrich_tw(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    if not TW_SOCIO.exists():
        return p
    socio = pd.read_csv(TW_SOCIO)
    # Static county income/density from latest available (2020 income, 2024 density preferred)
    inc = (
        socio.sort_values("election_year")
        .groupby("county_name", as_index=False)
        .agg(tw_mean_income=("mean_income", "last"), tw_pop_density=("population_density", "last"))
    )
    # Prefer density from years where present
    dens = (
        socio.dropna(subset=["population_density"])
        .sort_values("election_year")
        .groupby("county_name", as_index=False)
        .agg(tw_pop_density=("population_density", "last"))
    )
    inc = inc.drop(columns=["tw_pop_density"], errors="ignore").merge(dens, on="county_name", how="left")
    p = p.drop(columns=["tw_mean_income", "tw_pop_density", "tw_log_people"], errors="ignore")
    p = p.merge(inc, left_on="region_name", right_on="county_name", how="left").drop(
        columns=["county_name"], errors="ignore"
    )
    p["median_income"] = p["tw_mean_income"]  # proxy — different construct
    p["education_rate"] = np.nan  # no comparable TW county bachelor rate in pack
    p["log_density"] = np.log1p(p["tw_pop_density"].astype(float))
    p["urbanization"] = p["log_density"]
    p["unemployment"] = np.nan
    p["feature_set"] = "shared_history_size_socio_acs_ers"
    p["split_note"] = (
        "real_cec; income=tax_mean_income_static_county; density=static; edu/unemp unavailable"
    )
    return p


def within_country_z(df: pd.DataFrame, col: str, out: str) -> pd.DataFrame:
    def _z(s: pd.Series) -> pd.Series:
        mu, sd = s.mean(), s.std(ddof=0)
        if sd is None or sd == 0 or np.isnan(sd):
            return pd.Series(np.nan, index=s.index)
        return (s - mu) / sd

    df[out] = df.groupby("country")[col].transform(_z)
    return df


def main() -> None:
    us = enrich_us(pd.read_csv(CLEAN / "us_county_presidential_panel.csv"))
    tw = enrich_tw(pd.read_csv(CLEAN / "tw_county_presidential_panel_joint.csv"))
    us.to_csv(OUT / "us_county_presidential_panel.csv", index=False)
    tw.to_csv(OUT / "tw_county_presidential_panel_joint.csv", index=False)

    shared_cols = [
        "country",
        "region_id",
        "region_name",
        "year",
        "party_share_A",
        "prev_share_A",
        "log_total_votes",
        "median_income",
        "education_rate",
        "log_density",
        "urbanization",
        "unemployment",
        "feature_set",
        "split_note",
        "data_kind",
    ]
    for c in shared_cols:
        if c not in us.columns:
            us[c] = np.nan
        if c not in tw.columns:
            tw[c] = np.nan
    joint = pd.concat([us[shared_cols], tw[shared_cols]], ignore_index=True)
    # log income then z within country
    joint["log_income"] = np.log1p(pd.to_numeric(joint["median_income"], errors="coerce"))
    joint = within_country_z(joint, "log_income", "z_income")
    joint = within_country_z(joint, "log_density", "z_log_density")
    # education / unemployment z (mostly US)
    joint = within_country_z(joint, "education_rate", "z_education")
    joint = within_country_z(joint, "unemployment", "z_unemployment")

    joint.to_csv(OUT / "joint_real_shared_panel.csv", index=False)

    # coverage report
    cov = []
    for c in ["US", "TW"]:
        sub = joint[joint["country"] == c]
        for col in ["prev_share_A", "log_total_votes", "z_income", "z_log_density", "z_education", "z_unemployment"]:
            cov.append(
                {
                    "country": c,
                    "column": col,
                    "non_null": int(sub[col].notna().sum()),
                    "n": len(sub),
                    "frac": float(sub[col].notna().mean()),
                }
            )
    cov_df = pd.DataFrame(cov)
    cov_df.to_csv(OUT / "joint_socio_coverage.csv", index=False)

    feature_doc = pd.DataFrame(
        [
            {"feature": "prev_share_A", "shared": True, "note": "lagged party_share_A (4yr)"},
            {"feature": "log_total_votes", "shared": True, "note": "log1p turnout/size"},
            {
                "feature": "z_income",
                "shared": True,
                "note": "within-country z of log income; US=ERS2022 median HH or ACS2020; TW=tax mean_income static",
            },
            {
                "feature": "z_log_density",
                "shared": True,
                "note": "within-country z of log1p density; US=ACS2020; TW=static pop density",
            },
            {
                "feature": "z_education",
                "shared": False,
                "note": "US ERS bachelor% wave / ACS; TW missing — US-only / not in joint socio set",
            },
            {
                "feature": "z_unemployment",
                "shared": False,
                "note": "US ERS Unemployment_rate_year; TW missing",
            },
        ]
    )
    feature_doc.to_csv(OUT / "joint_shared_feature_list.csv", index=False)

    overlap = pd.DataFrame(
        [
            {"item": "shared_features_history", "value": "prev_share_A,log_total_votes"},
            {"item": "shared_features_socio", "value": "z_income,z_log_density"},
            {
                "item": "shared_features_full",
                "value": "prev_share_A,log_total_votes,z_income,z_log_density",
            },
            {"item": "us_socio_sources", "value": "ERS_unemp+ERS_edu+ERS_income2022+ACS2020_density_income_edu"},
            {"item": "tw_socio_sources", "value": "tax_mean_income_static+pop_density_static"},
            {"item": "construct_warning", "value": "TW income ≠ US median HH; z-score within country only"},
            {"item": "us_n", "value": str(len(us))},
            {"item": "tw_n", "value": str(len(tw))},
            {"item": "us_z_income_frac", "value": f"{us['median_income'].notna().mean():.3f}"},
            {"item": "tw_z_income_frac", "value": f"{tw['median_income'].notna().mean():.3f}"},
        ]
    )
    overlap.to_csv(OUT / "us_tw_overlap_report.csv", index=False)
    print(cov_df.to_string(index=False))
    print(overlap.to_string(index=False))
    print(f"Wrote joint_real_shared_panel.csv shape={joint.shape}")


if __name__ == "__main__":
    main()
