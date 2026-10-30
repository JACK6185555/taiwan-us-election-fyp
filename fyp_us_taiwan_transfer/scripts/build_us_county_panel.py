#!/usr/bin/env python3
"""Build cleaned US county presidential panel → shared schema.

Source: MEDSL mirror under raw/us/county-president.csv
Target: Dem two-party share (Dem / (Dem+Rep)) as party_share_A.
No invented socio features — ACS blocked; log_total_votes as size proxy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "us" / "county-president.csv"
OUT = ROOT / "cleaned"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_csv(RAW, low_memory=False)
    # Keep rows with FIPS
    df = df[df["county_fips"].notna()].copy()
    df["county_fips"] = df["county_fips"].astype(int).astype(str).str.zfill(5)
    # Aggregate votes by party (some years multiple candidates? MEDSL usually one Dem/Rep)
    party = df[df["party"].isin(["DEMOCRAT", "REPUBLICAN"])].copy()
    agg = (
        party.groupby(["year", "state_po", "county_fips", "county_name", "party"], as_index=False)[
            "candidatevotes"
        ]
        .sum()
    )
    pivot = agg.pivot_table(
        index=["year", "state_po", "county_fips", "county_name"],
        columns="party",
        values="candidatevotes",
        aggfunc="sum",
    ).reset_index()
    pivot.columns.name = None
    for col in ["DEMOCRAT", "REPUBLICAN"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["two_party"] = pivot["DEMOCRAT"].fillna(0) + pivot["REPUBLICAN"].fillna(0)
    pivot = pivot[pivot["two_party"] > 0].copy()
    pivot["party_share_A"] = pivot["DEMOCRAT"] / pivot["two_party"]
    # totalvotes from original (max per county-year)
    totals = (
        df.groupby(["year", "county_fips"], as_index=False)["totalvotes"].max()
    )
    pivot = pivot.merge(totals, on=["year", "county_fips"], how="left")
    pivot["log_total_votes"] = np.log1p(pivot["totalvotes"].fillna(pivot["two_party"]))
    pivot = pivot.sort_values(["county_fips", "year"])
    pivot["prev_share_A"] = pivot.groupby("county_fips")["party_share_A"].shift(1)
    # gap years check: shift is previous row — ensure year gap is 4
    pivot["prev_year"] = pivot.groupby("county_fips")["year"].shift(1)
    bad = pivot["prev_share_A"].notna() & ((pivot["year"] - pivot["prev_year"]) != 4)
    # still keep shift but flag
    pivot["prev_share_flag"] = np.where(bad, "non_4yr_gap", "ok")
    panel = pd.DataFrame(
        {
            "country": "US",
            "region_id": pivot["county_fips"],
            "region_name": pivot["county_name"].astype(str) + ", " + pivot["state_po"].astype(str),
            "year": pivot["year"].astype(int),
            "party_share_A": pivot["party_share_A"].astype(float),
            "prev_share_A": pivot["prev_share_A"].astype(float),
            "log_total_votes": pivot["log_total_votes"].astype(float),
            "dem_votes": pivot["DEMOCRAT"].astype(float),
            "rep_votes": pivot["REPUBLICAN"].astype(float),
            "two_party_votes": pivot["two_party"].astype(float),
            "total_votes": pivot["totalvotes"].astype(float),
            "state_po": pivot["state_po"].astype(str),
            "median_income": np.nan,
            "education_rate": np.nan,
            "urbanization": np.nan,
            "unemployment": np.nan,
            "feature_set": "shared_history_size",
            "split_note": "real_medsl; socio ACS blocked",
            "data_kind": "real",
            "prev_share_flag": pivot["prev_share_flag"],
        }
    )
    path = OUT / "us_county_presidential_panel.csv"
    panel.to_csv(path, index=False)
    summary = panel.groupby("year").agg(
        n_counties=("region_id", "nunique"),
        mean_dem_twoparty=("party_share_A", "mean"),
        frac_prev=("prev_share_A", lambda s: float(s.notna().mean())),
    )
    summary.to_csv(OUT / "us_county_panel_year_summary.csv")
    print(f"Wrote {path} shape={panel.shape}")
    print(summary)


if __name__ == "__main__":
    main()
