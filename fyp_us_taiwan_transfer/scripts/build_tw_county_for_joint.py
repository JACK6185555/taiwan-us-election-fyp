#!/usr/bin/env python3
"""Align Taiwan county presidential panel to shared schema for joint experiments.

Prefers tw2 long panel 2000–2024 (has country/region_id/target_DPP).
Falls back to cleaned/county_election_socio_panel_2016_2024.csv.
Documents year overlap vs US honestly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cleaned"
OUT.mkdir(parents=True, exist_ok=True)

TW2 = Path("/workspace/tw2/report/src/tw-election-forecast/data/county_panel_features_2000_2024.csv")
TW_SOCIO = Path("/workspace/taiwan-election-data/cleaned/county_election_socio_panel_2016_2024.csv")
US_PANEL = OUT / "us_county_presidential_panel.csv"


def from_tw2() -> pd.DataFrame:
    df = pd.read_csv(TW2)
    # target_DPP_vote_share; prev_DPP_vote_share
    panel = pd.DataFrame(
        {
            "country": "TW",
            "region_id": df["region_id"].astype(str),
            "region_name": df["region_name"].astype(str),
            "year": df["year"].astype(int),
            "party_share_A": df["target_DPP_vote_share"].astype(float),
            "prev_share_A": df["prev_DPP_vote_share"].astype(float),
            "log_total_votes": np.log1p(df["valid_votes"].astype(float)),
            "median_income": np.nan,  # not in this file as ACS-like median
            "education_rate": np.nan,
            "urbanization": np.nan,
            "unemployment": np.nan,
            "turnout": df["turnout"].astype(float),
            "feature_set": "shared_history_size",
            "split_note": "real_cec_tw2_panel_2000_2024",
            "data_kind": "real",
        }
    )
    return panel


def enrich_socio(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach TW county socio where available (2016–2024), marked country-specific."""
    if not TW_SOCIO.exists():
        return panel
    socio = pd.read_csv(TW_SOCIO)
    # map Chinese county name → we need match on region_name
    # tw2 uses Chinese names in region_name
    s = socio.rename(
        columns={
            "election_year": "year",
            "county_name": "region_name",
            "mean_income": "tw_mean_income",
            "population_density": "tw_pop_density",
        }
    )[["year", "region_name", "tw_mean_income", "tw_pop_density", "people_total"]]
    out = panel.merge(s, on=["year", "region_name"], how="left")
    # Do NOT copy into median_income for joint — keep as TW-specific columns
    out["tw_log_people"] = np.log1p(out["people_total"].astype(float))
    out = out.drop(columns=["people_total"])
    return out


def main() -> None:
    if TW2.exists():
        panel = from_tw2()
        src = str(TW2)
    else:
        raise SystemExit("TW2 county panel missing")
    panel = enrich_socio(panel)
    path = OUT / "tw_county_presidential_panel_joint.csv"
    panel.to_csv(path, index=False)

    # Joint shared panel (history features only)
    us = pd.read_csv(US_PANEL)
    shared_cols = [
        "country",
        "region_id",
        "region_name",
        "year",
        "party_share_A",
        "prev_share_A",
        "log_total_votes",
        "feature_set",
        "split_note",
        "data_kind",
    ]
    joint = pd.concat([us[shared_cols], panel[shared_cols]], ignore_index=True)
    joint.to_csv(OUT / "joint_real_shared_panel.csv", index=False)

    # Overlap report
    us_years = sorted(us["year"].unique())
    tw_years = sorted(panel["year"].unique())
    overlap = sorted(set(us_years) & set(tw_years))
    report = pd.DataFrame(
        [
            {"item": "us_years", "value": ",".join(map(str, us_years))},
            {"item": "tw_years", "value": ",".join(map(str, tw_years))},
            {"item": "year_overlap", "value": ",".join(map(str, overlap))},
            {"item": "us_n_regions_mean", "value": str(us.groupby("year")["region_id"].nunique().mean())},
            {"item": "tw_n_regions", "value": str(panel["region_id"].nunique())},
            {"item": "shared_features", "value": "prev_share_A,log_total_votes"},
            {
                "item": "feature_overlap_note",
                "value": "ACS US socio blocked; TW socio kept as tw_* country-specific only; joint uses history+size",
            },
            {"item": "tw_source", "value": src},
        ]
    )
    report.to_csv(OUT / "us_tw_overlap_report.csv", index=False)
    print(report.to_string(index=False))
    print(f"Wrote {path} and joint_real_shared_panel.csv")


if __name__ == "__main__":
    main()
