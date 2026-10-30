#!/usr/bin/env python3
"""Demo 1 — synthetic US + TW region×year panels with known spatial/structure shift.

Methodological harness only. Numbers are synthetic (seeded RNG), not elections.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cleaned"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def _regions(country: str, n: int) -> list[str]:
    return [f"{country}-R{i:03d}" for i in range(n)]


def make_country(
    country: str,
    n_regions: int,
    years: list[int],
    *,
    base: float,
    income_coef: float,
    edu_coef: float,
    urban_coef: float,
    spatial_ar: float,
    noise: float,
    shift_year: int | None = None,
    shift_delta: float = 0.0,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate panel with latent socio → share mapping; optional year structure shift."""
    rid = _regions(country, n_regions)
    # Fixed region traits
    income = rng.normal(50_000 if country == "US" else 35_000, 12_000, n_regions)
    edu = rng.uniform(0.15, 0.55, n_regions)
    urban = rng.uniform(0.1, 0.95, n_regions)
    # Spatial-ish: adjacent ids share a latent factor
    spatial = np.zeros(n_regions)
    spatial[0] = rng.normal()
    for i in range(1, n_regions):
        spatial[i] = spatial_ar * spatial[i - 1] + rng.normal(scale=np.sqrt(1 - spatial_ar**2))

    rows = []
    prev = np.full(n_regions, np.nan)
    for y in years:
        # structure shift after shift_year (different income coefficient)
        ic = income_coef + (shift_delta if shift_year and y > shift_year else 0.0)
        z = (
            base
            + ic * ((income - income.mean()) / income.std())
            + edu_coef * ((edu - edu.mean()) / edu.std())
            + urban_coef * ((urban - urban.mean()) / urban.std())
            + 0.08 * spatial
            + rng.normal(0, noise, n_regions)
        )
        # mild persistence
        if not np.isnan(prev).all():
            z = 0.65 * z + 0.35 * np.nan_to_num(prev, nan=base)
        share = 1 / (1 + np.exp(-z))
        # clip to realistic band
        share = np.clip(share, 0.15, 0.85)
        total_votes = rng.integers(5_000, 200_000, n_regions).astype(float)
        for i, r in enumerate(rid):
            rows.append(
                {
                    "country": country,
                    "region_id": r,
                    "region_name": f"{country} region {i}",
                    "year": y,
                    "party_share_A": float(share[i]),
                    "prev_share_A": float(prev[i]) if not np.isnan(prev[i]) else np.nan,
                    "log_total_votes": float(np.log1p(total_votes[i])),
                    "median_income": float(income[i] + rng.normal(0, 500)),
                    "education_rate": float(np.clip(edu[i] + rng.normal(0, 0.01), 0, 1)),
                    "urbanization": float(np.clip(urban[i] + rng.normal(0, 0.02), 0, 1)),
                    "unemployment": float(np.clip(0.06 + 0.04 * (1 - urban[i]) + rng.normal(0, 0.01), 0.01, 0.25)),
                    "feature_set": "synthetic_full",
                    "split_note": "synthetic",
                    "data_kind": "synthetic",
                }
            )
        prev = share.copy()
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    years_us = [2000, 2004, 2008, 2012, 2016, 2020, 2024]
    years_tw = [2000, 2004, 2008, 2012, 2016, 2020, 2024]

    # US: stronger income→Dem mapping; shift after 2012
    us = make_country(
        "US",
        n_regions=80,
        years=years_us,
        base=0.0,
        income_coef=0.35,
        edu_coef=0.45,
        urban_coef=0.30,
        spatial_ar=0.7,
        noise=0.25,
        shift_year=2012,
        shift_delta=-0.15,  # income coef weakens
        rng=rng,
    )
    # TW: education/urban dominate; different base; milder income; shift after 2016
    tw = make_country(
        "TW",
        n_regions=22,
        years=years_tw,
        base=0.15,
        income_coef=0.10,
        edu_coef=0.55,
        urban_coef=0.25,
        spatial_ar=0.55,
        noise=0.20,
        shift_year=2016,
        shift_delta=0.20,
        rng=rng,
    )
    panel = pd.concat([us, tw], ignore_index=True)
    path = OUT / "synthetic_us_tw_panel.csv"
    panel.to_csv(path, index=False)

    # Sanity check summary
    summary = (
        panel.groupby(["country", "year"], as_index=False)
        .agg(
            n=("region_id", "count"),
            mean_share=("party_share_A", "mean"),
            mean_prev=("prev_share_A", "mean"),
            mean_income=("median_income", "mean"),
        )
    )
    summary_path = OUT / "synthetic_us_tw_sanity_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Known shift diagnostic: correlation income vs share before/after
    diag_rows = []
    for c, shift_y in [("US", 2012), ("TW", 2016)]:
        sub = panel[panel["country"] == c]
        for label, mask in [
            ("pre_shift", sub["year"] <= shift_y),
            ("post_shift", sub["year"] > shift_y),
        ]:
            s = sub.loc[mask]
            if len(s) < 5:
                continue
            corr = s["median_income"].corr(s["party_share_A"])
            diag_rows.append({"country": c, "window": label, "corr_income_share": corr, "n": len(s)})
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(OUT / "synthetic_structure_shift_diag.csv", index=False)

    print(f"Wrote {path} shape={panel.shape}")
    print(summary.to_string(index=False))
    print(diag.to_string(index=False))


if __name__ == "__main__":
    main()
