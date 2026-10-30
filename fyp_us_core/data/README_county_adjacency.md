# US Census County Adjacency

**Source (official):** U.S. Census Bureau County Adjacency File  
**File used:** `county_adjacency2024.txt`  
**URL:** https://www2.census.gov/geo/docs/reference/county_adjacency/county_adjacency2024.txt  
**Index page:** https://www.census.gov/geographies/reference-files/time-series/geo/county-adjacency.html  
**Downloaded:** 2026-10-31 (Asia/Hong_Kong, UTC+8)

## Format

Pipe-delimited (`|`) with header:

`County Name|County GEOID|Neighbor Name|Neighbor GEOID`

- GEOID is 5-digit county FIPS (state+county).
- Each county includes a **self** row (County GEOID == Neighbor GEOID); scripts drop self-loops when building the undirected edge list.
- 2024 file: ~3236 unique GEOIDs (includes PR / territories); election panel uses contiguous US + AK/HI counties that appear in Core panel (~3111 with complete OOT years).

## Use in this FYP

- Build Queen-style contiguity for US RQ2 spatial / SGC / GCN-lite pilots.
- Prefer this official list over Plotly GeoJSON Queen when documenting adjacency provenance.
- Script: `fyp_us_taiwan_transfer/scripts/run_us_census_adjacency_pilot.py`
- Outputs: `results/us_census_adjacency_pilot/`

## Notes

- County boundaries / merges can differ by year; adjacency is a **2024 reference** snapshot. Panel FIPS that disappear after merges may be missing from the graph — see match-rate JSON in results.
- Do **not** invent edges for unmatched FIPS; report islands / dropouts honestly.
