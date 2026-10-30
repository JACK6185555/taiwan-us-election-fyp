# US national poll-as-forecast accuracy (Extension)

**Built:** 2026-10-30 14:05 Asia/Hong_Kong (UTC+8)  
**Scope:** National Dem two-party poll share vs MEDSL-aggregated national Dem two-party outcome.  
**NOT Core specification.** County polls not used.

## Sources
- Poll proxy: `data/us_parity_extension/us_national_poll_election_year_proxy.csv`
- 2004 fill: 538 GitHub `pres_pollaverages_1968-2016.csv` (last National modeldate in archive)
- Outcomes: `data/us_parity_extension/us_national_partisan_climate_proxy.csv` (MEDSL)

## Key numbers
- MAE (poll as forecast): **0.0142** over 7 years (2000,2004,2008,2012,2016,2020,2024)
- MAE (previous election share): **0.0221**
- Mean signed error (poll − actual): **0.0017**

## Caveats
- 2004 National last modeldate in 538 archive = **2004-11-02** (election day). Other legacy cycles: use last modeldate in file. 2020/2024 use separate averages files (2024 ends 2024-09-12).
- 2024 averages file in repo ends **2024-09-12** (pre-election).
- This is national Extension diagnostics only — does not enter Core specification.
