# Shared region × year schema (RQ3 / Guide §16-aligned minimal)

**Purpose:** one CSV layout for Demo 1 synthetic panels and for best-effort real US/TW county panels so country-only vs joint OOT harnesses share code.

**Caution (Guide §19):** columns with the same name are **not** claimed to be identical constructs across countries. `party_share_A` is Dem share (US) vs DPP share (TW) — analogues for transfer experiments, not the same party family.

## Required columns

| column | type | meaning |
|---|---|---|
| `country` | str | `US` or `TW` |
| `region_id` | str | stable region key (`FIPS5` for US; TW county roman/code or Chinese name key) |
| `region_name` | str | human-readable name |
| `year` | int | election year (presidential) |
| `party_share_A` | float | target share in \[0,1\]: US = Democrat / (Dem+Rep) two-party or Dem/valid as documented; TW = DPP / valid |
| `prev_share_A` | float | same construct, previous presidential cycle (NaN in first observed year) |
| `log_total_votes` | float | log(1 + total/valid votes); weak size proxy when socio blocked |
| `split_note` | str | how row may be used (`train_candidate`, `oot_test`, etc.) — optional metadata |

## Optional / country-specific (must be marked)

| column | availability | note |
|---|---|---|
| `median_income` | synthetic always; real TW partial; real US **blocked** this run | ACS API key / Census www2 403 |
| `education_rate` | synthetic; real gap as above | |
| `urbanization` | synthetic; TW uses density proxy; US gap | |
| `unemployment` | synthetic; real gap | |
| `feature_set` | str | `shared_history`, `shared_history_size`, `country_extended` |

## Split protocol (documented, not a column requirement)

1. **Country-only OOT:** expanding window within country; test whole election years never seen in train.
2. **Joint pooled:** same year windows where both countries have rows; else train on pooled history years available.
3. **Transfer:** train on country A all years ≤ T, test country B years > T using **shared feature subset only**.

## Files using this schema

- `cleaned/synthetic_us_tw_panel.csv`
- `cleaned/us_county_presidential_panel.csv`
- `cleaned/tw_county_presidential_panel_joint.csv`
- `cleaned/joint_real_shared_panel.csv`
