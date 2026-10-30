#!/usr/bin/env python3
"""Public-substitute analyses (Extension only) — do NOT modify Family A MAE.

1) Fill US 2004 national poll proxy from 538 GitHub pres_pollaverages_1968-2016.csv
2) Poll-as-forecast accuracy table (national two-party)
3) TW county tertiary education_rate from MOI micst → A+ style OOT with lag_education_rate
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/workspace/taiwan-election-data")
GH = Path("/workspace/taiwan-election-github")
TZ = ZoneInfo("Asia/Hong_Kong")
NOW = datetime.now(TZ).strftime("%Y-%m-%d %H:%M Asia/Hong_Kong (UTC+8)")
SEED = 42
TARGET = "target_DPP_vote_share"

ELECTORAL = [
    "prev_DPP_vote_share", "prev_KMT_vote_share", "prev_Other_vote_share",
    "prev2_DPP_vote_share", "prev2_KMT_vote_share", "prev2_Other_vote_share",
    "prev_turnout", "DPP_change_1", "KMT_change_1", "Other_change_1",
    "prev_DPP_KMT_margin", "prev_DPP_Other_margin", "DPP_KMT_margin_change_1",
    "previous_party_change_abs_mean", "previous_party_change_std", "turnout_change_1",
]
SOCIO = [
    "lag_unemployment_pct", "lag_log_mean_income", "lag_log_pop_density",
    "lag_taiwanese_id_pct", "lag_chinese_id_pct", "lag_both_id_pct",
    "lag_indep_lean_pct", "lag_status_quo_forever_pct", "lag_unif_lean_pct",
]
EDU_COL = "lag_education_rate"
A_PLUS = ELECTORAL + SOCIO
A_PLUS_EDU = A_PLUS + [EDU_COL]


def write_df(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {path} rows={len(df)}")


def mirror_file(src: Path, rel: str):
    if not GH.exists():
        return
    dst = GH / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    print(f"mirrored {dst}")


def scores(y, p):
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
    }


def make_model(name: str):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, random_state=SEED, n_jobs=1
        )
    return HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.06, max_leaf_nodes=7,
        min_samples_leaf=5, l2_regularization=1.0, random_state=SEED,
    )


def impute_train_median(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    tr, te = train.copy(), test.copy()
    for c in cols:
        if c not in tr.columns:
            continue
        med = tr[c].median()
        if pd.isna(med):
            med = 0.0
        tr[c] = tr[c].fillna(med)
        te[c] = te[c].fillna(med)
    return tr, te


# ---------------------------------------------------------------------------
# 1) Fill 2004 national poll from 538 historic averages
# ---------------------------------------------------------------------------
def fill_2004_poll():
    hist_path = ROOT / "raw/us_polls/pres_pollaverages_1968-2016.csv"
    hist = pd.read_csv(hist_path)
    nat = hist[(hist["cycle"] == 2004) & (hist["state"].astype(str).str.lower() == "national")].copy()
    if nat.empty:
        raise SystemExit("No 2004 National rows in 538 historic averages")

    nat["modeldate_dt"] = pd.to_datetime(nat["modeldate"])
    last_date = nat["modeldate_dt"].max()
    last = nat[nat["modeldate_dt"] == last_date].copy()

    def side(name: str) -> str:
        n = str(name).lower()
        if "kerry" in n or "gore" in n or "obama" in n or "clinton" in n or "harris" in n or "biden" in n:
            return "dem"
        if "bush" in n or "mccain" in n or "romney" in n or "trump" in n:
            return "gop"
        return "other"

    last["side"] = last["candidate_name"].map(side)
    dem_v = float(last.loc[last["side"] == "dem", "pct_estimate"].iloc[0])
    gop_v = float(last.loc[last["side"] == "gop", "pct_estimate"].iloc[0])
    dem_2p = dem_v / (dem_v + gop_v) if (dem_v + gop_v) > 0 else np.nan
    as_of = last_date.strftime("%Y-%m-%d")

    proxy_path = ROOT / "data/us_parity_extension/us_national_poll_election_year_proxy.csv"
    proxy = pd.read_csv(proxy_path)
    row = {
        "election_year": 2004,
        "as_of_date": as_of,
        "dem_pct": dem_v,
        "gop_pct": gop_v,
        "other_pct": np.nan,
        "dem_two_party_share_poll": dem_2p,
        "source": "fivethirtyeight_github_pres_pollaverages_1968-2016",
        "source_url": "https://raw.githubusercontent.com/fivethirtyeight/data/master/polls/pres_pollaverages_1968-2016.csv",
        "geography": "national",
        "note": (
            f"Last National modeldate in 538 1968-2016 archive for 2004 (= {as_of}); "
            "last National modeldate in file for this cycle; Extension only — verify as_of vs election day"
        ),
    }
    # replace 2004 row
    proxy = proxy[proxy["election_year"] != 2004]
    proxy = pd.concat([proxy, pd.DataFrame([row])], ignore_index=True)
    proxy = proxy.sort_values("election_year").reset_index(drop=True)
    write_df(proxy_path, proxy)
    mirror_file(proxy_path, "data/us_parity_extension/us_national_poll_election_year_proxy.csv")

    # also extract National daily series for 2000-2016 from historic file into extension
    daily_rows = []
    for cycle in [2000, 2004, 2008, 2012, 2016]:
        sub = hist[(hist["cycle"] == cycle) & (hist["state"].astype(str).str.lower() == "national")].copy()
        if sub.empty:
            continue
        sub["modeldate_dt"] = pd.to_datetime(sub["modeldate"])
        sub["side"] = sub["candidate_name"].map(side)
        for _, r in sub.iterrows():
            if r["side"] not in ("dem", "gop"):
                continue
            daily_rows.append({
                "election_year": cycle,
                "field_end": r["modeldate_dt"].strftime("%Y-%m-%d"),
                "target": "presidential",
                "side": r["side"],
                "party": "DEM" if r["side"] == "dem" else "GOP",
                "value": float(r["pct_estimate"]),
                "pct_trend_adjusted": float(r["pct_trend_adjusted"]) if pd.notna(r["pct_trend_adjusted"]) else np.nan,
                "pct_estimate": float(r["pct_estimate"]),
                "state": "National",
                "pollster": "538_model_average",
                "region": "national",
                "question_type": "candidate_pct",
                "source_url": "https://raw.githubusercontent.com/fivethirtyeight/data/master/polls/pres_pollaverages_1968-2016.csv",
                "candidate_name": r["candidate_name"],
            })
    daily_hist = pd.DataFrame(daily_rows)
    daily_path = ROOT / "data/us_parity_extension/us_polls_national_daily_538_legacy_1968_2016_national.csv"
    write_df(daily_path, daily_hist)
    mirror_file(daily_path, "data/us_parity_extension/us_polls_national_daily_538_legacy_1968_2016_national.csv")

    # update PROVENANCE
    prov_path = ROOT / "data/us_parity_extension/PROVENANCE.json"
    prov = json.loads(prov_path.read_text())
    src = "https://raw.githubusercontent.com/fivethirtyeight/data/master/polls/pres_pollaverages_1968-2016.csv"
    if src not in prov.get("sources", []):
        prov.setdefault("sources", []).append(src)
    prov["updated_at"] = NOW
    prov["notes_2004_fill"] = (
        "2004 National filled from last modeldate in pres_pollaverages_1968-2016.csv "
        f"({as_of}); archive truncates ~Sep 9 each cycle."
    )
    # remove ANES from not_used if present as "login" — keep as not-in-model
    prov["family_a"] = "NOT modified; Extension only"
    prov_path.write_text(json.dumps(prov, indent=2, ensure_ascii=False) + "\n")
    mirror_file(prov_path, "data/us_parity_extension/PROVENANCE.json")

    return proxy, row


# ---------------------------------------------------------------------------
# 2) Poll accuracy Extension table
# ---------------------------------------------------------------------------
def poll_accuracy(proxy: pd.DataFrame):
    outcome = pd.read_csv(ROOT / "data/us_parity_extension/us_national_partisan_climate_proxy.csv")
    m = proxy.merge(
        outcome[["election_year", "nat_dem_two_party_share"]],
        on="election_year",
        how="left",
    )
    m = m.rename(columns={"nat_dem_two_party_share": "dem_two_party_share_actual"})
    m["error_poll_minus_actual"] = m["dem_two_party_share_poll"] - m["dem_two_party_share_actual"]
    m["abs_error"] = m["error_poll_minus_actual"].abs()
    # also naive previous-election two-party as baseline (lag outcome)
    act = outcome.set_index("election_year")["nat_dem_two_party_share"]
    m["prev_election_two_party"] = m["election_year"].map(lambda y: act.get(y - 4, np.nan))
    m["abs_error_prev"] = (m["prev_election_two_party"] - m["dem_two_party_share_actual"]).abs()

    out_cols = [
        "election_year", "as_of_date", "dem_pct", "gop_pct",
        "dem_two_party_share_poll", "dem_two_party_share_actual",
        "error_poll_minus_actual", "abs_error",
        "prev_election_two_party", "abs_error_prev",
        "source", "note",
    ]
    detail = m[out_cols].sort_values("election_year")
    valid = detail.dropna(subset=["dem_two_party_share_poll", "dem_two_party_share_actual"])
    summary = pd.DataFrame([
        {
            "metric": "MAE_poll_as_forecast_two_party",
            "value": float(valid["abs_error"].mean()),
            "n_years": int(len(valid)),
            "years": ",".join(str(int(y)) for y in valid["election_year"]),
            "note": "Extension; national only; 2004/legacy as_of early-Sep in 538 archive; 2024 as_of 2024-09-12",
        },
        {
            "metric": "MAE_previous_election_two_party",
            "value": float(valid["abs_error_prev"].mean()),
            "n_years": int(valid["abs_error_prev"].notna().sum()),
            "years": ",".join(str(int(y)) for y in valid.loc[valid["abs_error_prev"].notna(), "election_year"]),
            "note": "Same years as poll MAE where prev exists",
        },
        {
            "metric": "mean_signed_error_poll_minus_actual",
            "value": float(valid["error_poll_minus_actual"].mean()),
            "n_years": int(len(valid)),
            "years": ",".join(str(int(y)) for y in valid["election_year"]),
            "note": "Positive = polls overstated Dem two-party share",
        },
    ])

    out_dir = ROOT / "results/extension_poll_platform"
    write_df(out_dir / "us_national_poll_vs_outcome.csv", detail)
    write_df(out_dir / "us_national_poll_accuracy_summary.csv", summary)
    readme = out_dir / "US_POLL_ACCURACY_README.md"
    readme.write_text(
        f"""# US national poll-as-forecast accuracy (Extension)

**Built:** {NOW}  
**Scope:** National Dem two-party poll share vs MEDSL-aggregated national Dem two-party outcome.  
**NOT Family A.** County polls not used.

## Sources
- Poll proxy: `data/us_parity_extension/us_national_poll_election_year_proxy.csv`
- 2004 fill: 538 GitHub `pres_pollaverages_1968-2016.csv` (last National modeldate in archive)
- Outcomes: `data/us_parity_extension/us_national_partisan_climate_proxy.csv` (MEDSL)

## Key numbers
- MAE (poll as forecast): **{summary.loc[0, 'value']:.4f}** over {int(summary.loc[0, 'n_years'])} years ({summary.loc[0, 'years']})
- MAE (previous election share): **{summary.loc[1, 'value']:.4f}**
- Mean signed error (poll − actual): **{summary.loc[2, 'value']:.4f}**

## Caveats
- 538 1968–2016 National archive truncates around **early September** each cycle (2004 as_of = last modeldate therein).
- 2024 averages file in repo ends **2024-09-12** (pre-election).
- This is national Extension diagnostics only — does not enter Family A.
""",
        encoding="utf-8",
    )
    print(f"wrote {readme}")
    for name in [
        "us_national_poll_vs_outcome.csv",
        "us_national_poll_accuracy_summary.csv",
        "US_POLL_ACCURACY_README.md",
    ]:
        mirror_file(out_dir / name, f"results/extension_poll_platform/{name}")
    return detail, summary


# ---------------------------------------------------------------------------
# 3) TW education_rate from MOI + A+ education Extension OOT
# ---------------------------------------------------------------------------
def build_education_panel():
    raw_path = ROOT / "raw/moi/education_rate/moi_c0110405.csv"
    text = raw_path.read_text(encoding="utf-8-sig")
    rows = []
    for line in text.strip().splitlines()[1:]:
        # "86年/ 新北市/ 性別總計",2691685,...,15.99
        m = re.match(
            r'"?(\d+)年/\s*([^/]+)/\s*([^"]+)"?\s*,([^,]*),([^,]*),([^,]*),([^,]*),([^,]*)\s*$',
            line.strip(),
        )
        if not m:
            continue
        roc = int(m.group(1))
        county = m.group(2).strip()
        sex = m.group(3).strip()
        if sex != "性別總計" or county == "區域別總計":
            continue
        def _f(x):
            x = (x or "").strip()
            if not x or x in {"-", "—", "…", "NA", "n/a"}:
                return np.nan
            try:
                return float(x)
            except ValueError:
                return np.nan
        rate = _f(m.group(8))
        total_tert = _f(m.group(4))
        rows.append({
            "roc_year": roc,
            "ce_year": roc + 1911,
            "region_name": county,
            "education_rate": rate,  # 15+ 大專以上比率 (%)
            "tertiary_pop": total_tert,
            "source": "MOI_micst_c0110405_data_gov_tw_14219",
            "source_url": "https://data.gov.tw/dataset/14219",
            "raw_file": "raw/moi/education_rate/moi_c0110405.csv",
        })
    edu = pd.DataFrame(rows).sort_values(["ce_year", "region_name"]).reset_index(drop=True)
    out = ROOT / "data/extension_education/tw_county_education_rate_moi.csv"
    write_df(out, edu)
    mirror_file(out, "data/extension_education/tw_county_education_rate_moi.csv")

    # provenance
    prov = {
        "built_at": NOW,
        "dataset": "data.gov.tw 14219 / MOI micst funid=c0110405",
        "definition": "15歲以上大專以上比率(%) by county, sex=總計",
        "download": (
            "https://statis.moi.gov.tw/micst/webMain.aspx?sys=220&kind=21&type=1"
            "&funid=c0110405&cycle=4&outmode=12&utf=1&compmode=0&outkind=3"
            "&fldspc=0,5,&codspc0=0,2,3,2,6,1,9,1,12,1,15,16,&codlst1=111"
            "&ym=8600&ymt=11300"
        ),
        "opdadm_resource_note": (
            "opdadm download URL returned small binary (not used as primary); "
            "micst CSV used instead"
        ),
        "family_a": "NOT modified; Extension / A+ education PROXY only",
        "alignment": "Intended parallel to US ERS education_rate (population attainment), NOT education-bureau politics",
    }
    prov_path = ROOT / "data/extension_education/PROVENANCE.json"
    prov_path.write_text(json.dumps(prov, indent=2, ensure_ascii=False) + "\n")
    mirror_file(prov_path, "data/extension_education/PROVENANCE.json")
    return edu


def attach_lag_education(panel: pd.DataFrame, edu: pd.DataFrame) -> pd.DataFrame:
    """Nearest ce_year < election year per county → lag_education_rate (percent)."""
    out = panel.copy()
    lags = []
    for _, r in out.iterrows():
        sub = edu[(edu["region_name"] == r["region_name"]) & (edu["ce_year"] < r["year"])]
        if sub.empty:
            lags.append(np.nan)
        else:
            lags.append(float(sub.sort_values("ce_year").iloc[-1]["education_rate"]))
    out[EDU_COL] = lags
    return out


def run_oot(panel: pd.DataFrame, feature_sets: dict[str, list[str]], test_years=(2012, 2016, 2020, 2024)):
    rows = []
    ridge_coefs = []
    for suite, feats in feature_sets.items():
        use_cols = [c for c in feats if c in panel.columns]
        for ty in test_years:
            train = panel[panel["year"] < ty].copy()
            test = panel[panel["year"] == ty].copy()
            # need prev for baseline
            train = train.dropna(subset=[TARGET, "prev_DPP_vote_share"])
            test = test.dropna(subset=[TARGET, "prev_DPP_vote_share"])
            if train.empty or test.empty:
                continue
            train, test = impute_train_median(train, test, use_cols)
            # Previous share baseline
            prev_pred = test["prev_DPP_vote_share"].values
            sc = scores(test[TARGET].values, prev_pred)
            rows.append({
                "suite": suite, "test_year": ty, "n_train": len(train), "n_test": len(test),
                "model": "Previous-election baseline", **sc,
            })
            for model_name in ["Ridge", "Random Forest", "HistGradientBoosting"]:
                Xtr = train[use_cols].astype(float)
                ytr = train[TARGET].astype(float)
                Xte = test[use_cols].astype(float)
                yte = test[TARGET].astype(float)
                model = make_model(model_name)
                model.fit(Xtr, ytr)
                pred = model.predict(Xte)
                sc = scores(yte, pred)
                rows.append({
                    "suite": suite, "test_year": ty, "n_train": len(train), "n_test": len(test),
                    "model": model_name, **sc,
                })
                if model_name == "Ridge":
                    # signed coefs after scaler
                    pipe = model
                    coefs = pipe.named_steps["ridge"].coef_
                    for f, c in zip(use_cols, coefs):
                        ridge_coefs.append({
                            "suite": suite, "test_year": ty, "feature": f, "ridge_coef_std": float(c),
                        })
    metrics = pd.DataFrame(rows)
    mean = (
        metrics.groupby(["suite", "model"], as_index=False)[["mae", "rmse"]]
        .mean()
        .sort_values(["suite", "mae"])
    )
    return metrics, mean, pd.DataFrame(ridge_coefs)


def education_a_plus_analysis(edu: pd.DataFrame):
    panel = pd.read_csv(ROOT / "data/county_panel_features_enriched_2000_2024.csv")
    panel = attach_lag_education(panel, edu)
    # coverage
    cov = (
        panel.groupby("year")[EDU_COL]
        .apply(lambda s: int(s.notna().sum()))
        .reset_index(name="n_non_null")
    )
    cov["n_total"] = panel.groupby("year").size().values

    out_data = ROOT / "data/extension_education/county_panel_with_lag_education_rate.csv"
    # save slim panel cols + edu
    keep = [
        "region_id", "region_name", "year", TARGET, "prev_DPP_vote_share", EDU_COL,
        "lag_unemployment_pct", "lag_log_mean_income", "lag_log_pop_density",
    ]
    write_df(out_data, panel[[c for c in keep if c in panel.columns]])
    mirror_file(out_data, "data/extension_education/county_panel_with_lag_education_rate.csv")
    write_df(ROOT / "data/extension_education/lag_education_coverage_by_year.csv", cov)
    mirror_file(
        ROOT / "data/extension_education/lag_education_coverage_by_year.csv",
        "data/extension_education/lag_education_coverage_by_year.csv",
    )

    feature_sets = {
        "family_a_electoral_only_rerun": ELECTORAL,
        "family_a_plus_enriched_rerun": A_PLUS,
        "extension_a_plus_education": A_PLUS_EDU,
    }
    # only use rows that have electoral features
    need = [TARGET] + ELECTORAL
    panel_m = panel.dropna(subset=[c for c in need if c in panel.columns]).copy()

    metrics, mean, coefs = run_oot(panel_m, feature_sets)
    out_dir = ROOT / "results/extension_education_a_plus"
    write_df(out_dir / "backtest_metrics_by_year.csv", metrics)
    write_df(out_dir / "mean_mae_summary.csv", mean)
    write_df(out_dir / "ridge_coefs_by_year.csv", coefs)

    # mean abs coef for education feature
    edu_coef = coefs[coefs["feature"] == EDU_COL].copy()
    write_df(out_dir / "ridge_education_coef_by_year.csv", edu_coef)

    # comparison table vs locked Family A / A+ published means (reference only)
    locked = pd.DataFrame([
        {"suite": "Family_A_LOCKED_published", "model": "Previous-election baseline", "mae": 0.0792648157080057, "note": "Do not overwrite"},
        {"suite": "Family_A_LOCKED_published", "model": "Ridge", "mae": 0.17857985899144863, "note": "Do not overwrite"},
        {"suite": "Family_A_PLUS_LOCKED_published", "model": "Previous-election baseline", "mae": 0.0792648157080057, "note": "Do not overwrite"},
        {"suite": "Family_A_PLUS_LOCKED_published", "model": "Ridge", "mae": 0.14250033498938372, "note": "Do not overwrite"},
    ])
    write_df(out_dir / "reference_locked_family_a_mae.csv", locked)

    readme = out_dir / "README.md"
    edu_mean = mean[mean["suite"] == "extension_a_plus_education"]
    aplus_mean = mean[mean["suite"] == "family_a_plus_enriched_rerun"]
    lines = [
        f"# TW education_rate Extension A+ (NOT Family A)",
        "",
        f"**Built:** {NOW}",
        "",
        "## Definition",
        "- `lag_education_rate` = MOI 15歲以上大專以上比率(%)，取最近一個 **ce_year < 選舉年** 的縣市值。",
        "- 來源：data.gov.tw dataset 14219 / MOI micst `c0110405`（`raw/moi/education_rate/moi_c0110405.csv`）。",
        "- 對齊美國 ERS `education_rate`（人口學歷），**唔係**教育長政治傾向。",
        "",
        "## Suites (OOT leave-one-election 2012/2016/2020/2024)",
        "1. `family_a_electoral_only_rerun` — 對照重跑（唔取代鎖定主表）",
        "2. `family_a_plus_enriched_rerun` — 既有 A+ 特徵重跑",
        "3. `extension_a_plus_education` — A+ + `lag_education_rate`",
        "",
        "## Mean OOT MAE (this run)",
        "```",
        mean.to_string(index=False),
        "```",
        "",
        "## Locked published reference (unchanged)",
        "- Family A Prev MAE **0.0793**; Family A Ridge **0.1786**",
        "- Family A+ Ridge **0.1425** (published enriched)",
        "",
        "## Caveats",
        "- 權重／係數**唔係**因果。",
        "- 教育率滯後對早期選舉覆蓋較佳（MOI 自民國86年起）；金門／連江亦有列。",
        "- **禁止**把本 Extension MAE 寫入 Family A 主表。",
        "",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {readme}")
    for name in [
        "backtest_metrics_by_year.csv",
        "mean_mae_summary.csv",
        "ridge_coefs_by_year.csv",
        "ridge_education_coef_by_year.csv",
        "reference_locked_family_a_mae.csv",
        "README.md",
    ]:
        mirror_file(out_dir / name, f"results/extension_education_a_plus/{name}")
    return mean, metrics, edu_coef, cov


def main():
    print("=== 1) Fill 2004 poll ===")
    proxy, row2004 = fill_2004_poll()
    print("2004 row:", row2004)

    print("=== 2) Poll accuracy ===")
    detail, summary = poll_accuracy(proxy)
    print(summary.to_string(index=False))

    print("=== 3) Education panel + A+ ===")
    edu = build_education_panel()
    print(f"education rows={len(edu)} years={edu.ce_year.min()}-{edu.ce_year.max()} counties={edu.region_name.nunique()}")
    mean, metrics, edu_coef, cov = education_a_plus_analysis(edu)
    print("coverage:\n", cov.to_string(index=False))
    print("mean MAE:\n", mean.to_string(index=False))
    print("edu ridge coefs:\n", edu_coef.to_string(index=False))

    # summary json for parent
    out = {
        "built_at": NOW,
        "poll_2004": {
            "as_of_date": row2004["as_of_date"],
            "dem_pct": row2004["dem_pct"],
            "gop_pct": row2004["gop_pct"],
            "dem_two_party_share_poll": row2004["dem_two_party_share_poll"],
        },
        "poll_mae": float(summary.loc[0, "value"]),
        "poll_prev_mae": float(summary.loc[1, "value"]),
        "education_a_plus_mean": mean[mean.suite == "extension_a_plus_education"][
            ["model", "mae"]
        ].to_dict(orient="records"),
        "family_a_locked_prev_mae": 0.0792648157080057,
    }
    path = ROOT / "results/extension_education_a_plus/run_summary.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    mirror_file(path, "results/extension_education_a_plus/run_summary.json")
    print("DONE", json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
