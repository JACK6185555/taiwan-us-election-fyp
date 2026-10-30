#!/usr/bin/env python3
"""Taiwan Core specification improve + data fill + feature weights + RQ4 conformal attempt.

- Adds lagged public socio/attitude features (unemployment, income, density, ESC national PROXY).
- Refits electoral-only Core specification and enriched Core+ (enriched).
- Exports Ridge signed coefs, RF/HGB impurity+permutation importance, ablation ΔMAE.
- Attempts improved conformal (train / calib year / test year).
- Does NOT invent TEDS microdata; Kinmen/Lienchiang unemployment missing → train-fold median.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_TREE = Path("/workspace/taiwan-election-data")
OUT_FW = ROOT / "results" / "feature_weights"
OUT_FA = ROOT / "results" / "family_a_county_presidential"
OUT_ENR = ROOT / "results" / "family_a_plus_enriched"
OUT_CONF = ROOT / "results" / "conformal_rq4_improved"
SEED = 42
TARGET = "target_DPP_vote_share"
ANCHOR = "prev_DPP_vote_share"

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
ENRICHED = ELECTORAL + SOCIO
LABELS = {
    "prev_DPP_vote_share": "Previous DPP share",
    "prev_KMT_vote_share": "Previous KMT share",
    "prev_Other_vote_share": "Previous other share",
    "prev2_DPP_vote_share": "DPP share two elections ago",
    "prev2_KMT_vote_share": "KMT share two elections ago",
    "prev2_Other_vote_share": "Other share two elections ago",
    "prev_turnout": "Previous turnout",
    "DPP_change_1": "DPP previous swing",
    "KMT_change_1": "KMT previous swing",
    "Other_change_1": "Other previous swing",
    "prev_DPP_KMT_margin": "Previous DPP-KMT margin",
    "prev_DPP_Other_margin": "Previous DPP-other margin",
    "DPP_KMT_margin_change_1": "DPP-KMT margin change",
    "previous_party_change_abs_mean": "Mean party swing magnitude",
    "previous_party_change_std": "Dispersion of party swings",
    "turnout_change_1": "Turnout change",
    "lag_unemployment_pct": "Lag county unemployment %",
    "lag_log_mean_income": "Lag log mean income",
    "lag_log_pop_density": "Lag log pop density",
    "lag_taiwanese_id_pct": "Lag national Taiwanese ID % (ESC PROXY)",
    "lag_chinese_id_pct": "Lag national Chinese ID % (ESC PROXY)",
    "lag_both_id_pct": "Lag national both ID % (ESC PROXY)",
    "lag_indep_lean_pct": "Lag national indep-lean % (ESC PROXY)",
    "lag_status_quo_forever_pct": "Lag national SQ-forever % (ESC PROXY)",
    "lag_unif_lean_pct": "Lag national unif-lean % (ESC PROXY)",
}
GROUPS_ELEC = {
    "Previous election": ["prev_DPP_vote_share", "prev_KMT_vote_share", "prev_Other_vote_share", "prev_turnout"],
    "Earlier election": ["prev2_DPP_vote_share", "prev2_KMT_vote_share", "prev2_Other_vote_share"],
    "Margins and swings": [
        "DPP_change_1", "KMT_change_1", "Other_change_1", "prev_DPP_KMT_margin",
        "prev_DPP_Other_margin", "DPP_KMT_margin_change_1",
        "previous_party_change_abs_mean", "previous_party_change_std",
    ],
    "Turnout change": ["turnout_change_1"],
}
GROUPS_ENR = {
    **GROUPS_ELEC,
    "County socio (lagged)": ["lag_unemployment_pct", "lag_log_mean_income", "lag_log_pop_density"],
    "National ESC attitude PROXY (lagged)": [
        "lag_taiwanese_id_pct", "lag_chinese_id_pct", "lag_both_id_pct",
        "lag_indep_lean_pct", "lag_status_quo_forever_pct", "lag_unif_lean_pct",
    ],
}
MODELS = ["Previous-election baseline", "Ridge", "Random Forest", "HistGradientBoosting"]


def scores(actual, pred):
    return {
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
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
    """Documented train-fold median fill for missing socio (e.g. Kinmen unemployment)."""
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


def nearest_prior(df: pd.DataFrame, county_col: str, year_col: str, value_cols: list[str], target_year: int):
    """For each county, take the latest observation with year < target_year."""
    sub = df[df[year_col] < target_year].copy()
    if sub.empty:
        return pd.DataFrame(columns=[county_col] + value_cols)
    sub = sub.sort_values(year_col).groupby(county_col, as_index=False).tail(1)
    return sub[[county_col] + value_cols]


def build_enriched_panel() -> tuple[pd.DataFrame, dict]:
    panel = pd.read_csv(ROOT / "data" / "county_panel_features_2000_2024.csv")
    unemp = pd.read_csv(ROOT / "data" / "economy" / "county_unemployment_annual_2010_2024.csv")
    # drop legacy county rows
    unemp = unemp[~unemp.county.str.contains("舊", na=False)].copy()

    income_path = DATA_TREE / "cleaned" / "income_county_panel.csv"
    if not income_path.exists():
        income_path = ROOT / "data" / "economy" / "income_county_panel.csv"
    income = pd.read_csv(income_path)

    pop_path = DATA_TREE / "cleaned" / "moi_pop_density_county_panel.csv"
    if not pop_path.exists():
        pop_path = ROOT / "data" / "moi_pop_density_county_panel.csv"
    pop = pd.read_csv(pop_path)

    # copy income/pop into github data/economy for pack completeness
    eco = ROOT / "data" / "economy"
    eco.mkdir(parents=True, exist_ok=True)
    shutil.copy2(income_path, eco / "income_county_panel.csv")
    shutil.copy2(pop_path, eco / "moi_pop_density_county_panel.csv")

    ident = pd.read_csv(ROOT / "data" / "esc_proxies" / "esc_identity_annual.csv")
    tongdu = pd.read_csv(ROOT / "data" / "esc_proxies" / "esc_tongdu_annual.csv")
    tongdu["indep_lean_pct"] = tongdu["indep_asap_pct"].fillna(0) + tongdu["status_quo_to_indep_pct"].fillna(0)
    tongdu["unif_lean_pct"] = tongdu["unif_asap_pct"].fillna(0) + tongdu["status_quo_to_unif_pct"].fillna(0)

    rows = []
    for _, r in panel.iterrows():
        y = int(r.year)
        county = r.region_name
        lag_y = y - 1
        # unemployment exact lag year
        u = unemp[(unemp.county == county) & (unemp.year == lag_y)]
        lag_u = float(u.unemployment_rate_pct.iloc[0]) if len(u) else np.nan

        # income / density: nearest prior CE year
        inc_n = nearest_prior(income, "county_name", "ce_year", ["mean_income"], y)
        inc_row = inc_n[inc_n.county_name == county]
        lag_inc = float(inc_row.mean_income.iloc[0]) if len(inc_row) else np.nan

        pop_n = nearest_prior(pop, "county_name", "ce_year", ["population_density"], y)
        pop_row = pop_n[pop_n.county_name == county]
        lag_dens = float(pop_row.population_density.iloc[0]) if len(pop_row) else np.nan

        # national ESC: prefer lag_y, else nearest prior
        id_sub = ident[ident.year <= lag_y].sort_values("year")
        td_sub = tongdu[tongdu.year <= lag_y].sort_values("year")
        if len(id_sub):
            idr = id_sub.iloc[-1]
            tw_id, ch_id, both_id = float(idr.taiwanese_pct), float(idr.chinese_pct), float(idr.both_pct)
        else:
            tw_id = ch_id = both_id = np.nan
        if len(td_sub):
            tdr = td_sub.iloc[-1]
            indep_l = float(tdr.indep_lean_pct)
            sqf = float(tdr.status_quo_forever_pct)
            unif_l = float(tdr.unif_lean_pct)
        else:
            indep_l = sqf = unif_l = np.nan

        out = r.to_dict()
        out.update({
            "lag_unemployment_pct": lag_u,
            "lag_log_mean_income": np.log(lag_inc) if pd.notna(lag_inc) and lag_inc > 0 else np.nan,
            "lag_log_pop_density": np.log(lag_dens) if pd.notna(lag_dens) and lag_dens > 0 else np.nan,
            "lag_taiwanese_id_pct": tw_id,
            "lag_chinese_id_pct": ch_id,
            "lag_both_id_pct": both_id,
            "lag_indep_lean_pct": indep_l,
            "lag_status_quo_forever_pct": sqf,
            "lag_unif_lean_pct": unif_l,
            "socio_lag_year": lag_y,
        })
        rows.append(out)

    enriched = pd.DataFrame(rows)
    out_csv = ROOT / "data" / "county_panel_features_enriched_2000_2024.csv"
    enriched.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # coverage stats
    cov = {
        c: {
            "n_non_null": int(enriched[c].notna().sum()),
            "n_total": int(len(enriched)),
            "by_year_non_null": {
                str(int(y)): int(enriched.loc[enriched.year == y, c].notna().sum())
                for y in sorted(enriched.year.unique())
            },
        }
        for c in SOCIO
    }
    provenance = {
        "panel_base": "data/county_panel_features_2000_2024.csv (CEC presidential county)",
        "unemployment": {
            "file": "data/economy/county_unemployment_annual_2010_2024.csv",
            "source": "data.gov.tw dataset 6640 DGBAS 人力資源調查縣市別失業率",
            "rule": "lag = election_year - 1; Kinmen/Lienchiang absent in source",
        },
        "income": {
            "file": "data/economy/income_county_panel.csv",
            "source": "MOI/Fiscal income tax unit township aggregate → county mean_income",
            "rule": "nearest ce_year < election_year; log(mean_income)",
            "available_ce_years": sorted(income.ce_year.unique().tolist()),
        },
        "pop_density": {
            "file": "data/economy/moi_pop_density_county_panel.csv",
            "source": "MOI township pop/area → county",
            "rule": "nearest ce_year < election_year; log(density)",
            "available_ce_years": sorted([int(x) for x in pop.ce_year.dropna().unique().tolist()]),
        },
        "esc_identity": {
            "file": "data/esc_proxies/esc_identity_annual.csv",
            "note": "PROXY transcribed from public ESC-citing tables; NOT official ESC Excel / NOT TEDS",
            "rule": "national series; use latest year <= election_year-1",
        },
        "esc_tongdu": {
            "file": "data/esc_proxies/esc_tongdu_annual.csv",
            "note": "PROXY; indep_lean = asap + SQ→indep; unif_lean = asap + SQ→unif",
            "rule": "national series; use latest year <= election_year-1",
        },
        "imputation": "Within each OOT fold, missing socio cells filled with TRAIN median (documented; not fabricated TEDS).",
        "coverage": cov,
        "written": str(out_csv.relative_to(ROOT)),
    }
    (ROOT / "data" / "DATA_FILL_PROVENANCE_enriched.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return enriched, provenance


def run_backtest(frame: pd.DataFrame, features: list[str], out_dir: Path, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    charts = out_dir / "charts"
    charts.mkdir(exist_ok=True)
    needed = [TARGET, *features]
    # keep rows that have electoral history; socio may be imputed later
    base_need = [TARGET, *ELECTORAL]
    frame = frame.dropna(subset=base_need).sort_values(["year", "region_id"]).copy()
    years = sorted(frame.year.unique())
    test_years = [y for y in years if y >= 2012]
    metric_rows, pred_rows = [], []
    for test_year in test_years:
        train = frame[frame.year < test_year]
        test = frame[frame.year == test_year]
        train, test = impute_train_median(train, test, features)
        actual = test[TARGET].to_numpy()
        for name in MODELS:
            if name == MODELS[0]:
                pred = test[ANCHOR].to_numpy()
            else:
                fitted = make_model(name).fit(train[features], train[TARGET])
                pred = fitted.predict(test[features])
            metric_rows.append({
                "suite": label, "test_year": int(test_year),
                "train_through": int(train.year.max()),
                "n_train": int(len(train)), "n_test": int(len(test)),
                "model": name, **scores(actual, pred),
            })
            for idx, (_, row) in enumerate(test.iterrows()):
                pred_rows.append({
                    "suite": label, "test_year": int(test_year),
                    "region_id": row.region_id, "region_name": row.region_name,
                    "model": name, "actual": float(actual[idx]),
                    "prediction": float(pred[idx]),
                    "error": float(pred[idx] - actual[idx]),
                })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "backtest_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pred_rows).to_csv(out_dir / "backtest_predictions.csv", index=False, encoding="utf-8-sig")
    mean_metrics = metrics.groupby("model", as_index=False)[["mae", "rmse"]].mean().sort_values("mae")
    mean_metrics.to_csv(out_dir / "family_a_mean_metrics.csv", index=False, encoding="utf-8-sig")
    return frame, test_years, metrics, mean_metrics


def run_ablation(frame, features, groups, test_years, out_dir: Path):
    print(f"  ablation -> {out_dir.name}", flush=True)
    sets = {"Previous-election only": [ANCHOR], "All features": features}
    for group, cols in groups.items():
        sets[f"All minus {group}"] = [c for c in features if c not in cols]
    rows = []
    for test_year in test_years:
        train, test = frame[frame.year < test_year], frame[frame.year == test_year]
        train, test = impute_train_median(train, test, features)
        y = test[TARGET].to_numpy()
        for set_name, cols in sets.items():
            for name in MODELS[1:]:
                pred = make_model(name).fit(train[cols], train[TARGET]).predict(test[cols])
                rows.append({
                    "test_year": int(test_year), "model": name, "feature_set": set_name,
                    "n_features": len(cols), **scores(y, pred),
                })
    ablation = pd.DataFrame(rows)
    ablation.to_csv(out_dir / "feature_ablation_by_year.csv", index=False, encoding="utf-8-sig")
    summary = ablation.groupby(["model", "feature_set", "n_features"], as_index=False)[["mae", "rmse"]].mean()
    # delta vs All features
    base = summary[summary.feature_set == "All features"][["model", "mae"]].rename(columns={"mae": "mae_all"})
    summary = summary.merge(base, on="model", how="left")
    summary["delta_mae_vs_all"] = summary["mae"] - summary["mae_all"]
    summary.to_csv(out_dir / "feature_ablation_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def export_weights(frame, features, test_years, out_prefix: str, suite_tag: str):
    """Export Ridge signed coefs (mean across OOT folds), RF/HGB impurity+perm importance."""
    OUT_FW.mkdir(parents=True, exist_ok=True)
    ridge_rows, rf_rows, hgb_rows, perm_rows = [], [], [], []

    for test_year in test_years:
        print(f"  weights fold {test_year} ({suite_tag})...", flush=True)
        train, test = frame[frame.year < test_year], frame[frame.year == test_year]
        train, test = impute_train_median(train, test, features)
        y_test = test[TARGET].to_numpy()

        # Ridge
        ridge = make_model("Ridge").fit(train[features], train[TARGET])
        coef = ridge[-1].coef_
        for feat, w in zip(features, coef):
            ridge_rows.append({
                "suite": suite_tag, "test_year": int(test_year), "feature": feat,
                "feature_label": LABELS.get(feat, feat),
                "standardized_coef_signed": float(w),
                "standardized_coef_abs": abs(float(w)),
                "aggregation_note": "per OOT fold; summary = mean across folds 2012-2024",
            })

        # RF
        rf = make_model("Random Forest").fit(train[features], train[TARGET])
        for feat, imp in zip(features, rf.feature_importances_):
            rf_rows.append({
                "suite": suite_tag, "test_year": int(test_year), "feature": feat,
                "feature_label": LABELS.get(feat, feat),
                "impurity_importance": float(imp),
            })
        pi_rf = permutation_importance(
            rf, test[features], y_test, scoring="neg_mean_absolute_error",
            n_repeats=15, random_state=SEED + int(test_year), n_jobs=1,
        )
        for j, feat in enumerate(features):
            perm_rows.append({
                "suite": suite_tag, "test_year": int(test_year), "model": "Random Forest",
                "feature": feat, "feature_label": LABELS.get(feat, feat),
                "permutation_mae_increase": float(pi_rf.importances_mean[j]),
                "permutation_std": float(pi_rf.importances_std[j]),
            })

        # HGB
        hgb = make_model("HistGradientBoosting").fit(train[features], train[TARGET])
        # HGB has no feature_importances_ in older sklearn; use permutation as primary
        # sklearn >=1.4 may have it via permutation only — check
        impurity = getattr(hgb, "feature_importances_", None)
        if impurity is None:
            # use train permutation as surrogate impurity-like; still export perm on test
            impurity = np.zeros(len(features))
        for feat, imp in zip(features, impurity):
            hgb_rows.append({
                "suite": suite_tag, "test_year": int(test_year), "feature": feat,
                "feature_label": LABELS.get(feat, feat),
                "impurity_importance": float(imp),
                "note": "zeros if estimator has no impurity importances; see permutation",
            })
        pi_hgb = permutation_importance(
            hgb, test[features], y_test, scoring="neg_mean_absolute_error",
            n_repeats=15, random_state=SEED + int(test_year), n_jobs=1,
        )
        for j, feat in enumerate(features):
            perm_rows.append({
                "suite": suite_tag, "test_year": int(test_year), "model": "HistGradientBoosting",
                "feature": feat, "feature_label": LABELS.get(feat, feat),
                "permutation_mae_increase": float(pi_hgb.importances_mean[j]),
                "permutation_std": float(pi_hgb.importances_std[j]),
            })
        # also Ridge perm for completeness
        pi_ridge = permutation_importance(
            ridge, test[features], y_test, scoring="neg_mean_absolute_error",
            n_repeats=15, random_state=SEED + int(test_year), n_jobs=1,
        )
        for j, feat in enumerate(features):
            perm_rows.append({
                "suite": suite_tag, "test_year": int(test_year), "model": "Ridge",
                "feature": feat, "feature_label": LABELS.get(feat, feat),
                "permutation_mae_increase": float(pi_ridge.importances_mean[j]),
                "permutation_std": float(pi_ridge.importances_std[j]),
            })

    ridge_df = pd.DataFrame(ridge_rows)
    rf_df = pd.DataFrame(rf_rows)
    hgb_df = pd.DataFrame(hgb_rows)
    perm_df = pd.DataFrame(perm_rows)

    ridge_mean = (
        ridge_df.groupby(["suite", "feature", "feature_label"], as_index=False)
        .agg(
            coef_mean_signed=("standardized_coef_signed", "mean"),
            coef_std_across_folds=("standardized_coef_signed", "std"),
            coef_mean_abs=("standardized_coef_abs", "mean"),
        )
        .sort_values("coef_mean_abs", ascending=False)
    )
    ridge_mean["rank_by_abs"] = range(1, len(ridge_mean) + 1)
    ridge_mean["aggregation"] = "mean across OOT folds 2012/2016/2020/2024 (not last-fold only)"

    rf_mean = (
        rf_df.groupby(["suite", "feature", "feature_label"], as_index=False)
        .agg(
            impurity_mean=("impurity_importance", "mean"),
            impurity_std=("impurity_importance", "std"),
        )
    )
    rf_perm = (
        perm_df[perm_df.model == "Random Forest"]
        .groupby(["suite", "feature", "feature_label"], as_index=False)
        .agg(
            permutation_mae_increase_mean=("permutation_mae_increase", "mean"),
            permutation_std_across_folds=("permutation_mae_increase", "std"),
        )
    )
    rf_out = rf_mean.merge(rf_perm, on=["suite", "feature", "feature_label"])
    rf_out = rf_out.sort_values("permutation_mae_increase_mean", ascending=False)
    rf_out["rank"] = range(1, len(rf_out) + 1)

    hgb_mean = (
        hgb_df.groupby(["suite", "feature", "feature_label"], as_index=False)
        .agg(
            impurity_mean=("impurity_importance", "mean"),
            impurity_std=("impurity_importance", "std"),
        )
    )
    hgb_perm = (
        perm_df[perm_df.model == "HistGradientBoosting"]
        .groupby(["suite", "feature", "feature_label"], as_index=False)
        .agg(
            permutation_mae_increase_mean=("permutation_mae_increase", "mean"),
            permutation_std_across_folds=("permutation_mae_increase", "std"),
        )
    )
    hgb_out = hgb_mean.merge(hgb_perm, on=["suite", "feature", "feature_label"])
    hgb_out = hgb_out.sort_values("permutation_mae_increase_mean", ascending=False)
    hgb_out["rank"] = range(1, len(hgb_out) + 1)

    # required filenames for Taiwan county (electoral suite uses required names)
    if out_prefix == "taiwan_county":
        ridge_mean.to_csv(OUT_FW / "taiwan_county_ridge_coefs.csv", index=False, encoding="utf-8-sig")
        rf_out.to_csv(OUT_FW / "taiwan_county_rf_importance.csv", index=False, encoding="utf-8-sig")
        hgb_out.to_csv(OUT_FW / "taiwan_county_hgb_importance.csv", index=False, encoding="utf-8-sig")
        ridge_df.to_csv(OUT_FW / "taiwan_county_ridge_coefs_by_year.csv", index=False, encoding="utf-8-sig")
        perm_df.to_csv(OUT_FW / "taiwan_county_permutation_by_year.csv", index=False, encoding="utf-8-sig")
    else:
        ridge_mean.to_csv(OUT_FW / f"{out_prefix}_ridge_coefs.csv", index=False, encoding="utf-8-sig")
        rf_out.to_csv(OUT_FW / f"{out_prefix}_rf_importance.csv", index=False, encoding="utf-8-sig")
        hgb_out.to_csv(OUT_FW / f"{out_prefix}_hgb_importance.csv", index=False, encoding="utf-8-sig")

    return ridge_mean, rf_out, hgb_out, perm_df


def run_conformal_improved(frame: pd.DataFrame, features: list[str]):
    """Improved RQ4: train years < calib_year < test_year; residual absolute conformal.

    Protocols tried:
      A) calib = previous election only (strict year split)
      B) expanding: all years except last pre-test as train; last pre-test as calib
      C) Jackknife+-style year leave-one-out residual quantile on train years (for last test)
    """
    OUT_CONF.mkdir(parents=True, exist_ok=True)
    frame = frame.dropna(subset=[TARGET, *ELECTORAL]).copy()
    alpha = 0.1
    target_cov = 1 - alpha
    protocols = []

    test_years = [2016, 2020, 2024]  # need at least one calib year before test after min train
    for test_year in test_years:
        pre = sorted(y for y in frame.year.unique() if y < test_year)
        if len(pre) < 2:
            continue
        calib_year = pre[-1]
        train_years = [y for y in pre if y < calib_year]
        if not train_years:
            continue
        for model_name in ["Ridge", "Random Forest", "HistGradientBoosting"]:
            train = frame[frame.year.isin(train_years)]
            calib = frame[frame.year == calib_year]
            test = frame[frame.year == test_year]
            train, calib = impute_train_median(train, calib, features)
            _, test = impute_train_median(train, test, features)
            fitted = make_model(model_name).fit(train[features], train[TARGET])
            calib_resid = np.abs(calib[TARGET].to_numpy() - fitted.predict(calib[features]))
            # finite-sample conformal quantile
            n = len(calib_resid)
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            q_level = min(q_level, 1.0)
            q = float(np.quantile(calib_resid, q_level, method="higher"))
            pred = fitted.predict(test[features])
            actual = test[TARGET].to_numpy()
            lo, hi = pred - q, pred + q
            cover = float(np.mean((actual >= lo) & (actual <= hi)))
            protocols.append({
                "protocol": "A_prev_election_calib",
                "model": model_name,
                "train_years": ",".join(map(str, train_years)),
                "calib_year": int(calib_year),
                "test_year": int(test_year),
                "alpha": alpha,
                "quantile": q,
                "coverage": cover,
                "target_coverage": target_cov,
                "avg_width": float(2 * q),
                "n_train": int(len(train)),
                "n_calib": int(len(calib)),
                "n_test": int(len(test)),
                "success": cover >= target_cov - 0.05,
            })
            # save 2024 intervals for RF as example
            if test_year == 2024 and model_name == "Random Forest":
                iv = test[["region_id", "region_name", TARGET]].copy()
                iv["prediction"] = pred
                iv["lo"] = lo
                iv["hi"] = hi
                iv["covered"] = (actual >= lo) & (actual <= hi)
                iv.to_csv(OUT_CONF / "intervals_2024_rf_protocol_A.csv", index=False, encoding="utf-8-sig")

    # Protocol B: demeaned residual (county-mean removed on train window) — sometimes helps shift
    for test_year in test_years:
        pre = sorted(y for y in frame.year.unique() if y < test_year)
        if len(pre) < 2:
            continue
        calib_year = pre[-1]
        train_years = [y for y in pre if y < calib_year]
        train = frame[frame.year.isin(train_years)].copy()
        calib = frame[frame.year == calib_year].copy()
        test = frame[frame.year == test_year].copy()
        # county means from train only
        cm = train.groupby("region_id")[TARGET].mean()
        global_m = float(train[TARGET].mean())
        for df in (train, calib, test):
            df["_mu"] = df.region_id.map(cm).fillna(global_m)
            df["_y_dm"] = df[TARGET] - df["_mu"]
        train, calib = impute_train_median(train, calib, features)
        _, test = impute_train_median(train, test, features)
        for model_name in ["Ridge", "Random Forest"]:
            fitted = make_model(model_name).fit(train[features], train["_y_dm"])
            calib_resid = np.abs(calib["_y_dm"].to_numpy() - fitted.predict(calib[features]))
            n = len(calib_resid)
            q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
            q = float(np.quantile(calib_resid, q_level, method="higher"))
            pred_dm = fitted.predict(test[features])
            pred = pred_dm + test["_mu"].to_numpy()
            actual = test[TARGET].to_numpy()
            lo, hi = pred - q, pred + q
            cover = float(np.mean((actual >= lo) & (actual <= hi)))
            protocols.append({
                "protocol": "B_demeaned_prev_calib",
                "model": model_name,
                "train_years": ",".join(map(str, train_years)),
                "calib_year": int(calib_year),
                "test_year": int(test_year),
                "alpha": alpha,
                "quantile": q,
                "coverage": cover,
                "target_coverage": target_cov,
                "avg_width": float(2 * q),
                "n_train": int(len(train)),
                "n_calib": int(len(calib)),
                "n_test": int(len(test)),
                "success": cover >= target_cov - 0.05,
            })

    # Protocol C: pooled absolute residual quantile from leave-one-year-out on all pre-test years
    for test_year in [2024]:
        pre = sorted(y for y in frame.year.unique() if y < test_year)
        if len(pre) < 3:
            continue
        for model_name in ["Ridge", "Random Forest"]:
            loo_resid = []
            for cy in pre:
                tr_y = [y for y in pre if y != cy]
                train = frame[frame.year.isin(tr_y)]
                calib = frame[frame.year == cy]
                train, calib = impute_train_median(train, calib, features)
                fitted = make_model(model_name).fit(train[features], train[TARGET])
                loo_resid.append(np.abs(calib[TARGET].to_numpy() - fitted.predict(calib[features])))
            resid = np.concatenate(loo_resid)
            n = len(resid)
            q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
            q = float(np.quantile(resid, q_level, method="higher"))
            train_all = frame[frame.year.isin(pre)]
            test = frame[frame.year == test_year]
            train_all, test = impute_train_median(train_all, test, features)
            fitted = make_model(model_name).fit(train_all[features], train_all[TARGET])
            pred = fitted.predict(test[features])
            actual = test[TARGET].to_numpy()
            lo, hi = pred - q, pred + q
            cover = float(np.mean((actual >= lo) & (actual <= hi)))
            protocols.append({
                "protocol": "C_year_LOO_pooled_calib",
                "model": model_name,
                "train_years": ",".join(map(str, pre)),
                "calib_year": -1,
                "test_year": int(test_year),
                "alpha": alpha,
                "quantile": q,
                "coverage": cover,
                "target_coverage": target_cov,
                "avg_width": float(2 * q),
                "n_train": int(len(train_all)),
                "n_calib": int(n),
                "n_test": int(len(test)),
                "success": cover >= target_cov - 0.05,
            })

    meta = pd.DataFrame(protocols)
    meta.to_csv(OUT_CONF / "conformal_protocols_summary.csv", index=False, encoding="utf-8-sig")
    best = meta.sort_values(["success", "coverage"], ascending=[False, False]).head(5)
    summary = {
        "target_coverage": target_cov,
        "n_protocols_tried": len(meta),
        "any_success_near_90pct": bool(meta.success.any()),
        "best_rows": best.to_dict("records"),
        "honest_note": (
            "If success is false, RQ4 remains uncalibrated for product use. "
            "Few election years + national swing make exchangeability fragile."
        ),
    }
    (OUT_CONF / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8"
    )
    return meta, summary


def write_summaries(ridge_elec, rf_elec, hgb_elec, ridge_enr, rf_enr, hgb_enr,
                    mean_elec, mean_enr, abl_elec, abl_enr, conf_meta, conf_sum, provenance):
    OUT_FW.mkdir(parents=True, exist_ok=True)

    # US placeholder / merge
    us_paths = [
        ROOT / "results" / "feature_weights" / "us_county_ridge_coefs.csv",
        ROOT / "results" / "family_a_us_county_presidential" / "feature_importance_summary.csv",
        DATA_TREE / "fyp_us_core" / "results" / "feature_importance_summary.csv",
    ]
    us_status = "pending_sibling"
    us_note = "US Core/RQ2 sibling still running or weights not yet on disk; placeholders below."
    us_top = []
    for p in us_paths:
        if p.exists():
            us_status = f"found:{p}"
            try:
                udf = pd.read_csv(p)
                us_top = udf.head(10).to_dict("records")
                us_note = f"Merged preview from {p}"
            except Exception as e:
                us_note = f"Found {p} but read failed: {e}"
            break

    compare_rows = []
    for _, r in ridge_elec.head(10).iterrows():
        compare_rows.append({
            "side": "TW_electoral", "model": "Ridge", "feature": r.feature,
            "feature_label": r.feature_label, "weight": r.coef_mean_signed,
            "weight_kind": "standardized_coef_signed_mean_OOT",
        })
    for _, r in rf_elec.head(10).iterrows():
        compare_rows.append({
            "side": "TW_electoral", "model": "Random Forest", "feature": r.feature,
            "feature_label": r.feature_label, "weight": r.permutation_mae_increase_mean,
            "weight_kind": "permutation_mae_increase_mean_OOT",
        })
    if us_top:
        for row in us_top[:10]:
            compare_rows.append({"side": "US", **{k: row.get(k) for k in row}})
    else:
        compare_rows.append({
            "side": "US", "model": None, "feature": None, "feature_label": None,
            "weight": None, "weight_kind": None, "status": us_status, "note": us_note,
        })
    pd.DataFrame(compare_rows).to_csv(OUT_FW / "tw_us_feature_weights_comparison.csv", index=False, encoding="utf-8-sig")

    # English summary md
    lines = [
        "# Feature weights summary (Taiwan county presidential)",
        "",
        "**Interpretation:** weights = model dependence in out-of-time folds — **not causal**, **not** reasons a party wins.",
        "",
        f"Aggregation: **mean across OOT folds** 2012, 2016, 2020, 2024 (unless noted).",
        "",
        "## Data added this round",
        "",
        "- Lagged county unemployment (DGBAS / data.gov.tw 6640), election_year−1",
        "- Lagged log mean income (MOI fiscal county panel; nearest prior CE year)",
        "- Lagged log population density (MOI; nearest prior CE year)",
        "- Lagged national ESC PROXY identity + tongdu (public transcription; not official Excel / not TEDS)",
        "- Kinmen/Lienchiang unemployment missing in source → train-fold median imputation",
        "",
        "## Core specification electoral-only mean OOT MAE",
        "",
        mean_elec.to_string(index=False),
        "",
        "## Core+ (enriched) enriched mean OOT MAE",
        "",
        mean_enr.to_string(index=False),
        "",
        "## Top Ridge standardized coefficients (electoral; signed mean)",
        "",
        ridge_elec.head(10)[["feature", "feature_label", "coef_mean_signed", "coef_mean_abs", "rank_by_abs"]].to_string(index=False),
        "",
        "## Top RF permutation importance (electoral)",
        "",
        rf_elec.head(10)[["feature", "feature_label", "permutation_mae_increase_mean", "impurity_mean", "rank"]].to_string(index=False),
        "",
        "## Top HGB permutation importance (electoral)",
        "",
        hgb_elec.head(10)[["feature", "feature_label", "permutation_mae_increase_mean", "rank"]].to_string(index=False),
        "",
        "## Top Ridge coefs (enriched A+)",
        "",
        ridge_enr.head(10)[["feature", "feature_label", "coef_mean_signed", "coef_mean_abs"]].to_string(index=False),
        "",
        "## RQ4 conformal",
        "",
        f"- Any success near 90%: **{conf_sum.get('any_success_near_90pct')}**",
        f"- Protocols tried: {conf_sum.get('n_protocols_tried')}",
        "",
        "Best rows:",
        "",
        "```",
        conf_meta.sort_values("coverage", ascending=False).head(8).to_string(index=False),
        "```",
        "",
        f"## US weights: {us_status}",
        "",
        us_note,
        "",
        "## Files",
        "",
        "- `taiwan_county_ridge_coefs.csv`",
        "- `taiwan_county_rf_importance.csv`",
        "- `taiwan_county_hgb_importance.csv`",
        "- `taiwan_county_plus_ridge_coefs.csv` (enriched)",
        "- `feature_weights_summary.md` (this file)",
        "- `docs/特徵權重說明.md`",
        "",
    ]
    (OUT_FW / "feature_weights_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Traditional Chinese teacher-facing doc
    top_ridge = ridge_elec.head(10)
    top_rf = rf_elec.head(10)
    zh = []
    zh.append("# 特徵權重說明（臺灣縣市總統票｜Core specification）")
    zh.append("")
    zh.append("更新：2026-10-30（Asia/Hong_Kong）")
    zh.append("")
    zh.append("## 必讀：權重唔係因果、唔係勝選原因")
    zh.append("")
    zh.append("本文件嘅「權重／重要性」只描述模型喺**整屆留出（out-of-time）**回測入面**點樣依賴特徵**。")
    zh.append("")
    zh.append("- **唔係**因果效應（唔代表改呢個變數就會改得票）")
    zh.append("- **唔係**選情斷言或勝選原因")
    zh.append("- Ridge 係**標準化後有號係數**（跨 fold 平均）；RF／HGB 係**置換重要性**（打亂特徵後 OOT MAE 升幾多）同／或 impurity")
    zh.append("- 細樣本（四個測試屆）下，排名會有波動")
    zh.append("")
    zh.append("## 本輪補上嘅公開資料（無捏造、無爬 TEDS）")
    zh.append("")
    zh.append("| 特徵 | 來源 | 規則 |")
    zh.append("|------|------|------|")
    zh.append("| `lag_unemployment_pct` | 主計總處縣市失業率（data.gov.tw 6640） | 選舉年前一年 |")
    zh.append("| `lag_log_mean_income` | 財政／MOI 所得縣市彙總 | 最近一個早於選舉年嘅 CE 年；取 log |")
    zh.append("| `lag_log_pop_density` | MOI 人口密度 | 同上 |")
    zh.append("| ESC 認同／統獨 PROXY | `data/esc_proxies/*` 公開轉錄 | 全國序列；≤選舉年−1；**唔係**官方 Excel／TEDS |")
    zh.append("")
    zh.append("金門／連江失業率來源缺列 → **訓練折內中位數填補**（已記錄），唔係虛構民調。")
    zh.append("")
    zh.append("## Core specification（純選舉史）平均 OOT MAE")
    zh.append("")
    zh.append("```")
    zh.append(mean_elec.to_string(index=False))
    zh.append("```")
    zh.append("")
    zh.append("## Core+ (enriched)（加滯後社經／態度 PROXY）平均 OOT MAE")
    zh.append("")
    zh.append("```")
    zh.append(mean_enr.to_string(index=False))
    zh.append("```")
    zh.append("")
    zh.append("若 A+ 未穩贏 A／基準，結論仍係：**基準 PreviousShare 好強**；新特徵屬探索。")
    zh.append("")
    zh.append("## Top 10：Ridge 標準化係數（有號；OOT fold 平均）")
    zh.append("")
    zh.append("| 排名 | 特徵 | 中文標籤 | 係數（有號） | |係數| |")
    zh.append("|------|------|----------|-------------|---------|")
    for i, r in enumerate(top_ridge.itertuples(), 1):
        zh.append(f"| {i} | `{r.feature}` | {r.feature_label} | {r.coef_mean_signed:.6f} | {r.coef_mean_abs:.6f} |")
    zh.append("")
    zh.append("## Top 10：Random Forest 置換重要性（OOT MAE 升幅）")
    zh.append("")
    zh.append("| 排名 | 特徵 | 標籤 | perm MAE↑ | impurity |")
    zh.append("|------|------|------|-----------|----------|")
    for i, r in enumerate(top_rf.itertuples(), 1):
        zh.append(
            f"| {i} | `{r.feature}` | {r.feature_label} | {r.permutation_mae_increase_mean:.6f} | {r.impurity_mean:.6f} |"
        )
    zh.append("")
    zh.append("## Top 10：HGB 置換重要性")
    zh.append("")
    zh.append("| 排名 | 特徵 | 標籤 | perm MAE↑ |")
    zh.append("|------|------|------|-----------|")
    for i, r in enumerate(hgb_elec.head(10).itertuples(), 1):
        zh.append(f"| {i} | `{r.feature}` | {r.feature_label} | {r.permutation_mae_increase_mean:.6f} |")
    zh.append("")
    zh.append("## 特徵組 ablation（ΔMAE vs 全特徵）")
    zh.append("")
    zh.append("正數 ΔMAE = 去掉該組後誤差變差（模型較依賴該組）。見 `results/family_a_county_presidential/feature_ablation_summary.csv` 同 A+ 目錄。")
    zh.append("")
    zh.append("## RQ4 預測區間（改良協定）")
    zh.append("")
    zh.append(f"- 目標覆蓋約 90%；本輪有否接近達標：**{conf_sum.get('any_success_near_90pct')}**")
    zh.append(f"- 試過協議數：{conf_sum.get('n_protocols_tried')}（前屆校準／去均值殘差／年 LOO 匯總）")
    zh.append("- 細節：`results/conformal_rq4_improved/`")
    if not conf_sum.get("any_success_near_90pct"):
        zh.append("- **誠實結論：區間仍未校準到可產品化**；全國位移＋少屆數令 exchangeability 脆弱。")
    zh.append("")
    zh.append("## 美國對照權重")
    zh.append("")
    zh.append(f"狀態：`{us_status}` — {us_note}")
    zh.append("")
    zh.append("## 檔案路徑")
    zh.append("")
    zh.append("- `results/feature_weights/taiwan_county_ridge_coefs.csv`")
    zh.append("- `results/feature_weights/taiwan_county_rf_importance.csv`")
    zh.append("- `results/feature_weights/taiwan_county_hgb_importance.csv`")
    zh.append("- `results/feature_weights/feature_weights_summary.md`")
    zh.append("- `data/county_panel_features_enriched_2000_2024.csv`")
    zh.append("")
    (ROOT / "docs" / "特徵權重說明.md").write_text("\n".join(zh) + "\n", encoding="utf-8")


def mirror_to_data_tree():
    """Mirror key outputs into /workspace/taiwan-election-data for consistency."""
    pairs = [
        (OUT_FW, DATA_TREE / "analysis" / "feature_weights"),
        (OUT_ENR, DATA_TREE / "fyp_taiwan_county_core_plus"),
        (OUT_CONF, DATA_TREE / "analysis" / "conformal_rq4_improved"),
        (OUT_FA, DATA_TREE / "fyp_taiwan_county_core"),
    ]
    for src, dst in pairs:
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    # enriched panel + provenance
    for name in [
        "county_panel_features_enriched_2000_2024.csv",
        "DATA_FILL_PROVENANCE_enriched.json",
    ]:
        s = ROOT / "data" / name
        if s.exists():
            shutil.copy2(s, DATA_TREE / "cleaned" / name if name.endswith(".csv") else DATA_TREE / "docs" / name)
            shutil.copy2(s, DATA_TREE / "data" / name if (DATA_TREE / "data").exists() else s)
    docs_src = ROOT / "docs" / "特徵權重說明.md"
    if docs_src.exists():
        (DATA_TREE / "docs").mkdir(exist_ok=True)
        shutil.copy2(docs_src, DATA_TREE / "docs" / "特徵權重說明.md")


def update_data_gaps():
    path = ROOT / "docs" / "DATA_GAPS.md"
    text = path.read_text(encoding="utf-8")
    block = (
        "\n## 2026-10-30 追加公開填補（特徵權重輪）\n\n"
        "| 項目 | 位置 |\n|------|------|\n"
        "| 縣市所得／密度併入 github economy | `data/economy/income_county_panel.csv`, `moi_pop_density_county_panel.csv` |\n"
        "| 富化縣市面板（滯後社經＋ESC PROXY） | `data/county_panel_features_enriched_2000_2024.csv` |\n"
        "| 出處 JSON | `data/DATA_FILL_PROVENANCE_enriched.json` |\n"
        "| 特徵權重匯出 | `results/feature_weights/` |\n"
        "| RQ4 改良 conformal | `results/conformal_rq4_improved/` |\n\n"
        "仍需申請：TEDS 微資料、ESC 官方 Excel（政黨偏好完整序列）。\n"
    )
    if "2026-10-30 追加公開填補" not in text:
        path.write_text(text.rstrip() + "\n" + block, encoding="utf-8")


def main():
    print("Building enriched panel...", flush=True)
    enriched, provenance = build_enriched_panel()
    print("Coverage sample:", {k: provenance["coverage"][k]["n_non_null"] for k in SOCIO})

    print("Refitting Core specification electoral...", flush=True)
    frame_e, test_years, metrics_e, mean_e = run_backtest(
        enriched, ELECTORAL, OUT_FA, "family_a_electoral"
    )
    print(mean_e.to_string(index=False))
    abl_e = run_ablation(frame_e, ELECTORAL, GROUPS_ELEC, test_years, OUT_FA)
    ridge_e, rf_e, hgb_e, _ = export_weights(frame_e, ELECTORAL, test_years, "taiwan_county", "electoral")

    # also refresh family_a importance summary for compatibility
    # (permutation already in feature_weights; copy mean metrics note)
    (OUT_FA / "REPRODUCIBILITY.md").write_text(
        "# Core specification refresh\n\n"
        "Refit by `scripts/run_taiwan_feature_weights_and_improve.py`.\n"
        "Canonical electoral features unchanged; see also Core+ (enriched) in "
        "`results/family_a_plus_enriched/`.\n"
        "Feature weights: `results/feature_weights/`.\n",
        encoding="utf-8",
    )

    print("Refitting Core+ (enriched) enriched...", flush=True)
    frame_p, test_years2, metrics_p, mean_p = run_backtest(
        enriched, ENRICHED, OUT_ENR, "family_a_plus_enriched"
    )
    print(mean_p.to_string(index=False))
    abl_p = run_ablation(frame_p, ENRICHED, GROUPS_ENR, test_years2, OUT_ENR)
    ridge_p, rf_p, hgb_p, _ = export_weights(
        frame_p, ENRICHED, test_years2, "taiwan_county_plus", "enriched"
    )

    print("RQ4 improved conformal...", flush=True)
    # Use enriched features but conformal should also run on electoral for fair RQ4
    conf_meta, conf_sum = run_conformal_improved(frame_e, ELECTORAL)
    conf_meta2, conf_sum2 = run_conformal_improved(frame_p, ENRICHED)
    # merge both
    conf_meta["feature_suite"] = "electoral"
    conf_meta2["feature_suite"] = "enriched"
    conf_all = pd.concat([conf_meta, conf_meta2], ignore_index=True)
    conf_all.to_csv(OUT_CONF / "conformal_protocols_summary.csv", index=False, encoding="utf-8-sig")
    conf_sum = {
        "target_coverage": 0.9,
        "n_protocols_tried": int(len(conf_all)),
        "any_success_near_90pct": bool(conf_all.success.any()),
        "best_rows": conf_all.sort_values(["success", "coverage"], ascending=[False, False]).head(8).to_dict("records"),
        "honest_note": conf_sum["honest_note"],
    }
    (OUT_CONF / "summary.json").write_text(
        json.dumps(conf_sum, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print("Conformal any success:", conf_sum["any_success_near_90pct"])
    print(conf_all.sort_values("coverage", ascending=False).head(6).to_string(index=False))

    write_summaries(
        ridge_e, rf_e, hgb_e, ridge_p, rf_p, hgb_p,
        mean_e, mean_p, abl_e, abl_p, conf_all, conf_sum, provenance,
    )
    update_data_gaps()
    mirror_to_data_tree()

    # also mirror analysis/taiwan_core
    ac = ROOT / "analysis" / "taiwan_core"
    if OUT_FA.exists():
        if ac.exists():
            shutil.rmtree(ac)
        shutil.copytree(OUT_FA, ac)

    print("DONE")
    print("Electoral mean MAE:\n", mean_e.to_string(index=False))
    print("Enriched mean MAE:\n", mean_p.to_string(index=False))
    print("Top Ridge electoral:\n", ridge_e.head(10).to_string(index=False))
    print("Top RF electoral:\n", rf_e.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
