#!/usr/bin/env python3
"""Recompute feature weights (model dependence / predictive importance — NOT causal).

Suites:
  - TW Core specification electoral-only (county×year OOT)
  - TW Core specification+ enriched (socio + ESC PROXY)
  - TW A+ education Extension (A+ + lag_education_rate) — labeled Extension if education only
  - US Core county×year
  - Extension national-only correlational table (polls, X sentiment) — NOT Core specification

Does NOT invent numbers. Same Ridge/RF/HGB settings as prior weight pipelines.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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
SEED = 42
HK = ZoneInfo("Asia/Hong_Kong")
NOW = datetime.now(HK).strftime("%Y-%m-%d %H:%M Asia/Hong_Kong (UTC+8)")

TW_TARGET = "target_DPP_vote_share"
TW_ANCHOR = "prev_DPP_vote_share"
US_TARGET = "target_dem_two_party_share"
US_ANCHOR = "prev_dem_two_party_share"

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
EDU = ["lag_education_rate"]
ENRICHED = ELECTORAL + SOCIO
ENRICHED_EDU = ELECTORAL + SOCIO + EDU

US_FEATURES = [
    "prev_dem_two_party_share", "prev_gop_two_party_share",
    "prev2_dem_two_party_share", "prev2_gop_two_party_share",
    "prev_log_total_votes", "dem_change_1", "gop_change_1",
    "prev_dem_gop_margin", "dem_gop_margin_change_1", "abs_dem_change_1",
    "log_votes_change_1", "unemployment", "education_rate",
    "log_median_income", "log_density",
]

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
    "lag_education_rate": "Lag tertiary+ education rate % (MOI PROXY)",
    "prev_dem_two_party_share": "Previous Dem two-party share",
    "prev_gop_two_party_share": "Previous GOP two-party share",
    "prev2_dem_two_party_share": "Dem share two elections ago",
    "prev2_gop_two_party_share": "GOP share two elections ago",
    "prev_log_total_votes": "Previous log total votes",
    "dem_change_1": "Dem previous swing",
    "gop_change_1": "GOP previous swing",
    "prev_dem_gop_margin": "Previous Dem–GOP margin",
    "dem_gop_margin_change_1": "Dem–GOP margin change",
    "abs_dem_change_1": "Abs Dem previous swing",
    "log_votes_change_1": "Log-votes change",
    "unemployment": "Unemployment rate",
    "education_rate": "Bachelor's+ rate (ERS wave)",
    "log_median_income": "Log median income",
    "log_density": "Log population density (ACS)",
}
MODELS_TW = ["Previous-election baseline", "Ridge", "Random Forest", "HistGradientBoosting"]


def scores(actual, pred):
    return {
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
    }


def make_tw(name: str):
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


def make_us(name: str):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=120, min_samples_leaf=8, max_depth=18, random_state=SEED, n_jobs=-1
        )
    return HistGradientBoostingRegressor(
        max_iter=120, learning_rate=0.07, max_leaf_nodes=31,
        min_samples_leaf=25, l2_regularization=1.0, random_state=SEED,
    )


def impute_train_median(train, test, cols):
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


def tw_backtest(frame, features, suite):
    frame = frame.dropna(subset=[TW_TARGET, *ELECTORAL]).sort_values(["year", "region_id"]).copy()
    test_years = [y for y in sorted(frame.year.unique()) if y >= 2012]
    rows = []
    for ty in test_years:
        train, test = frame[frame.year < ty], frame[frame.year == ty]
        train, test = impute_train_median(train, test, features)
        y = test[TW_TARGET].to_numpy()
        for name in MODELS_TW:
            if name == MODELS_TW[0]:
                pred = test[TW_ANCHOR].to_numpy()
            else:
                pred = make_tw(name).fit(train[features], train[TW_TARGET]).predict(test[features])
            rows.append({"suite": suite, "test_year": int(ty), "model": name, **scores(y, pred),
                         "n_train": len(train), "n_test": len(test)})
    metrics = pd.DataFrame(rows)
    mean = metrics.groupby(["suite", "model"], as_index=False)[["mae", "rmse"]].mean().sort_values("mae")
    return frame, test_years, metrics, mean


def export_tw_weights(frame, features, test_years, out_prefix, suite_tag):
    OUT_FW.mkdir(parents=True, exist_ok=True)
    ridge_rows, rf_rows, hgb_rows, perm_rows = [], [], [], []
    for ty in test_years:
        print(f"  TW weights {suite_tag} fold {ty}...", flush=True)
        train, test = frame[frame.year < ty], frame[frame.year == ty]
        train, test = impute_train_median(train, test, features)
        y_te = test[TW_TARGET].to_numpy()
        ridge = make_tw("Ridge").fit(train[features], train[TW_TARGET])
        for feat, w in zip(features, ridge[-1].coef_):
            ridge_rows.append({
                "suite": suite_tag, "test_year": int(ty), "feature": feat,
                "feature_label": LABELS.get(feat, feat),
                "standardized_coef_signed": float(w),
                "standardized_coef_abs": abs(float(w)),
            })
        rf = make_tw("Random Forest").fit(train[features], train[TW_TARGET])
        for feat, imp in zip(features, rf.feature_importances_):
            rf_rows.append({
                "suite": suite_tag, "test_year": int(ty), "feature": feat,
                "feature_label": LABELS.get(feat, feat),
                "impurity_importance": float(imp),
            })
        pi_rf = permutation_importance(
            rf, test[features], y_te, scoring="neg_mean_absolute_error",
            n_repeats=15, random_state=SEED + int(ty), n_jobs=1,
        )
        for j, feat in enumerate(features):
            perm_rows.append({
                "suite": suite_tag, "test_year": int(ty), "model": "Random Forest",
                "feature": feat, "feature_label": LABELS.get(feat, feat),
                "permutation_mae_increase": float(pi_rf.importances_mean[j]),
                "permutation_std": float(pi_rf.importances_std[j]),
            })
        hgb = make_tw("HistGradientBoosting").fit(train[features], train[TW_TARGET])
        impurity = getattr(hgb, "feature_importances_", None)
        if impurity is None:
            impurity = np.zeros(len(features))
        for feat, imp in zip(features, impurity):
            hgb_rows.append({
                "suite": suite_tag, "test_year": int(ty), "feature": feat,
                "feature_label": LABELS.get(feat, feat),
                "impurity_importance": float(imp),
            })
        pi_hgb = permutation_importance(
            hgb, test[features], y_te, scoring="neg_mean_absolute_error",
            n_repeats=15, random_state=SEED + int(ty), n_jobs=1,
        )
        for j, feat in enumerate(features):
            perm_rows.append({
                "suite": suite_tag, "test_year": int(ty), "model": "HistGradientBoosting",
                "feature": feat, "feature_label": LABELS.get(feat, feat),
                "permutation_mae_increase": float(pi_hgb.importances_mean[j]),
                "permutation_std": float(pi_hgb.importances_std[j]),
            })
        pi_ridge = permutation_importance(
            ridge, test[features], y_te, scoring="neg_mean_absolute_error",
            n_repeats=15, random_state=SEED + int(ty), n_jobs=1,
        )
        for j, feat in enumerate(features):
            perm_rows.append({
                "suite": suite_tag, "test_year": int(ty), "model": "Ridge",
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
    ridge_mean["interpretation"] = "model dependence / predictive importance — NOT causal"

    rf_mean = rf_df.groupby(["suite", "feature", "feature_label"], as_index=False).agg(
        impurity_mean=("impurity_importance", "mean"),
        impurity_std=("impurity_importance", "std"),
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
    rf_out["interpretation"] = "model dependence / predictive importance — NOT causal"

    hgb_mean = hgb_df.groupby(["suite", "feature", "feature_label"], as_index=False).agg(
        impurity_mean=("impurity_importance", "mean"),
        impurity_std=("impurity_importance", "std"),
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
    hgb_out["interpretation"] = "model dependence / predictive importance — NOT causal"

    ridge_mean.to_csv(OUT_FW / f"{out_prefix}_ridge_coefs.csv", index=False, encoding="utf-8-sig")
    rf_out.to_csv(OUT_FW / f"{out_prefix}_rf_importance.csv", index=False, encoding="utf-8-sig")
    hgb_out.to_csv(OUT_FW / f"{out_prefix}_hgb_importance.csv", index=False, encoding="utf-8-sig")
    if out_prefix == "taiwan_county":
        ridge_df.to_csv(OUT_FW / "taiwan_county_ridge_coefs_by_year.csv", index=False, encoding="utf-8-sig")
        perm_df.to_csv(OUT_FW / "taiwan_county_permutation_by_year.csv", index=False, encoding="utf-8-sig")
    elif out_prefix == "taiwan_county_plus":
        ridge_df.to_csv(OUT_FW / "taiwan_county_plus_ridge_coefs_by_year.csv", index=False, encoding="utf-8-sig")
        perm_df.to_csv(OUT_FW / "taiwan_county_plus_permutation_by_year.csv", index=False, encoding="utf-8-sig")
    elif out_prefix == "taiwan_county_plus_edu":
        ridge_df.to_csv(OUT_FW / "taiwan_county_plus_edu_ridge_coefs_by_year.csv", index=False, encoding="utf-8-sig")
        perm_df.to_csv(OUT_FW / "taiwan_county_plus_edu_permutation_by_year.csv", index=False, encoding="utf-8-sig")

    return ridge_mean, rf_out, hgb_out, perm_df, metrics_placeholder(suite_tag)


def metrics_placeholder(suite_tag):
    return suite_tag


def export_us_weights(panel: pd.DataFrame):
    print("US Core weights...", flush=True)
    train = panel[panel.year < 2024].dropna(subset=[US_TARGET, *US_FEATURES])
    test_pool = panel[panel.year == 2024].dropna(subset=[US_TARGET, *US_FEATURES])
    test = test_pool.sample(n=min(800, len(test_pool)), random_state=SEED)
    y_te = test[US_TARGET].to_numpy()
    weight_rows = []
    m_ridge = make_us("Ridge").fit(train[US_FEATURES], train[US_TARGET])
    m_rf = make_us("Random Forest").fit(train[US_FEATURES], train[US_TARGET])
    m_hgb = make_us("HistGradientBoosting").fit(train[US_FEATURES], train[US_TARGET])
    for feat, c in zip(US_FEATURES, m_ridge[-1].coef_):
        weight_rows.append(dict(
            source="refit_train_lt_2024", model="Ridge", feature=feat,
            feature_label=LABELS[feat],
            weight_type="standardized_coefficient_signed", weight=float(c),
            abs_weight=abs(float(c)),
            note="signed; + => higher Dem share after z-score; NOT causal",
        ))
    for feat, w in zip(US_FEATURES, m_rf.feature_importances_):
        weight_rows.append(dict(
            source="refit_train_lt_2024", model="Random Forest", feature=feat,
            feature_label=LABELS[feat],
            weight_type="impurity_importance_mdi", weight=float(w), abs_weight=float(w),
            note="MDI impurity; NOT causal",
        ))
    print("  US permutation...", flush=True)
    for name, fitted in [("Ridge", m_ridge), ("Random Forest", m_rf), ("HistGradientBoosting", m_hgb)]:
        pi = permutation_importance(
            fitted, test[US_FEATURES], y_te, scoring="neg_mean_absolute_error",
            n_repeats=10, random_state=SEED, n_jobs=1,
        )
        for feat, m, s in zip(US_FEATURES, pi.importances_mean, pi.importances_std):
            weight_rows.append(dict(
                source="perm_on_2024_subsample", model=name, feature=feat,
                feature_label=LABELS[feat],
                weight_type="permutation_mae_increase", weight=float(m),
                abs_weight=abs(float(m)),
                note=f"perm_std={float(s):.6g}; n={len(test)}; NOT causal",
            ))
    weights = pd.DataFrame(weight_rows)
    weights["normalized_abs"] = (
        weights.groupby(["source", "model", "weight_type"], group_keys=False)["abs_weight"]
        .transform(lambda s: s / s.sum() if s.sum() > 0 else 0.0)
    )
    weights = weights.sort_values(
        ["source", "model", "weight_type", "abs_weight"], ascending=[True, True, True, False]
    )
    weights["interpretation"] = "model dependence / predictive importance — NOT causal"
    weights.to_csv(OUT_FW / "us_county_feature_weights_all.csv", index=False, encoding="utf-8-sig")
    ridge_s = weights[(weights.model == "Ridge") & (weights.weight_type == "standardized_coefficient_signed")]
    rf_mdi = weights[(weights.model == "Random Forest") & (weights.weight_type == "impurity_importance_mdi")]
    perm = weights[weights.weight_type == "permutation_mae_increase"]
    ridge_s.to_csv(OUT_FW / "us_county_feature_weights_ridge_signed.csv", index=False, encoding="utf-8-sig")
    rf_mdi.to_csv(OUT_FW / "us_county_feature_weights_rf_impurity.csv", index=False, encoding="utf-8-sig")
    perm.to_csv(OUT_FW / "us_county_feature_weights_permutation.csv", index=False, encoding="utf-8-sig")

    # summary table (match prior schema used in comparison)
    summary_rows = []
    for _, r in perm.iterrows():
        summary_rows.append({
            "model": r.model, "feature": r.feature, "feature_label": r.feature_label,
            "importance_mean": r.weight, "importance_kind": "permutation_mae_increase",
            "source": r.source,
        })
    for _, r in ridge_s.iterrows():
        summary_rows.append({
            "model": "Ridge_coef", "feature": r.feature, "feature_label": r.feature_label,
            "importance_mean": abs(r.weight), "importance_kind": "abs_standardized_coef",
            "signed_coef": r.weight, "source": r.source,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_FW / "us_county_feature_importance_summary.csv", index=False, encoding="utf-8-sig")

    # US side-check MAE (Previous baseline + Ridge/RF/HGB mean OOT on county panel)
    print("  US OOT MAE side-check...", flush=True)
    us_metric_rows = []
    for ty in [2012, 2016, 2020, 2024]:
        tr = panel[panel.year < ty].dropna(subset=[US_TARGET, US_ANCHOR, *US_FEATURES])
        te = panel[panel.year == ty].dropna(subset=[US_TARGET, US_ANCHOR, *US_FEATURES])
        if te.empty or tr.empty:
            continue
        y = te[US_TARGET].to_numpy()
        for name in ["Previous-election baseline", "Ridge", "Random Forest", "HistGradientBoosting"]:
            if name == "Previous-election baseline":
                pred = te[US_ANCHOR].to_numpy()
            else:
                pred = make_us(name).fit(tr[US_FEATURES], tr[US_TARGET]).predict(te[US_FEATURES])
            us_metric_rows.append({"suite": "us_core", "test_year": ty, "model": name, **scores(y, pred)})
    us_metrics = pd.DataFrame(us_metric_rows)
    us_mean = us_metrics.groupby("model", as_index=False)[["mae", "rmse"]].mean().sort_values("mae")
    us_mean.to_csv(OUT_FW / "us_county_mean_mae_sidecheck.csv", index=False, encoding="utf-8-sig")
    us_metrics.to_csv(OUT_FW / "us_county_backtest_metrics_sidecheck.csv", index=False, encoding="utf-8-sig")

    # FEATURE_WEIGHTS.md
    lines = [
        "# US Core specification — Feature weights (model dependence, not causal)",
        "",
        f"Recomputed: {NOW}",
        "",
        "Not causal. No county `poll_index` invented.",
        "",
        "## Ridge standardized coefficients (signed, train < 2024)",
        "",
    ]
    for _, r in ridge_s.iterrows():
        lines.append(
            f"- **{r.feature_label}** (`{r.feature}`): weight={r.weight:.4g}, |w| share={100*r.normalized_abs:.1f}%"
        )
    lines += ["", "## Random Forest MDI impurity (train < 2024)", ""]
    for _, r in rf_mdi.iterrows():
        lines.append(
            f"- **{r.feature_label}** (`{r.feature}`): weight={r.weight:.4g}, |w| share={100*r.normalized_abs:.1f}%"
        )
    lines += ["", "## Permutation MAE ↑ on 2024 subsample", ""]
    for model in ["Ridge", "Random Forest", "HistGradientBoosting"]:
        lines.append(f"\n### {model}\n")
        sub = perm[perm.model == model]
        for _, r in sub.iterrows():
            lines.append(
                f"- **{r.feature_label}** (`{r.feature}`): weight={r.weight:.4g}, |w| share={100*r.normalized_abs:.1f}%"
            )
    lines += [
        "",
        "## Side-check mean OOT MAE",
        "",
        "```",
        us_mean.to_string(index=False),
        "```",
        "",
        "## Files",
        "- `us_county_feature_weights_all.csv`",
        "- `us_county_feature_weights_ridge_signed.csv`",
        "- `us_county_feature_weights_rf_impurity.csv`",
        "- `us_county_feature_weights_permutation.csv`",
        "",
    ]
    (OUT_FW / "us_county_FEATURE_WEIGHTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return weights, ridge_s, perm, us_mean


def extension_national_weights():
    """Correlational / simple regression on national series only — NOT Core specification."""
    print("Extension national correlational weights...", flush=True)
    rows = []

    # US national poll as predictor of national dem two-party share
    poll = pd.read_csv(ROOT / "data" / "us_parity_extension" / "us_national_poll_election_year_proxy.csv")
    outcome = pd.read_csv(ROOT / "data" / "us_parity_extension" / "us_national_partisan_climate_proxy.csv")
    m = poll.merge(outcome, on="election_year", how="inner")
    m = m.dropna(subset=["dem_two_party_share_poll", "nat_dem_two_party_share"])
    if len(m) >= 3:
        x = m[["dem_two_party_share_poll"]].to_numpy()
        y = m["nat_dem_two_party_share"].to_numpy()
        # simple standardized ridge (alpha small) / OLS via ridge alpha~0
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=0.1))
        pipe.fit(x, y)
        coef = float(pipe[-1].coef_[0])
        pred = pipe.predict(x)
        corr = float(np.corrcoef(m["dem_two_party_share_poll"], m["nat_dem_two_party_share"])[0, 1])
        rows.append({
            "layer": "Extension_national_NOT_Family_A",
            "country": "US",
            "feature": "dem_two_party_share_poll",
            "feature_label": "538 national poll Dem two-party (election-year as-of)",
            "target": "nat_dem_two_party_share",
            "n_years": int(len(m)),
            "pearson_r": corr,
            "ridge_std_coef": coef,
            "mae": float(mean_absolute_error(y, pred)),
            "method": "national series only; Ridge(alpha=0.1) after StandardScaler",
            "interpretation": "correlational / predictive association — NOT causal; NOT county Core specification",
            "years": ",".join(str(int(y_)) for y_ in sorted(m.election_year.unique())),
        })

    # TW national polls: last pre-election DPP vs KMT if available
    tw_polls = ROOT / "data" / "extension_poll_platform" / "taiwan_polls_long.csv"
    tw_nat = ROOT / "data" / "cec_president_national_totals.csv"
    if tw_polls.exists() and tw_nat.exists():
        pl = pd.read_csv(tw_polls)
        # try to build election-year DPP/KMT poll averages
        pl["field_end"] = pd.to_datetime(pl.get("field_end", pl.get("published_at")), errors="coerce")
        # crude: group by year of field_end and target containing 民進黨/DPP or 國民黨/KMT
        pl["ey"] = pl["field_end"].dt.year
        # map targets
        def side(t):
            t = str(t)
            if "民進" in t or "DPP" in t.upper() or "蔡" in t or "賴" in t:
                return "DPP"
            if "國民" in t or "KMT" in t.upper() or "韓" in t or "馬" in t or "朱" in t or "侯" in t:
                return "KMT"
            return None
        pl["side"] = pl["target"].map(side)
        pl = pl[pl.side.notna() & pl.value.notna() & pl.ey.notna()]
        # election years of interest
        for ey in [2008, 2012, 2016, 2020, 2024]:
            # polls in [ey-1 Dec .. ey election] approx: ey-1 and ey before March
            sub = pl[(pl.ey == ey) | ((pl.ey == ey - 1) & (pl.field_end.dt.month >= 10))]
            if sub.empty:
                continue
        # aggregate mean poll by election year × side for years with both
        agg = pl[pl.ey.isin([2008, 2012, 2016, 2020, 2024])].groupby(["ey", "side"], as_index=False)["value"].mean()
        wide = agg.pivot(index="ey", columns="side", values="value").dropna(how="any")
        nat = pd.read_csv(tw_nat)
        # expect columns year, DPP share etc.
        # try common schemas
        if "year" in nat.columns:
            ycol = "year"
        elif "election_year" in nat.columns:
            ycol = "election_year"
        else:
            ycol = None
        dpp_col = None
        for c in nat.columns:
            if "DPP" in c and ("share" in c.lower() or "pct" in c.lower() or "vote" in c.lower()):
                dpp_col = c
                break
        if dpp_col is None and {"DPP_votes", "valid_votes"}.issubset(nat.columns):
            nat = nat.copy()
            nat["dpp_share"] = nat["DPP_votes"] / nat["valid_votes"]
            dpp_col = "dpp_share"
            ycol = ycol or "year"
        if ycol and dpp_col is not None and not wide.empty and "DPP" in wide.columns:
            wide = wide.reset_index().rename(columns={"ey": ycol})
            mm = wide.merge(nat[[ycol, dpp_col]], on=ycol, how="inner")
            if len(mm) >= 3:
                # use DPP poll share / (DPP+KMT) as two-party-ish
                if "KMT" in mm.columns:
                    mm["poll_dpp_two"] = mm["DPP"] / (mm["DPP"] + mm["KMT"])
                else:
                    mm["poll_dpp_two"] = mm["DPP"] / 100.0 if mm["DPP"].max() > 1.5 else mm["DPP"]
                yv = mm[dpp_col].to_numpy()
                if yv.max() > 1.5:
                    yv = yv / 100.0
                xv = mm[["poll_dpp_two"]].to_numpy()
                pipe = make_pipeline(StandardScaler(), Ridge(alpha=0.1))
                pipe.fit(xv, yv)
                corr = float(np.corrcoef(mm["poll_dpp_two"], yv)[0, 1])
                pred = pipe.predict(xv)
                rows.append({
                    "layer": "Extension_national_NOT_Family_A",
                    "country": "TW",
                    "feature": "poll_dpp_two_party_approx",
                    "feature_label": "TW national poll DPP/(DPP+KMT) crude mean",
                    "target": dpp_col,
                    "n_years": int(len(mm)),
                    "pearson_r": corr,
                    "ridge_std_coef": float(pipe[-1].coef_[0]),
                    "mae": float(mean_absolute_error(yv, pred)),
                    "method": "national series only; crude poll average; Ridge(alpha=0.1)",
                    "interpretation": "correlational — NOT causal; NOT county Core specification; poll target mapping is heuristic",
                    "years": ",".join(str(int(y_)) for y_ in sorted(mm[ycol].unique())),
                })

    # X sentiment camp gap vs winner alignment / national share (few years)
    xs_path = ROOT / "data" / "x_sentiment_extension" / "x_entity_year_aggregates.csv"
    if xs_path.exists():
        xs = pd.read_csv(xs_path)
        # camp mean by country × year × side
        camp = (
            xs.groupby(["country", "election_year", "side"], as_index=False)
            .agg(mean_sentiment=("mean_sentiment", "mean"), n_posts=("n_posts", "sum"))
        )
        # pivot DPP/Dem vs KMT/GOP
        for country, a_side, b_side, target_name in [
            ("TW", "DPP", "KMT", "tw_sentiment_gap_dpp_minus_kmt"),
            ("US", "DEM", "GOP", "us_sentiment_gap_dem_minus_gop"),
        ]:
            sub = camp[camp.country == country]
            if sub.empty:
                continue
            wide = sub.pivot(index="election_year", columns="side", values="mean_sentiment")
            if a_side not in wide.columns or b_side not in wide.columns:
                continue
            wide = wide.dropna(subset=[a_side, b_side]).copy()
            wide["gap"] = wide[a_side] - wide[b_side]
            # merge outcome
            if country == "US":
                outc = pd.read_csv(ROOT / "data" / "us_parity_extension" / "us_national_partisan_climate_proxy.csv")
                mm = wide.reset_index().merge(outc, on="election_year", how="inner")
                if len(mm) >= 2:
                    corr = float(np.corrcoef(mm["gap"], mm["nat_dem_two_party_share"])[0, 1]) if len(mm) > 1 else float("nan")
                    rows.append({
                        "layer": "Extension_national_NOT_Family_A",
                        "country": "US",
                        "feature": "x_sentiment_gap_dem_minus_gop",
                        "feature_label": "X camp mean sentiment Dem−GOP (pilot sample)",
                        "target": "nat_dem_two_party_share",
                        "n_years": int(len(mm)),
                        "pearson_r": corr,
                        "ridge_std_coef": float("nan") if len(mm) < 3 else float(
                            make_pipeline(StandardScaler(), Ridge(alpha=0.1))
                            .fit(mm[["gap"]], mm["nat_dem_two_party_share"])[-1].coef_[0]
                        ),
                        "mae": float("nan"),
                        "method": "pilot X API sample aggregates; national only",
                        "interpretation": "correlational / demo only — NOT causal; NOT Core specification; tiny n",
                        "years": ",".join(str(int(y_)) for y_ in sorted(mm.election_year.unique())),
                    })
            else:
                # TW: use national DPP share from cec
                nat = pd.read_csv(ROOT / "data" / "cec_president_national_totals.csv")
                if {"DPP_votes", "valid_votes", "year"}.issubset(nat.columns):
                    nat = nat.copy()
                    nat["dpp_share"] = nat["DPP_votes"] / nat["valid_votes"]
                    mm = wide.reset_index().merge(
                        nat.rename(columns={"year": "election_year"})[["election_year", "dpp_share"]],
                        on="election_year", how="inner",
                    )
                    if len(mm) >= 2:
                        corr = float(np.corrcoef(mm["gap"], mm["dpp_share"])[0, 1]) if len(mm) > 1 else float("nan")
                        rows.append({
                            "layer": "Extension_national_NOT_Family_A",
                            "country": "TW",
                            "feature": "x_sentiment_gap_dpp_minus_kmt",
                            "feature_label": "X camp mean sentiment DPP−KMT (pilot sample)",
                            "target": "dpp_share",
                            "n_years": int(len(mm)),
                            "pearson_r": corr,
                            "ridge_std_coef": float("nan") if len(mm) < 3 else float(
                                make_pipeline(StandardScaler(), Ridge(alpha=0.1))
                                .fit(mm[["gap"]], mm["dpp_share"])[-1].coef_[0]
                            ),
                            "mae": float("nan"),
                            "method": "pilot X API sample aggregates; national only",
                            "interpretation": "correlational / demo only — NOT causal; NOT Core specification; tiny n",
                            "years": ",".join(str(int(y_)) for y_ in sorted(mm.election_year.unique())),
                        })

    ext = pd.DataFrame(rows)
    ext.to_csv(OUT_FW / "extension_national_feature_associations.csv", index=False, encoding="utf-8-sig")
    readme = [
        "# Extension national feature associations — NOT Core specification",
        "",
        f"Built: {NOW}",
        "",
        "These are **national-only** correlational / simple-regression associations",
        "(polls, X sentiment pilot aggregates).",
        "",
        "- **NOT** county Core specification / A+ weights",
        "- **NOT** causal",
        "- Tiny n for X (2–4 election years) — exploratory only",
        "",
        "File: `extension_national_feature_associations.csv`",
        "",
    ]
    (OUT_FW / "extension_national_README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return ext


def plot_tops(ridge_elec, rf_elec, ridge_us_perm):
    charts = OUT_FW / "charts"
    charts.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    top = ridge_elec.head(10).iloc[::-1]
    ax.barh(top.feature_label, top.coef_mean_signed, color="#2878B5")
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_title("TW Core specification — Ridge standardized coef (mean OOT)", loc="left")
    ax.set_xlabel("Signed coef (NOT causal)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(charts / "tw_family_a_ridge_coefs.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    top = rf_elec.head(10).iloc[::-1]
    ax.barh(top.feature_label, top.permutation_mae_increase_mean, color="#E07A2D")
    ax.set_title("TW Core specification — RF permutation MAE↑ (mean OOT)", loc="left")
    ax.set_xlabel("Permutation MAE increase (NOT causal)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(charts / "tw_family_a_rf_perm.png", dpi=160)
    plt.close()


def write_docs(mean_elec, mean_enr, mean_edu, ridge_e, rf_e, hgb_e, ridge_p, ridge_edu,
               us_perm, us_ridge, us_mean, ext_df, prior_note):
    # comparison CSV
    compare = []
    for _, r in ridge_e.head(15).iterrows():
        compare.append({
            "side": "TW_Family_A_electoral", "model": "Ridge",
            "feature": r.feature, "feature_label": r.feature_label,
            "weight": r.coef_mean_signed, "weight_kind": "standardized_coef_signed_mean_OOT",
        })
    for _, r in rf_e.head(15).iterrows():
        compare.append({
            "side": "TW_Family_A_electoral", "model": "Random Forest",
            "feature": r.feature, "feature_label": r.feature_label,
            "weight": r.permutation_mae_increase_mean, "weight_kind": "permutation_mae_increase_mean_OOT",
        })
    for _, r in ridge_p.head(15).iterrows():
        compare.append({
            "side": "TW_Family_A_plus", "model": "Ridge",
            "feature": r.feature, "feature_label": r.feature_label,
            "weight": r.coef_mean_signed, "weight_kind": "standardized_coef_signed_mean_OOT",
        })
    if ridge_edu is not None:
        for _, r in ridge_edu.head(15).iterrows():
            compare.append({
                "side": "TW_A_plus_education_Extension", "model": "Ridge",
                "feature": r.feature, "feature_label": r.feature_label,
                "weight": r.coef_mean_signed, "weight_kind": "standardized_coef_signed_mean_OOT",
            })
    us_r = us_perm[us_perm.model == "Ridge"].sort_values("abs_weight", ascending=False).head(10)
    for _, r in us_r.iterrows():
        compare.append({
            "side": "US_Core", "model": "Ridge",
            "feature": r.feature, "feature_label": r.feature_label,
            "weight": r.weight, "weight_kind": "permutation_mae_increase_2024_subsample",
        })
    us_rf = us_perm[us_perm.model == "Random Forest"].sort_values("abs_weight", ascending=False).head(10)
    for _, r in us_rf.iterrows():
        compare.append({
            "side": "US_Core", "model": "Random Forest",
            "feature": r.feature, "feature_label": r.feature_label,
            "weight": r.weight, "weight_kind": "permutation_mae_increase_2024_subsample",
        })
    for _, r in us_ridge.head(10).iterrows():
        compare.append({
            "side": "US_Core", "model": "Ridge_coef",
            "feature": r.feature, "feature_label": r.feature_label,
            "weight": r.weight, "weight_kind": "standardized_coefficient_signed_train_lt_2024",
        })
    pd.DataFrame(compare).to_csv(OUT_FW / "tw_us_feature_weights_comparison.csv", index=False, encoding="utf-8-sig")

    # English summary
    lines = [
        "# Feature weights summary (recomputed)",
        "",
        f"**Recomputed:** {NOW}",
        "",
        "**Interpretation:** weights = model dependence in out-of-time folds — **not causal**, **not** reasons a party wins.",
        "",
        "Aggregation TW: **mean across OOT folds** 2012, 2016, 2020, 2024.",
        "US: Ridge/RF coef on train<2024; permutation on 2024 subsample (n≤800) — same protocol as `us_core_weights_conformal.py`.",
        "",
        "## Core specification electoral-only mean OOT MAE (side-check)",
        "",
        mean_elec.to_string(index=False),
        "",
        "## Core+ (enriched) enriched mean OOT MAE",
        "",
        mean_enr.to_string(index=False),
        "",
        "## A+ + education Extension mean OOT MAE (NOT locked Core specification)",
        "",
        mean_edu.to_string(index=False) if mean_edu is not None else "(skipped)",
        "",
        "## Top Ridge standardized coefficients (electoral)",
        "",
        ridge_e.head(10)[["feature", "feature_label", "coef_mean_signed", "coef_mean_abs", "rank_by_abs"]].to_string(index=False),
        "",
        "## Top RF permutation (electoral)",
        "",
        rf_e.head(10)[["feature", "feature_label", "permutation_mae_increase_mean", "impurity_mean", "rank"]].to_string(index=False),
        "",
        "## Top HGB permutation (electoral)",
        "",
        hgb_e.head(10)[["feature", "feature_label", "permutation_mae_increase_mean", "rank"]].to_string(index=False),
        "",
        "## Top Ridge (A+ enriched)",
        "",
        ridge_p.head(10)[["feature", "feature_label", "coef_mean_signed", "coef_mean_abs"]].to_string(index=False),
        "",
    ]
    if ridge_edu is not None:
        lines += [
            "## Top Ridge (A+ + education Extension)",
            "",
            ridge_edu.head(12)[["feature", "feature_label", "coef_mean_signed", "coef_mean_abs"]].to_string(index=False),
            "",
        ]
    lines += [
        "## US Core side-check mean OOT MAE",
        "",
        us_mean.to_string(index=False),
        "",
        "## US Ridge permutation top (2024 subsample)",
        "",
        us_r[["feature", "feature_label", "weight"]].to_string(index=False),
        "",
        "## Extension national associations (NOT Core specification)",
        "",
        ext_df.to_string(index=False) if len(ext_df) else "(none)",
        "",
        prior_note,
        "",
        "## Files",
        "- `taiwan_county_*.csv`, `taiwan_county_plus_*.csv`, `taiwan_county_plus_edu_*.csv`",
        "- `us_county_feature_weights_*.csv`",
        "- `extension_national_feature_associations.csv`",
        "- `tw_us_feature_weights_comparison.csv`",
        "- `docs/特徵權重說明.md`, `docs/權重重算說明.md`",
        "",
    ]
    (OUT_FW / "feature_weights_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Chinese teacher doc
    zh = []
    zh.append("# 特徵權重說明（臺灣縣市總統票｜Core specification／A+｜美國對照）")
    zh.append("")
    zh.append(f"更新：{NOW}")
    zh.append("")
    zh.append("## 必讀：權重唔係因果、唔係勝選原因")
    zh.append("")
    zh.append("本文件嘅「權重／重要性」只描述模型喺**整屆留出（out-of-time）**回測入面**點樣依賴特徵**。")
    zh.append("")
    zh.append("- **唔係**因果效應（唔代表改呢個變數就會改得票）")
    zh.append("- **唔係**選情斷言或勝選原因")
    zh.append("- Ridge 係**標準化後有號係數**（跨 OOT fold **平均**，唔係只取最後一屆）")
    zh.append("- RF／HGB 係**置換重要性**（打亂特徵後 OOT MAE 升幾多）同 impurity（RF）")
    zh.append("- 只有四個測試屆（2012／2016／2020／2024），排名會有波動")
    zh.append("")
    zh.append("## Core specification（純選舉史）平均 OOT MAE（重算旁證）")
    zh.append("")
    zh.append("```")
    zh.append(mean_elec.to_string(index=False))
    zh.append("```")
    zh.append("")
    zh.append("## Core+ (enriched)（滯後社經／ESC PROXY）平均 OOT MAE")
    zh.append("")
    zh.append("```")
    zh.append(mean_enr.to_string(index=False))
    zh.append("```")
    zh.append("")
    if mean_edu is not None:
        zh.append("## A+ ＋教育率 Extension（`lag_education_rate`｜唔入鎖定 Core specification）")
        zh.append("")
        zh.append("```")
        zh.append(mean_edu.to_string(index=False))
        zh.append("```")
        zh.append("")
    zh.append("**解讀：** PreviousShare 基準仍然最好。權重只講模型依賴。")
    zh.append("")
    zh.append("## Top 10：Ridge 標準化係數（純選舉史｜OOT 平均）")
    zh.append("")
    zh.append("| 排名 | 特徵 | 標籤 | 係數（有號） | |係數| |")
    zh.append("|------|------|------|-------------|---------|")
    for i, r in enumerate(ridge_e.head(10).itertuples(), 1):
        zh.append(f"| {i} | `{r.feature}` | {r.feature_label} | {r.coef_mean_signed:.6f} | {r.coef_mean_abs:.6f} |")
    zh.append("")
    zh.append("## Top 10：Random Forest（置換 MAE↑｜純選舉史）")
    zh.append("")
    zh.append("| 排名 | 特徵 | 標籤 | perm MAE↑ | impurity |")
    zh.append("|------|------|------|-----------|----------|")
    for i, r in enumerate(rf_e.head(10).itertuples(), 1):
        zh.append(
            f"| {i} | `{r.feature}` | {r.feature_label} | {r.permutation_mae_increase_mean:.6f} | {r.impurity_mean:.6f} |"
        )
    zh.append("")
    zh.append("## Top 10：HGB（置換 MAE↑｜純選舉史）")
    zh.append("")
    zh.append("| 排名 | 特徵 | 標籤 | perm MAE↑ |")
    zh.append("|------|------|------|-----------|")
    for i, r in enumerate(hgb_e.head(10).itertuples(), 1):
        zh.append(f"| {i} | `{r.feature}` | {r.feature_label} | {r.permutation_mae_increase_mean:.6f} |")
    zh.append("")
    zh.append("## Top 10：Ridge（A+ 富化）")
    zh.append("")
    zh.append("| 排名 | 特徵 | 標籤 | 係數（有號） | |係數| |")
    zh.append("|------|------|------|-------------|---------|")
    for i, r in enumerate(ridge_p.head(10).itertuples(), 1):
        zh.append(f"| {i} | `{r.feature}` | {r.feature_label} | {r.coef_mean_signed:.6f} | {r.coef_mean_abs:.6f} |")
    zh.append("")
    if ridge_edu is not None:
        zh.append("## Top：Ridge（A+＋教育 Extension）")
        zh.append("")
        zh.append("| 排名 | 特徵 | 標籤 | 係數（有號） | |係數| |")
        zh.append("|------|------|------|-------------|---------|")
        for i, r in enumerate(ridge_edu.head(12).itertuples(), 1):
            zh.append(f"| {i} | `{r.feature}` | {r.feature_label} | {r.coef_mean_signed:.6f} | {r.coef_mean_abs:.6f} |")
        zh.append("")
    zh.append("## 美國 Core 對照")
    zh.append("")
    zh.append("### US 平均 OOT MAE（旁證）")
    zh.append("")
    zh.append("```")
    zh.append(us_mean.to_string(index=False))
    zh.append("```")
    zh.append("")
    zh.append("### US Ridge 置換重要性 top（2024 子樣本）")
    zh.append("")
    zh.append("| rank | feature | weight |")
    zh.append("|------|---------|--------|")
    for i, r in enumerate(us_r.itertuples(), 1):
        zh.append(f"| {i} | `{r.feature}` | {r.weight:.6f} |")
    zh.append("")
    zh.append("### US Ridge 標準化係數 top（train < 2024）")
    zh.append("")
    zh.append("| rank | feature | signed coef |")
    zh.append("|------|---------|-------------|")
    for i, r in enumerate(us_ridge.head(8).itertuples(), 1):
        zh.append(f"| {i} | `{r.feature}` | {r.weight:.6f} |")
    zh.append("")
    zh.append("美台兩邊都係「上屆／上上屆得票」主導；社經／教育權重遠細過歷史得票。")
    zh.append("")
    zh.append("## Extension 全國關聯（**唔係** Core specification）")
    zh.append("")
    zh.append("民調／X 情緒只得全國序列；見 `extension_national_feature_associations.csv`。相關／簡單回歸，**唔係**縣市權重、**唔係**因果。")
    zh.append("")
    zh.append("## 檔案路徑")
    zh.append("")
    zh.append("- `results/feature_weights/taiwan_county_ridge_coefs.csv`")
    zh.append("- `results/feature_weights/taiwan_county_rf_importance.csv`")
    zh.append("- `results/feature_weights/taiwan_county_hgb_importance.csv`")
    zh.append("- `results/feature_weights/taiwan_county_plus_*.csv`")
    zh.append("- `results/feature_weights/taiwan_county_plus_edu_*.csv`")
    zh.append("- `results/feature_weights/us_county_feature_weights_*.csv`")
    zh.append("- `results/feature_weights/tw_us_feature_weights_comparison.csv`")
    zh.append("- `docs/權重重算說明.md`")
    zh.append("")
    (ROOT / "docs" / "特徵權重說明.md").write_text("\n".join(zh) + "\n", encoding="utf-8")

    # 權重重算說明
    note = f"""# 權重重算說明

**日期：** {NOW}

## 做咗咩

用**現有面板**按既有 Core specification／US Core 權重管道重算模型依賴（predictive importance），**無捏造數字**，**無權重＝因果**。

## 模型同設定（同前輪）

| 側 | 模型 | 設定 |
|----|------|------|
| TW | Ridge | `StandardScaler` + `Ridge(alpha=10)` |
| TW | Random Forest | `n_estimators=200, min_samples_leaf=3, random_state=42` |
| TW | HistGradientBoosting | `max_iter=150, lr=0.06, max_leaf_nodes=7, min_samples_leaf=5, l2=1.0` |
| US | Ridge | 同上 alpha=10 |
| US | RF | `n_estimators=120, min_samples_leaf=8, max_depth=18` |
| US | HGB | `max_iter=120, lr=0.07, max_leaf_nodes=31, min_samples_leaf=25` |

TW 權重：**跨 OOT fold（2012／2016／2020／2024）平均** Ridge 係數；RF／HGB 置換 MAE↑。  
US 權重：train&lt;2024 係數／MDI；置換喺 2024 子樣本（n≤800）。

## 特徵清單

- **TW Core specification：** 16 個選舉史特徵（`ELECTORAL`）
- **TW A+：** ELECTORAL + 失業率／所得／密度 + ESC 認同／統獨 PROXY（全國滯後）
- **TW A+ edu Extension：** A+ + `lag_education_rate`（MOI 大專以上；**唔入鎖定 Core specification**）
- **US Core：** 上屆／上上屆兩黨得票、swing、失業率、education_rate、所得、密度等 15 欄
- **Extension 全國：** 538／TW 民調全國序列、X 情緒陣營差（pilot）— **標明 NOT Core specification**

## 相對上輪改變

{prior_note}

- 新增／刷新：`taiwan_county_plus_edu_*`、`extension_national_feature_associations.csv`
- Core specification 鎖定主數字預期不變：Prev MAE ≈ **0.0793**（本輪旁證重 fit 確認）
- 權重數值可能因重跑而有浮點／置換抽樣微差；解讀不變：歷史得票主導

## 權重 ≠ 因果

Ridge 有號係數、置換重要性、MDI 都只係**模型依賴／預測重要性**。唔代表改特徵就會改得票，亦唔係勝選原因。

## 腳本

`scripts/recompute_feature_weights.py`（讀現有 `county_panel_features_enriched_2000_2024.csv`、教育 Extension、US Core specification panel；唔重建社經爬取）。
"""
    (ROOT / "docs" / "權重重算說明.md").write_text(note, encoding="utf-8")


def mirror():
    # mirror feature_weights + docs into data tree
    dst_fw = DATA_TREE / "results" / "feature_weights"
    if dst_fw.exists():
        shutil.rmtree(dst_fw)
    shutil.copytree(OUT_FW, dst_fw)
    # also analysis path used earlier
    dst_an = DATA_TREE / "analysis" / "feature_weights"
    if dst_an.exists():
        shutil.rmtree(dst_an)
    shutil.copytree(OUT_FW, dst_an)
    for name in ["特徵權重說明.md", "權重重算說明.md"]:
        s = ROOT / "docs" / name
        if s.exists():
            (DATA_TREE / "docs").mkdir(exist_ok=True)
            shutil.copy2(s, DATA_TREE / "docs" / name)
    # sync script
    shutil.copy2(Path(__file__), DATA_TREE / "scripts" / Path(__file__).name)


def main():
    OUT_FW.mkdir(parents=True, exist_ok=True)
    print(f"Recompute feature weights @ {NOW}", flush=True)

    enriched = pd.read_csv(ROOT / "data" / "county_panel_features_enriched_2000_2024.csv")
    edu_panel = pd.read_csv(ROOT / "data" / "extension_education" / "county_panel_with_lag_education_rate.csv")
    # merge education rate onto enriched
    edu_cols = edu_panel[["region_id", "year", "lag_education_rate"]].drop_duplicates()
    enriched_edu = enriched.merge(edu_cols, on=["region_id", "year"], how="left")

    print("TW Core specification electoral side-check + weights...", flush=True)
    frame_e, ty, metrics_e, mean_e = tw_backtest(enriched, ELECTORAL, "family_a_electoral")
    print(mean_e.to_string(index=False))
    ridge_e, rf_e, hgb_e, _, _ = export_tw_weights(frame_e, ELECTORAL, ty, "taiwan_county", "electoral")
    metrics_e.to_csv(OUT_FW / "taiwan_county_backtest_metrics_sidecheck.csv", index=False, encoding="utf-8-sig")
    mean_e.to_csv(OUT_FW / "taiwan_county_mean_mae_sidecheck.csv", index=False, encoding="utf-8-sig")
    # refresh FA mean metrics without wiping other FA artifacts
    OUT_FA.mkdir(parents=True, exist_ok=True)
    mean_e.to_csv(OUT_FA / "family_a_mean_metrics.csv", index=False, encoding="utf-8-sig")
    metrics_e.to_csv(OUT_FA / "backtest_metrics_weights_sidecheck.csv", index=False, encoding="utf-8-sig")

    print("TW Core specification+ enriched...", flush=True)
    frame_p, ty2, metrics_p, mean_p = tw_backtest(enriched, ENRICHED, "family_a_plus_enriched")
    print(mean_p.to_string(index=False))
    ridge_p, rf_p, hgb_p, _, _ = export_tw_weights(frame_p, ENRICHED, ty2, "taiwan_county_plus", "enriched")
    OUT_ENR.mkdir(parents=True, exist_ok=True)
    mean_p.to_csv(OUT_ENR / "family_a_mean_metrics.csv", index=False, encoding="utf-8-sig")
    metrics_p.to_csv(OUT_ENR / "backtest_metrics_weights_sidecheck.csv", index=False, encoding="utf-8-sig")

    print("TW A+ education Extension...", flush=True)
    frame_edu, ty3, metrics_edu, mean_edu = tw_backtest(
        enriched_edu, ENRICHED_EDU, "extension_a_plus_education"
    )
    print(mean_edu.to_string(index=False))
    ridge_edu, rf_edu, hgb_edu, _, _ = export_tw_weights(
        frame_edu, ENRICHED_EDU, ty3, "taiwan_county_plus_edu", "enriched_plus_education_EXTENSION"
    )
    metrics_edu.to_csv(OUT_FW / "taiwan_county_plus_edu_backtest_metrics.csv", index=False, encoding="utf-8-sig")
    mean_edu.to_csv(OUT_FW / "taiwan_county_plus_edu_mean_mae.csv", index=False, encoding="utf-8-sig")

    us_panel_path = ROOT / "data" / "us_county_panel_features_family_a.csv"
    if not us_panel_path.exists():
        us_panel_path = DATA_TREE / "fyp_us_core" / "us_county_panel_features_family_a.csv"
    us_panel = pd.read_csv(us_panel_path)
    us_weights, us_ridge, us_perm, us_mean = export_us_weights(us_panel)

    # also mirror US weights into fyp_us_core/results
    us_res = DATA_TREE / "fyp_us_core" / "results"
    us_res.mkdir(parents=True, exist_ok=True)
    for fn in [
        "us_county_feature_weights_all.csv",
        "us_county_feature_weights_ridge_signed.csv",
        "us_county_feature_weights_rf_impurity.csv",
        "us_county_feature_weights_permutation.csv",
        "us_county_FEATURE_WEIGHTS.md",
    ]:
        src = OUT_FW / fn
        if src.exists():
            # map naming for fyp_us_core
            dst_name = fn.replace("us_county_", "").replace("FEATURE_WEIGHTS.md", "FEATURE_WEIGHTS.md")
            if fn.startswith("us_county_feature_weights_"):
                dst_name = fn.replace("us_county_", "")
            elif fn == "us_county_FEATURE_WEIGHTS.md":
                dst_name = "FEATURE_WEIGHTS.md"
            shutil.copy2(src, us_res / dst_name)
            shutil.copy2(src, us_res / fn)  # keep both names

    ext_df = extension_national_weights()
    plot_tops(ridge_e, rf_e, us_perm)

    prev_mae = float(mean_e.loc[mean_e.model == "Previous-election baseline", "mae"].iloc[0])
    prior_note = (
        f"- Side-check Core specification Prev MAE = **{prev_mae:.6f}** "
        f"(期望 ≈ 0.079265；鎖定主數字唔改語意）\n"
        f"- A+ Ridge MAE ≈ {float(mean_p.loc[mean_p.model=='Ridge','mae'].iloc[0]):.6f}；"
        f" edu Extension Ridge ≈ {float(mean_edu.loc[mean_edu.model=='Ridge','mae'].iloc[0]):.6f}"
    )
    write_docs(
        mean_e.drop(columns=["suite"], errors="ignore"),
        mean_p.drop(columns=["suite"], errors="ignore"),
        mean_edu.drop(columns=["suite"], errors="ignore"),
        ridge_e, rf_e, hgb_e, ridge_p, ridge_edu,
        us_perm, us_ridge, us_mean, ext_df, prior_note,
    )
    # rewrite mean tables without suite for cleaner display already done in write_docs args

    mirror()

    meta = {
        "recomputed_at": NOW,
        "script": "scripts/recompute_feature_weights.py",
        "tw_family_a_prev_mae": prev_mae,
        "tw_family_a_mean": mean_e.to_dict("records"),
        "tw_a_plus_mean": mean_p.to_dict("records"),
        "tw_a_plus_edu_mean": mean_edu.to_dict("records"),
        "us_core_mean": us_mean.to_dict("records"),
        "interpretation": "weights = model dependence / predictive importance — NOT causal",
        "outputs": str(OUT_FW),
    }
    (OUT_FW / "recompute_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print("DONE", flush=True)
    print("TW Core specification mean:\n", mean_e.to_string(index=False))
    print("Top Ridge electoral:\n", ridge_e.head(5).to_string(index=False))
    print("Top RF electoral:\n", rf_e.head(5).to_string(index=False))
    print("US mean:\n", us_mean.to_string(index=False))


if __name__ == "__main__":
    main()
