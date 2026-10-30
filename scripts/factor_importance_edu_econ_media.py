#!/usr/bin/env python3
"""Factor-group importance: Education vs Economic vs Media (Extension tables).

NOT Family A re-fit of locked headline MAE. Weights / ΔMAE = predictive dependence
in OOT / national leave-one-out — NOT causal.

Groups
------
Education (county): TW lag_education_rate; US education_rate
Economic (county): TW lag_unemployment_pct + lag_log_mean_income;
                   US unemployment + log_median_income
Economic (national fundamentals PROXY): unrate, cpi_yoy_pct, umcsent (where available)
Media (national Extension only): X camp sentiment gap; issue volume/sentiment
  — tiny n; honest limits

Outputs → results/factor_importance_edu_econ_media/
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/workspace/taiwan-election-data")
OUT = ROOT / "results" / "factor_importance_edu_econ_media"
OUT.mkdir(parents=True, exist_ok=True)
TZ = ZoneInfo("Asia/Hong_Kong")
NOW = datetime.now(TZ).strftime("%Y-%m-%d %H:%M Asia/Hong_Kong (UTC+8)")
SEED = 42

TW_TARGET = "target_DPP_vote_share"
TW_PREV = "prev_DPP_vote_share"
US_TARGET = "target_dem_two_party_share"
US_PREV = "prev_dem_two_party_share"

TW_EDU = ["lag_education_rate"]
TW_ECON = ["lag_unemployment_pct", "lag_log_mean_income"]
US_EDU = ["education_rate"]
US_ECON = ["unemployment", "log_median_income"]


def mae(y, p):
    return float(mean_absolute_error(y, p))


def impute(train, test, cols):
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


def ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def rf_tw():
    return RandomForestRegressor(
        n_estimators=200, min_samples_leaf=3, random_state=SEED, n_jobs=1
    )


def rf_us():
    return RandomForestRegressor(
        n_estimators=120, min_samples_leaf=8, max_depth=18, random_state=SEED, n_jobs=-1
    )


def county_group_backtest(frame, target, prev, groups: dict, test_years, make_rf, suite):
    """Expanding-window OOT: Prev-only vs Prev+group(s)."""
    need = [target, prev]
    for cols in groups.values():
        need.extend(cols)
    # keep rows with prev+target; impute group cols per fold
    frame = frame.dropna(subset=[target, prev]).sort_values(["year"]).copy()
    rows_mae, rows_coef, rows_perm, rows_group_perm = [], [], [], []

    for ty in test_years:
        train = frame[frame.year < ty]
        test = frame[frame.year == ty]
        if len(train) < 10 or len(test) < 5:
            continue
        y = test[target].to_numpy()
        prev_mae = mae(y, test[prev].to_numpy())
        rows_mae.append({
            "suite": suite, "grain": "county", "test_year": int(ty),
            "spec": "PreviousShare_only", "features": prev,
            "model": "baseline", "mae": prev_mae, "n_train": len(train), "n_test": len(test),
            "delta_mae_vs_prev": 0.0, "note": "baseline",
        })

        specs = {
            "Prev+Education": [prev] + groups["education"],
            "Prev+Economic": [prev] + groups["economic"],
            "Prev+Education+Economic": [prev] + groups["education"] + groups["economic"],
            "Education_only": groups["education"],
            "Economic_only": groups["economic"],
        }
        for spec_name, feats in specs.items():
            tr, te = impute(train, test, feats)
            # drop if all missing after impute somehow
            for model_name, model in [("Ridge", ridge()), ("RandomForest", make_rf())]:
                m = model.fit(tr[feats], tr[target])
                pred = m.predict(te[feats])
                m_mae = mae(y, pred)
                rows_mae.append({
                    "suite": suite, "grain": "county", "test_year": int(ty),
                    "spec": spec_name, "features": ",".join(feats),
                    "model": model_name, "mae": m_mae,
                    "n_train": len(tr), "n_test": len(te),
                    "delta_mae_vs_prev": m_mae - prev_mae,
                    "note": "negative delta = better than PreviousShare",
                })
                if model_name == "Ridge":
                    coefs = m[-1].coef_
                    for f, w in zip(feats, coefs):
                        group = (
                            "previous" if f == prev
                            else "education" if f in groups["education"]
                            else "economic" if f in groups["economic"]
                            else "other"
                        )
                        rows_coef.append({
                            "suite": suite, "test_year": int(ty), "spec": spec_name,
                            "feature": f, "factor_group": group,
                            "standardized_coef_signed": float(w),
                            "standardized_coef_abs": abs(float(w)),
                        })
                if model_name == "RandomForest" and spec_name == "Prev+Education+Economic":
                    pi = permutation_importance(
                        m, te[feats], y, scoring="neg_mean_absolute_error",
                        n_repeats=12, random_state=SEED + int(ty), n_jobs=1,
                    )
                    for j, f in enumerate(feats):
                        group = (
                            "previous" if f == prev
                            else "education" if f in groups["education"]
                            else "economic" if f in groups["economic"]
                            else "other"
                        )
                        rows_perm.append({
                            "suite": suite, "test_year": int(ty), "spec": spec_name,
                            "feature": f, "factor_group": group,
                            "permutation_mae_increase": float(pi.importances_mean[j]),
                            "permutation_std": float(pi.importances_std[j]),
                        })
                    # group-level: permute all features in group together
                    Xte = te[feats].to_numpy().copy()
                    base_pred = m.predict(te[feats])
                    base_mae = mae(y, base_pred)
                    for gname, gcols in {
                        "previous": [prev],
                        "education": groups["education"],
                        "economic": groups["economic"],
                    }.items():
                        idxs = [feats.index(c) for c in gcols if c in feats]
                        if not idxs:
                            continue
                        incs = []
                        rng = np.random.RandomState(SEED + int(ty) + hash(gname) % 1000)
                        for _ in range(12):
                            Xp = Xte.copy()
                            perm = rng.permutation(len(Xp))
                            for ix in idxs:
                                Xp[:, ix] = Xp[perm, ix]
                            incs.append(mae(y, m.predict(Xp)) - base_mae)
                        rows_group_perm.append({
                            "suite": suite, "test_year": int(ty),
                            "factor_group": gname,
                            "group_permutation_mae_increase_mean": float(np.mean(incs)),
                            "group_permutation_mae_increase_std": float(np.std(incs)),
                        })

    return (
        pd.DataFrame(rows_mae),
        pd.DataFrame(rows_coef),
        pd.DataFrame(rows_perm),
        pd.DataFrame(rows_group_perm),
    )


def national_media_and_fundamentals():
    """National Extension: media + econ fundamentals leave-one-out (tiny n)."""
    # National vote means from county panels (unweighted county mean — PROXY for climate)
    tw = pd.read_csv(ROOT / "data/county_panel_features_enriched_2000_2024.csv")
    us = pd.read_csv(ROOT / "fyp_us_core/us_county_panel_features_family_a.csv")
    tw_nat = tw.groupby("year", as_index=False).agg(
        vote=("target_DPP_vote_share", "mean"),
        prev=("prev_DPP_vote_share", "mean"),
    )
    tw_nat["country"] = "TW"
    us_nat = us.groupby("year", as_index=False).agg(
        vote=("target_dem_two_party_share", "mean"),
        prev=("prev_dem_two_party_share", "mean"),
    )
    us_nat["country"] = "US"

    fund = pd.read_csv(ROOT / "data/extension_must_have/national_fundamentals_enriched_extension.csv")
    fund = fund.rename(columns={"year": "year"})

    xs = pd.read_csv(ROOT / "data/x_sentiment_extension/x_entity_year_aggregates.csv")
    # camp gap: side mean sentiment (party+person) green/dem minus blue/gop
    camp_rows = []
    for (c, y), g in xs[xs["kind"].isin(["party", "person"])].groupby(["country", "election_year"]):
        # map sides
        if c == "TW":
            left = g[g["side"] == "DPP"]["mean_sentiment"]
            right = g[g["side"] == "KMT"]["mean_sentiment"]
        else:
            left = g[g["side"] == "DEM"]["mean_sentiment"]
            right = g[g["side"] == "GOP"]["mean_sentiment"]
        if len(left) == 0 or len(right) == 0:
            continue
        camp_rows.append({
            "country": c, "year": int(y),
            "media_camp_sentiment_gap": float(left.mean() - right.mean()),
            "media_n_posts_camp": int(g["n_posts"].sum()),
            "media_mean_sentiment_all": float(g["mean_sentiment"].mean()),
        })
    camp = pd.DataFrame(camp_rows)

    iss = pd.read_csv(ROOT / "data/extension_issue_news/issue_year_aggregates.csv")
    # top-issue sentiment + volume HHI-ish: mean sentiment of rank-1 issue; share of top issue
    top = iss.sort_values(["country", "year", "volume_rank"]).groupby(["country", "year"], as_index=False).first()
    top = top.rename(columns={
        "mean_sentiment": "media_top_issue_sentiment",
        "volume_share": "media_top_issue_volume_share",
        "n_posts": "media_top_issue_n_posts",
        "issue_en": "media_top_issue",
    })[["country", "year", "media_top_issue", "media_top_issue_sentiment",
        "media_top_issue_volume_share", "media_top_issue_n_posts"]]
    # also mean issue sentiment
    iss_mean = iss.groupby(["country", "year"], as_index=False).agg(
        media_issue_mean_sentiment=("mean_sentiment", "mean"),
        media_issue_total_posts=("n_posts", "sum"),
    )

    rows_assoc, rows_loo, rows_rank = [], [], []

    for country, nat in [("TW", tw_nat), ("US", us_nat)]:
        df = nat.merge(fund[fund.country == country][
            ["year", "unrate", "cpi_yoy_pct", "umcsent", "presidential_approval"]
        ], on="year", how="left")
        df = df.merge(camp[camp.country == country].drop(columns=["country"]), on="year", how="left")
        df = df.merge(top[top.country == country].drop(columns=["country"]), on="year", how="left")
        df = df.merge(iss_mean[iss_mean.country == country].drop(columns=["country"]), on="year", how="left")
        df.to_csv(OUT / f"national_merged_{country}.csv", index=False)

        # Factor associations (Pearson) — descriptive
        for feat, group in [
            ("unrate", "economic_national"),
            ("cpi_yoy_pct", "economic_national"),
            ("umcsent", "economic_national"),
            ("presidential_approval", "economic_national"),
            ("media_camp_sentiment_gap", "media"),
            ("media_mean_sentiment_all", "media"),
            ("media_top_issue_sentiment", "media"),
            ("media_top_issue_volume_share", "media"),
            ("media_issue_mean_sentiment", "media"),
            ("prev", "previous"),
        ]:
            sub = df[["vote", feat]].dropna()
            if len(sub) < 2:
                continue
            r = float(sub["vote"].corr(sub[feat]))
            rows_assoc.append({
                "country": country, "feature": feat, "factor_group": group,
                "n_years": len(sub), "pearson_r_vs_national_vote_PROXY": r,
                "years": ",".join(str(int(y)) for y in df.loc[sub.index, "year"]) if False else
                         ",".join(str(int(y)) for y in df.loc[df[feat].notna() & df["vote"].notna(), "year"]),
                "note": "county-mean national PROXY; correlational NOT causal; media n tiny",
            })

        # Leave-one-year MAE: Prev vs Prev+econ_nat vs Prev+media (Ridge alpha=1)
        def loo_specs(frame):
            specs = {
                "PreviousShare_only": ["prev"],
                "Prev+EconNational": ["prev", "unrate", "cpi_yoy_pct"],
                "EconNational_only": ["unrate", "cpi_yoy_pct"],
            }
            # umcsent mostly US
            if frame["umcsent"].notna().sum() >= 3:
                specs["Prev+EconNational+CCI"] = ["prev", "unrate", "cpi_yoy_pct", "umcsent"]
            if frame["media_camp_sentiment_gap"].notna().sum() >= 2:
                specs["Prev+MediaCampGap"] = ["prev", "media_camp_sentiment_gap"]
                specs["MediaCampGap_only"] = ["media_camp_sentiment_gap"]
            if frame["media_issue_mean_sentiment"].notna().sum() >= 2:
                specs["Prev+MediaIssueSent"] = ["prev", "media_issue_mean_sentiment"]
            return specs

        specs = loo_specs(df)
        for spec, feats in specs.items():
            use = df.dropna(subset=["vote"] + feats).copy()
            if len(use) < 2:
                continue
            abs_errs = []
            for i, row in use.iterrows():
                train = use.drop(index=i)
                test = use.loc[[i]]
                if len(train) < 1:
                    continue
                if spec == "PreviousShare_only":
                    pred = float(test["prev"].iloc[0])
                else:
                    # impute train median
                    tr, te = train.copy(), test.copy()
                    for c in feats:
                        med = tr[c].median()
                        tr[c] = tr[c].fillna(med)
                        te[c] = te[c].fillna(med)
                    m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                    m.fit(tr[feats], tr["vote"])
                    pred = float(m.predict(te[feats])[0])
                abs_errs.append(abs(float(test["vote"].iloc[0]) - pred))
            rows_loo.append({
                "country": country, "grain": "national_PROXY",
                "spec": spec, "features": ",".join(feats),
                "model": "baseline" if spec == "PreviousShare_only" else "Ridge_LOO",
                "n_years": len(use), "mae_loo": float(np.mean(abs_errs)),
                "note": "LEAVE-ONE-YEAR national; media/econ fundamentals Extension; NOT Family A; NOT causal",
            })

    assoc = pd.DataFrame(rows_assoc)
    loo = pd.DataFrame(rows_loo)

    # Ranking summary for national
    if not loo.empty:
        for country, g in loo.groupby("country"):
            base = g.loc[g["spec"] == "PreviousShare_only", "mae_loo"]
            base_v = float(base.iloc[0]) if len(base) else np.nan
            for _, r in g.iterrows():
                rows_rank.append({
                    "country": country, "level": "national_PROXY",
                    "spec": r["spec"], "mae": r["mae_loo"],
                    "delta_vs_prev": r["mae_loo"] - base_v if pd.notna(base_v) else np.nan,
                    "n": r["n_years"],
                    "factor_focus": (
                        "media" if "Media" in r["spec"]
                        else "economic" if "Econ" in r["spec"]
                        else "previous"
                    ),
                })
    return assoc, loo, pd.DataFrame(rows_rank), camp, top


def summarize_county(mae_df, coef_df, gperm_df, suite):
    mean_mae = (
        mae_df.groupby(["spec", "model"], as_index=False)["mae"]
        .mean()
        .sort_values("mae")
    )
    mean_delta = (
        mae_df[mae_df["model"] != "baseline"]
        .groupby(["spec", "model"], as_index=False)["delta_mae_vs_prev"]
        .mean()
        .sort_values("delta_mae_vs_prev")
    )
    # mean |coef| by factor group for Prev+Edu+Econ Ridge
    coef_sub = coef_df[coef_df["spec"] == "Prev+Education+Economic"]
    if not coef_sub.empty:
        coef_g = (
            coef_sub.groupby("factor_group", as_index=False)["standardized_coef_abs"]
            .mean()
            .sort_values("standardized_coef_abs", ascending=False)
        )
    else:
        coef_g = pd.DataFrame()
    if not gperm_df.empty:
        gperm_mean = (
            gperm_df.groupby("factor_group", as_index=False)["group_permutation_mae_increase_mean"]
            .mean()
            .sort_values("group_permutation_mae_increase_mean", ascending=False)
        )
    else:
        gperm_mean = pd.DataFrame()
    return mean_mae, mean_delta, coef_g, gperm_mean


def main():
    print("=== TW county Education vs Economic ===", flush=True)
    tw_edu = pd.read_csv(ROOT / "data/extension_education/county_panel_with_lag_education_rate.csv")
    # merge income/unemp from enriched if missing
    enr = pd.read_csv(ROOT / "data/county_panel_features_enriched_2000_2024.csv")
    for c in ["lag_unemployment_pct", "lag_log_mean_income", "prev_DPP_vote_share"]:
        if c not in tw_edu.columns:
            tw_edu = tw_edu.merge(
                enr[["region_id", "year", c]], on=["region_id", "year"], how="left", suffixes=("", "_e")
            )
    # ensure prev
    if "prev_DPP_vote_share" not in tw_edu.columns or tw_edu["prev_DPP_vote_share"].isna().all():
        tw_edu = tw_edu.drop(columns=[c for c in tw_edu.columns if c.endswith("_e")], errors="ignore")
        tw_edu = tw_edu.merge(
            enr[["region_id", "year", "prev_DPP_vote_share", "lag_unemployment_pct", "lag_log_mean_income"]],
            on=["region_id", "year"], how="left", suffixes=("", "_y")
        )
    # prefer non-null columns
    for c in TW_ECON + [TW_PREV]:
        alt = c + "_y"
        if alt in tw_edu.columns:
            tw_edu[c] = tw_edu[c].fillna(tw_edu[alt]) if c in tw_edu.columns else tw_edu[alt]

    tw_years = [2012, 2016, 2020, 2024]
    tw_mae, tw_coef, tw_perm, tw_gperm = county_group_backtest(
        tw_edu, TW_TARGET, TW_PREV,
        {"education": TW_EDU, "economic": TW_ECON},
        tw_years, rf_tw, "TW_county",
    )
    tw_mae.to_csv(OUT / "tw_county_spec_mae_by_year.csv", index=False)
    tw_coef.to_csv(OUT / "tw_county_ridge_coefs_by_year.csv", index=False)
    tw_perm.to_csv(OUT / "tw_county_permutation_by_feature.csv", index=False)
    tw_gperm.to_csv(OUT / "tw_county_permutation_by_group.csv", index=False)
    tw_mean, tw_delta, tw_coef_g, tw_gperm_m = summarize_county(tw_mae, tw_coef, tw_gperm, "TW")
    tw_mean.to_csv(OUT / "tw_county_mean_mae_by_spec.csv", index=False)
    tw_delta.to_csv(OUT / "tw_county_mean_delta_mae_vs_prev.csv", index=False)
    tw_coef_g.to_csv(OUT / "tw_county_mean_abs_ridge_coef_by_group.csv", index=False)
    tw_gperm_m.to_csv(OUT / "tw_county_mean_group_permutation.csv", index=False)

    print("=== US county Education vs Economic ===", flush=True)
    us = pd.read_csv(ROOT / "fyp_us_core/us_county_panel_features_family_a.csv")
    us_years = [y for y in sorted(us.year.unique()) if y >= 2012]
    us_mae, us_coef, us_perm, us_gperm = county_group_backtest(
        us, US_TARGET, US_PREV,
        {"education": US_EDU, "economic": US_ECON},
        us_years, rf_us, "US_county",
    )
    us_mae.to_csv(OUT / "us_county_spec_mae_by_year.csv", index=False)
    us_coef.to_csv(OUT / "us_county_ridge_coefs_by_year.csv", index=False)
    us_perm.to_csv(OUT / "us_county_permutation_by_feature.csv", index=False)
    us_gperm.to_csv(OUT / "us_county_permutation_by_group.csv", index=False)
    us_mean, us_delta, us_coef_g, us_gperm_m = summarize_county(us_mae, us_coef, us_gperm, "US")
    us_mean.to_csv(OUT / "us_county_mean_mae_by_spec.csv", index=False)
    us_delta.to_csv(OUT / "us_county_mean_delta_mae_vs_prev.csv", index=False)
    us_coef_g.to_csv(OUT / "us_county_mean_abs_ridge_coef_by_group.csv", index=False)
    us_gperm_m.to_csv(OUT / "us_county_mean_group_permutation.csv", index=False)

    print("=== National media + fundamentals Extension ===", flush=True)
    assoc, loo, rank, camp, top = national_media_and_fundamentals()
    assoc.to_csv(OUT / "national_factor_associations.csv", index=False)
    loo.to_csv(OUT / "national_loo_mae_by_spec.csv", index=False)
    rank.to_csv(OUT / "national_loo_ranking.csv", index=False)
    camp.to_csv(OUT / "media_camp_sentiment_gap.csv", index=False)
    top.to_csv(OUT / "media_top_issue_by_year.csv", index=False)

    # Overall ranking narrative table
    narrative = []
    # TW county: compare mean RF/Ridge delta for Prev+Edu vs Prev+Econ
    for label, mean_df, gperm_m, country in [
        ("TW_county", tw_mean, tw_gperm_m, "TW"),
        ("US_county", us_mean, us_gperm_m, "US"),
    ]:
        base = mean_df[(mean_df.spec == "PreviousShare_only") & (mean_df.model == "baseline")]
        base_mae = float(base.mae.iloc[0]) if len(base) else np.nan
        for spec in ["Prev+Education", "Prev+Economic", "Prev+Education+Economic",
                     "Education_only", "Economic_only"]:
            for model in ["Ridge", "RandomForest"]:
                sub = mean_df[(mean_df.spec == spec) & (mean_df.model == model)]
                if sub.empty:
                    continue
                narrative.append({
                    "country": country, "level": "county",
                    "spec": spec, "model": model,
                    "mean_mae": float(sub.mae.iloc[0]),
                    "delta_vs_PreviousShare": float(sub.mae.iloc[0] - base_mae),
                    "PreviousShare_mae": base_mae,
                    "interpretation": "ΔMAE>0 means worse than PreviousShare in OOT; NOT causal",
                })
        if not gperm_m.empty:
            for _, r in gperm_m.iterrows():
                narrative.append({
                    "country": country, "level": "county_group_permutation",
                    "spec": f"group:{r['factor_group']}", "model": "RandomForest",
                    "mean_mae": np.nan,
                    "delta_vs_PreviousShare": float(r["group_permutation_mae_increase_mean"]),
                    "PreviousShare_mae": base_mae,
                    "interpretation": "group permutation MAE increase on Prev+Edu+Econ RF; higher=more predictive dependence; NOT causal",
                })
    for _, r in rank.iterrows():
        narrative.append({
            "country": r["country"], "level": "national_PROXY",
            "spec": r["spec"], "model": "Ridge_LOO",
            "mean_mae": float(r["mae"]),
            "delta_vs_PreviousShare": float(r["delta_vs_prev"]) if pd.notna(r["delta_vs_prev"]) else np.nan,
            "PreviousShare_mae": np.nan,
            "interpretation": "national leave-one-year; media tiny n; NOT Family A; NOT causal",
        })
    narr = pd.DataFrame(narrative)
    narr.to_csv(OUT / "factor_importance_summary_table.csv", index=False)

    # Rank groups by evidence
    ranking_md = []
    ranking_md.append("# Education vs Economic vs Media — factor importance (Extension)\n")
    ranking_md.append(f"Built: {NOW}\n")
    ranking_md.append("**Weights / ΔMAE = predictive dependence in OOT or national LOO — NOT causal.**\n")
    ranking_md.append("Family A locked headline (Prev MAE 0.0793) is **unchanged**; this folder is Extension analysis for teacher feedback.\n")

    ranking_md.append("\n## 1. Taiwan county (OOT 2012–2024)\n")
    ranking_md.append("| spec | model | mean MAE | Δ vs PreviousShare |\n|---|---|---:|---:|\n")
    for _, r in tw_mean.iterrows():
        base = float(tw_mean[(tw_mean.spec == "PreviousShare_only")].mae.iloc[0])
        ranking_md.append(
            f"| {r['spec']} | {r['model']} | {r['mae']:.4f} | {r['mae']-base:+.4f} |\n"
        )
    if not tw_gperm_m.empty:
        ranking_md.append("\nGroup permutation (RF on Prev+Edu+Econ; MAE increase):\n\n")
        ranking_md.append("| factor_group | mean MAE increase |\n|---|---:|\n")
        for _, r in tw_gperm_m.iterrows():
            ranking_md.append(
                f"| {r['factor_group']} | {r['group_permutation_mae_increase_mean']:.4f} |\n"
            )
    if not tw_coef_g.empty:
        ranking_md.append("\nMean |standardized Ridge coef| by group (Prev+Edu+Econ):\n\n")
        ranking_md.append("| factor_group | mean |coef| |\n|---|---:|\n")
        for _, r in tw_coef_g.iterrows():
            ranking_md.append(f"| {r['factor_group']} | {r['standardized_coef_abs']:.4f} |\n")

    ranking_md.append("\n## 2. US county (OOT ≥2012)\n")
    ranking_md.append("| spec | model | mean MAE | Δ vs PreviousShare |\n|---|---|---:|---:|\n")
    for _, r in us_mean.iterrows():
        base = float(us_mean[(us_mean.spec == "PreviousShare_only")].mae.iloc[0])
        ranking_md.append(
            f"| {r['spec']} | {r['model']} | {r['mae']:.4f} | {r['mae']-base:+.4f} |\n"
        )
    if not us_gperm_m.empty:
        ranking_md.append("\nGroup permutation (RF on Prev+Edu+Econ; MAE increase):\n\n")
        ranking_md.append("| factor_group | mean MAE increase |\n|---|---:|\n")
        for _, r in us_gperm_m.iterrows():
            ranking_md.append(
                f"| {r['factor_group']} | {r['group_permutation_mae_increase_mean']:.4f} |\n"
            )
    if not us_coef_g.empty:
        ranking_md.append("\nMean |standardized Ridge coef| by group (Prev+Edu+Econ):\n\n")
        ranking_md.append("| factor_group | mean |coef| |\n|---|---:|\n")
        for _, r in us_coef_g.iterrows():
            ranking_md.append(f"| {r['factor_group']} | {r['standardized_coef_abs']:.4f} |\n")

    ranking_md.append("\n## 3. National media / fundamentals (Extension; LOO)\n")
    if not loo.empty:
        ranking_md.append("| country | spec | n_years | MAE_LOO |\n|---|---|---:|---:|\n")
        for _, r in loo.sort_values(["country", "mae_loo"]).iterrows():
            ranking_md.append(
                f"| {r['country']} | {r['spec']} | {r['n_years']} | {r['mae_loo']:.4f} |\n"
            )
    ranking_md.append(
        "\nMedia uses official X search pilot samples (small n posts/years). "
        "Do **not** treat Pearson≈1 on n=2 as evidence.\n"
    )

    ranking_md.append("\n## 4. Empirical ranking (honest)\n")
    ranking_md.append(
        "1. **Previous / partisan baseline** dominates county OOT in both TW and US "
        "(group permutation and |coef|).\n"
        "2. Between **education** and **economic** county features added to PreviousShare: "
        "compare ΔMAE and group permutation in tables above — typically both add little "
        "vs PreviousShare alone; relative rank may differ by country.\n"
        "3. **Media** is national Extension only; cannot enter county Family A; "
        "n_years≤2 for camp gap → descriptive / pilot only "
        "(cf. Brito et al. social-media prediction caution).\n"
    )

    (OUT / "README.md").write_text("".join(ranking_md), encoding="utf-8")

    summary = {
        "built_at": NOW,
        "family_a_locked": True,
        "family_a_prev_mae_tw": 0.0793,
        "outputs": sorted(p.name for p in OUT.glob("*")),
        "disclaimer": "predictive dependence / OOT ΔMAE — NOT causal",
    }
    # add quick ranks from group perm
    if not tw_gperm_m.empty:
        summary["tw_county_group_perm_rank"] = tw_gperm_m.to_dict(orient="records")
    if not us_gperm_m.empty:
        summary["us_county_group_perm_rank"] = us_gperm_m.to_dict(orient="records")
    summary["tw_county_mean_mae"] = tw_mean.to_dict(orient="records")
    summary["us_county_mean_mae"] = us_mean.to_dict(orient="records")
    summary["national_loo"] = loo.to_dict(orient="records") if not loo.empty else []
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])
    print("DONE →", OUT)


if __name__ == "__main__":
    main()
