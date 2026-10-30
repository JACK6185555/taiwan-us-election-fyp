"""Fast finish: feature weights + conformal + Ridge-only ablation + docs. Uses existing backtest metrics."""
from __future__ import annotations
import json, shutil
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

DATA = Path("/workspace/taiwan-election-data")
OUT = DATA / "fyp_us_core"
RES = OUT / "results"
CH = OUT / "charts"
GH = Path("/workspace/taiwan-election-github")
MIRROR = GH / "results/family_a_us_county_presidential"
PANEL = OUT / "us_county_panel_features_family_a.csv"
SEED = 42
TARGET = "target_dem_two_party_share"
ANCHOR = "prev_dem_two_party_share"
TEST_YEARS = [2012, 2016, 2020, 2024]
FEATURES = [
    "prev_dem_two_party_share", "prev_gop_two_party_share",
    "prev2_dem_two_party_share", "prev2_gop_two_party_share",
    "prev_log_total_votes", "dem_change_1", "gop_change_1",
    "prev_dem_gop_margin", "dem_gop_margin_change_1", "abs_dem_change_1",
    "log_votes_change_1", "unemployment", "education_rate",
    "log_median_income", "log_density",
]
LABELS = {
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
MODELS = ["Previous-election baseline", "Ridge", "Random Forest", "HistGradientBoosting"]

def ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
def rf():
    return RandomForestRegressor(n_estimators=120, min_samples_leaf=8, max_depth=18, random_state=SEED, n_jobs=-1)
def hgb():
    return HistGradientBoostingRegressor(max_iter=120, learning_rate=0.07, max_leaf_nodes=31, min_samples_leaf=25, l2_regularization=1.0, random_state=SEED)

def scores(y, p):
    return {"mae": float(mean_absolute_error(y, p)), "rmse": float(np.sqrt(mean_squared_error(y, p)))}

def main():
    RES.mkdir(parents=True, exist_ok=True); CH.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(PANEL)
    metrics = pd.read_csv(RES / "backtest_metrics.csv")
    mean_metrics = pd.read_csv(RES / "family_a_us_mean_metrics.csv")
    mean_metrics.to_csv(OUT / "family_a_us_mean_metrics.csv", index=False, encoding="utf-8-sig")

    # charts if missing mae already exists
    for metric, title, fn in [("mae","US county OOT MAE by election year","backtest_mae.png"),
                              ("rmse","US county OOT RMSE by election year","backtest_rmse.png")]:
        pivot = metrics.pivot(index="test_year", columns="model", values=metric).reindex(columns=MODELS)
        ax = pivot.plot(kind="bar", figsize=(9,4.8), width=.78, color=["#555555","#2878B5","#E07A2D","#4B9B70"])
        ax.set_title(title, loc="left"); ax.set_xlabel("Test election year"); ax.set_ylabel(metric.upper())
        ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", color="#e5e5e5"); ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False); plt.xticks(rotation=0); plt.tight_layout()
        plt.savefig(CH/fn, dpi=180); plt.close()

    print("weights...", flush=True)
    train = frame[frame.year < 2024]
    test = frame[frame.year == 2024].sample(n=min(800, (frame.year==2024).sum()), random_state=SEED)
    y_te = test[TARGET].to_numpy()
    weight_rows = []
    m_ridge = ridge().fit(train[FEATURES], train[TARGET])
    m_rf = rf().fit(train[FEATURES], train[TARGET])
    m_hgb = hgb().fit(train[FEATURES], train[TARGET])
    print("  fitted ridge/rf/hgb", flush=True)
    for feat, c in zip(FEATURES, m_ridge[-1].coef_):
        weight_rows.append(dict(source="refit_train_lt_2024", model="Ridge", feature=feat, feature_label=LABELS[feat],
            weight_type="standardized_coefficient_signed", weight=float(c), abs_weight=abs(float(c)),
            note="signed; + => higher Dem share prediction after z-score; not causal"))
    for feat, w in zip(FEATURES, m_rf.feature_importances_):
        weight_rows.append(dict(source="refit_train_lt_2024", model="Random Forest", feature=feat, feature_label=LABELS[feat],
            weight_type="impurity_importance_mdi", weight=float(w), abs_weight=float(w),
            note="MDI impurity; not causal"))
    print("  permutation...", flush=True)
    for name, fitted in [("Ridge", m_ridge), ("Random Forest", m_rf), ("HistGradientBoosting", m_hgb)]:
        pi = permutation_importance(fitted, test[FEATURES], y_te, scoring="neg_mean_absolute_error",
                                    n_repeats=10, random_state=SEED, n_jobs=1)
        for feat, m, s in zip(FEATURES, pi.importances_mean, pi.importances_std):
            weight_rows.append(dict(source="perm_on_2024_subsample", model=name, feature=feat, feature_label=LABELS[feat],
                weight_type="permutation_mae_increase", weight=float(m), abs_weight=abs(float(m)),
                note=f"perm_std={float(s):.6g}; n={len(test)}; not causal"))
    weights = pd.DataFrame(weight_rows)
    def norm(g):
        s = g["abs_weight"].sum(); g=g.copy()
        g["normalized_abs"] = g["abs_weight"]/s if s>0 else 0.0
        return g
    weights = weights.groupby(["source","model","weight_type"], group_keys=False).apply(norm)
    weights = weights.sort_values(["source","model","weight_type","abs_weight"], ascending=[True,True,True,False])
    weights.to_csv(RES/"feature_weights_all.csv", index=False, encoding="utf-8-sig")
    ridge_s = weights[(weights.model=="Ridge")&(weights.weight_type=="standardized_coefficient_signed")]
    rf_mdi = weights[(weights.model=="Random Forest")&(weights.weight_type=="impurity_importance_mdi")]
    perm = weights[weights.weight_type=="permutation_mae_increase"]
    ridge_s.to_csv(RES/"feature_weights_ridge_signed.csv", index=False, encoding="utf-8-sig")
    rf_mdi.to_csv(RES/"feature_weights_rf_impurity.csv", index=False, encoding="utf-8-sig")
    perm.to_csv(RES/"feature_weights_permutation.csv", index=False, encoding="utf-8-sig")
    # also as "importance summary" fold-proxy: use perm table ranks
    imp_sum_rows = []
    for name in ["Ridge","Random Forest","HistGradientBoosting"]:
        sub = perm[perm.model==name].copy()
        sub = sub.sort_values("abs_weight", ascending=False)
        for rank, (_, r) in enumerate(sub.iterrows(), 1):
            imp_sum_rows.append(dict(model=name, feature=r.feature, feature_label=r.feature_label,
                importance_mean=r.weight, rank=rank, source="perm_2024_subsample"))
    # ridge abs coef ranks
    for rank, (_, r) in enumerate(ridge_s.sort_values("abs_weight", ascending=False).iterrows(), 1):
        imp_sum_rows.append(dict(model="Ridge standardized coefficient", feature=r.feature, feature_label=r.feature_label,
            importance_mean=r.abs_weight, rank=rank, source="refit_lt_2024"))
    pd.DataFrame(imp_sum_rows).to_csv(RES/"feature_importance_summary.csv", index=False, encoding="utf-8-sig")
    perm.to_csv(RES/"feature_importance_by_year.csv", index=False, encoding="utf-8-sig")  # proxy note in README

    def tops(df, k=8):
        lines=[]
        for _, r in df.head(k).iterrows():
            lines.append(f"- **{r.feature_label}** (`{r.feature}`): weight={r.weight:.4g}, |w| share={r.normalized_abs:.1%}")
        return "\n".join(lines)
    md = ["# US Core specification — Feature weights (model dependence, not causal)\n",
          "Not causal. Socio may be wave-static. No county `poll_index` (Guide gap — document only).\n",
          "## Ridge standardized coefficients (signed, train < 2024)\n",
          tops(ridge_s.sort_values("abs_weight", ascending=False)),
          "\n## Random Forest MDI impurity (train < 2024)\n",
          tops(rf_mdi.sort_values("abs_weight", ascending=False)),
          "\n## Permutation MAE ↑ on 2024 subsample\n"]
    for name in ["Ridge","Random Forest","HistGradientBoosting"]:
        md.append(f"\n### {name}\n"); md.append(tops(perm[perm.model==name].sort_values("abs_weight", ascending=False)))
    md.append("\n## Files\n- feature_weights_all.csv\n- feature_weights_ridge_signed.csv\n- feature_weights_rf_impurity.csv\n- feature_weights_permutation.csv\n")
    md.append("\n## Socio / polls gap\n- Census ACS API requires key (blocked).\n- ERS/BLS bulk URLs 404 or Access Denied from this host.\n- Panel keeps ERS **yearly unemployment**; income/education/density largely wave-static.\n- **No county poll_index** constructed — would invent polls; gap documented.\n")
    (RES/"FEATURE_WEIGHTS.md").write_text("\n".join(md), encoding="utf-8")
    (OUT/"FEATURE_WEIGHTS.md").write_text("\n".join(md), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8,5.5))
    d = ridge_s.sort_values("weight")
    ax.barh(d.feature_label, d.weight, color=["#c0392b" if v<0 else "#2878B5" for v in d.weight])
    ax.axvline(0, color="#333", lw=.8); ax.set_title("Ridge standardized coefficients (signed) — US Core specification", loc="left")
    ax.set_xlabel("Standardized coefficient"); ax.grid(axis="x", color="#e5e5e5"); ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False); plt.tight_layout(); plt.savefig(CH/"feature_weights_ridge_signed.png", dpi=180); plt.close()

    fig, axes = plt.subplots(1,3, figsize=(14,5))
    for ax, name, color in zip(axes, ["Ridge","Random Forest","HistGradientBoosting"], ["#397AA8","#D1813A","#4B9B70"]):
        d = perm[perm.model==name].sort_values("weight").tail(10)
        ax.barh(d.feature_label, d.weight, color=color); ax.set_title(name, loc="left")
        ax.set_xlabel("Perm MAE increase"); ax.grid(axis="x", color="#e5e5e5"); ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("US permutation importance 2024 subsample — not causal", x=.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(CH/"feature_importance_models.png", dpi=180); plt.close()
    print("weights on disk", flush=True)

    # Ridge-only ablation (fast)
    print("ablation ridge/hgb...", flush=True)
    sets = {"Previous-election only":[ANCHOR], "All features":FEATURES,
            "History only":[c for c in FEATURES if c not in ("unemployment","education_rate","log_median_income","log_density")],
            "All minus socio":[c for c in FEATURES if c not in ("unemployment","education_rate","log_median_income","log_density")],
            "All minus previous election":[c for c in FEATURES if c not in ("prev_dem_two_party_share","prev_gop_two_party_share","prev_log_total_votes")]}
    rows=[]
    for ty in TEST_YEARS:
        tr, te = frame[frame.year<ty], frame[frame.year==ty]
        y = te[TARGET].to_numpy()
        for sn, cols in sets.items():
            for name, factory in [("Ridge", ridge), ("HistGradientBoosting", hgb)]:
                p = factory().fit(tr[cols], tr[TARGET]).predict(te[cols])
                rows.append({"test_year":ty,"model":name,"feature_set":sn,"n_features":len(cols), **scores(y,p)})
    abl = pd.DataFrame(rows)
    abl.to_csv(RES/"feature_ablation_by_year.csv", index=False, encoding="utf-8-sig")
    abls = abl.groupby(["model","feature_set","n_features"], as_index=False)[["mae","rmse"]].mean()
    abls.to_csv(RES/"feature_ablation_summary.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1,2, figsize=(12,5), sharey=True)
    for ax, name in zip(axes, ["Ridge","HistGradientBoosting"]):
        d = abls[abls.model==name].sort_values("mae")
        ax.barh(d.feature_set, d.mae, color="#438B78"); ax.set_title(name, loc="left"); ax.set_xlabel("Mean MAE")
        ax.grid(axis="x", color="#e5e5e5"); ax.set_axisbelow(True); ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("US ablation (Ridge/HGB) — not causal", x=.01, ha="left"); fig.tight_layout()
    fig.savefig(CH/"feature_ablation.png", dpi=180); plt.close()

    # Conformal
    print("conformal...", flush=True)
    alpha=0.1
    crow=[]; iv=[]
    for ty in TEST_YEARS:
        prior = sorted(y for y in frame.year.unique() if y < ty)
        if len(prior)<2: continue
        cy = prior[-1]
        tr, ca, te = frame[frame.year<cy], frame[frame.year==cy], frame[frame.year==ty]
        for name, factory in [("Previous-election baseline", None), ("Ridge", ridge), ("Random Forest", rf), ("HistGradientBoosting", hgb)]:
            if name.startswith("Previous"):
                resid = np.abs(ca[TARGET]-ca[ANCHOR]).to_numpy(); pred = te[ANCHOR].to_numpy()
            else:
                fit = factory().fit(tr[FEATURES], tr[TARGET])
                resid = np.abs(ca[TARGET].to_numpy()-fit.predict(ca[FEATURES]))
                pred = fit.predict(te[FEATURES])
            n=len(resid); q=float(np.quantile(resid, min(1.0, np.ceil((n+1)*(1-alpha))/n), method="higher"))
            actual=te[TARGET].to_numpy(); lo, hi = pred-q, pred+q
            cov = float(((actual>=lo)&(actual<=hi)).mean())
            crow.append(dict(test_year=int(ty), calib_year=int(cy), model=name, alpha=alpha, target_coverage=1-alpha,
                             empirical_coverage=cov, avg_width=float(np.mean(hi-lo)), quantile=q,
                             n_train=len(tr), n_calib=len(ca), n_test=len(te)))
            if ty==2024:
                chunk=te[["region_id","region_name"]].copy()
                chunk["test_year"]=2024; chunk["calib_year"]=int(cy); chunk["model"]=name
                chunk["actual"]=actual; chunk["prediction"]=pred; chunk["lower"]=lo; chunk["upper"]=hi
                chunk["width"]=hi-lo; chunk["covered"]=(actual>=lo)&(actual<=hi)
                iv.append(chunk)
        print(f"  conformal {ty} done", flush=True)
    conf=pd.DataFrame(crow)
    conf.to_csv(RES/"conformal_coverage_by_year.csv", index=False, encoding="utf-8-sig")
    conf.groupby("model", as_index=False).agg(mean_coverage=("empirical_coverage","mean"), mean_width=("avg_width","mean"),
        min_coverage=("empirical_coverage","min"), max_coverage=("empirical_coverage","max")).to_csv(
        RES/"conformal_coverage_summary.csv", index=False, encoding="utf-8-sig")
    if iv: pd.concat(iv, ignore_index=True).to_csv(RES/"conformal_intervals_2024.csv", index=False, encoding="utf-8-sig")
    (RES/"conformal_meta.json").write_text(json.dumps({
        "method":"split conformal; calib=prior election; train<calib",
        "alpha":alpha,"target_coverage":0.9,
        "by_model_mean": conf.groupby("model")["empirical_coverage"].mean().to_dict(),
        "note":"If coverage << 0.9, miscalibrated (honest like TW RQ4).",
    }, indent=2)+"\n", encoding="utf-8")
    pivot=conf.pivot(index="test_year", columns="model", values="empirical_coverage")
    ax=pivot.reindex(columns=[c for c in MODELS if c in pivot.columns]).plot(kind="bar", figsize=(9,4.5), width=.78,
        color=["#555555","#2878B5","#E07A2D","#4B9B70"])
    ax.axhline(0.9, color="#c0392b", ls="--", lw=1); ax.set_ylim(0,1.05)
    ax.set_title("US split-conformal empirical coverage (target 0.90)", loc="left")
    ax.set_ylabel("Coverage"); ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", color="#e5e5e5")
    ax.set_axisbelow(True); ax.spines[["top","right"]].set_visible(False); plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(CH/"conformal_coverage.png", dpi=180); plt.close()

    cov_mean = conf.groupby("model")["empirical_coverage"].mean()
    payload = {
        "family":"US Core specification", "target":TARGET, "unit":"US county; Dem two-party share",
        "test_years":TEST_YEARS, "mean_metrics": mean_metrics.to_dict("records"),
        "conformal_mean_coverage": cov_mean.to_dict(),
        "poll_index":"NOT AVAILABLE — no county poll series; gap documented (do not invent)",
        "socio_gap":"ACS key required; ERS/BLS bulk blocked; yearly unemployment retained; income/edu/density wave-static",
    }
    (RES/"summary.json").write_text(json.dumps(payload, indent=2, default=str)+"\n", encoding="utf-8")

    # README
    mm = mean_metrics.set_index("model")
    conf_lines = "\n".join(f"| {m} | {v:.4f} |" for m,v in cov_mean.items())
    readme = f"""# US Core specification — County Presidential Core

| Item | Value |
|---|---|
| Unit | US county (~3100) |
| Target | Dem **two-party** share |
| Test years | 2012–2024 OOT expanding window |

## Mean OOT MAE

| Model | MAE | RMSE |
|---|---:|---:|
| Previous-election baseline | {mm.loc['Previous-election baseline','mae_4dp']} | {mm.loc['Previous-election baseline','rmse_4dp']} |
| Random Forest | {mm.loc['Random Forest','mae_4dp']} | {mm.loc['Random Forest','rmse_4dp']} |
| Ridge | {mm.loc['Ridge','mae_4dp']} | {mm.loc['Ridge','rmse_4dp']} |
| HistGradientBoosting | {mm.loc['HistGradientBoosting','mae_4dp']} | {mm.loc['HistGradientBoosting','rmse_4dp']} |

## Conformal coverage (target 0.90)

| Model | Mean coverage |
|---|---:|
{conf_lines}

## Feature weights
See `FEATURE_WEIGHTS.md` + `results/feature_weights_*.csv`.

## Gaps
- No county `poll_index` (not invented).
- ACS API key required; ERS/BLS bulk fetch failed from this host.
- Yearly unemployment OK; income/edu/density wave-static.

Weights/ablation/importance = **model dependence**, not causal.
"""
    (OUT/"README.md").write_text(readme, encoding="utf-8")
    (RES/"REPRODUCIBILITY.md").write_text("# US Core specification\n\nScripts: us_core_backtest.py, us_core_weights_conformal.py\n", encoding="utf-8")

    # DATA_GAPS
    (OUT/"DATA_GAPS.md").write_text(
        "# US Core data gaps\n\n"
        "- **poll_index**: none at county level in pack — not invented.\n"
        "- **ACS yearly income/education/density**: Census API redirects to missing_key.\n"
        "- **ERS/BLS bulk**: USDA pages 404/antibot; BLS download Access Denied.\n"
        "- Retained: MEDSL votes; ERS yearly unemployment already in panel.\n", encoding="utf-8")

    # mirror
    MIRROR.mkdir(parents=True, exist_ok=True); (MIRROR/"charts").mkdir(exist_ok=True)
    for p in list(RES.glob("*"))+[OUT/"README.md", OUT/"FEATURE_WEIGHTS.md", OUT/"family_a_us_mean_metrics.csv", OUT/"DATA_GAPS.md"]:
        if p.is_file(): shutil.copy2(p, MIRROR/p.name)
    for p in CH.glob("*"):
        if p.is_file(): shutil.copy2(p, MIRROR/"charts"/p.name)
    (GH/"scripts").mkdir(exist_ok=True)
    for s in ["us_core_backtest.py","us_core_weights_conformal.py"]:
        src = OUT/"scripts"/s
        if src.exists(): shutil.copy2(src, GH/"scripts"/s)
    if PANEL.exists():
        shutil.copy2(PANEL, GH/"data"/"us_county_panel_features_family_a.csv")

    print("DONE")
    print(mean_metrics.to_string(index=False))
    print("coverage:\n", cov_mean.to_string())
    print("top ridge:\n", ridge_s.sort_values("abs_weight", ascending=False).head(5)[["feature","weight","normalized_abs"]].to_string(index=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
