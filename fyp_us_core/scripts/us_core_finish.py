"""Finish US Core specification after backtest metrics already written: charts, ablation (light),
feature weights, conformal, README, mirror."""
from __future__ import annotations

import json
import shutil
import sys
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

# import constants/helpers from main script
sys.path.insert(0, str(Path(__file__).resolve().parent))
import us_core_backtest as core

OUT, CHARTS, RESULTS, MIRROR = core.OUT, core.CHARTS, core.RESULTS, core.MIRROR
FEATURES, LABELS, GROUPS, MODELS = core.FEATURES, core.LABELS, core.GROUPS, core.MODELS
TARGET, ANCHOR, TEST_YEARS, SEED = core.TARGET, core.ANCHOR, core.TEST_YEARS, core.SEED


def model_light(name: str):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=80, min_samples_leaf=8, random_state=SEED, n_jobs=-1, max_depth=20,
        )
    return HistGradientBoostingRegressor(
        max_iter=100, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=30, l2_regularization=1.0, random_state=SEED,
    )


def scores(actual, pred):
    return {
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(OUT / "us_county_panel_features_family_a.csv")
    metrics = pd.read_csv(RESULTS / "backtest_metrics.csv")
    mean_metrics = pd.read_csv(RESULTS / "family_a_us_mean_metrics.csv")
    core.plot_metrics(metrics, CHARTS)

    # Light ablation: Ridge + HGB full; RF only on key sets
    print("ablation light...", flush=True)
    sets = {"Previous-election only": [ANCHOR], "All features": FEATURES}
    for group, cols in GROUPS.items():
        sets[f"All minus {group}"] = [c for c in FEATURES if c not in cols]
    ablation_rows = []
    for test_year in TEST_YEARS:
        train, test = frame[frame.year < test_year], frame[frame.year == test_year]
        y = test[TARGET].to_numpy()
        for set_name, cols in sets.items():
            for name in ["Ridge", "HistGradientBoosting"]:
                pred = model_light(name).fit(train[cols], train[TARGET]).predict(test[cols])
                ablation_rows.append({
                    "test_year": int(test_year), "model": name, "feature_set": set_name,
                    "n_features": len(cols), "rf_note": "RF ablation skipped for runtime; see weights",
                    **scores(y, pred),
                })
            # RF only for All features + Previous-election only
            if set_name in ("Previous-election only", "All features"):
                name = "Random Forest"
                pred = model_light(name).fit(train[cols], train[TARGET]).predict(test[cols])
                ablation_rows.append({
                    "test_year": int(test_year), "model": name, "feature_set": set_name,
                    "n_features": len(cols), "rf_note": "light RF (80 trees)",
                    **scores(y, pred),
                })
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(RESULTS / "feature_ablation_by_year.csv", index=False, encoding="utf-8-sig")
    ablation_summary = ablation.groupby(
        ["model", "feature_set", "n_features"], as_index=False,
    )[["mae", "rmse"]].mean()
    ablation_summary.to_csv(RESULTS / "feature_ablation_summary.csv", index=False, encoding="utf-8-sig")
    models_plot = ["Ridge", "Random Forest", "HistGradientBoosting"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=False)
    for ax, name in zip(axes, models_plot):
        d = ablation_summary[ablation_summary.model == name].sort_values("mae")
        if d.empty:
            ax.set_title(name + " (no rows)"); continue
        ax.barh(d.feature_set, d.mae, color="#438B78")
        ax.set_title(name, loc="left"); ax.set_xlabel("Mean MAE")
        ax.grid(axis="x", color="#e5e5e5"); ax.set_axisbefore = True
        ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "US feature ablation (OOT, light): mean MAE — not causal (RF only key sets)",
        x=0.01, ha="left", fontsize=11,
    )
    fig.tight_layout(); fig.savefig(CHARTS / "feature_ablation.png", dpi=180); plt.close(fig)

    # Feature weights (refit + permutation)
    print("feature_weights...", flush=True)
    train_refit = frame[frame.year < 2024]
    weight_rows = []
    ridge = core.model("Ridge").fit(train_refit[FEATURES], train_refit[TARGET])
    for feat, c in zip(FEATURES, ridge[-1].coef_):
        weight_rows.append({
            "source": "refit_train_lt_2024", "model": "Ridge", "feature": feat,
            "feature_label": LABELS[feat], "weight_type": "standardized_coefficient_signed",
            "weight": float(c), "abs_weight": abs(float(c)), "normalized_abs": np.nan,
            "note": "sign: + raises predicted Dem share when feature increases (z-scored)",
        })
    rf = core.model("Random Forest").fit(train_refit[FEATURES], train_refit[TARGET])
    for feat, w in zip(FEATURES, rf.feature_importances_):
        weight_rows.append({
            "source": "refit_train_lt_2024", "model": "Random Forest", "feature": feat,
            "feature_label": LABELS[feat], "weight_type": "impurity_importance_mdi",
            "weight": float(w), "abs_weight": float(w), "normalized_abs": np.nan,
            "note": "non-negative MDI; model dependence not causal",
        })
    hgb = core.model("HistGradientBoosting").fit(train_refit[FEATURES], train_refit[TARGET])
    test_2024 = frame[frame.year == 2024]
    test_w = test_2024.sample(n=min(1000, len(test_2024)), random_state=SEED)
    y_w = test_w[TARGET].to_numpy()
    for name, fitted in [("Ridge", ridge), ("Random Forest", rf), ("HistGradientBoosting", hgb)]:
        pi = permutation_importance(
            fitted, test_w[FEATURES], y_w, scoring="neg_mean_absolute_error",
            n_repeats=12, random_state=SEED, n_jobs=1,
        )
        for feat, m, s in zip(FEATURES, pi.importances_mean, pi.importances_std):
            weight_rows.append({
                "source": "perm_on_2024_subsample", "model": name, "feature": feat,
                "feature_label": LABELS[feat], "weight_type": "permutation_mae_increase",
                "weight": float(m), "abs_weight": abs(float(m)), "normalized_abs": np.nan,
                "note": f"perm_std={float(s):.6g}; n_test={len(test_w)}; not causal",
            })
    # also fold-averaged perm importance (lighter: 5 repeats, 500 subsample) for summary parity with TW
    print("importance folds...", flush=True)
    importance_rows = []
    for test_year in TEST_YEARS:
        train, test = frame[frame.year < test_year], frame[frame.year == test_year]
        test_imp = test.sample(n=min(500, len(test)), random_state=SEED + int(test_year))
        y = test_imp[TARGET].to_numpy()
        for name in MODELS[1:]:
            fitted = model_light(name).fit(train[FEATURES], train[TARGET])
            result = permutation_importance(
                fitted, test_imp[FEATURES], y, scoring="neg_mean_absolute_error",
                n_repeats=5, random_state=SEED + int(test_year), n_jobs=1,
            )
            for j, feat in enumerate(FEATURES):
                importance_rows.append({
                    "test_year": int(test_year), "model": name, "feature": feat,
                    "feature_label": LABELS[feat],
                    "permutation_mae_increase": float(result.importances_mean[j]),
                    "permutation_std": float(result.importances_std[j]),
                    "n_test_subsample": int(len(test_imp)),
                })
            if name == "Ridge":
                coef = fitted[-1].coef_
                for feat, weight in zip(FEATURES, coef):
                    importance_rows.append({
                        "test_year": int(test_year),
                        "model": "Ridge standardized coefficient",
                        "feature": feat, "feature_label": LABELS[feat],
                        "permutation_mae_increase": abs(float(weight)),
                        "permutation_std": np.nan,
                        "n_test_subsample": int(len(test_imp)),
                    })
    imp = pd.DataFrame(importance_rows)
    imp.to_csv(RESULTS / "feature_importance_by_year.csv", index=False, encoding="utf-8-sig")
    summary = (
        imp.groupby(["model", "feature", "feature_label"], as_index=False)
        .agg(
            importance_mean=("permutation_mae_increase", "mean"),
            importance_std_across_folds=("permutation_mae_increase", "std"),
        )
        .sort_values(["model", "importance_mean"], ascending=[True, False])
    )
    summary["rank"] = summary.groupby("model")["importance_mean"].rank(
        method="min", ascending=False,
    ).astype(int)
    summary.to_csv(RESULTS / "feature_importance_summary.csv", index=False, encoding="utf-8-sig")

    weights = pd.DataFrame(weight_rows)

    def _norm(g):
        s = g["abs_weight"].sum()
        g = g.copy()
        g["normalized_abs"] = g["abs_weight"] / s if s > 0 else 0.0
        return g

    weights = weights.groupby(["source", "model", "weight_type"], group_keys=False).apply(_norm)
    weights = weights.sort_values(
        ["source", "model", "weight_type", "abs_weight"], ascending=[True, True, True, False],
    )
    weights.to_csv(RESULTS / "feature_weights_all.csv", index=False, encoding="utf-8-sig")
    ridge_signed = weights[
        (weights.model == "Ridge") & (weights.weight_type == "standardized_coefficient_signed")
    ].copy()
    ridge_signed.to_csv(RESULTS / "feature_weights_ridge_signed.csv", index=False, encoding="utf-8-sig")
    rf_mdi = weights[
        (weights.model == "Random Forest") & (weights.weight_type == "impurity_importance_mdi")
    ].copy()
    rf_mdi.to_csv(RESULTS / "feature_weights_rf_impurity.csv", index=False, encoding="utf-8-sig")
    perm = weights[weights.weight_type == "permutation_mae_increase"].copy()
    perm.to_csv(RESULTS / "feature_weights_permutation.csv", index=False, encoding="utf-8-sig")

    def top_lines(df, k=8):
        lines = []
        for _, r in df.head(k).iterrows():
            lines.append(
                f"- **{r.feature_label}** (`{r.feature}`): weight={r.weight:.4g}, "
                f"|w| share={r.normalized_abs:.1%}"
            )
        return "\n".join(lines)

    md = [
        "# US Core specification — Feature weights (model dependence, not causal)\n",
        "Weights describe how each fitted model **uses** features. **Not** causal voter effects.\n",
        "## Ridge standardized coefficients (signed, train years < 2024)\n",
        "Positive ⇒ higher feature → higher predicted Dem two-party share (after z-scoring).\n",
        top_lines(ridge_signed.sort_values("abs_weight", ascending=False)),
        "\n## Random Forest impurity importance (MDI, train < 2024)\n",
        top_lines(rf_mdi.sort_values("abs_weight", ascending=False)),
        "\n## Permutation MAE increase on 2024 subsample (n≤1000)\n",
    ]
    for name in ["Ridge", "Random Forest", "HistGradientBoosting"]:
        sub = perm[perm.model == name].sort_values("abs_weight", ascending=False)
        md.append(f"\n### {name}\n")
        md.append(top_lines(sub))
    md.append("\n## Files\n")
    md.append(
        "- `feature_weights_all.csv`\n"
        "- `feature_weights_ridge_signed.csv`\n"
        "- `feature_weights_rf_impurity.csv`\n"
        "- `feature_weights_permutation.csv`\n"
        "- `feature_importance_summary.csv` (fold-averaged perm)\n"
    )
    md.append(
        "\n**Data gap:** Census ACS API now requires a key (HTTP 302 → missing_key). "
        "Socio columns keep ERS yearly unemployment + wave-static ACS/ERS income/density/education. "
        "No yearly ACS refit without a key.\n"
    )
    md.append("\n**Disclaimer:** Do not treat ranks as causes of Dem share.\n")
    (RESULTS / "FEATURE_WEIGHTS.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "FEATURE_WEIGHTS.md").write_text("\n".join(md), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    d = ridge_signed.sort_values("weight")
    colors = ["#c0392b" if v < 0 else "#2878B5" for v in d.weight]
    ax.barh(d.feature_label, d.weight, color=colors)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_title("Ridge standardized coefficients (signed) — US Core specification", loc="left")
    ax.set_xlabel("Standardized coefficient")
    ax.grid(axis="x", color="#e5e5e5"); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(CHARTS / "feature_weights_ridge_signed.png", dpi=180); plt.close()

    # importance chart
    plot_models = ["Ridge", "Ridge standardized coefficient", "Random Forest", "HistGradientBoosting"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, name in zip(axes.flat, plot_models):
        d = summary[summary.model == name].nlargest(10, "importance_mean").sort_values("importance_mean")
        color = "#397AA8" if name.startswith("Ridge") else "#D1813A" if name == "Random Forest" else "#4B9B70"
        if d.empty:
            ax.set_title(name); continue
        ax.barh(d.feature_label, d.importance_mean, color=color)
        ax.set_title(name, loc="left")
        ax.set_xlabel(
            "|standardized coef|" if name == "Ridge standardized coefficient"
            else "Mean OOT MAE increase (model dependence)"
        )
        ax.grid(axis="x", color="#e5e5e5"); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "US model dependence in OOT (2012–2024) — not causal",
        x=0.01, ha="left", fontsize=11,
    )
    fig.tight_layout(); fig.savefig(CHARTS / "feature_importance_models.png", dpi=180); plt.close()

    print("conformal...", flush=True)
    # Use light models inside conformal for speed but report with core.model for Ridge/HGB/Prev
    # Override core.run_conformal to use model_light for RF/HGB by temporarily patching
    _orig = core.model

    def _m(name):
        if name in ("Random Forest", "HistGradientBoosting"):
            return model_light(name)
        return _orig(name)

    core.model = _m
    conf = core.run_conformal(frame, RESULTS)
    core.model = _orig

    years = sorted(frame.year.unique())
    socio_note = (
        "ACS API requires key (blocked 2026-10-30). unemployment=ERS yearly; "
        "education/income/density largely wave-static ACS2020/ERS2022."
    )
    payload = {
        "family": "US Core specification county presidential (Core parity with TW Core specification)",
        "target": TARGET,
        "unit": "US county-year; Democratic two-party presidential share",
        "panel_years": [int(y) for y in years],
        "test_years": TEST_YEARS,
        "fold_rule": "expanding window; train year < test year",
        "n_features": len(FEATURES),
        "features": FEATURES,
        "seed": SEED,
        "mean_metrics": mean_metrics.to_dict("records"),
        "conformal_mean_coverage": (
            conf.groupby("model")["empirical_coverage"].mean().to_dict() if not conf.empty else {}
        ),
        "socio_gap": socio_note,
        "limitations": [
            "US county N≈3100 vs TW 22 — MAE not directly comparable.",
            "Dem two-party share vs TW DPP multi-party valid share.",
            socio_note,
            "Ablation RF limited to key feature sets for runtime; Ridge/HGB full ablation.",
            "Importance/weights = model dependence, not causal.",
        ],
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2,
                   default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n",
        encoding="utf-8",
    )

    mm = mean_metrics.set_index("model")
    conf_s = ""
    if not conf.empty:
        for model_name, cov in conf.groupby("model")["empirical_coverage"].mean().items():
            conf_s += f"| {model_name} | {cov:.4f} |\n"

    readme = f"""# US Core specification — County Presidential Core（美國核心設定）

**English:** Expanding-window OOT backtest for US counties, mirroring Taiwan Core specification.

**中文：** 美國縣級總統票 Core，對齊臺灣核心設定（PreviousShare／Ridge／RF／HGB、ablation、importance／weights、conformal）。

## Setup

| Item | Value |
|---|---|
| Unit | US county (~3,100) |
| Target | Democratic **two-party** presidential share |
| Test years | 2012, 2016, 2020, 2024 |
| Fold | expanding window: train years < test year |
| Features | {len(FEATURES)} history + socio |
| Source | `fyp_us_taiwan_transfer/cleaned/us_county_presidential_panel.csv` |

## Mean OOT MAE (Core specification)

| Model | MAE (4dp) | RMSE (4dp) |
|---|---:|---:|
| Previous-election baseline | {mm.loc['Previous-election baseline','mae_4dp']} | {mm.loc['Previous-election baseline','rmse_4dp']} |
| Random Forest | {mm.loc['Random Forest','mae_4dp']} | {mm.loc['Random Forest','rmse_4dp']} |
| Ridge | {mm.loc['Ridge','mae_4dp']} | {mm.loc['Ridge','rmse_4dp']} |
| HistGradientBoosting | {mm.loc['HistGradientBoosting','mae_4dp']} | {mm.loc['HistGradientBoosting','rmse_4dp']} |

## Conformal mean coverage (target 0.90)

| Model | Mean coverage |
|---|---:|
{conf_s}

## Feature weights

See `FEATURE_WEIGHTS.md` and `results/feature_weights_*.csv` (Ridge signed coefs, RF MDI, permutation).

## Gaps

- Census ACS API requires key → yearly ACS not refreshed this run.
- TW-only items (TEDS, HK, platforms): **N/A** on US checklist.

## Reproduce

```bash
.venv/bin/python fyp_us_core/scripts/us_core_backtest.py   # full backtest
.venv/bin/python fyp_us_core/scripts/us_core_finish.py     # ablation/weights/conformal
```

**I-C:** weights／permutation／ablation = **model dependence**, not causal.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    (RESULTS / "REPRODUCIBILITY.md").write_text(
        "# US Core specification reproducibility\n\n"
        "- Backtest: `us_core_backtest.py` → `family_a_us_mean_metrics.csv`\n"
        "- Finish: `us_core_finish.py` → ablation, weights, conformal\n"
        f"- Seed: {SEED}\n",
        encoding="utf-8",
    )

    # mirror
    MIRROR.mkdir(parents=True, exist_ok=True)
    (MIRROR / "charts").mkdir(parents=True, exist_ok=True)
    for p in list(RESULTS.glob("*")) + [OUT / "README.md", OUT / "FEATURE_WEIGHTS.md",
                                         OUT / "family_a_us_mean_metrics.csv"]:
        if p.exists() and p.is_file():
            shutil.copy2(p, MIRROR / p.name)
    for p in CHARTS.glob("*"):
        if p.is_file():
            shutil.copy2(p, MIRROR / "charts" / p.name)
    shutil.copy2(Path(__file__).resolve().parent / "us_core_backtest.py",
                 core.GITHUB / "scripts" / "us_core_backtest.py")
    shutil.copy2(Path(__file__).resolve(), core.GITHUB / "scripts" / "us_core_finish.py")
    feat = OUT / "us_county_panel_features_family_a.csv"
    if feat.exists():
        shutil.copy2(feat, core.GITHUB / "data" / "us_county_panel_features_family_a.csv")

    print("DONE mean MAE:")
    print(mean_metrics.to_string(index=False))
    if not conf.empty:
        print("\nConformal mean coverage:")
        print(conf.groupby("model")["empirical_coverage"].mean().to_string())
    print("\nTop Ridge |coef|:")
    print(ridge_signed.sort_values("abs_weight", ascending=False).head(5)[
        ["feature", "weight", "normalized_abs"]
    ].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
