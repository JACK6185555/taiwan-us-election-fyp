"""US Core specification county presidential Core — parity with Taiwan Core specification.

Expanding-window leave-one-election-year-out for test years 2012/2016/2020/2024.
Models: PreviousShare, Ridge, RandomForest, HistGradientBoosting.
Also: feature ablation, permutation/model importance (model dependence, not causal),
split-conformal intervals (honest coverage report).

Run: /workspace/taiwan-election-data/.venv/bin/python fyp_us_core/scripts/us_core_backtest.py
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

DATA_ROOT = Path("/workspace/taiwan-election-data")
GITHUB = Path("/workspace/taiwan-election-github")
PANEL = DATA_ROOT / "fyp_us_taiwan_transfer/cleaned/us_county_presidential_panel.csv"
OUT = DATA_ROOT / "fyp_us_core"
CHARTS = OUT / "charts"
RESULTS = OUT / "results"
MIRROR = GITHUB / "results/family_a_us_county_presidential"
SEED = 42
TARGET = "target_dem_two_party_share"
ANCHOR = "prev_dem_two_party_share"
TEST_YEARS = [2012, 2016, 2020, 2024]
MODELS = ["Previous-election baseline", "Ridge", "Random Forest", "HistGradientBoosting"]

HISTORY_FEATURES = [
    "prev_dem_two_party_share",
    "prev_gop_two_party_share",
    "prev2_dem_two_party_share",
    "prev2_gop_two_party_share",
    "prev_log_total_votes",
    "dem_change_1",
    "gop_change_1",
    "prev_dem_gop_margin",
    "dem_gop_margin_change_1",
    "abs_dem_change_1",
    "log_votes_change_1",
]
SOCIO_FEATURES = [
    "unemployment",
    "education_rate",
    "log_median_income",
    "log_density",
]
FEATURES = HISTORY_FEATURES + SOCIO_FEATURES
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
GROUPS = {
    "Previous election": [
        "prev_dem_two_party_share", "prev_gop_two_party_share", "prev_log_total_votes",
    ],
    "Earlier election": [
        "prev2_dem_two_party_share", "prev2_gop_two_party_share",
    ],
    "Margins and swings": [
        "dem_change_1", "gop_change_1", "prev_dem_gop_margin",
        "dem_gop_margin_change_1", "abs_dem_change_1", "log_votes_change_1",
    ],
    "Socioeconomic": SOCIO_FEATURES,
}


def model(name: str):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "Random Forest":
        # Larger N than TW: slightly higher leaf size for speed/stability; same spirit.
        return RandomForestRegressor(
            n_estimators=150, min_samples_leaf=5, random_state=SEED, n_jobs=-1,
        )
    return HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0, random_state=SEED,
    )


def scores(actual, pred):
    return {
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
    }


def build_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["region_id"] = df["region_id"].astype(str)
    df = df.sort_values(["region_id", "year"])
    g = df.groupby("region_id", sort=False)

    dem = df["party_share_A"].astype(float)
    prev = df["prev_share_A"].astype(float)
    # When prev missing, try lag within county
    lag1 = g["party_share_A"].shift(1)
    lag2 = g["party_share_A"].shift(2)
    prev = prev.fillna(lag1)
    prev2 = lag2

    logv = df["log_total_votes"].astype(float)
    prev_logv = g["log_total_votes"].shift(1)
    prev2_logv = g["log_total_votes"].shift(2)

    out = pd.DataFrame({
        "region_id": df["region_id"].values,
        "region_name": df["region_name"].values,
        "state_po": df["state_po"].values,
        "year": df["year"].astype(int).values,
        TARGET: dem.values,
        ANCHOR: prev.values,
        "prev_dem_two_party_share": prev.values,
        "prev_gop_two_party_share": (1.0 - prev).values,
        "prev2_dem_two_party_share": prev2.values,
        "prev2_gop_two_party_share": (1.0 - prev2).values,
        "prev_log_total_votes": prev_logv.values,
        "dem_change_1": (prev - prev2).values,
        "gop_change_1": ((1.0 - prev) - (1.0 - prev2)).values,
        "prev_dem_gop_margin": (2.0 * prev - 1.0).values,
        "dem_gop_margin_change_1": (2.0 * (prev - prev2)).values,
        "abs_dem_change_1": (prev - prev2).abs().values,
        "log_votes_change_1": (prev_logv - prev2_logv).values,
        "unemployment": df["unemployment"].astype(float).values,
        "education_rate": df["education_rate"].astype(float).values,
        "median_income": df["median_income"].astype(float).values,
        "log_density": df["log_density"].astype(float).values,
        "log_total_votes": logv.values,
    })
    out["log_median_income"] = np.log(out["median_income"].clip(lower=1.0))
    # Complete rows for Core specification features
    need = [TARGET, *FEATURES]
    before = len(out)
    out = out.dropna(subset=need).copy()
    out["_dropped_incomplete"] = before - len(out)
    return out


def plot_metrics(metrics: pd.DataFrame, charts: Path):
    for metric, title, filename in [
        ("mae", "US county OOT MAE by election year (Dem two-party share)", "backtest_mae.png"),
        ("rmse", "US county OOT RMSE by election year", "backtest_rmse.png"),
    ]:
        pivot = metrics.pivot(index="test_year", columns="model", values=metric).reindex(columns=MODELS)
        ax = pivot.plot(
            kind="bar", figsize=(9, 4.8), width=0.78,
            color=["#555555", "#2878B5", "#E07A2D", "#4B9B70"],
        )
        ax.set_title(title, loc="left")
        ax.set_xlabel("Test election year")
        ax.set_ylabel(metric.upper() + " (vote-share points)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", color="#e5e5e5")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(charts / filename, dpi=180)
        plt.close()


def run_conformal(frame: pd.DataFrame, results: Path) -> pd.DataFrame:
    """Split conformal: train < calib_year < test_year; calib = prior election year."""
    rows = []
    interval_rows = []
    alpha = 0.1
    for test_year in TEST_YEARS:
        # calib = previous presidential year present in frame
        prior = sorted(y for y in frame.year.unique() if y < test_year)
        if len(prior) < 2:
            continue
        calib_year = prior[-1]
        train = frame[frame.year < calib_year]
        calib = frame[frame.year == calib_year]
        test = frame[frame.year == test_year]
        if train.empty or calib.empty or test.empty:
            continue
        for name in MODELS[1:]:
            fitted = model(name).fit(train[FEATURES], train[TARGET])
            calib_pred = fitted.predict(calib[FEATURES])
            residuals = np.abs(calib[TARGET].to_numpy() - calib_pred)
            # finite-sample quantile
            n = len(residuals)
            q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
            q = float(np.quantile(residuals, q_level, method="higher"))
            test_pred = fitted.predict(test[FEATURES])
            actual = test[TARGET].to_numpy()
            lower = test_pred - q
            upper = test_pred + q
            covered = (actual >= lower) & (actual <= upper)
            cov = float(covered.mean())
            width = float(np.mean(upper - lower))
            rows.append({
                "test_year": int(test_year),
                "calib_year": int(calib_year),
                "model": name,
                "alpha": alpha,
                "target_coverage": 1 - alpha,
                "empirical_coverage": cov,
                "avg_width": width,
                "quantile": q,
                "n_train": int(len(train)),
                "n_calib": int(len(calib)),
                "n_test": int(len(test)),
            })
            # store a sample of intervals for latest year only (full for 2024)
            if test_year == 2024:
                chunk = test[["region_id", "region_name"]].copy()
                chunk["test_year"] = 2024
                chunk["calib_year"] = int(calib_year)
                chunk["model"] = name
                chunk["actual"] = actual
                chunk["prediction"] = test_pred
                chunk["lower"] = lower
                chunk["upper"] = upper
                chunk["width"] = upper - lower
                chunk["covered"] = covered
                interval_rows.append(chunk)
        # PreviousShare conformal (residual around anchor)
        train_ps = frame[frame.year < calib_year]
        calib_ps = frame[frame.year == calib_year]
        test_ps = frame[frame.year == test_year]
        if not train_ps.empty:
            residuals = np.abs(calib_ps[TARGET].to_numpy() - calib_ps[ANCHOR].to_numpy())
            n = len(residuals)
            q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
            q = float(np.quantile(residuals, q_level, method="higher"))
            pred = test_ps[ANCHOR].to_numpy()
            actual = test_ps[TARGET].to_numpy()
            lower, upper = pred - q, pred + q
            covered = (actual >= lower) & (actual <= upper)
            rows.append({
                "test_year": int(test_year),
                "calib_year": int(calib_year),
                "model": MODELS[0],
                "alpha": alpha,
                "target_coverage": 1 - alpha,
                "empirical_coverage": float(covered.mean()),
                "avg_width": float(np.mean(upper - lower)),
                "quantile": q,
                "n_train": int(len(train_ps)),
                "n_calib": int(len(calib_ps)),
                "n_test": int(len(test_ps)),
            })
            if test_year == 2024:
                chunk = test_ps[["region_id", "region_name"]].copy()
                chunk["test_year"] = 2024
                chunk["calib_year"] = int(calib_year)
                chunk["model"] = MODELS[0]
                chunk["actual"] = actual
                chunk["prediction"] = pred
                chunk["lower"] = lower
                chunk["upper"] = upper
                chunk["width"] = upper - lower
                chunk["covered"] = covered
                interval_rows.append(chunk)
    meta = pd.DataFrame(rows)
    meta.to_csv(results / "conformal_coverage_by_year.csv", index=False, encoding="utf-8-sig")
    if interval_rows:
        pd.concat(interval_rows, ignore_index=True).to_csv(
            results / "conformal_intervals_2024.csv", index=False, encoding="utf-8-sig",
        )
    summary = meta.groupby("model", as_index=False).agg(
        mean_coverage=("empirical_coverage", "mean"),
        mean_width=("avg_width", "mean"),
        min_coverage=("empirical_coverage", "min"),
        max_coverage=("empirical_coverage", "max"),
    )
    summary.to_csv(results / "conformal_coverage_summary.csv", index=False, encoding="utf-8-sig")
    (results / "conformal_meta.json").write_text(
        json.dumps({
            "method": "split conformal; calib = prior election year; train years < calib",
            "alpha": alpha,
            "target_coverage": 1 - alpha,
            "note": "If empirical coverage << 0.9, intervals are miscalibrated (same honesty as TW RQ4).",
            "by_model_mean": summary.to_dict("records"),
            "by_year": meta.to_dict("records"),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # chart
    if not meta.empty:
        pivot = meta.pivot(index="test_year", columns="model", values="empirical_coverage")
        ax = pivot.reindex(columns=[c for c in MODELS if c in pivot.columns]).plot(
            kind="bar", figsize=(9, 4.5), width=0.78,
            color=["#555555", "#2878B5", "#E07A2D", "#4B9B70"],
        )
        ax.axhline(0.9, color="#c0392b", ls="--", lw=1, label="target 90%")
        ax.set_title("US split-conformal empirical coverage (target 0.90)", loc="left")
        ax.set_ylabel("Coverage"); ax.set_xlabel("Test year")
        ax.legend(frameon=False, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color="#e5e5e5"); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=0); plt.tight_layout()
        plt.savefig(CHARTS / "conformal_coverage.png", dpi=180); plt.close()
    return meta


def mirror_outputs():
    MIRROR.mkdir(parents=True, exist_ok=True)
    (MIRROR / "charts").mkdir(parents=True, exist_ok=True)
    for p in RESULTS.glob("*"):
        if p.is_file():
            shutil.copy2(p, MIRROR / p.name)
    for p in CHARTS.glob("*"):
        if p.is_file():
            shutil.copy2(p, MIRROR / "charts" / p.name)
    # also copy feature panel used
    panel_out = GITHUB / "data" / "us_county_panel_features_family_a.csv"
    src = OUT / "us_county_panel_features_family_a.csv"
    if src.exists():
        shutil.copy2(src, panel_out)
    script_dst = GITHUB / "scripts" / "us_core_backtest.py"
    shutil.copy2(Path(__file__).resolve(), script_dst)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(PANEL)
    frame = build_frame(raw)
    frame.to_csv(OUT / "us_county_panel_features_family_a.csv", index=False, encoding="utf-8-sig")

    years = sorted(frame.year.unique())
    metric_rows, pred_rows = [], []
    for test_year in TEST_YEARS:
        train = frame[frame.year < test_year]
        test = frame[frame.year == test_year]
        if train.empty or test.empty:
            continue
        actual = test[TARGET].to_numpy()
        for name in MODELS:
            if name == MODELS[0]:
                pred = test[ANCHOR].to_numpy()
            else:
                fitted = model(name).fit(train[FEATURES], train[TARGET])
                pred = fitted.predict(test[FEATURES])
            metric_rows.append({
                "test_year": int(test_year),
                "train_through": int(train.year.max()),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "model": name,
                **scores(actual, pred),
            })
            # store predictions (all counties — large but needed)
            chunk = test[["region_id", "region_name", "state_po"]].copy()
            chunk["test_year"] = int(test_year)
            chunk["model"] = name
            chunk["actual"] = actual
            chunk["prediction"] = pred
            chunk["error"] = pred - actual
            pred_rows.append(chunk)
        print(f"backtest fold {test_year}: n_train={len(train)} n_test={len(test)}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS / "backtest_metrics.csv", index=False, encoding="utf-8-sig")
    preds_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    preds_df.to_csv(RESULTS / "backtest_predictions.csv", index=False, encoding="utf-8-sig")
    plot_metrics(metrics, CHARTS)

    mean_metrics = (
        metrics.groupby("model", as_index=False)[["mae", "rmse"]].mean()
        .sort_values("mae")
    )
    mean_metrics["mae_4dp"] = mean_metrics["mae"].round(4)
    mean_metrics["rmse_4dp"] = mean_metrics["rmse"].round(4)
    mean_metrics.to_csv(RESULTS / "family_a_us_mean_metrics.csv", index=False, encoding="utf-8-sig")
    # convenience copy at OUT root
    mean_metrics.to_csv(OUT / "family_a_us_mean_metrics.csv", index=False, encoding="utf-8-sig")

    # Feature ablation
    print("ablation...", flush=True)
    sets = {"Previous-election only": [ANCHOR], "All features": FEATURES}
    for group, cols in GROUPS.items():
        sets[f"All minus {group}"] = [c for c in FEATURES if c not in cols]
    ablation_rows = []
    for test_year in TEST_YEARS:
        train, test = frame[frame.year < test_year], frame[frame.year == test_year]
        y = test[TARGET].to_numpy()
        for set_name, cols in sets.items():
            for name in MODELS[1:]:
                pred = model(name).fit(train[cols], train[TARGET]).predict(test[cols])
                ablation_rows.append({
                    "test_year": int(test_year), "model": name, "feature_set": set_name,
                    "n_features": len(cols), **scores(y, pred),
                })
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(RESULTS / "feature_ablation_by_year.csv", index=False, encoding="utf-8-sig")
    ablation_summary = ablation.groupby(
        ["model", "feature_set", "n_features"], as_index=False,
    )[["mae", "rmse"]].mean()
    ablation_summary.to_csv(RESULTS / "feature_ablation_summary.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True)
    for ax, name in zip(axes, MODELS[1:]):
        d = ablation_summary[ablation_summary.model == name].sort_values("mae")
        ax.barh(d.feature_set, d.mae, color="#438B78")
        ax.set_title(name, loc="left")
        ax.set_xlabel("Mean MAE")
        ax.grid(axis="x", color="#e5e5e5")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "US feature ablation (OOT): mean MAE — MAE change when groups removed (not causal)",
        x=0.01, ha="left", fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(CHARTS / "feature_ablation.png", dpi=180)
    plt.close(fig)

    # Permutation importance — subsample test counties for speed; document
    print("importance...", flush=True)
    importance_rows = []
    rng = np.random.RandomState(SEED)
    for test_year in TEST_YEARS:
        train, test = frame[frame.year < test_year], frame[frame.year == test_year]
        if len(test) > 800:
            test_imp = test.sample(n=800, random_state=SEED + int(test_year))
        else:
            test_imp = test
        y = test_imp[TARGET].to_numpy()
        for name in MODELS[1:]:
            fitted = model(name).fit(train[FEATURES], train[TARGET])
            result = permutation_importance(
                fitted, test_imp[FEATURES], y, scoring="neg_mean_absolute_error",
                n_repeats=10, random_state=SEED + int(test_year), n_jobs=1,
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
    plot_models = ["Ridge", "Ridge standardized coefficient", "Random Forest", "HistGradientBoosting"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False)
    for ax, name in zip(axes.flat, plot_models):
        d = summary[summary.model == name].nlargest(10, "importance_mean").sort_values("importance_mean")
        color = "#397AA8" if name.startswith("Ridge") else "#D1813A" if name == "Random Forest" else "#4B9B70"
        ax.barh(d.feature_label, d.importance_mean, color=color)
        ax.set_title(name, loc="left")
        ax.set_xlabel(
            "Mean OOT MAE increase (model dependence)"
            if name != "Ridge standardized coefficient"
            else "|standardized coef| (association, not causal)"
        )
        ax.grid(axis="x", color="#e5e5e5")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "US model dependence in OOT (2012–2024): permutation MAE increase — not causal",
        x=0.01, ha="left", fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(CHARTS / "feature_importance_models.png", dpi=180)
    plt.close(fig)


    # Explicit feature weights / importances (signed Ridge; RF impurity; HGB/RF permutation)
    print("feature_weights...", flush=True)
    # Fit on all pre-2024 rows for a single "refit" weight table; plus fold-averaged
    train_refit = frame[frame.year < 2024]
    weight_rows = []
    # Ridge signed standardized coefficients
    ridge = model("Ridge").fit(train_refit[FEATURES], train_refit[TARGET])
    coefs = ridge[-1].coef_
    scale = ridge[0].scale_
    for feat, c, s in zip(FEATURES, coefs, scale):
        weight_rows.append({
            "source": "refit_train_lt_2024",
            "model": "Ridge",
            "feature": feat,
            "feature_label": LABELS[feat],
            "weight_type": "standardized_coefficient_signed",
            "weight": float(c),
            "abs_weight": abs(float(c)),
            "normalized_abs": np.nan,  # filled later
            "note": "sign: + raises predicted Dem share when feature increases (after standardization)",
        })
    # RF impurity (MDI)
    rf = model("Random Forest").fit(train_refit[FEATURES], train_refit[TARGET])
    for feat, w in zip(FEATURES, rf.feature_importances_):
        weight_rows.append({
            "source": "refit_train_lt_2024",
            "model": "Random Forest",
            "feature": feat,
            "feature_label": LABELS[feat],
            "weight_type": "impurity_importance_mdi",
            "weight": float(w),
            "abs_weight": float(w),
            "normalized_abs": np.nan,
            "note": "non-negative; biased toward high-cardinality; model dependence not causal",
        })
    # Permutation on 2024 holdout (full or subsample) for RF + HGB + Ridge
    test_2024 = frame[frame.year == 2024]
    test_w = test_2024.sample(n=min(1000, len(test_2024)), random_state=SEED)
    y_w = test_w[TARGET].to_numpy()
    for name, fitted in [
        ("Ridge", ridge),
        ("Random Forest", rf),
        ("HistGradientBoosting", model("HistGradientBoosting").fit(train_refit[FEATURES], train_refit[TARGET])),
    ]:
        pi = permutation_importance(
            fitted, test_w[FEATURES], y_w, scoring="neg_mean_absolute_error",
            n_repeats=15, random_state=SEED, n_jobs=1,
        )
        for feat, m, s in zip(FEATURES, pi.importances_mean, pi.importances_std):
            weight_rows.append({
                "source": "perm_on_2024_subsample",
                "model": name,
                "feature": feat,
                "feature_label": LABELS[feat],
                "weight_type": "permutation_mae_increase",
                "weight": float(m),
                "abs_weight": abs(float(m)),
                "normalized_abs": np.nan,
                "note": f"perm_std={float(s):.6g}; n_test={len(test_w)}; model dependence not causal",
            })
    weights = pd.DataFrame(weight_rows)
    # normalize abs within model+weight_type+source
    def _norm(g):
        s = g["abs_weight"].sum()
        g = g.copy()
        g["normalized_abs"] = g["abs_weight"] / s if s > 0 else 0.0
        return g
    weights = weights.groupby(["source", "model", "weight_type"], group_keys=False).apply(_norm)
    weights = weights.sort_values(
        ["source", "model", "weight_type", "abs_weight"], ascending=[True, True, True, False]
    )
    weights.to_csv(RESULTS / "feature_weights_all.csv", index=False, encoding="utf-8-sig")
    # Wide convenience tables
    ridge_signed = weights[(weights.model == "Ridge") & (weights.weight_type == "standardized_coefficient_signed")].copy()
    ridge_signed.to_csv(RESULTS / "feature_weights_ridge_signed.csv", index=False, encoding="utf-8-sig")
    rf_mdi = weights[(weights.model == "Random Forest") & (weights.weight_type == "impurity_importance_mdi")].copy()
    rf_mdi.to_csv(RESULTS / "feature_weights_rf_impurity.csv", index=False, encoding="utf-8-sig")
    perm = weights[weights.weight_type == "permutation_mae_increase"].copy()
    perm.to_csv(RESULTS / "feature_weights_permutation.csv", index=False, encoding="utf-8-sig")
    # Short markdown interpretation
    def top_lines(df, k=5):
        lines = []
        for _, r in df.head(k).iterrows():
            lines.append(f"- **{r.feature_label}** (`{r.feature}`): weight={r.weight:.4g}, |w| share={r.normalized_abs:.1%}")
        return "\n".join(lines)
    md = []
    md.append("# US Core specification — Feature weights (model dependence, not causal)\n")
    md.append("Weights describe how each fitted model **uses** features in this OOT setup. They are **not** causal effects on voters.\n")
    md.append("## Ridge standardized coefficients (signed, train years < 2024)\n")
    md.append("Positive ⇒ higher feature value associated with higher predicted Dem two-party share (after z-scoring).\n")
    md.append(top_lines(ridge_signed.sort_values("abs_weight", ascending=False)))
    md.append("\n## Random Forest impurity importance (MDI, train < 2024)\n")
    md.append(top_lines(rf_mdi.sort_values("abs_weight", ascending=False)))
    md.append("\n## Permutation MAE increase on 2024 subsample\n")
    for name in ["Previous-election baseline", "Ridge", "Random Forest", "HistGradientBoosting"]:
        pass
    for name in ["Ridge", "Random Forest", "HistGradientBoosting"]:
        sub = perm[perm.model == name].sort_values("abs_weight", ascending=False)
        md.append(f"\n### {name}\n")
        md.append(top_lines(sub))
    md.append("\n## Files\n")
    md.append("- `feature_weights_all.csv` — all weight types\n- `feature_weights_ridge_signed.csv`\n- `feature_weights_rf_impurity.csv`\n- `feature_weights_permutation.csv`\n")
    md.append("\n**Disclaimer:** Do not interpret ranks as “what causes Dem share.” Socio features may be wave-static.\n")
    (RESULTS / "FEATURE_WEIGHTS.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "FEATURE_WEIGHTS.md").write_text("\n".join(md), encoding="utf-8")
    # chart: ridge signed
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

    print("conformal...", flush=True)

    conf = run_conformal(frame, RESULTS)

    # summary.json
    socio_note = (
        "Socio columns (unemployment yearly ERS; education/income/density often wave-static "
        "ACS2020/ERS2022) — not fully year-synchronized; treat as weak covariates."
    )
    payload = {
        "family": "US Core specification county presidential (Core parity with TW Core specification)",
        "target": TARGET,
        "unit": "US county-year; Democratic two-party presidential share",
        "panel_years": [int(y) for y in years],
        "test_years": TEST_YEARS,
        "fold_rule": "expanding window; train year < test year; all counties with complete features",
        "n_features": len(FEATURES),
        "features": FEATURES,
        "seed": SEED,
        "permutation_repeats": 10,
        "permutation_test_subsample_cap": 800,
        "mean_metrics": mean_metrics.to_dict("records"),
        "conformal_mean_coverage": conf.groupby("model")["empirical_coverage"].mean().to_dict() if not conf.empty else {},
        "socio_gap": socio_note,
        "limitations": [
            "Unit is US county (~3100), not TW county/city (22) — MAE not directly comparable in magnitude alone.",
            "Target is Dem two-party share; TW Core specification uses DPP valid-vote share (multi-party).",
            socio_note,
            "Permutation importance uses up to 800-county test subsample and 10 repeats for runtime.",
            "Importance/ablation describe model dependence in OOT — not causal.",
        ],
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2,
                   default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n",
        encoding="utf-8",
    )

    # README bilingual
    mm = mean_metrics.set_index("model")
    conf_s = ""
    if not conf.empty:
        for _, r in conf.groupby("model", as_index=False)["empirical_coverage"].mean().iterrows():
            conf_s += f"| {r['model']} | {r['empirical_coverage']:.4f} |\n"
    readme = f"""# US Core specification — County Presidential Core（美國核心設定）

**English:** Expanding-window out-of-time backtest for US counties, mirroring Taiwan Core specification Core.

**中文：** 美國縣級總統票 Core 回測，結構對齊臺灣核心設定（PreviousShare／Ridge／RF／HGB、ablation、importance、conformal）。

## Setup

| Item | Value |
|---|---|
| Unit | US county (~3,100) |
| Target | Democratic **two-party** presidential share |
| Test years | 2012, 2016, 2020, 2024 |
| Fold | expanding window: train years < test year |
| Features | {len(FEATURES)} history + socio (see gaps) |
| Source panel | `fyp_us_taiwan_transfer/cleaned/us_county_presidential_panel.csv` |

## Mean OOT MAE (Core specification)

| Model | MAE (4dp) | RMSE (4dp) |
|---|---:|---:|
| Previous-election baseline | {mm.loc[MODELS[0], 'mae_4dp'] if MODELS[0] in mm.index else 'n/a'} | {mm.loc[MODELS[0], 'rmse_4dp'] if MODELS[0] in mm.index else 'n/a'} |
| Random Forest | {mm.loc['Random Forest', 'mae_4dp'] if 'Random Forest' in mm.index else 'n/a'} | {mm.loc['Random Forest', 'rmse_4dp'] if 'Random Forest' in mm.index else 'n/a'} |
| HistGradientBoosting | {mm.loc['HistGradientBoosting', 'mae_4dp'] if 'HistGradientBoosting' in mm.index else 'n/a'} | {mm.loc['HistGradientBoosting', 'rmse_4dp'] if 'HistGradientBoosting' in mm.index else 'n/a'} |
| Ridge | {mm.loc['Ridge', 'mae_4dp'] if 'Ridge' in mm.index else 'n/a'} | {mm.loc['Ridge', 'rmse_4dp'] if 'Ridge' in mm.index else 'n/a'} |

完整數字：`results/family_a_us_mean_metrics.csv` / `family_a_us_mean_metrics.csv`

## Conformal (RQ4) — mean empirical coverage (target 0.90)

| Model | Mean coverage |
|---|---:|
{conf_s}
If coverage ≪ 0.90, intervals are **miscalibrated** — report honestly (same as TW).

## Gaps / 資料缺口

- Socio: unemployment 多為年度 ERS；education／income／density 常為 ACS2020／ERS2022 **波次靜態**，非逐年同步。
- 無 TEDS／政綱／香港因素（臺灣專屬）→ US checklist 標 N/A。
- Importance：測試縣 subsample ≤800、permutation repeats=10（runtime）。

## Reproduce

```bash
/workspace/taiwan-election-data/.venv/bin/python \\
  /workspace/taiwan-election-data/fyp_us_core/scripts/us_core_backtest.py
```

Outputs: `results/`, `charts/`, mirrored to `taiwan-election-github/results/family_a_us_county_presidential/`.

**I-C:** permutation／ablation／Ridge 權重僅 **model dependence in OOT**，**非因果**。
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    (RESULTS / "REPRODUCIBILITY.md").write_text(
        f"""# US Core specification reproducibility

- Script: `fyp_us_core/scripts/us_core_backtest.py`
- Panel: `{PANEL}`
- sklearn via `/workspace/taiwan-election-data/.venv`
- Mean metrics: `family_a_us_mean_metrics.csv`
- Seed: {SEED}
""",
        encoding="utf-8",
    )

    mirror_outputs()
    print("US Core specification mean MAE:")
    print(mean_metrics.to_string(index=False))
    if not conf.empty:
        print("\nConformal mean coverage:")
        print(conf.groupby("model")["empirical_coverage"].mean().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
