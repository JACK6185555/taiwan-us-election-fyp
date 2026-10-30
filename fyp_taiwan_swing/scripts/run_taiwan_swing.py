#!/usr/bin/env python3
"""Taiwan FYP Core — SWING (ΔV) as primary target.

ΔV(i,t) = V(i,t) − V(i,t−1), V = DPP presidential vote share.
Reuses harmonized panel from fyp_taiwan/harmonized_tw_regions.csv.
Out-of-time backtest, ablation, RF importance, split conformal α=0.10.
Does NOT invent unemployment or poll_index.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path("/workspace/taiwan-election-data")
LEVEL_DIR = ROOT / "fyp_taiwan"
OUT = ROOT / "fyp_taiwan_swing"
CHARTS = OUT / "charts"
OUT.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET = "swing"
HARM_PATH = LEVEL_DIR / "harmonized_tw_regions.csv"

MODEL_FEATURES = [
    "prev_vote_share",
    "prev_swing",
    "prev_turnout",
    "median_income",
    "urbanization",
    "education_rate",
    "log_people",
]

FEATURE_SETS = {
    "History": ["prev_vote_share", "prev_turnout", "prev_swing"],
    "History+Economy": ["prev_vote_share", "prev_turnout", "prev_swing", "median_income"],
    "History+Demographics": [
        "prev_vote_share",
        "prev_turnout",
        "prev_swing",
        "urbanization",
        "education_rate",
        "log_people",
    ],
    "History+Economy+Demo": [
        "prev_vote_share",
        "prev_turnout",
        "prev_swing",
        "median_income",
        "urbanization",
        "education_rate",
        "log_people",
    ],
    "All_available": MODEL_FEATURES,
}


def get_models():
    return {
        "Ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1
        ),
        "HistGB": HistGradientBoostingRegressor(
            max_depth=5, learning_rate=0.06, random_state=SEED
        ),
    }


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(HARM_PATH)
    # Ensure swing is defined where prev exists
    if "swing" not in df.columns or df["swing"].isna().all():
        df["swing"] = df["target_vote_share"] - df["prev_vote_share"]
    # Require target swing + prev share
    df = df.dropna(subset=["prev_vote_share", "swing", "target_vote_share"]).copy()
    print(
        f"panel rows with swing={len(df)} years={sorted(df.year.unique())} "
        f"swing mean by year:\n{df.groupby('year')['swing'].agg(['mean','std','count'])}"
    )
    return df


def impute_features(train: pd.DataFrame, test: pd.DataFrame, feats: list[str]):
    tr, te = train.copy(), test.copy()
    for c in feats:
        if c not in tr.columns:
            continue
        fill = tr[c].median()
        if pd.isna(fill):
            fill = 0.0
        tr[c] = tr[c].fillna(fill)
        te[c] = te[c].fillna(fill)
    return tr, te


def backtest(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    years = sorted(d["year"].unique())
    models = get_models()
    feats = MODEL_FEATURES
    results = []

    for test_year in years:
        train = d[d["year"] < test_year]
        test = d[d["year"] == test_year]
        if len(train) == 0 or len(test) == 0:
            continue
        tr, te = impute_features(train, test, feats)
        tr = tr.dropna(subset=feats + [TARGET])
        te = te.dropna(subset=feats + [TARGET])
        if len(tr) < 50 or len(te) < 20:
            print(f"  skip {test_year}: train={len(tr)} test={len(te)}")
            continue

        y_test = te[TARGET]

        # Baseline A: predict ΔV = 0 (no change in vote share)
        pred_zero = np.zeros(len(te))
        results.append(
            {
                "country": "TW",
                "test_year": test_year,
                "model": "ZeroSwing",
                "n_train": len(tr),
                "n_test": len(te),
                "MAE": mean_absolute_error(y_test, pred_zero),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, pred_zero))),
                "R2": r2_score(y_test, pred_zero),
            }
        )

        # Baseline B: PreviousElection-for-swing = persist prior swing ΔV(t)=ΔV(t−1)
        # Documented as PreviousSwing; when prev_swing missing, treated as 0 after impute
        pred_prev_sw = te["prev_swing"].values
        results.append(
            {
                "country": "TW",
                "test_year": test_year,
                "model": "PreviousSwing",
                "n_train": len(tr),
                "n_test": len(te),
                "MAE": mean_absolute_error(y_test, pred_prev_sw),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, pred_prev_sw))),
                "R2": r2_score(y_test, pred_prev_sw),
            }
        )

        X_train, y_train = tr[feats], tr[TARGET]
        X_test = te[feats]
        for name, model in models.items():
            m = clone(model)
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            results.append(
                {
                    "country": "TW",
                    "test_year": test_year,
                    "model": name,
                    "n_train": len(tr),
                    "n_test": len(te),
                    "MAE": mean_absolute_error(y_test, pred),
                    "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                    "R2": r2_score(y_test, pred),
                }
            )
        print(
            f"  backtest test_year={test_year} train_years={sorted(tr.year.unique())} "
            f"n_test={len(te)} mean_swing={y_test.mean():.4f}"
        )

    return pd.DataFrame(results)


def ablation(data: pd.DataFrame, test_year: int = 2024) -> pd.DataFrame:
    d = data.copy()
    train = d[d["year"] < test_year]
    test = d[d["year"] == test_year]
    rows = []
    for name, feats in FEATURE_SETS.items():
        tr, te = impute_features(train, test, feats)
        tr = tr.dropna(subset=feats + [TARGET])
        te = te.dropna(subset=feats + [TARGET])
        if len(tr) < 50 or len(te) < 20:
            continue
        model = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1
        )
        model.fit(tr[feats], tr[TARGET])
        pred = model.predict(te[feats])
        mae = mean_absolute_error(te[TARGET], pred)
        rows.append(
            {
                "country": "TW",
                "test_year": test_year,
                "features": name,
                "n_features": len(feats),
                "feature_list": ",".join(feats),
                "MAE": mae,
                "RMSE": float(np.sqrt(mean_squared_error(te[TARGET], pred))),
                "R2": r2_score(te[TARGET], pred),
            }
        )
        print(f"  ablation {name}: MAE={mae:.4f}")
    return pd.DataFrame(rows)


def feature_importance(data: pd.DataFrame, test_year: int = 2024) -> pd.DataFrame:
    d = data.copy()
    feats = MODEL_FEATURES
    train = d[d["year"] < test_year]
    test = d[d["year"] == test_year]
    tr, te = impute_features(train, test, feats)
    tr = tr.dropna(subset=feats + [TARGET])
    te = te.dropna(subset=feats + [TARGET])
    model = RandomForestRegressor(
        n_estimators=500, min_samples_leaf=3, random_state=SEED, n_jobs=-1
    )
    model.fit(tr[feats], tr[TARGET])
    imp = pd.DataFrame({"feature": feats, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp["test_year"] = test_year
    imp["country"] = "TW"
    imp["target"] = TARGET
    return imp


def conformal(data: pd.DataFrame, test_year: int = 2024, alpha: float = 0.10):
    """Split conformal on swing; no [0,1] clip (swings can be negative)."""
    d = data.copy()
    feats = MODEL_FEATURES
    years_before = sorted(d[d["year"] < test_year]["year"].unique())
    if len(years_before) < 2:
        raise ValueError("Need ≥2 years before test for conformal calibration")
    calibration_year = years_before[-1]
    train = d[d["year"] < calibration_year]
    calibration = d[d["year"] == calibration_year]
    test = d[d["year"] == test_year]

    tr, cal = impute_features(train, calibration, feats)
    _, te = impute_features(train, test, feats)
    tr = tr.dropna(subset=feats + [TARGET])
    cal = cal.dropna(subset=feats + [TARGET])
    te = te.dropna(subset=feats + [TARGET])

    model = RandomForestRegressor(
        n_estimators=500, min_samples_leaf=3, random_state=SEED, n_jobs=-1
    )
    model.fit(tr[feats], tr[TARGET])
    cal_pred = model.predict(cal[feats])
    residuals = np.abs(cal[TARGET].values - cal_pred)
    q = float(np.quantile(residuals, 1 - alpha))

    pred = model.predict(te[feats])
    lower, upper = pred - q, pred + q
    # Clip to plausible swing range [-1, 1]
    lower = np.clip(lower, -1.0, 1.0)
    upper = np.clip(upper, -1.0, 1.0)
    actual = te[TARGET].values
    coverage = float(np.mean((actual >= lower) & (actual <= upper)))
    avg_width = float(np.mean(upper - lower))

    meta = {
        "target": TARGET,
        "calibration_year": int(calibration_year),
        "test_year": int(test_year),
        "alpha": alpha,
        "quantile": q,
        "coverage": coverage,
        "avg_width": avg_width,
        "n_train": len(tr),
        "n_calib": len(cal),
        "n_test": len(te),
        "target_coverage": 1 - alpha,
        "mean_actual_swing": float(np.mean(actual)),
        "mean_pred_swing": float(np.mean(pred)),
    }
    intervals = pd.DataFrame(
        {
            "country": "TW",
            "region": te["region"].values,
            "year": test_year,
            "actual_swing": actual,
            "prediction": pred,
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
            "covered": (actual >= lower) & (actual <= upper),
            "prev_vote_share": te["prev_vote_share"].values,
            "actual_vote_share": te["target_vote_share"].values,
        }
    )
    print(
        f"  conformal calib={calibration_year} q={q:.4f} "
        f"coverage={coverage:.4f} (target {1-alpha:.2f}) width={avg_width:.4f}"
    )
    return intervals, meta


def make_charts(back: pd.DataFrame, abl: pd.DataFrame, imp: pd.DataFrame, level_back_path: Path):
    # MAE by year (swing models)
    plt.figure(figsize=(9, 5))
    for model in back["model"].unique():
        m = back[back["model"] == model].sort_values("test_year")
        plt.plot(m["test_year"], m["MAE"], marker="o", label=model)
    plt.xlabel("Test election year")
    plt.ylabel("Swing MAE |ΔV − ΔV̂|")
    plt.title("TW: Out-of-time swing (ΔV) forecast error")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHARTS / "mae_by_year_swing.png", dpi=150)
    plt.close()

    # Ablation
    abl_s = abl.sort_values("MAE")
    plt.figure(figsize=(8, 5))
    plt.barh(abl_s["features"], abl_s["MAE"], color="steelblue")
    plt.xlabel("MAE (test 2024, swing)")
    plt.title("TW swing: Feature ablation (Random Forest)")
    plt.tight_layout()
    plt.savefig(CHARTS / "ablation_mae_swing.png", dpi=150)
    plt.close()

    # Importance
    imp_s = imp.sort_values("importance")
    plt.figure(figsize=(8, 5))
    plt.barh(imp_s["feature"], imp_s["importance"], color="darkorange")
    plt.xlabel("RF impurity importance")
    plt.title("TW swing: RF feature importance (train < 2024)")
    plt.tight_layout()
    plt.savefig(CHARTS / "feature_importance_swing.png", dpi=150)
    plt.close()

    # Compare level vs swing: for fair compare, convert level MAE is on V;
    # swing ZeroSwing MAE == level PreviousElection MAE (same errors).
    # Plot side-by-side best ML MAE for swing vs ZeroSwing.
    if level_back_path.exists():
        level = pd.read_csv(level_back_path)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
        for ax, df, title, ylab in [
            (
                axes[0],
                level,
                "Level target V",
                "Vote-share MAE",
            ),
            (
                axes[1],
                back,
                "Swing target ΔV",
                "Swing MAE",
            ),
        ]:
            for model in df["model"].unique():
                m = df[df["model"] == model].sort_values("test_year")
                ax.plot(m["test_year"], m["MAE"], marker="o", label=model)
            ax.set_title(title)
            ax.set_xlabel("Test year")
            ax.set_ylabel(ylab)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle("TW: Level vs Swing out-of-time MAE")
        fig.tight_layout()
        fig.savefig(CHARTS / "level_vs_swing_mae.png", dpi=150)
        plt.close()


def write_readme(
    back: pd.DataFrame,
    abl: pd.DataFrame,
    imp: pd.DataFrame,
    conf_meta: dict,
    data: pd.DataFrame,
    level_back: pd.DataFrame | None,
):
    swing_stats = (
        data.groupby("year")["swing"]
        .agg(["mean", "std", "count"])
        .round(4)
        .reset_index()
        .to_string(index=False)
    )
    pivot = back.pivot_table(index="model", columns="test_year", values="MAE").round(4)

    level_note = ""
    if level_back is not None and len(level_back):
        lp = level_back.pivot_table(index="model", columns="test_year", values="MAE").round(4)
        level_note = (
            "\n## Comparison: swing vs level prediction\n\n"
            "Level-model MAE (from `fyp_taiwan/backtest_results.csv`):\n\n```\n"
            + lp.to_string()
            + "\n```\n\n"
            "Key identities / findings:\n\n"
            "1. **ZeroSwing MAE on ΔV equals PreviousElection MAE on V** "
            "(same absolute errors: |V−V_prev| = |ΔV−0|).\n"
            "2. Predicting **level** is dominated by `prev_vote_share` (importance ~0.96). "
            "Predicting **swing** removes that mechanical lag; importance redistributes "
            "toward `prev_swing`, demographics, and income — or remains weak if ΔV is "
            "mostly a national shock.\n"
            "3. **2024** national DPP drop (~−0.15) is a large common shock: "
            "township features explain little of the *level shift*; ML on swing still "
            "struggles vs ZeroSwing when the mean swing is far from historical train means.\n"
            "4. If ML beats ZeroSwing on swing MAE, features add value beyond persistence; "
            "if not, ecological covariates are insufficient for out-of-time ΔV.\n"
        )

    lines = [
        "# Taiwan FYP Core — Swing (ΔV) Forecasting",
        "",
        "Primary target: **ΔV(i,t) = V(i,t) − V(i,t−1)** where V = DPP presidential share.",
        "Panel: `../fyp_taiwan/harmonized_tw_regions.csv` (same township × year).",
        "",
        "## Baselines (documented)",
        "",
        "| Model | Definition |",
        "|-------|------------|",
        "| **ZeroSwing** | Predict ΔV̂ = 0 (no change). Primary naive baseline for swing. "
        "Equivalent errors to level PreviousElection. |",
        "| **PreviousSwing** | Predict ΔV̂(t) = prev_swing = ΔV(t−1). "
        "FYP “PreviousElection” analogue for swings (persistence of change). |",
        "| Ridge / RandomForest / HistGB | Supervised models on available features |",
        "",
        "## Features (no invented unemployment / poll)",
        "",
        ", ".join(MODEL_FEATURES),
        "",
        "`prev_swing` missing on earliest train rows → median-imputed from training fold only.",
        "",
        "## Swing distribution (this panel)",
        "",
        "```",
        swing_stats,
        "```",
        "",
        "## Methods",
        "",
        "- Out-of-time only (train years < test year)",
        "- Ablation: History / +Economy / +Demographics / combined",
        "- RF impurity importance, train < 2024",
        "- Split conformal α=0.10; calibration = last year before test; intervals on ΔV clipped to [-1,1]",
        "",
        "## Backtest MAE (swing)",
        "",
        "```",
        pivot.to_string(),
        "```",
        "",
        "## Ablation ranking (2024 holdout, swing)",
        "",
        abl.sort_values("MAE")[["features", "MAE", "RMSE", "R2"]].round(4).to_string(index=False),
        "",
        "## Top features (RF swing, train < 2024)",
        "",
        imp.head(10).to_string(index=False),
        "",
        "## Conformal (2024 swing, α=0.10)",
        "",
        json.dumps(conf_meta, indent=2),
        "",
        level_note,
        "## Caveats",
        "",
        "1. Ecological / township model — not individual TEDS.",
        "2. 2024 three-candidate race creates large negative mean swing vs 2012–2020.",
        "3. Naive conformal calibrated on 2020 may undercover 2024 if residual scale shifts.",
        "4. Same data gaps as level model (no unemployment, no poll_index).",
        "",
        "## Reproduce",
        "",
        "```bash",
        "/workspace/taiwan-election-data/.venv/bin/python \\",
        "  /workspace/taiwan-election-data/fyp_taiwan_swing/scripts/run_taiwan_swing.py",
        "```",
        "",
        "## Outputs",
        "",
        "- `backtest_results.csv`",
        "- `ablation_results.csv`",
        "- `feature_importance.csv`",
        "- `conformal_intervals_2024.csv`",
        "- `conformal_meta_2024.json`",
        "- `charts/*.png`",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("=== Load harmonized panel ===")
    data = load_panel()
    # Save a copy pointer / symlink note
    data.to_csv(OUT / "harmonized_tw_regions_used.csv", index=False)

    print("=== Out-of-time swing backtest ===")
    back = backtest(data)
    back.to_csv(OUT / "backtest_results.csv", index=False)
    print(back.to_string(index=False))

    print("=== Feature ablation (test 2024) ===")
    abl_all = []
    for y in sorted(data.year.unique()):
        if (data["year"] < y).sum() >= 50 and (data["year"] == y).sum() >= 20:
            print(f"=== Feature ablation (test {y}) ===")
            abl_all.append(ablation(data, test_year=int(y)))
    abl_full = pd.concat(abl_all, ignore_index=True) if abl_all else pd.DataFrame()
    abl_full.to_csv(OUT / "ablation_results.csv", index=False)
    abl_2024 = abl_full[abl_full["test_year"] == 2024] if len(abl_full) else abl_full

    print("=== Feature importance ===")
    imp = feature_importance(data, test_year=2024)
    imp.to_csv(OUT / "feature_importance.csv", index=False)
    print(imp.to_string(index=False))

    print("=== Conformal intervals 2024 (swing) ===")
    intervals, meta = conformal(data, test_year=2024, alpha=0.10)
    intervals.to_csv(OUT / "conformal_intervals_2024.csv", index=False)
    (OUT / "conformal_meta_2024.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    level_back_path = LEVEL_DIR / "backtest_results.csv"
    level_back = pd.read_csv(level_back_path) if level_back_path.exists() else None

    print("=== Charts ===")
    make_charts(
        back,
        abl_2024 if len(abl_2024) else abl_full,
        imp,
        level_back_path,
    )

    print("=== README ===")
    write_readme(back, abl_2024 if len(abl_2024) else abl_full, imp, meta, data, level_back)

    summary = {
        "target": TARGET,
        "backtest_mae_pivot": back.pivot_table(index="model", columns="test_year", values="MAE")
        .round(6)
        .to_dict(),
        "ablation_2024": abl_2024.sort_values("MAE")[["features", "MAE"]].round(6).to_dict(
            orient="records"
        )
        if len(abl_2024)
        else [],
        "top_features": imp.head(7).to_dict(orient="records"),
        "conformal": meta,
        "n_rows": len(data),
        "years": sorted(int(y) for y in data.year.unique()),
        "baselines": {
            "ZeroSwing": "predict ΔV=0",
            "PreviousSwing": "predict ΔV(t)=prev_swing=ΔV(t-1)",
        },
    }
    (OUT / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print("DONE →", OUT)


if __name__ == "__main__":
    main()
