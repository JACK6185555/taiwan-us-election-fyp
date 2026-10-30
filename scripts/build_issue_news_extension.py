#!/usr/bin/env python3
"""News / Policy Issue Salience Extension — national×year only. Does NOT touch Core specification."""
from __future__ import annotations
import json, re, csv
from pathlib import Path
from collections import defaultdict
from statistics import mean
import math

ROOT = Path("/workspace/taiwan-election-data")
RAW = ROOT / "data/extension_issue_news/raw"
OUT_DATA = ROOT / "data/extension_issue_news"
OUT_RES = ROOT / "results/extension_issue_news"
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_RES.mkdir(parents=True, exist_ok=True)

FILE_META = {
    "tw2024_issue_crossstrait": dict(country="TW", year=2024, issue="兩岸", issue_en="cross_strait",
                                     window="2023-12-01..2024-01-12"),
    "tw2024_issue_housing": dict(country="TW", year=2024, issue="居住", issue_en="housing",
                                 window="2023-12-01..2024-01-12"),
    "tw2024_issue_energy": dict(country="TW", year=2024, issue="能源", issue_en="energy",
                                window="2023-12-01..2024-01-12"),
    "tw2024_issue_economy": dict(country="TW", year=2024, issue="經濟", issue_en="economy",
                                 window="2023-12-01..2024-01-12"),
    "tw2020_issue_crossstrait": dict(country="TW", year=2020, issue="兩岸", issue_en="cross_strait",
                                     window="2019-12-01..2020-01-10"),
    "tw2020_issue_housing": dict(country="TW", year=2020, issue="居住", issue_en="housing",
                                 window="2019-12-01..2020-01-10"),
    "tw2020_issue_energy": dict(country="TW", year=2020, issue="能源", issue_en="energy",
                                window="2019-12-01..2020-01-10"),
    "tw2020_issue_economy": dict(country="TW", year=2020, issue="經濟", issue_en="economy",
                                 window="2019-12-01..2020-01-10"),
    "tw2020_issue_pandemic": dict(country="TW", year=2020, issue="防疫", issue_en="pandemic",
                                  window="2019-12-01..2020-01-10"),
    "us2024_issue_economy": dict(country="US", year=2024, issue="economy", issue_en="economy",
                                 window="2024-10-01..2024-11-04"),
    "us2024_issue_immigration": dict(country="US", year=2024, issue="immigration", issue_en="immigration",
                                     window="2024-10-01..2024-11-04"),
    "us2024_issue_abortion": dict(country="US", year=2024, issue="abortion", issue_en="abortion",
                                  window="2024-10-01..2024-11-04"),
    "us2024_issue_healthcare": dict(country="US", year=2024, issue="healthcare", issue_en="healthcare",
                                    window="2024-10-01..2024-11-04"),
    "us2024_issue_democracy": dict(country="US", year=2024, issue="democracy", issue_en="democracy",
                                   window="2024-10-01..2024-11-04"),
    "us2020_issue_economy": dict(country="US", year=2020, issue="economy", issue_en="economy",
                                 window="2020-10-01..2020-11-02"),
    "us2020_issue_immigration": dict(country="US", year=2020, issue="immigration", issue_en="immigration",
                                     window="2020-10-01..2020-11-02"),
    "us2020_issue_abortion": dict(country="US", year=2020, issue="abortion", issue_en="abortion",
                                  window="2020-10-01..2020-11-02"),
    "us2020_issue_healthcare": dict(country="US", year=2020, issue="healthcare", issue_en="healthcare",
                                    window="2020-10-01..2020-11-02"),
    "us2020_issue_democracy": dict(country="US", year=2020, issue="democracy", issue_en="democracy",
                                   window="2020-10-01..2020-11-02"),
}

TW_OUT = {
    2020: {"winner_side": "DPP", "winner": "蔡英文", "dpp_share": 0.571307, "kmt_share": 0.386137,
           "outcome_share": 0.571307},
    2024: {"winner_side": "DPP", "winner": "賴清德", "dpp_share": 0.400503, "kmt_share": 0.334900,
           "outcome_share": 0.400503},
}
US_OUT = {
    2020: {"winner_side": "DEM", "winner": "Biden", "dem_two_party": 0.522578, "outcome_share": 0.522578},
    2024: {"winner_side": "GOP", "winner": "Trump", "dem_two_party": 0.492378, "outcome_share": 0.492378},
}

ZH_POS = "支持 凍蒜 加油 勝利 當選 優秀 信任 值得 團結 進步 讚 棒 看好 穩了 必勝 愛台灣".split()
ZH_NEG = "賣台 腐敗 貪污 黑金 可惡 垃圾 騙子 舔共 戰爭 完蛋 無能 違建 抄襲 拒投 下台 背叛 謊言 骯髒".split()


def load_vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def load_snow():
    from snownlp import SnowNLP
    return SnowNLP


def zh_sent(text, SnowNLP):
    try:
        base = SnowNLP(text).sentiments
    except Exception:
        base = 0.5
    nudge = 0.0
    for w in ZH_POS:
        if w in text:
            nudge += 0.07
    for w in ZH_NEG:
        if w in text:
            nudge -= 0.07
    s01 = max(0.0, min(1.0, base + nudge))
    return 2 * s01 - 1


def en_sent(text, sia):
    return float(sia.polarity_scores(text)["compound"])


def parse_raw_file(path: Path) -> list[dict]:
    t = path.read_text(encoding="utf-8", errors="replace")
    # strip untrusted fence if present
    t2 = re.sub(r"</?cursor_untrusted_data_1337[^>]*>", "", t)
    m = re.search(r"\{.*\}", t2, re.S)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    posts = obj.get("data") or []
    out = []
    for p in posts:
        pm = p.get("public_metrics") or {}
        eng = (
            int(pm.get("like_count") or 0)
            + int(pm.get("retweet_count") or 0)
            + int(pm.get("reply_count") or 0)
            + int(pm.get("quote_count") or 0)
        )
        out.append({
            "id": p.get("id"),
            "text": p.get("text") or "",
            "created_at": p.get("created_at"),
            "lang": p.get("lang"),
            "author_id": p.get("author_id"),
            "like_count": pm.get("like_count"),
            "retweet_count": pm.get("retweet_count"),
            "reply_count": pm.get("reply_count"),
            "quote_count": pm.get("quote_count"),
            "impression_count": pm.get("impression_count"),
            "engagement": eng,
            "url": p.get("url"),
        })
    return out


def stem_key(name: str):
    m = re.match(r"(.+)_p\d+\.json\.txt$", name)
    return m.group(1) if m else None


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def write_csv(path: Path, rows: list[dict], fieldnames=None):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def platform_issue_tables():
    import pandas as pd
    tw = pd.read_csv(ROOT / "data/extension_poll_platform/taiwan_platforms_long.csv")
    us = pd.read_csv(ROOT / "data/us_parity_extension/us_platforms_long.csv")
    rows = []
    for year in sorted(tw["year"].unique()):
        sub = tw[tw["year"] == year]
        n = len(sub)
        for cat, g in sub.groupby("issue_category"):
            rows.append({
                "country": "TW", "year": int(year), "source": "taiwan_platforms_long",
                "issue_category": cat, "n_rows": int(len(g)),
                "share_of_year_rows": round(len(g) / n, 4) if n else None,
            })
    for year in sorted(us["year"].unique()):
        sub = us[us["year"] == year]
        n = len(sub)
        for cat, g in sub.groupby("issue_category"):
            rows.append({
                "country": "US", "year": int(year), "source": "us_platforms_long",
                "issue_category": cat, "n_rows": int(len(g)),
                "share_of_year_rows": round(len(g) / n, 4) if n else None,
            })
    write_csv(OUT_DATA / "platform_issue_salience_by_year.csv", rows)
    write_csv(OUT_RES / "platform_issue_salience_by_year.csv", rows)
    # focus 2020/2024
    focus = [r for r in rows if r["year"] in (2020, 2024)]
    write_csv(OUT_RES / "platform_issue_salience_2020_2024.csv", focus)
    return rows


def national_tiny_regression(agg_rows):
    """National grain only: outcome ~ prev_share + top-issue volume share. Label Extension."""
    # Build year-level features: volume share of top issues
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    df = pd.DataFrame(agg_rows)
    out_rows = []
    notes = []

    for country, years_out in [("TW", TW_OUT), ("US", US_OUT)]:
        sub = df[df["country"] == country].copy()
        if sub.empty:
            continue
        # pivot volume
        piv = sub.pivot_table(index="year", columns="issue_en", values="n_posts", aggfunc="sum").fillna(0)
        piv["total"] = piv.sum(axis=1)
        for c in [c for c in piv.columns if c != "total"]:
            piv[f"volshare_{c}"] = piv[c] / piv["total"].replace(0, np.nan)
        # mean tone overall
        tone = sub.groupby("year")["mean_sentiment"].mean()
        piv["mean_tone_all"] = tone
        # outcome + prev
        y, prev, feat_names = [], [], []
        Xlist = []
        for yr in sorted(piv.index):
            if yr not in years_out:
                continue
            o = years_out[yr]
            y.append(o["outcome_share"])
            # previous share
            prev_yr = yr - 4
            if prev_yr in years_out:
                prev.append(years_out[prev_yr]["outcome_share"])
            else:
                # TW 2020 prev = 2016 from cec
                if country == "TW" and yr == 2020:
                    prev.append(0.561234)
                elif country == "US" and yr == 2020:
                    prev.append(0.511043)  # 2016 Dem two-party from poll_vs_outcome
                else:
                    prev.append(float("nan"))
            # features: volume shares + tone
            feats = []
            names = []
            for col in sorted([c for c in piv.columns if c.startswith("volshare_")]):
                feats.append(float(piv.loc[yr, col]) if not pd.isna(piv.loc[yr, col]) else 0.0)
                names.append(col)
            feats.append(float(piv.loc[yr, "mean_tone_all"]) if not pd.isna(piv.loc[yr, "mean_tone_all"]) else 0.0)
            names.append("mean_tone_all")
            Xlist.append(feats)
            feat_names = names

        # With n=2 years per country, cannot fit reliable multi-feature Ridge.
        # Report associations only + a toy "prev vs prev+tone" note.
        notes.append({
            "country": country,
            "n_years": len(y),
            "note": "n_years<=2 → 唔做多特徵 Ridge 擬合；只報相關／描述。若強行擬合會過擬合。",
            "features_available": ",".join(feat_names),
            "outcomes": ",".join(f"{yr}:{years_out[yr]['outcome_share']:.4f}" for yr in sorted(piv.index) if yr in years_out),
        })

        # Cross-issue within-year salience ranks already in aggregates.
        # Country-pooled across 4 elections for volume of shared economy issue vs outcome:
        # skip causal.

    write_csv(OUT_RES / "national_extension_regression_NOTE.csv", notes)

    # Pooled descriptive: for each election, top issue by n_posts and by |mean_sentiment|
    top = []
    for (country, year), g in df.groupby(["country", "year"]):
        g2 = g.sort_values("n_posts", ascending=False)
        top_vol = g2.iloc[0]
        g3 = g.assign(abs_tone=g["mean_sentiment"].abs()).sort_values("abs_tone", ascending=False)
        top_tone = g3.iloc[0]
        out = TW_OUT.get(year) if country == "TW" else US_OUT.get(year)
        top.append({
            "country": country, "year": int(year),
            "top_volume_issue": top_vol["issue"], "top_volume_n": int(top_vol["n_posts"]),
            "top_volume_mean_sent": round(float(top_vol["mean_sentiment"]), 4),
            "most_negative_or_polarized_issue": top_tone["issue"],
            "that_mean_sent": round(float(top_tone["mean_sentiment"]), 4),
            "winner": out["winner"] if out else None,
            "outcome_share": out["outcome_share"] if out else None,
            "interpretation": "議題氣候 PROXY；唔等於中間選民個人轉向",
        })
    write_csv(OUT_RES / "top_issue_signals_by_election.csv", top)
    return top, notes


def associations(agg_rows):
    """Simple associations — n small, say so."""
    import pandas as pd
    df = pd.DataFrame(agg_rows)
    rows = []
    # Within each country-year: rank volume vs nothing (descriptive)
    # Across elections: economy volume share vs outcome (pooled TW+US = 4 pts)
    for country in ["TW", "US"]:
        sub = df[df["country"] == country]
        years = sorted(sub["year"].unique())
        outs = TW_OUT if country == "TW" else US_OUT
        # For each issue present in both years, correlate mean_sentiment with outcome_share (n=2 → nan)
        for issue in sorted(sub["issue_en"].unique()):
            iss = sub[sub["issue_en"] == issue]
            xs, ys, vols = [], [], []
            for _, r in iss.iterrows():
                yr = int(r["year"])
                if yr not in outs:
                    continue
                xs.append(float(r["mean_sentiment"]))
                ys.append(outs[yr]["outcome_share"])
                vols.append(float(r["n_posts"]))
            # volume share within year
            volshares = []
            for _, r in iss.iterrows():
                yr = int(r["year"])
                tot = sub[sub["year"] == yr]["n_posts"].sum()
                volshares.append(float(r["n_posts"]) / tot if tot else float("nan"))
            rows.append({
                "country": country,
                "issue": issue,
                "n_elections_with_issue": len(xs),
                "corr_mean_sent_vs_outcome": round(pearson(xs, ys), 4) if len(xs) >= 3 else "n<3_not_reported",
                "corr_volshare_vs_outcome": round(pearson(volshares, ys), 4) if len(volshares) >= 3 else "n<3_not_reported",
                "mean_sent_by_year": ";".join(f"{int(r['year'])}:{r['mean_sentiment']:.3f}" for _, r in iss.iterrows()),
                "n_posts_by_year": ";".join(f"{int(r['year'])}:{int(r['n_posts'])}" for _, r in iss.iterrows()),
                "caveat": "n 極細；相關唔等於因果；X 樣本有偏",
            })
    # Pooled 4 elections economy tone vs Dem/DPP share
    econ = df[df["issue_en"] == "economy"]
    xs, ys = [], []
    for _, r in econ.iterrows():
        c, yr = r["country"], int(r["year"])
        o = TW_OUT.get(yr) if c == "TW" else US_OUT.get(yr)
        if not o:
            continue
        xs.append(float(r["mean_sentiment"]))
        ys.append(o["outcome_share"])
    rows.append({
        "country": "TW+US_pooled",
        "issue": "economy",
        "n_elections_with_issue": len(xs),
        "corr_mean_sent_vs_outcome": round(pearson(xs, ys), 4) if len(xs) >= 3 else "n<3_not_reported",
        "corr_volshare_vs_outcome": "n/a_mixed_denominators",
        "mean_sent_by_year": ";".join(f"{r['country']}{int(r['year'])}:{r['mean_sentiment']:.3f}" for _, r in econ.iterrows()),
        "n_posts_by_year": ";".join(f"{r['country']}{int(r['year'])}:{int(r['n_posts'])}" for _, r in econ.iterrows()),
        "caveat": "跨國混合 DPP share 同 Dem two-party；只作敏感度",
    })
    write_csv(OUT_RES / "issue_outcome_associations.csv", rows)
    return rows


def main():
    sia = load_vader()
    SnowNLP = load_snow()

    posts_rows = []
    by_key = defaultdict(list)

    for path in sorted(RAW.glob("*.json.txt")):
        key = stem_key(path.name)
        if key not in FILE_META:
            print("skip unknown", path.name)
            continue
        meta = FILE_META[key]
        posts = parse_raw_file(path)
        for p in posts:
            text = p["text"]
            if meta["country"] == "TW":
                s = zh_sent(text, SnowNLP)
            else:
                s = en_sent(text, sia)
            row = {**meta, **p, "sentiment": s, "source_file": path.name}
            posts_rows.append(row)
            by_key[key].append(row)

    # long scored
    write_csv(OUT_DATA / "issue_posts_scored_long.csv", posts_rows)

    # aggregates
    agg = []
    for key, rows in sorted(by_key.items()):
        meta = FILE_META[key]
        sents = [r["sentiment"] for r in rows]
        n = len(sents)
        pos = sum(1 for s in sents if s > 0.05) / n if n else 0
        neg = sum(1 for s in sents if s < -0.05) / n if n else 0
        eng = sum(r["engagement"] or 0 for r in rows)
        agg.append({
            **meta,
            "n_posts": n,
            "mean_sentiment": round(mean(sents), 4) if sents else None,
            "pct_pos": round(pos, 4),
            "pct_neg": round(neg, 4),
            "engagement_sum": eng,
            "volume_rank_note": "within-country-year rank computed below",
        })

    # volume share within country-year
    from collections import Counter
    totals = Counter()
    for a in agg:
        totals[(a["country"], a["year"])] += a["n_posts"]
    for a in agg:
        tot = totals[(a["country"], a["year"])]
        a["volume_share"] = round(a["n_posts"] / tot, 4) if tot else None

    # ranks
    by_cy = defaultdict(list)
    for a in agg:
        by_cy[(a["country"], a["year"])].append(a)
    for _, lst in by_cy.items():
        lst_sorted = sorted(lst, key=lambda x: -x["n_posts"])
        for i, a in enumerate(lst_sorted, 1):
            a["volume_rank"] = i

    write_csv(OUT_DATA / "issue_year_aggregates.csv", agg)
    write_csv(OUT_RES / "issue_salience_tone_by_election.csv", agg)

    platform_issue_tables()
    assoc = associations(agg)
    top, notes = national_tiny_regression(agg)

    summary = {
        "date": "2026-10-30",
        "timezone": "Asia/Hong_Kong",
        "n_posts_scored": len(posts_rows),
        "n_issue_year_cells": len(agg),
        "x_api_calls_this_extension": 19,
        "gdelt": "HTTP 429 — not used; see data/extension_issue_news/gdelt/GDELT_STATUS.md",
        "family_a_unchanged": True,
        "top_signals": top,
        "regression_note": notes,
        "caveat": "National Extension PROXY for campaign issue climate; cannot ID individual swing voters",
    }
    (OUT_RES / "pilot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DATA / "PROVENANCE.json").write_text(json.dumps({
        "source": "Grok Bot built-in X MCP search_posts_all (full archive); platform tags from extension_poll_platform / us_parity_extension",
        "not_html_scrape": True,
        "windows_utc": {
            "TW2024": "2023-12-01..2024-01-12",
            "TW2020": "2019-12-01..2020-01-10",
            "US2024": "2024-10-01..2024-11-04",
            "US2020": "2020-10-01..2020-11-02",
        },
        "sentiment": {"TW": "SnowNLP+lexicon nudge", "US": "VADER compound"},
        "gdelt": "429 failed",
        "family_a": "NOT modified",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DATA / "README.md").write_text(
        "# Extension: News / Policy Issue Salience\n\n"
        "National×election-year volume + tone from X public search + platform issue tags.\n"
        "**Not Core specification.** See `docs/新聞政策議題特徵Extension說明.md`.\n",
        encoding="utf-8",
    )
    print(f"scored {len(posts_rows)} posts, {len(agg)} issue×year cells")
    for a in sorted(agg, key=lambda x: (x["country"], x["year"], -x["n_posts"])):
        print(f"  {a['country']}{a['year']} {a['issue']}: n={a['n_posts']} mean={a['mean_sentiment']} volshare={a['volume_share']}")


if __name__ == "__main__":
    main()
