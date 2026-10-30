#!/usr/bin/env python3
"""Pilot: aggregate X (Twitter) public-search sentiment for TW/US election Extension.
Does NOT touch Family A. Uses Grok Bot X MCP search_posts_all dumps only.
"""
from __future__ import annotations
import json, re, csv
from pathlib import Path
from collections import defaultdict
from statistics import mean

ROOT = Path("/workspace/taiwan-election-data")
RAW = ROOT / "data/x_sentiment_extension/raw"
OUT_DATA = ROOT / "data/x_sentiment_extension"
OUT_RES = ROOT / "results/x_sentiment_extension"
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_RES.mkdir(parents=True, exist_ok=True)

# File stem -> metadata
FILE_META = {
    "tw2024_lai": dict(country="TW", year=2024, entity="賴清德", side="DPP", kind="person",
                       window="2023-12-01..2024-01-12"),
    "tw2024_hou": dict(country="TW", year=2024, entity="侯友宜", side="KMT", kind="person",
                       window="2023-12-01..2024-01-12"),
    "tw2024_ko": dict(country="TW", year=2024, entity="柯文哲", side="TPP", kind="person",
                      window="2023-12-01..2024-01-12"),
    "tw2024_dpp": dict(country="TW", year=2024, entity="民進黨", side="DPP", kind="party",
                       window="2023-12-01..2024-01-12"),
    "tw2024_kmt": dict(country="TW", year=2024, entity="國民黨", side="KMT", kind="party",
                       window="2023-12-01..2024-01-12"),
    "tw2024_policy_crossstrait": dict(country="TW", year=2024, entity="兩岸", side="TOPIC", kind="policy",
                                      window="2023-12-01..2024-01-12"),
    "tw2020_tsai": dict(country="TW", year=2020, entity="蔡英文", side="DPP", kind="person",
                        window="2019-12-01..2020-01-10"),
    "tw2020_dpp": dict(country="TW", year=2020, entity="民進黨", side="DPP", kind="party",
                       window="2019-12-01..2020-01-10"),
    "tw2020_kmt": dict(country="TW", year=2020, entity="國民黨", side="KMT", kind="party",
                       window="2019-12-01..2020-01-10"),
    "us2024_trump": dict(country="US", year=2024, entity="Trump", side="GOP", kind="person",
                         window="2024-10-01..2024-11-04"),
    "us2024_harris": dict(country="US", year=2024, entity="Harris", side="DEM", kind="person",
                          window="2024-10-01..2024-11-04"),
    "us2024_dem": dict(country="US", year=2024, entity="Democrats", side="DEM", kind="party",
                       window="2024-10-01..2024-11-04"),
    "us2024_gop": dict(country="US", year=2024, entity="Republicans", side="GOP", kind="party",
                       window="2024-10-01..2024-11-04"),
    "us2024_policy_immigration": dict(country="US", year=2024, entity="immigration", side="TOPIC", kind="policy",
                                      window="2024-10-01..2024-11-04"),
    "us2020_trump": dict(country="US", year=2020, entity="Trump", side="GOP", kind="person",
                         window="2020-10-01..2020-11-02"),
    "us2020_biden": dict(country="US", year=2020, entity="Biden", side="DEM", kind="person",
                         window="2020-10-01..2020-11-02"),
    "us2020_dem": dict(country="US", year=2020, entity="Democrats", side="DEM", kind="party",
                       window="2020-10-01..2020-11-02"),
    "us2020_gop": dict(country="US", year=2020, entity="Republicans", side="GOP", kind="party",
                       window="2020-10-01..2020-11-02"),
}

# Outcomes (national)
# TW vote_share from cec_president_national_totals; winner = highest share
TW_OUT = {
    2020: {"winner_side": "DPP", "winner": "蔡英文", "dpp_share": 0.571307, "kmt_share": 0.386137},
    2024: {"winner_side": "DPP", "winner": "賴清德", "dpp_share": 0.400503, "kmt_share": 0.334900, "tpp_share": 0.264597},
}
# US two-party Dem share from us_national_poll_vs_outcome actual
US_OUT = {
    2020: {"winner_side": "DEM", "winner": "Biden", "dem_two_party": 0.522578},
    2024: {"winner_side": "GOP", "winner": "Trump", "dem_two_party": 0.492378},
}

ZH_POS = "支持 凍蒜 加油 勝利 當選 優秀 信任 值得 團結 進步 讚 棒 看好 穩了 必勝 愛台灣".split()
ZH_NEG = "賣台 腐敗 貪污 黑金 可惡 垃圾 騙子 舔共 戰爭 完蛋 無能 違建 抄襲 拒投 下台 背叛 謊言 骯髒".split()

def load_vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()

def load_snow():
    from snownlp import SnowNLP
    return SnowNLP

def zh_sent(text: str, SnowNLP) -> float:
    """Return score in [-1,1]: SnowNLP 0..1 mapped + lexicon nudge."""
    try:
        base = SnowNLP(text).sentiments  # 0..1
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
    return 2 * s01 - 1  # [-1,1]

def en_sent(text: str, sia) -> float:
    return float(sia.polarity_scores(text)["compound"])

def parse_raw_file(path: Path) -> list[dict]:
    t = path.read_text(encoding="utf-8", errors="replace")
    # strip untrusted fence wrappers if present
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return []
    obj = json.loads(m.group(0))
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

def stem_key(name: str) -> str | None:
    # tw2024_lai_p1.json.txt -> tw2024_lai
    m = re.match(r"(.+)_p\d+\.json\.txt$", name)
    return m.group(1) if m else None

def main():
    sia = load_vader()
    SnowNLP = load_snow()

    posts_rows = []
    by_key = defaultdict(list)

    for path in sorted(RAW.glob("*.json.txt")):
        key = stem_key(path.name)
        if key not in FILE_META:
            continue
        meta = FILE_META[key]
        posts = parse_raw_file(path)
        for p in posts:
            lang = (p.get("lang") or "").lower()
            text = p["text"]
            if meta["country"] == "TW" or lang.startswith("zh"):
                score = zh_sent(text, SnowNLP)
                method = "snownlp+zh_lexicon"
            else:
                score = en_sent(text, sia)
                method = "vader"
            label = "pos" if score > 0.05 else ("neg" if score < -0.05 else "neu")
            row = {
                **{k: meta[k] for k in ("country", "year", "entity", "side", "kind", "window")},
                "source_file": path.name,
                "post_id": p["id"],
                "created_at": p["created_at"],
                "lang": p["lang"],
                "text": text.replace("\n", " ")[:500],
                "sentiment": round(score, 4),
                "sent_label": label,
                "method": method,
                "engagement": p["engagement"],
                "like_count": p["like_count"],
                "url": p["url"],
            }
            posts_rows.append(row)
            by_key[key].append(row)

    # Deduplicate by post_id within entity
    seen = set()
    deduped = []
    for r in posts_rows:
        k = (r["country"], r["year"], r["entity"], r["post_id"])
        if k in seen:
            continue
        seen.add(k)
        deduped.append(r)
    posts_rows = deduped

    # Write posts long
    posts_csv = OUT_DATA / "x_posts_scored_long.csv"
    fields = list(posts_rows[0].keys()) if posts_rows else []
    with posts_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(posts_rows)

    # Aggregate by country-year-entity
    agg = defaultdict(list)
    for r in posts_rows:
        agg[(r["country"], r["year"], r["entity"], r["side"], r["kind"], r["window"])].append(r)

    agg_rows = []
    for (country, year, entity, side, kind, window), rows in sorted(agg.items()):
        scores = [r["sentiment"] for r in rows]
        n = len(scores)
        n_pos = sum(1 for r in rows if r["sent_label"] == "pos")
        n_neg = sum(1 for r in rows if r["sent_label"] == "neg")
        eng = sum(r["engagement"] or 0 for r in rows)
        agg_rows.append({
            "country": country,
            "election_year": year,
            "entity": entity,
            "side": side,
            "kind": kind,
            "window_utc": window,
            "n_posts": n,
            "mean_sentiment": round(mean(scores), 4) if scores else None,
            "pct_pos": round(n_pos / n, 4) if n else None,
            "pct_neg": round(n_neg / n, 4) if n else None,
            "engagement_sum": eng,
            "mean_engagement": round(eng / n, 2) if n else None,
        })

    agg_csv = OUT_DATA / "x_entity_year_aggregates.csv"
    with agg_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(agg_rows)

    # Differentials vs winners
    diff_rows = []
    # TW: DPP - KMT mean (party + person blended separately)
    for year in (2020, 2024):
        rows = [r for r in agg_rows if r["country"] == "TW" and r["election_year"] == year]
        def side_mean(side, kinds=None):
            sel = [r for r in rows if r["side"] == side and (kinds is None or r["kind"] in kinds)]
            if not sel:
                return None, 0
            # volume-weighted
            tot_n = sum(r["n_posts"] for r in sel)
            if tot_n == 0:
                return None, 0
            m = sum(r["mean_sentiment"] * r["n_posts"] for r in sel) / tot_n
            return m, tot_n
        dpp_m, dpp_n = side_mean("DPP")
        kmt_m, kmt_n = side_mean("KMT")
        person_dpp, _ = side_mean("DPP", {"person"})
        person_kmt, _ = side_mean("KMT", {"person"})
        out = TW_OUT[year]
        diff = None if dpp_m is None or kmt_m is None else round(dpp_m - kmt_m, 4)
        tracks = None
        if diff is not None:
            # positive diff = DPP more positive online → should match DPP win
            tracks = (diff > 0 and out["winner_side"] == "DPP") or (diff < 0 and out["winner_side"] == "KMT")
        diff_rows.append({
            "country": "TW",
            "election_year": year,
            "metric": "mean_sent_DPP_minus_KMT_volume_weighted",
            "diff": diff,
            "dpp_mean": None if dpp_m is None else round(dpp_m, 4),
            "kmt_mean": None if kmt_m is None else round(kmt_m, 4),
            "dpp_n": dpp_n,
            "kmt_n": kmt_n,
            "person_dpp_mean": None if person_dpp is None else round(person_dpp, 4),
            "person_kmt_mean": None if person_kmt is None else round(person_kmt, 4),
            "national_winner_side": out["winner_side"],
            "national_winner": out["winner"],
            "dpp_vote_share": out["dpp_share"],
            "kmt_vote_share": out["kmt_share"],
            "diff_sign_matches_winner": tracks,
            "note": "Pilot sample (≤2 pages/entity); X TW user base skewed; not causal",
        })

    for year in (2020, 2024):
        rows = [r for r in agg_rows if r["country"] == "US" and r["election_year"] == year]
        def side_mean(side):
            sel = [r for r in rows if r["side"] == side]
            tot_n = sum(r["n_posts"] for r in sel)
            if tot_n == 0:
                return None, 0
            m = sum(r["mean_sentiment"] * r["n_posts"] for r in sel) / tot_n
            return m, tot_n
        dem_m, dem_n = side_mean("DEM")
        gop_m, gop_n = side_mean("GOP")
        out = US_OUT[year]
        diff = None if dem_m is None or gop_m is None else round(dem_m - gop_m, 4)
        tracks = None
        if diff is not None:
            tracks = (diff > 0 and out["winner_side"] == "DEM") or (diff < 0 and out["winner_side"] == "GOP")
        diff_rows.append({
            "country": "US",
            "election_year": year,
            "metric": "mean_sent_DEM_minus_GOP_volume_weighted",
            "diff": diff,
            "dem_mean": None if dem_m is None else round(dem_m, 4),
            "gop_mean": None if gop_m is None else round(gop_m, 4),
            "dem_n": dem_n,
            "gop_n": gop_n,
            "person_dpp_mean": "",
            "person_kmt_mean": "",
            "national_winner_side": out["winner_side"],
            "national_winner": out["winner"],
            "dpp_vote_share": "",  # reuse cols loosely
            "kmt_vote_share": "",
            "dem_two_party_actual": out["dem_two_party"],
            "diff_sign_matches_winner": tracks,
            "note": "Pilot sample; English VADER; X platform skew; not causal",
        })

    # Normalize diff CSV columns
    diff_fields = [
        "country", "election_year", "metric", "diff",
        "dpp_mean", "kmt_mean", "dpp_n", "kmt_n",
        "dem_mean", "gop_mean", "dem_n", "gop_n",
        "person_dpp_mean", "person_kmt_mean",
        "national_winner_side", "national_winner",
        "dpp_vote_share", "kmt_vote_share", "dem_two_party_actual",
        "diff_sign_matches_winner", "note",
    ]
    for r in diff_rows:
        for k in diff_fields:
            r.setdefault(k, "")

    diff_csv = OUT_RES / "sentiment_differential_vs_winner.csv"
    with diff_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=diff_fields)
        w.writeheader()
        w.writerows(diff_rows)

    # Also copy aggregates to results
    (OUT_RES / "x_entity_year_aggregates.csv").write_text(agg_csv.read_text(encoding="utf-8"), encoding="utf-8")

    # Tiny note regression-style: correlate |diff| with margin — n=4 only descriptive
    summary_path = OUT_RES / "pilot_summary.json"
    summary = {
        "n_posts_scored": len(posts_rows),
        "n_entity_year_rows": len(agg_rows),
        "x_api_calls_approx": 24,  # documented estimate for filtered pilot pages
        "family_a_unchanged": True,
        "grain": "election-year × entity (national); no county join (no reliable geo)",
        "differentials": diff_rows,
        "aggregates_preview": agg_rows,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Provenance
    prov = {
        "source": "Grok Bot built-in X MCP search_posts_all (full archive public read)",
        "no_html_scrape": True,
        "no_user_api_keys": True,
        "filters": "-is:retweet -is:reply; lang:zh (TW) / lang:en (US); election keywords",
        "sentiment": {
            "zh": "SnowNLP (0..1→[-1,1]) + small political pos/neg lexicon nudge",
            "en": "vaderSentiment compound [-1,1]",
        },
        "limits": "max 25/page; pilot 1–2 pages/entity; platform skew; spam residual; not population-representative",
        "not_family_a": True,
        "raw_dir": str(RAW.relative_to(ROOT)),
        "date_built": "2026-10-30",
        "timezone_note": "Windows are UTC as in search_posts_all start_time/end_time",
    }
    (OUT_DATA / "PROVENANCE.json").write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")

    print("POSTS", len(posts_rows))
    print("AGG rows", len(agg_rows))
    for r in agg_rows:
        print(f"  {r['country']}{r['election_year']} {r['entity']}: n={r['n_posts']} mean={r['mean_sentiment']} pos%={r['pct_pos']} neg%={r['pct_neg']}")
    print("--- DIFF ---")
    for r in diff_rows:
        print(r["country"], r["election_year"], "diff=", r["diff"], "tracks=", r["diff_sign_matches_winner"], "winner=", r["national_winner"])

if __name__ == "__main__":
    main()
