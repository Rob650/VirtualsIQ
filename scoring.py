"""
VirtualsIQ — Scoring Engine v4 (benchmark-based)

Ranking hierarchy (weights):
  1. Idea × Market Fit  50%  — uniqueness + TAM (geometric mean)
  2. Moat               20%  — defensibility + community + age
  3. Market             25%  — on-chain validation vs benchmark top-20
  4. Execution          10%  — team signals, presence (tiebreaker, NOT a gate)

Benchmark approach:
  - Identify the top ~20 real projects (high holders, volume, market cap).
    These define what a "90-point" market score looks like.
  - Score every agent's market metrics relative to those benchmarks on a
    log scale — so small projects aren't crushed to zero, but the gap
    between a 1M-holder project and a 50-holder project is meaningful.
  - Benchmarks are derived from live data (excl. 365love whose Virtuals
    API market cap is known to be wrong).

Philosophy:
  - A unique idea in a large market scores well even with zero doxx.
  - Moat separates durable from fragile projects; community size counts.
  - Market metrics confirm the idea resonates with real users.
  - NULL / empty fields → factor is SKIPPED (no penalty for absence).
    Remaining factors are re-weighted proportionally.
  - doxx_tier = 3 (anonymous) IS real data and scores accordingly; it is
    NOT treated as "missing." True missing = null/empty field.

Score range: 8 (floor) → 95 (elite)
dead_flagged / strong_flagged are UI badges only — zero scoring impact.

Expected distribution with this calibration:
  85–95  Top-tier: elite idea + proven market (AI-analyzed + benchmark MC/holders)
  65–84  Strong: good idea or strong market, solid moat
  40–64  Moderate: decent idea, limited market validation
  20–39  Weak: poor idea/execution, minimal market signal
  8–19   Avoid: meme/generic + near-zero market presence
"""

from __future__ import annotations

import math
from datetime import datetime


# ---------------------------------------------------------------------------
# Display labels
# ---------------------------------------------------------------------------

TIER_LABELS = {
    "idea":      "Idea × Market Fit",
    "moat":      "Moat",
    "execution": "Execution",
    "market":    "Market",
}

FACTOR_LABELS = {
    "F_idea_market_fit": "Idea × Market Fit",
    "F_moat":            "Moat / Defensibility",
    "F_execution":       "Execution Signals",
    "F_holders":         "Holder Count",
    "F_mcap":            "Market Cap",
    "F_volume":          "24h Volume",
    "F_efficiency":      "Trading Efficiency",
    "F_momentum":        "Holder Momentum",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(value, default=0.0) -> float:
    try:
        v = float(value)
        return v if not math.isnan(v) else default
    except (TypeError, ValueError):
        return default


def _clamp(v: float, lo=0.0, hi=100.0) -> float:
    return max(lo, min(hi, v))


def _days_since(date_str) -> float | None:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(str(date_str)[:26], fmt[:len(str(date_str)[:26])])
            return (datetime.utcnow() - dt).total_seconds() / 86400
        except ValueError:
            continue
    return None


def _log_score(value: float, breakpoints: list) -> float:
    """Log-interpolated score from sorted (threshold, score) breakpoints."""
    if value <= 0:
        return breakpoints[0][1] if breakpoints else 0.0
    for i, (thresh, score) in enumerate(breakpoints):
        if value <= thresh:
            if i == 0:
                return score
            prev_thresh, prev_score = breakpoints[i - 1]
            lv = math.log10(max(value, 1))
            ll = math.log10(max(prev_thresh, 1))
            lh = math.log10(max(thresh, 1))
            if lh == ll:
                return score
            t = (lv - ll) / (lh - ll)
            return prev_score + t * (score - prev_score)
    return breakpoints[-1][1]


# ---------------------------------------------------------------------------
# Factor 1 — UNIQUENESS  (40%)
# ---------------------------------------------------------------------------

# Category crowdedness → uniqueness base score
# Lower = more saturated / "another X" territory
# Higher = more differentiated / novel space
_CATEGORY_UNIQUENESS = {
    # Very saturated
    "meme":          12.0,
    "ai assistant":  15.0,
    "trading bot":   18.0,
    "ai agent":      20.0,
    "assistant":     22.0,
    # Saturated but with real utility
    "trading":       35.0,
    "defi":          40.0,
    "yield":         42.0,
    "lending":       45.0,
    "finance":       38.0,
    "nft":           30.0,
    "social":        35.0,
    "community":     33.0,
    "entertainment": 30.0,
    # Less crowded, more differentiated
    "gaming":        55.0,
    "game":          55.0,
    "creator":       50.0,
    "art":           47.0,
    "productivity":  58.0,
    "tools":         60.0,
    "analytics":     63.0,
    "data":          63.0,
    # Specialized / hard-to-replicate
    "infrastructure": 72.0,
    "infra":          72.0,
    "protocol":       75.0,
    "security":       78.0,
    "bridge":         70.0,
    "dev":            67.0,
    "oracle":         75.0,
}


def _category_uniqueness_score(agent: dict, ai: dict) -> float:
    cat = str(
        agent.get("category") or agent.get("agent_type") or
        ai.get("category") or ""
    ).lower().strip()
    for key in sorted(_CATEGORY_UNIQUENESS, key=len, reverse=True):
        if key in cat:
            return _CATEGORY_UNIQUENESS[key]
    return 42.0  # "other" / unknown


def _description_specificity(agent: dict) -> float | None:
    bio = str(agent.get("biography") or agent.get("description") or "").strip()
    if not bio:
        return None
    chars = len(bio)
    if chars >= 300: return 85.0
    if chars >= 150: return 68.0
    if chars >= 80:  return 52.0
    if chars >= 30:  return 35.0
    return 20.0


def _f_uniqueness(agent: dict, ai: dict) -> float:
    """
    Primary score driver (40%). Always returns a value.

    Priority:
      1. AI first_mover signals (most accurate when available)
      2. Category crowdedness score + description specificity proxy
    """
    fm = ai.get("first_mover", {})
    ai_cat_unique = fm.get("category_unique")
    ai_novel      = fm.get("approach_novel")

    if ai_cat_unique is not None or ai_novel is not None:
        base = 50.0
        if ai_cat_unique is True:   base += 30.0
        elif ai_cat_unique is False: base -= 20.0
        if ai_novel is True:        base += 20.0
        elif ai_novel is False:     base -= 10.0
        days_ahead = fm.get("days_ahead_of_competitor")
        if days_ahead is not None:
            d = float(days_ahead)
            base += (10.0 if d > 180 else 7.0 if d > 90 else 3.0 if d > 30 else 0.0)
        return _clamp(round(base, 1))

    cat_score  = _category_uniqueness_score(agent, ai)
    desc_score = _description_specificity(agent)

    if desc_score is not None:
        blended = 0.60 * cat_score + 0.40 * desc_score
    else:
        blended = cat_score

    return _clamp(round(blended, 1))


# ---------------------------------------------------------------------------
# Factor 2 — TAM  (25%)
# ---------------------------------------------------------------------------

# TAM in USD mapped from category
_TAM_BY_CAT = {
    "defi":           100_000_000_000,
    "trading":        100_000_000_000,
    "finance":         90_000_000_000,
    "infrastructure":  80_000_000_000,
    "infra":           80_000_000_000,
    "protocol":        80_000_000_000,
    "security":        70_000_000_000,
    "gaming":          50_000_000_000,
    "game":            50_000_000_000,
    "social":          30_000_000_000,
    "community":       30_000_000_000,
    "entertainment":   20_000_000_000,
    "creative":        20_000_000_000,
    "art":             20_000_000_000,
    "nft":             20_000_000_000,
    "productivity":    15_000_000_000,
    "tools":           15_000_000_000,
    "analytics":       25_000_000_000,
    "data":            25_000_000_000,
    "meme":             5_000_000_000,
}

_TAM_BP = [
    (5_000_000_000,   35.0),
    (15_000_000_000,  55.0),
    (20_000_000_000,  62.0),
    (30_000_000_000,  70.0),
    (50_000_000_000,  80.0),
    (80_000_000_000,  90.0),
    (100_000_000_000, 95.0),
]


def _f_tam(agent: dict, ai: dict) -> float:
    """
    Addressable market score (25%). Always returns a value.

    Priority:
      1. AI tam_score (agent-specific TAM from AI analysis)
      2. Category → generic TAM lookup
    """
    ai_tam = ai.get("market", {}).get("tam_score")
    if ai_tam is not None:
        return _clamp(float(ai_tam))

    cat = str(
        agent.get("category") or agent.get("agent_type") or
        ai.get("category") or ""
    ).lower().strip()

    tam = 25_000_000_000  # default "other"
    for key, val in _TAM_BY_CAT.items():
        if key in cat:
            tam = val
            break

    return _clamp(_log_score(tam, _TAM_BP))


# ---------------------------------------------------------------------------
# Factor 3 — MOAT  (15%)
# ---------------------------------------------------------------------------

def _f_moat(agent: dict, ai: dict) -> float | None:
    """
    Defensibility/moat signals. Returns None if no signals available.
    Sources: AI defensibility_score, GitHub community, project age.
    """
    signals = []

    # AI defensibility
    ai_def = ai.get("first_mover", {}).get("defensibility_score")
    if ai_def is not None:
        signals.append((_clamp(float(ai_def)), 3.0))   # weight 3×

    # GitHub community = hard-to-replicate social moat
    stars        = _safe(agent.get("github_stars"), 0)
    contributors = _safe(agent.get("github_contributors"), 0)
    commits      = _safe(agent.get("github_commits_30d"), 0)
    if stars > 0 or contributors > 0 or commits > 0:
        gh_score  = min(60.0, math.log10(max(stars, 1)) * 22.0)
        gh_score += min(20.0, contributors * 5.0)
        gh_score += min(20.0, commits * 1.0)
        signals.append((_clamp(gh_score), 2.0))          # weight 2×

    # Project longevity (age) as proxy for sustained commitment
    days = _days_since(agent.get("creation_date") or agent.get("first_seen"))
    if days is not None:
        age_score = (80.0 if days > 365 else
                     65.0 if days > 180 else
                     48.0 if days > 90  else
                     30.0 if days > 30  else 15.0)
        signals.append((age_score, 1.0))                 # weight 1×

    # AI partnership score as additional moat signal
    partnership = ai.get("product", {}).get("partnership_score")
    if partnership is not None:
        signals.append((_clamp(float(partnership)), 1.5))

    # Community size: a large holder base is a hard-to-replicate network moat.
    # Only count if ≥ 1 000 holders (below that it's noise, not community).
    holders = _safe(agent.get("holder_count"), 0)
    if holders >= 1_000:
        # log10(1000)=3 → 20*3=60; log10(1M)=6 → 20*6=120 capped at 90
        community_moat = _clamp(20.0 * math.log10(holders))
        signals.append((community_moat, 2.0))  # weight 2× — community is durable

    if not signals:
        return None  # no defensibility data at all — skip

    w_sum   = sum(s * w for s, w in signals)
    w_total = sum(w       for _, w in signals)
    return _clamp(round(w_sum / w_total, 1))


# ---------------------------------------------------------------------------
# Factor 4 — EXECUTION SIGNALS  (10%)
# ---------------------------------------------------------------------------

def _f_execution(agent: dict, ai: dict) -> float | None:
    """
    Secondary tiebreaker: online presence + product stage + weak doxx signal.
    Returns None if no signals at all (no penalty for absence).
    Doxx is intentionally low-weighted here — a weak tiebreaker, not a gate.
    """
    points   = 0.0
    max_pts  = 0.0
    has_any  = False

    # Website (strong signal — shows real product intent)
    website = str(agent.get("linked_website") or "").strip()
    max_pts += 35.0
    if website and len(website) > 5:
        has_any = True
        points += 35.0 if "github.com" not in website else 15.0

    # Twitter presence
    twitter = str(agent.get("linked_twitter") or "").strip()
    max_pts += 25.0
    if twitter and len(twitter) > 5:
        has_any = True
        points += 20.0
        followers = _safe(agent.get("twitter_followers"), 0)
        if followers > 10_000: points += 5.0
        elif followers > 1_000: points += 3.0

    # Product stage
    ai_status = str(ai.get("product", {}).get("status", "")).lower().replace(" ", "_")
    stage_score = {
        "live": 40, "production": 40, "beta": 30, "mainnet_beta": 30,
        "testnet": 18, "alpha": 18, "pre-product": 8, "pre_product": 8,
        "development": 8, "vaporware": 0, "concept": 0,
    }.get(ai_status, -1)
    if stage_score >= 0:
        has_any = True
        max_pts += 40.0
        points  += float(stage_score)
    elif agent.get("status") == "Sentient":
        has_any = True
        max_pts += 40.0
        points  += 30.0

    # Doxx tier — weak positive signal only (tier 1 or 2; tier 3 doesn't add pts)
    # Anonymous (tier 3) is noted via the doxx_detail field but does NOT factor
    # into execution — this ensures no penalty for anonymous teams.
    doxx_raw = agent.get("doxx_tier") or ai.get("team", {}).get("doxx_tier")
    if doxx_raw is not None:
        doxx_tier_val = int(doxx_raw)
        if doxx_tier_val in (1, 2):
            max_pts += 10.0
            has_any  = True
            if doxx_tier_val == 1:
                points += 10.0
            else:
                points += 6.0

    if not has_any:
        return None

    raw = (points / max_pts * 100.0) if max_pts > 0 else 0.0
    return _clamp(round(raw, 1))


# ---------------------------------------------------------------------------
# Factor 5-9 — MARKET VALIDATION  (collectively 25%)
# Benchmark-relative scoring derived from the "$5M+ cohort":
#   All agents with market cap > $5M, excluding 365love (whose Virtuals API
#   MC is known to be wrong).  These 8 projects represent proven successes.
#
# $5M+ cohort (live data, April 2026):
#   Ribbita ($113M, 71K h, $74K vol), Toshi ($75M, 1.08M h, $3.8K vol),
#   Fabric Protocol ($37M, 1.8K h, $28K vol), aixbt ($25M, 413K h, $18K vol),
#   G.A.M.E ($8.6M, 282K h, $13K vol), Mamo ($7.7M, 72K h, $4.4K vol),
#   Luna ($6M, 458K h, $14K vol), Keyboard Cat ($5.2M, 908K h, $115K vol)
#
# Benchmark = median of each metric in this cohort:
#   holders  → 347 618  (median of 8 values)
#   MC       → $16 831 178
#   volume   → $16 108/day
#
# _BM_* values are the "90-score" reference points.  Projects at or above the
# benchmark score ≥ 90; those well below fall proportionally on a power-law
# log scale (exponent 1.5 for steeper drop-off below benchmark).
# ---------------------------------------------------------------------------

_BM_HOLDERS = 347_618    # median holders of $5M+ cohort
_BM_MCAP    = 16_831_178 # median MC of $5M+ cohort
_BM_VOLUME  = 16_108     # median 24h volume of $5M+ cohort


def _bm_log_score(value: float, benchmark: float, floor: float = 5.0) -> float:
    """
    Benchmark-relative log score on [floor, 95].

    value >= benchmark  → 90–95  (above-benchmark tail scales linearly to 95)
    value = benchmark   → 90
    value = 0           → floor (5)

    Uses exponent 1.5 on the log ratio below benchmark so the drop-off is
    steeper than pure log but not as harsh as quadratic.
    """
    if value <= 0:
        return floor
    log_ratio = math.log10(value + 1) / math.log10(benchmark + 1)
    if log_ratio >= 1.0:
        # Above-benchmark bonus: up to +5 for projects well above benchmark
        return _clamp(90.0 + (log_ratio - 1.0) * 10.0)
    score = floor + (90.0 - floor) * (log_ratio ** 1.5)
    return _clamp(score)


def _f_holders(agent: dict, ai: dict) -> float | None:
    h = _safe(agent.get("holder_count"), -1)
    if h < 0:
        return None
    return _bm_log_score(h, _BM_HOLDERS) if h > 0 else 5.0


def _f_mcap(agent: dict, ai: dict) -> float | None:
    m = _safe(agent.get("market_cap"), -1)
    if m < 0:
        return None
    return _bm_log_score(m, _BM_MCAP) if m > 0 else 5.0


def _f_volume(agent: dict, ai: dict) -> float | None:
    v = _safe(agent.get("volume_24h"), -1)
    if v < 0:
        return None
    return _bm_log_score(v, _BM_VOLUME) if v > 0 else 5.0


def _f_efficiency(agent: dict, ai: dict) -> float | None:
    """Vol/MCap ratio — healthy trading activity indicator."""
    vol  = _safe(agent.get("volume_24h"), -1)
    mcap = _safe(agent.get("market_cap"),  0)
    if vol < 0 or mcap <= 0: return None
    r = vol / mcap
    if r > 2.0:   return 18.0   # likely wash trading
    if r > 0.5:   return 38.0
    if r > 0.1:   return 72.0
    if r > 0.05:  return 62.0
    if r > 0.01:  return 50.0
    if r > 0.001: return 32.0
    return 15.0


def _f_momentum(agent: dict, ai: dict) -> float | None:
    """Holder change 24h."""
    change = agent.get("holder_count_change_24h")
    if change is None: return None
    change  = _safe(change, 0)
    holders = max(_safe(agent.get("holder_count"), 1), 1)
    pct = change / holders * 100.0
    if pct > 5:   return 90.0
    if pct > 2:   return 75.0
    if pct > 0:   return 60.0
    if pct == 0:  return 45.0
    if pct > -2:  return 28.0
    return 12.0


# ---------------------------------------------------------------------------
# FACTORS registry
# Weights sum to 1.0:  uniqueness=0.40, tam=0.25, moat=0.15, execution=0.10,
#                      market sub-factors collectively 0.10
# ---------------------------------------------------------------------------

def _f_idea_market_fit(agent: dict, ai: dict) -> float:
    """
    Combined idea × market-fit score (65% of total).

    Uses geometric mean of uniqueness × TAM so BOTH must be strong for a
    high score. A generic idea in a huge market scores the same as a unique
    idea in a tiny market — you need both to reach the top tier.

    Always returns a value (floor ~15 for meme/unknown category with no bio).
    """
    uniq = _f_uniqueness(agent, ai)
    tam  = _f_tam(agent, ai)
    # Geometric mean: sqrt(u × t) preserves scale nicely
    return _clamp(round(math.sqrt(uniq * tam), 1))


FACTORS = [
    # (function, weight, name, tier)
    # Within-tier weights determine the weighted average inside each tier;
    # cross-tier contributions are controlled by _TIER_WEIGHTS below.
    (_f_idea_market_fit, 1.00, "F_idea_market_fit", "idea"),      # sole factor in tier

    (_f_moat,            1.00, "F_moat",            "moat"),      # sole factor in tier

    (_f_execution,       1.00, "F_execution",        "execution"), # sole factor in tier

    # Market sub-factors — within-tier weights reflect relative importance
    (_f_holders,         0.42, "F_holders",         "market"),
    (_f_mcap,            0.34, "F_mcap",            "market"),
    (_f_volume,          0.14, "F_volume",          "market"),
    (_f_efficiency,      0.05, "F_efficiency",      "market"),
    (_f_momentum,        0.05, "F_momentum",        "market"),
]

_TIER_WEIGHTS = {
    "idea":      0.45,   # was 0.65 — still dominant but market gets more say
    "moat":      0.20,   # was 0.15 — community moat now explicit
    "execution": 0.10,   # tiebreaker, not gate (unchanged)
    "market":    0.25,   # was 0.10 — benchmark-relative, core differentiator
}


# ---------------------------------------------------------------------------
# Badge detection (UI only, zero scoring impact)
# ---------------------------------------------------------------------------

def _is_dead_agent(agent: dict) -> bool:
    holders = _safe(agent.get("holder_count"), 0)
    mcap    = _safe(agent.get("market_cap"),   0)
    vol     = _safe(agent.get("volume_24h"),   0)
    if mcap > 0 and mcap < 5_000:
        return True
    if holders > 0 and holders < 50 and mcap > 0 and mcap < 10_000:
        return True
    days = _days_since(agent.get("creation_date") or agent.get("first_seen"))
    if days is not None and days > 90 and 0 < vol < 1_000:
        return True
    change = agent.get("holder_count_change_24h")
    if change is not None and _safe(change, 0) < 0 and vol < 500:
        return True
    return False


def _is_strong_investment(agent: dict) -> bool:
    return (
        _safe(agent.get("holder_count"), 0) >= 2_000 and
        _safe(agent.get("market_cap"),   0) >= 5_000_000 and
        _safe(agent.get("volume_24h"),   0) >= 20_000 and
        _safe(agent.get("holder_count_change_24h"), 0) >= 0
    )


# ---------------------------------------------------------------------------
# doxx_tier2 detail (UI breakdown only — not used for scoring)
# ---------------------------------------------------------------------------

def score_doxx_tier2(agent: dict) -> dict:
    twitter_age   = _safe(agent.get("twitter_account_age"),       0)
    followers     = _safe(agent.get("twitter_followers"),         0)
    engagement    = _safe(agent.get("twitter_engagement_rate"),   0)
    creation_date = agent.get("creation_date")
    components    = {}

    age_score = (25.0 if twitter_age > 730 else 20.0 if twitter_age > 365 else
                 14.0 if twitter_age > 180 else  8.0 if twitter_age > 30  else 2.0)
    components["account_age"] = round(age_score, 1)

    if followers > 50_000:  fq = 25.0
    elif followers > 10_000: fq = 20.0
    elif followers > 5_000:  fq = 16.0
    elif followers > 1_000:  fq = 12.0
    elif followers > 100:    fq = 6.0
    else:                    fq = 1.0
    if followers > 1_000 and engagement > 15.0:
        fq = max(0, fq - 8.0)
    components["follower_quality"] = round(fq, 1)

    if   1.0 <= engagement <= 5.0:  eng = 25.0
    elif 0.5 <= engagement <= 8.0:  eng = 18.0
    elif 0.1 <= engagement <  0.5:  eng = 10.0
    elif engagement > 15.0:         eng = 3.0
    elif engagement > 8.0:          eng = 8.0
    else:                           eng = 2.0
    components["engagement_authenticity"] = round(eng, 1)

    pre = 12.5
    if creation_date and twitter_age > 0:
        pd = _days_since(creation_date)
        if pd:
            if twitter_age > pd + 180:  pre = 25.0
            elif twitter_age > pd + 90:  pre = 22.0
            elif twitter_age > pd:       pre = 16.0
            elif twitter_age > pd - 30:  pre = 10.0
            else:                        pre = 3.0
    components["pre_project_existence"] = round(pre, 1)

    total = _clamp(age_score + fq + eng + pre)
    return {
        "total_score":      round(total, 1),
        "components":       components,
        "twitter_age_days": int(twitter_age),
        "followers":        int(followers),
        "engagement_rate":  round(engagement, 2),
    }


# ---------------------------------------------------------------------------
# Classification / narrative
# ---------------------------------------------------------------------------

def _classify_tier(score: float) -> str:
    if score >= 80: return "Top Tier"
    if score >= 65: return "Strong"
    if score >= 45: return "Moderate"
    if score >= 25: return "Weak"
    return "Avoid"


def _build_one_liner(tier_scores: dict, agent_data: dict) -> str:
    if not tier_scores:
        return ""
    valid = {k: v for k, v in tier_scores.items() if v is not None and v > 0}
    if not valid:
        return ""
    sorted_tiers = sorted(valid.items(), key=lambda x: x[1], reverse=True)
    strong = [TIER_LABELS.get(t, t) for t, s in sorted_tiers if s >= 60]
    weak   = [TIER_LABELS.get(t, t) for t, s in sorted_tiers if s < 40]
    parts  = []
    if strong: parts.append("Strong on " + " and ".join(strong[:2]))
    if weak:   parts.append("weak on "   + " and ".join(weak[:2]))
    if parts:  return ", ".join(parts)
    best  = TIER_LABELS.get(sorted_tiers[0][0],  sorted_tiers[0][0])
    worst = TIER_LABELS.get(sorted_tiers[-1][0], sorted_tiers[-1][0])
    return f"Balanced profile, strongest in {best}, watch {worst}"


def _build_score_narrative(agent_data: dict, ai_analysis: dict, scores: dict,
                           composite: float, tier_scores: dict) -> str:
    parts = []

    doxx_raw = agent_data.get("doxx_tier") or ai_analysis.get("team", {}).get("doxx_tier")
    if doxx_raw is not None:
        if int(doxx_raw) == 1:
            parts.append("Team is fully doxxed")
        elif int(doxx_raw) == 3:
            parts.append("Team is anonymous")

    vol     = _safe(agent_data.get("volume_24h"),    0)
    holders = int(_safe(agent_data.get("holder_count"), 0))
    mcap    = _safe(agent_data.get("market_cap"),    0)

    if 0 < vol < 10_000:
        parts.append(f"Very low 24h volume (${vol:,.0f})")
    if 0 < holders < 100:
        parts.append(f"Only {holders} holders")
    elif holders > 5_000:
        parts.append(f"{holders:,} holders")
    if mcap > 10_000_000:
        parts.append(f"${mcap / 1_000_000:.1f}M market cap")

    if tier_scores:
        valid = {k: v for k, v in tier_scores.items() if v}
        if len(valid) >= 2:
            strongest = max(valid, key=valid.get)
            weakest   = min(valid, key=valid.get)
            if strongest != weakest:
                parts.append(
                    f"Strongest: {TIER_LABELS.get(strongest, strongest)} "
                    f"({valid[strongest]:.0f}). "
                    f"Weakest: {TIER_LABELS.get(weakest, weakest)} "
                    f"({valid[weakest]:.0f})"
                )

    return ". ".join(parts) + "." if parts else ""


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def calculate_composite_score(agent_data: dict, ai_analysis: dict) -> dict:
    """
    Compute composite score from all factors.

    Missing data rule:
      - Factors return None when their input data is absent.
      - Skipped factors are excluded from the weighted average.
      - The remaining factors' weights are renormalized within each tier,
        and each tier's contribution is renormalized at the tier level.
      - This ensures no factor's absence drags the score down.

    Final formula:
        raw = weighted_avg over present tiers, respecting tier weights
        composite = 8 + (raw / 100) * 87   →  [0,100] → [8, 95]

    Returns full breakdown dict compatible with all downstream consumers.
    """
    scores         = {}
    factor_details = []
    tier_acc: dict[str, list] = {t: [] for t in _TIER_WEIGHTS}

    for fn, weight, name, tier in FACTORS:
        try:
            raw = fn(agent_data, ai_analysis)
        except Exception:
            raw = None

        if raw is None:
            scores[name] = None
            factor_details.append({
                "factor": name, "label": FACTOR_LABELS.get(name, name),
                "score": None, "weight": weight, "tier": tier,
                "contribution": 0.0, "skipped": True,
            })
        else:
            s = _clamp(raw)
            scores[name] = round(s, 1)
            tier_acc[tier].append((s, weight))
            factor_details.append({
                "factor": name, "label": FACTOR_LABELS.get(name, name),
                "score": round(s, 1), "weight": weight, "tier": tier,
                "contribution": 0.0, "skipped": False,
            })

    # ── Per-tier weighted averages ───────────────────────────────────────────
    tier_scores: dict[str, float | None] = {}
    for tier_name, items in tier_acc.items():
        if items:
            w_sum   = sum(s * w for s, w in items)
            w_total = sum(w       for _, w in items)
            tier_scores[tier_name] = round(w_sum / w_total, 1)
        else:
            tier_scores[tier_name] = None  # entire tier missing — will be skipped

    # ── Combine tiers; skip tiers with no data ───────────────────────────────
    comb_sum = 0.0
    comb_w   = 0.0
    for tier_name, tw in _TIER_WEIGHTS.items():
        ts = tier_scores.get(tier_name)
        if ts is not None:
            comb_sum += ts * tw
            comb_w   += tw

    raw_composite = (comb_sum / comb_w) if comb_w > 0 else 20.0

    # ── Map to [8, 95] ───────────────────────────────────────────────────────
    composite = 8.0 + (raw_composite / 100.0) * 87.0
    composite = round(_clamp(composite, 8.0, 95.0), 1)

    # ── Back-fill factor contributions ──────────────────────────────────────
    for fd in factor_details:
        if not fd["skipped"]:
            t = fd["tier"]
            t_items   = tier_acc[t]
            t_w_total = sum(w for _, w in t_items)
            if t_w_total > 0:
                tier_gw      = _TIER_WEIGHTS[t]
                factor_share = fd["weight"] / t_w_total
                fd["contribution"] = round(
                    (fd["score"] - 50.0) * factor_share * tier_gw * 0.87, 2
                )

    # ── Badges (display only, zero scoring impact) ──────────────────────────
    dead_flagged   = _is_dead_agent(agent_data)
    strong_flagged = _is_strong_investment(agent_data)

    classification = _classify_tier(composite)

    # ── Top helped / hurt ───────────────────────────────────────────────────
    active     = [fd for fd in factor_details if not fd["skipped"]]
    by_contrib = sorted(active, key=lambda x: x["contribution"], reverse=True)
    top_helped = [
        {"factor": f["factor"], "score": f["score"], "label": f["label"]}
        for f in by_contrib[:3] if f["contribution"] > 0
    ]
    top_hurt = [
        {"factor": f["factor"], "score": f["score"], "label": f["label"]}
        for f in by_contrib[-3:] if f["contribution"] < 0
    ]
    top_hurt.reverse()

    # Replace None tier scores with 0.0 for display
    tier_scores_display = {k: (v if v is not None else 0.0) for k, v in tier_scores.items()}

    one_liner = _build_one_liner(tier_scores_display, agent_data)
    narrative = _build_score_narrative(
        agent_data, ai_analysis, scores, composite, tier_scores_display
    )

    # ── doxx detail ─────────────────────────────────────────────────────────
    doxx_raw      = agent_data.get("doxx_tier") or ai_analysis.get("team", {}).get("doxx_tier")
    doxx_tier_val = int(doxx_raw) if doxx_raw is not None else 3
    doxx_detail   = {
        "tier":   doxx_tier_val,
        "label":  {1: "Full Doxx", 2: "Social Presence", 3: "Anonymous"}.get(doxx_tier_val, "Anonymous"),
        "score":  scores.get("F_execution"),
        "reason": {
            1: "Fully verified identity — public team with verifiable credentials",
            2: "Social presence only — pseudonymous with active social accounts",
            3: "Anonymous — no verifiable identity found",
        }.get(doxx_tier_val, "Unknown"),
    }
    if doxx_tier_val == 2:
        doxx_detail["tier2_breakdown"] = score_doxx_tier2(agent_data)

    factors_scored = sum(1 for k, v in scores.items()
                         if not k.startswith("_") and v is not None)

    # Attach metadata for downstream consumers
    scores["_tier_scores"]      = tier_scores_display
    scores["_one_liner"]        = one_liner
    scores["_top_helped"]       = top_helped
    scores["_top_hurt"]         = top_hurt
    scores["_doxx_tier_detail"] = doxx_detail
    scores["_dead_flagged"]     = dead_flagged
    scores["_strong_flagged"]   = strong_flagged
    scores["_factors_scored"]   = factors_scored

    return {
        "composite_score":    composite,
        "tier_classification": classification,
        "scores":              scores,
        "tier_scores":         tier_scores_display,
        "first_mover":         False,           # kept for API compat
        "score_narrative":     narrative,
        "one_liner":           one_liner,
        "top_helped":          top_helped,
        "top_hurt":            top_hurt,
        "doxx_tier_detail":    doxx_detail,
        "dead_flagged":        dead_flagged,
        "strong_flagged":      strong_flagged,
        "factors_scored":      factors_scored,
    }
