"""
VirtualsIQ — Scoring Engine v1.1 (Phase-Aware, Risk-Adjusted)

This engine implements a dual-ranking system built around the lifecycle phase
of each agent. Every agent is classified into one of three phases:
  Phase 1 — Speculation Window (days 0-21 since launch)
  Phase 2 — Transition (days 22-60)
  Phase 3 — Mature (day 61+)

Four sub-scores are computed:
  Quality Score  — how good is the idea, team, and product? (weekly)
  Upside Score   — phase-specific upside potential (daily/weekly)
  Momentum Score — current on-chain + social momentum (daily)
  Risk Score     — risk-adjusted multiplier (higher = safer)

Composite formula:
  base = 0.30 × Quality + 0.45 × Upside + 0.25 × Momentum
  risk_mult = (0.20 + 0.80 × Risk/100) for Phase 1
              (0.30 + 0.70 × Risk/100) for Phase 2/3
  composite = base × risk_mult, clamped to [8, 95]

Missing data rule:
  Any missing metric is assigned the median score (50) for that attribute.
  All scoring functions track which attributes had real data via `score_evidence`.

Backward compatibility:
  calculate_composite_score() still works and is used by bulk_score_agents().
  It now calls score_agent() internally and returns the legacy shape plus new fields.
"""

from __future__ import annotations

import math
from datetime import datetime


# ---------------------------------------------------------------------------
# Display labels (kept for backward compatibility)
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
    s = str(date_str).strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(s, fmt)
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


def _median_or_default(values: list, default: float = 50.0) -> float:
    """Return median of non-None values, or default if list is empty."""
    valids = [v for v in values if v is not None]
    if not valids:
        return default
    s = sorted(valids)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0


# ---------------------------------------------------------------------------
# Phase Classification
# ---------------------------------------------------------------------------

def classify_phase(agent_data: dict, momentum_override: bool = False) -> int:
    """
    Classify agent into lifecycle phase 1, 2, or 3.

    Phase 1: days 0-21  (Speculation Window)
    Phase 2: days 22-60 (Transition)
    Phase 3: day 61+    (Mature)

    momentum_override: if True, a Phase 3 agent may be promoted to Phase 2
    upside calculation when all four momentum-break conditions are met.
    That check is done separately; this function returns the base phase.
    """
    days = _days_since(agent_data.get("creation_date") or agent_data.get("first_seen"))
    if days is None:
        # No launch date — treat conservatively as Phase 2
        return 2
    if days <= 21:
        return 1
    if days <= 60:
        return 2
    return 3


def check_momentum_break(agent_data: dict, ai_analysis: dict) -> bool:
    """
    Check whether a Phase 3 agent qualifies for momentum-break promotion.

    All FOUR conditions must hold over a rolling 14-day window:
      1. Holder count grows 5%+ (using holder_count_change_7d or 24h proxy)
      2. Smart money net inflow is positive (or data missing → skip condition)
      3. At least 1 verifiable team action per week (github_commits_30d proxy)
      4. Quality score stable or improving vs 30-day prior (scores_json proxy)

    Returns True only when ALL available conditions are satisfied.
    Missing data degrades gracefully (condition skipped, not failed).
    """
    conditions_met = []
    conditions_total = 0

    # Condition 1: holder growth 5%+ over ~7 days
    holder_change = agent_data.get("holder_count_change_24h")
    if holder_change is not None:
        holders = max(_safe(agent_data.get("holder_count"), 1), 1)
        change_pct_24h = _safe(holder_change, 0) / holders * 100.0
        # Annualise to ~7d: multiply 24h rate × 7
        growth_7d_est = change_pct_24h * 7
        conditions_total += 1
        conditions_met.append(growth_7d_est >= 5.0)

    # Condition 2: smart money net inflow (field not yet populated — skip)
    smart_flow = agent_data.get("smart_money_net_flow_14d")
    if smart_flow is not None:
        conditions_total += 1
        conditions_met.append(_safe(smart_flow, 0) > 0)

    # Condition 3: verifiable team activity — github commits as proxy
    commits = _safe(agent_data.get("github_commits_30d"), 0)
    if agent_data.get("github_commits_30d") is not None:
        conditions_total += 1
        conditions_met.append(commits >= 4)  # ≥ 4 commits/30d ≈ 1/week

    # Condition 4: quality score not declining (scores_json → tier_scores)
    scores_json = agent_data.get("scores_json") or {}
    if isinstance(scores_json, dict) and "quality_score" in scores_json:
        prev_quality = _safe(scores_json.get("quality_score_30d_prior"), 0)
        curr_quality = _safe(scores_json.get("quality_score"), 0)
        if prev_quality > 0 and curr_quality > 0:
            conditions_total += 1
            conditions_met.append(curr_quality >= prev_quality * 0.95)

    # Must have at least 2 evaluable conditions and ALL must pass
    if conditions_total < 2:
        return False
    return all(conditions_met)


# ---------------------------------------------------------------------------
# Quality Score  (0-100, recomputed weekly)
# ---------------------------------------------------------------------------
# Weights:
#   Idea × TAM Fit             25%
#   Product Stage & Shipping   20%
#   Team Signal                15%
#   Moat & Defensibility       15%
#   Category Fit               10%
#   Narrative Alignment        10%
#   Partnerships & Integrations 5%

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
    # Less crowded
    "gaming":        55.0,
    "game":          55.0,
    "creator":       50.0,
    "art":           47.0,
    "productivity":  58.0,
    "tools":         60.0,
    "analytics":     63.0,
    "data":          63.0,
    # Specialized
    "infrastructure": 72.0,
    "infra":          72.0,
    "protocol":       75.0,
    "security":       78.0,
    "bridge":         70.0,
    "dev":            67.0,
    "oracle":         75.0,
}

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


def _category_uniqueness_score(agent: dict, ai: dict) -> float:
    cat = str(
        agent.get("category") or agent.get("agent_type") or
        ai.get("category") or ""
    ).lower().strip()
    for key in sorted(_CATEGORY_UNIQUENESS, key=len, reverse=True):
        if key in cat:
            return _CATEGORY_UNIQUENESS[key]
    return 42.0


def _score_quality(agent_data: dict, ai_analysis: dict) -> tuple[float, dict]:
    """
    Compute Quality Score (0-100) and return (score, evidence_dict).

    Uses weighted sub-components; missing data defaults to median (50).
    """
    evidence = {}
    components: list[tuple[float, float, str]] = []  # (score, weight, label)

    # ── 1. Idea × TAM Fit (25%) ───────────────────────────────────────────
    fm = ai_analysis.get("first_mover", {})
    ai_cat_unique = fm.get("category_unique")
    ai_novel      = fm.get("approach_novel")

    if ai_cat_unique is not None or ai_novel is not None:
        base = 50.0
        if ai_cat_unique is True:    base += 30.0
        elif ai_cat_unique is False: base -= 20.0
        if ai_novel is True:         base += 20.0
        elif ai_novel is False:      base -= 10.0
        days_ahead = fm.get("days_ahead_of_competitor")
        if days_ahead is not None:
            d = float(days_ahead)
            base += (10.0 if d > 180 else 7.0 if d > 90 else 3.0 if d > 30 else 0.0)
        idea_score = _clamp(base)
        evidence["idea_tam_source"] = "ai_analysis"
    else:
        cat = str(
            agent_data.get("category") or agent_data.get("agent_type") or
            ai_analysis.get("category") or ""
        ).lower().strip()
        cat_score = _category_uniqueness_score(agent_data, ai_analysis)

        bio = str(agent_data.get("biography") or agent_data.get("description") or "").strip()
        desc_score: float | None = None
        if bio:
            chars = len(bio)
            desc_score = (85.0 if chars >= 300 else 68.0 if chars >= 150 else
                          52.0 if chars >= 80 else 35.0 if chars >= 30 else 20.0)

        if desc_score is not None:
            blended = 0.60 * cat_score + 0.40 * desc_score
        else:
            blended = cat_score

        ai_tam = ai_analysis.get("market", {}).get("tam_score")
        if ai_tam is not None:
            tam_score = _clamp(float(ai_tam))
        else:
            cat_lower = cat
            tam = 25_000_000_000
            for key, val in _TAM_BY_CAT.items():
                if key in cat_lower:
                    tam = val
                    break
            tam_score = _clamp(_log_score(tam, _TAM_BP))

        idea_score = _clamp(round(math.sqrt(blended * tam_score), 1))
        evidence["idea_tam_source"] = "category_lookup"

    components.append((idea_score, 0.25, "idea_tam"))
    evidence["idea_tam_score"] = round(idea_score, 1)

    # ── 2. Product Stage & Shipping Cadence (20%) ─────────────────────────
    ai_status = str(ai_analysis.get("product", {}).get("status", "")).lower().replace(" ", "_")
    stage_raw_score = {
        "live": 100, "production": 100, "beta": 60, "mainnet_beta": 60,
        "testnet": 30, "alpha": 30, "pre-product": 0, "pre_product": 0,
        "development": 0, "vaporware": 0, "concept": 0,
    }.get(ai_status)

    if stage_raw_score is None and agent_data.get("status") == "Sentient":
        stage_raw_score = 80
    elif stage_raw_score is None:
        stage_raw_score = None  # will use median

    if stage_raw_score is not None:
        # Freshness factor: multiply by recency of last meaningful update
        days = _days_since(
            agent_data.get("github_last_commit") or
            agent_data.get("last_analyzed") or
            agent_data.get("creation_date")
        )
        freshness = 1.0
        if days is not None:
            freshness = (1.0 if days < 7 else 0.9 if days < 30 else
                         0.75 if days < 90 else 0.5)
        product_score = _clamp(float(stage_raw_score) * freshness)
        evidence["product_stage"] = ai_status or "sentient"
        evidence["product_stage_source"] = "real"
    else:
        product_score = 50.0
        evidence["product_stage"] = "unknown"
        evidence["product_stage_source"] = "default"

    components.append((product_score, 0.20, "product_stage"))
    evidence["product_stage_score"] = round(product_score, 1)

    # ── 3. Team Signal (15%) ──────────────────────────────────────────────
    doxx_raw = agent_data.get("doxx_tier") or ai_analysis.get("team", {}).get("doxx_tier")
    if doxx_raw is not None:
        d = int(doxx_raw)
        # doxx_tier=1 → full doxx=100; tier2 strong=70; tier2 weak=40; tier3=30
        if d == 1:
            team_score = 100.0
        elif d == 2:
            # Use twitter quality to differentiate tier2 strong vs weak
            followers = _safe(agent_data.get("twitter_followers"), 0)
            team_score = 70.0 if followers > 5_000 else 40.0
        else:
            team_score = 30.0
        evidence["team_doxx_tier"] = d
        evidence["team_source"] = "real"
    else:
        team_score = 50.0
        evidence["team_doxx_tier"] = None
        evidence["team_source"] = "default"

    components.append((team_score, 0.15, "team_signal"))
    evidence["team_signal_score"] = round(team_score, 1)

    # ── 4. Moat & Defensibility (15%) ─────────────────────────────────────
    moat_signals = []

    ai_def = fm.get("defensibility_score")
    if ai_def is not None:
        moat_signals.append((_clamp(float(ai_def)), 3.0))

    stars        = _safe(agent_data.get("github_stars"), 0)
    contributors = _safe(agent_data.get("github_contributors"), 0)
    commits      = _safe(agent_data.get("github_commits_30d"), 0)
    if stars > 0 or contributors > 0 or commits > 0:
        gh_score  = min(60.0, math.log10(max(stars, 1)) * 22.0)
        gh_score += min(20.0, contributors * 5.0)
        gh_score += min(20.0, commits * 1.0)
        moat_signals.append((_clamp(gh_score), 2.0))

    days_age = _days_since(agent_data.get("creation_date") or agent_data.get("first_seen"))
    if days_age is not None:
        age_score = (80.0 if days_age > 365 else 65.0 if days_age > 180 else
                     48.0 if days_age > 90 else 30.0 if days_age > 30 else 15.0)
        moat_signals.append((age_score, 1.0))

    partnership = ai_analysis.get("product", {}).get("partnership_score")
    if partnership is not None:
        moat_signals.append((_clamp(float(partnership)), 1.5))

    holders = _safe(agent_data.get("holder_count"), 0)
    if holders >= 1_000:
        community_moat = _clamp(20.0 * math.log10(holders))
        moat_signals.append((community_moat, 2.0))

    if moat_signals:
        w_sum   = sum(s * w for s, w in moat_signals)
        w_total = sum(w for _, w in moat_signals)
        moat_score = _clamp(round(w_sum / w_total, 1))
        evidence["moat_source"] = "real"
    else:
        moat_score = 50.0
        evidence["moat_source"] = "default"

    components.append((moat_score, 0.15, "moat"))
    evidence["moat_score"] = round(moat_score, 1)

    # ── 5. Category Fit (10%) ─────────────────────────────────────────────
    cat = str(
        agent_data.get("category") or agent_data.get("agent_type") or
        ai_analysis.get("category") or ""
    ).lower().strip()
    # Category fit: how well the project fits an established Virtuals category
    if cat and cat not in ("other", "unknown", "ip", ""):
        cat_fit_score = 75.0  # known category → good fit
        for saturated in ("meme", "ai agent", "trading bot", "assistant"):
            if saturated in cat:
                cat_fit_score = 45.0
                break
        evidence["category_fit_source"] = "real"
    else:
        cat_fit_score = 50.0
        evidence["category_fit_source"] = "default"

    components.append((cat_fit_score, 0.10, "category_fit"))
    evidence["category_fit_score"] = round(cat_fit_score, 1)

    # ── 6. Narrative Alignment (10%) ──────────────────────────────────────
    narrative_score_ai = ai_analysis.get("narrative_alignment_score")
    if narrative_score_ai is not None:
        narrative_score = _clamp(float(narrative_score_ai))
        evidence["narrative_source"] = "ai_analysis"
    else:
        # Infer from bio length + AI keywords if available
        bio = str(agent_data.get("biography") or "").lower()
        narrative_keywords = ["autonomous", "ai", "agent", "protocol", "on-chain",
                               "blockchain", "defi", "nft", "web3", "base"]
        hits = sum(1 for kw in narrative_keywords if kw in bio)
        narrative_score = min(90.0, 30.0 + hits * 8.0) if bio else 50.0
        evidence["narrative_source"] = "keyword_proxy" if bio else "default"

    components.append((narrative_score, 0.10, "narrative"))
    evidence["narrative_score"] = round(narrative_score, 1)

    # ── 7. Partnerships & Integrations (5%) ───────────────────────────────
    partnership_score_ai = ai_analysis.get("product", {}).get("partnership_score")
    if partnership_score_ai is not None:
        partnership_score = _clamp(float(partnership_score_ai))
        evidence["partnership_source"] = "ai_analysis"
    else:
        # Proxy: telegram/website existence = integration effort signal
        has_telegram = bool(str(agent_data.get("linked_telegram") or "").strip())
        has_website  = bool(str(agent_data.get("linked_website") or "").strip())
        partnership_score = 50.0 + (15.0 if has_telegram else 0) + (10.0 if has_website else 0)
        evidence["partnership_source"] = "presence_proxy"

    components.append((partnership_score, 0.05, "partnerships"))
    evidence["partnership_score"] = round(partnership_score, 1)

    # ── Weighted average ──────────────────────────────────────────────────
    total_w = sum(w for _, w, _ in components)
    quality = sum(s * w for s, w, _ in components) / total_w if total_w > 0 else 50.0
    quality = _clamp(round(quality, 1))

    evidence["quality_score"] = quality
    return quality, evidence


# ---------------------------------------------------------------------------
# Upside Score — Phase-Aware  (0-100)
# ---------------------------------------------------------------------------

_BM_HOLDERS = 347_618
_BM_MCAP    = 16_831_178
_BM_VOLUME  = 16_108


def _bm_log_score(value: float, benchmark: float, floor: float = 5.0) -> float:
    if value <= 0:
        return floor
    log_ratio = math.log10(value + 1) / math.log10(benchmark + 1)
    if log_ratio >= 1.0:
        return _clamp(90.0 + (log_ratio - 1.0) * 10.0)
    score = floor + (90.0 - floor) * (log_ratio ** 1.5)
    return _clamp(score)


def _score_phase1_upside(agent_data: dict, ai_analysis: dict) -> tuple[float, dict]:
    """
    Phase 1 Upside (days 0-21):
      Holder Growth Velocity 7d      30%
      Smart Money Inflow Accel 7d    25%
      Social Engagement Velocity 7d  20%
      Team Showing-Up Signal         15%
      Market Cap Floor (inverted)    10%

    Then multiply by age decay:
      days 0-7  → 1.0
      days 8-14 → 0.8
      days 15-21 → 0.6

    Missing data → 50 (median).
    """
    evidence = {}
    components: list[tuple[float, float]] = []  # (score, weight)

    # Holder Growth Velocity 7d (30%)
    holder_change = agent_data.get("holder_count_change_24h")
    if holder_change is not None:
        holders = max(_safe(agent_data.get("holder_count"), 1), 1)
        pct_24h  = _safe(holder_change, 0) / holders * 100.0
        pct_7d   = pct_24h * 7
        hg_score = _clamp(50.0 + pct_7d * 5.0)  # 1% 7d growth → 55, 10% → 100
        evidence["holder_growth_7d_pct"] = round(pct_7d, 2)
        evidence["holder_growth_source"] = "real"
    else:
        hg_score = 50.0
        evidence["holder_growth_source"] = "default"
    components.append((hg_score, 0.30))
    evidence["holder_growth_score"] = round(hg_score, 1)

    # Smart Money Inflow Acceleration 7d (25%) — gracefully degrade
    smart_flow = agent_data.get("smart_money_net_flow_14d")
    sm_accel = agent_data.get("smart_money_acceleration")
    if smart_flow is not None or sm_accel is not None:
        flow = _safe(smart_flow, 0)
        mcap = max(_safe(agent_data.get("market_cap"), 1), 1)
        flow_pct = flow / mcap * 100
        base_sm = _clamp(50.0 + flow_pct * 10.0)
        # Acceleration bonus/penalty: positive acceleration boosts score
        if sm_accel is not None:
            accel = _safe(sm_accel, 0)
            accel_boost = _clamp(accel * 5.0, -20.0, 20.0)
            sm_score = _clamp(base_sm + accel_boost)
        else:
            sm_score = base_sm
        evidence["smart_money_source"] = "real"
        evidence["smart_money_acceleration"] = round(_safe(sm_accel, 0), 4)
    else:
        sm_score = 50.0
        evidence["smart_money_source"] = "default"
    components.append((sm_score, 0.25))
    evidence["smart_money_score"] = round(sm_score, 1)

    # Social Engagement Velocity 7d (20%)
    tw_followers = _safe(agent_data.get("twitter_followers"), 0)
    tw_engagement = _safe(agent_data.get("twitter_engagement_rate"), 0)
    if tw_followers > 0:
        # Score based on follower count and engagement rate together
        follower_score = min(70.0, math.log10(max(tw_followers, 1)) / math.log10(100_000) * 70.0)
        engagement_bonus = min(30.0, tw_engagement * 4.0) if 0 < tw_engagement <= 10 else 0.0
        social_score = _clamp(follower_score + engagement_bonus)
        evidence["social_source"] = "real"
    else:
        social_score = 50.0
        evidence["social_source"] = "default"
    components.append((social_score, 0.20))
    evidence["social_score"] = round(social_score, 1)

    # Team Showing-Up Signal (15%) — days out of 7 with verifiable team action
    commits_30d = _safe(agent_data.get("github_commits_30d"), 0)
    if agent_data.get("github_commits_30d") is not None:
        # ~1 commit/day = active; map 0-30 commits/month to 0-100
        team_show = _clamp(min(commits_30d / 30.0, 1.0) * 100.0)
        evidence["team_activity_source"] = "github"
    elif str(agent_data.get("linked_twitter") or "").strip():
        # Has twitter → some showing-up signal
        team_show = 55.0
        evidence["team_activity_source"] = "twitter_proxy"
    else:
        team_show = 50.0
        evidence["team_activity_source"] = "default"
    components.append((team_show, 0.15))
    evidence["team_activity_score"] = round(team_show, 1)

    # Market Cap Floor — inverted, smaller mcap = higher potential score (10%)
    mcap = _safe(agent_data.get("market_cap"), 0)
    if mcap > 0:
        # Below $100K → high potential (90); above $10M → low remaining upside (20)
        if mcap < 100_000:          mcap_floor_score = 90.0
        elif mcap < 500_000:        mcap_floor_score = 75.0
        elif mcap < 1_000_000:      mcap_floor_score = 65.0
        elif mcap < 5_000_000:      mcap_floor_score = 50.0
        elif mcap < 10_000_000:     mcap_floor_score = 35.0
        else:                       mcap_floor_score = 20.0
        evidence["mcap_floor_source"] = "real"
    else:
        mcap_floor_score = 50.0
        evidence["mcap_floor_source"] = "default"
    components.append((mcap_floor_score, 0.10))
    evidence["mcap_floor_score"] = round(mcap_floor_score, 1)

    # Weighted average
    total_w = sum(w for _, w in components)
    upside = sum(s * w for s, w in components) / total_w if total_w > 0 else 50.0

    # Age decay multiplier
    days = _days_since(agent_data.get("creation_date") or agent_data.get("first_seen"))
    if days is None:
        age_decay = 0.8
    elif days <= 7:
        age_decay = 1.0
    elif days <= 14:
        age_decay = 0.8
    else:
        age_decay = 0.6
    evidence["age_decay"] = age_decay

    upside = _clamp(round(upside * age_decay, 1))
    evidence["phase1_upside_score"] = upside
    return upside, evidence


def _score_phase2_upside(agent_data: dict, ai_analysis: dict,
                          quality_score: float) -> tuple[float, dict]:
    """
    Phase 2 Upside (days 22-60):
      Momentum Continuation (50%)
        — weighted avg of holder growth, smart money flow, social velocity
          over last 14d vs prior 14d
      Fundamentals Validation (50%)
        — change in Quality score + Edge calculation

    Divergence Penalty: if attention declining AND fundamentals weak → × 0.5
    """
    evidence = {}

    # ── Momentum Continuation (50%) ───────────────────────────────────────
    momentum_signals = []

    # Holder growth signal (14d estimated from 24h)
    holder_change = agent_data.get("holder_count_change_24h")
    if holder_change is not None:
        holders = max(_safe(agent_data.get("holder_count"), 1), 1)
        pct_24h = _safe(holder_change, 0) / holders * 100.0
        h_signal = _clamp(50.0 + pct_24h * 5.0)
        momentum_signals.append((h_signal, 0.40))
        evidence["holder_momentum_source"] = "real"
    else:
        momentum_signals.append((50.0, 0.40))
        evidence["holder_momentum_source"] = "default"

    # Smart money (if available)
    smart_flow = agent_data.get("smart_money_net_flow_14d")
    if smart_flow is not None:
        flow = _safe(smart_flow, 0)
        mcap = max(_safe(agent_data.get("market_cap"), 1), 1)
        flow_pct = flow / mcap * 100
        sm_signal = _clamp(50.0 + flow_pct * 10.0)
        momentum_signals.append((sm_signal, 0.40))
        evidence["smart_money_source"] = "real"
    else:
        momentum_signals.append((50.0, 0.40))
        evidence["smart_money_source"] = "default"

    # Social velocity
    tw_followers = _safe(agent_data.get("twitter_followers"), 0)
    tw_engagement = _safe(agent_data.get("twitter_engagement_rate"), 0)
    if tw_followers > 0:
        follower_score = min(70.0, math.log10(max(tw_followers, 1)) / math.log10(100_000) * 70.0)
        engagement_bonus = min(30.0, tw_engagement * 4.0) if 0 < tw_engagement <= 10 else 0.0
        social_signal = _clamp(follower_score + engagement_bonus)
        momentum_signals.append((social_signal, 0.20))
        evidence["social_source"] = "real"
    else:
        momentum_signals.append((50.0, 0.20))
        evidence["social_source"] = "default"

    total_w = sum(w for _, w in momentum_signals)
    momentum_continuation = sum(s * w for s, w in momentum_signals) / total_w if total_w > 0 else 50.0
    evidence["momentum_continuation"] = round(momentum_continuation, 1)

    # ── Fundamentals Validation (50%) ─────────────────────────────────────
    # Quality component: current quality score vs baseline of 50
    quality_component = quality_score  # 0-100, already computed
    evidence["quality_component"] = round(quality_component, 1)

    # Edge: quality_percentile - mcap_percentile (50 = neutral, >50 = underpriced)
    mcap = _safe(agent_data.get("market_cap"), 0)
    mcap_score = _bm_log_score(mcap, _BM_MCAP) if mcap > 0 else 50.0
    # edge = quality vs market cap valuation gap
    raw_edge = quality_score - mcap_score
    # Map raw_edge [-100, +100] → edge score [0, 100] where 50 = neutral
    edge_component = _clamp(50.0 + raw_edge * 0.5)
    evidence["edge_component"] = round(edge_component, 1)
    evidence["quality_vs_mcap_gap"] = round(raw_edge, 1)

    fundamentals = _clamp(0.50 * quality_component + 0.50 * edge_component)
    evidence["fundamentals_validation"] = round(fundamentals, 1)

    # ── Combine ───────────────────────────────────────────────────────────
    upside = 0.50 * momentum_continuation + 0.50 * fundamentals

    # Divergence penalty: attention declining AND fundamentals weak
    attention_declining = (
        holder_change is not None and _safe(holder_change, 0) < 0 and
        tw_followers < 500
    )
    fundamentals_weak = fundamentals < 40.0
    if attention_declining and fundamentals_weak:
        upside *= 0.5
        evidence["divergence_penalty"] = True
    else:
        evidence["divergence_penalty"] = False

    upside = _clamp(round(upside, 1))
    evidence["phase2_upside_score"] = upside
    return upside, evidence


def _score_phase3_upside(agent_data: dict, ai_analysis: dict,
                          quality_score: float,
                          category_agents: list | None = None) -> tuple[float, dict]:
    """
    Phase 3 Upside (day 61+):
      Edge Calculation (100%)
        — quality_percentile_in_category - mcap_percentile_in_category
        — mapped to [0, 100] where 50=neutral, >50=underpriced, <50=overpriced

    category_agents: list of dicts with quality_score and market_cap for peers.
    When not available, falls back to benchmark-relative calculation.
    """
    evidence = {}

    mcap = _safe(agent_data.get("market_cap"), 0)

    if category_agents and len(category_agents) >= 3:
        # True percentile ranking within category
        cat_mcaps     = sorted([_safe(a.get("market_cap"), 0) for a in category_agents])
        cat_qualities = sorted([_safe(a.get("quality_score") or
                                      a.get("composite_score"), 50) for a in category_agents])

        def _percentile(val: float, sorted_list: list) -> float:
            """Percentile rank of val within sorted_list (0-100)."""
            n = len(sorted_list)
            if n == 0:
                return 50.0
            below = sum(1 for x in sorted_list if x < val)
            return (below / n) * 100.0

        mcap_pct = _percentile(mcap, cat_mcaps)
        quality_pct = _percentile(quality_score, cat_qualities)
        evidence["percentile_source"] = "category_peers"
        evidence["peer_count"] = len(category_agents)
    else:
        # Benchmark-relative fallback
        mcap_pct    = _bm_log_score(mcap, _BM_MCAP) if mcap > 0 else 50.0
        quality_pct = quality_score  # already 0-100, treat as percentile proxy
        evidence["percentile_source"] = "benchmark_relative"

    # Edge: how much quality exceeds market cap (positive = underpriced)
    raw_edge = quality_pct - mcap_pct
    # Map [-100, +100] → [0, 100] where 50 = neutral
    upside = _clamp(50.0 + raw_edge * 0.5)

    evidence["quality_percentile"] = round(quality_pct, 1)
    evidence["mcap_percentile"]    = round(mcap_pct, 1)
    evidence["raw_edge"]           = round(raw_edge, 1)

    upside = _clamp(round(upside, 1))
    evidence["phase3_upside_score"] = upside
    return upside, evidence


def _score_upside(agent_data: dict, ai_analysis: dict, phase: int,
                   quality_score: float, momentum_break_active: bool,
                   category_agents: list | None = None) -> tuple[float, dict, float, float, float]:
    """
    Route to phase-specific upside scoring.
    Returns (upside, evidence, p1_upside, p2_upside, p3_upside).
    """
    p1, p1_ev = _score_phase1_upside(agent_data, ai_analysis)
    p2, p2_ev = _score_phase2_upside(agent_data, ai_analysis, quality_score)
    p3, p3_ev = _score_phase3_upside(agent_data, ai_analysis, quality_score, category_agents)

    if phase == 1:
        upside = p1
        evidence = p1_ev
    elif phase == 2 or momentum_break_active:
        upside = p2
        evidence = p2_ev
    else:
        upside = p3
        evidence = p3_ev

    if momentum_break_active and phase == 3:
        evidence["momentum_break_override"] = True

    evidence["active_phase"] = phase if not momentum_break_active else "2_override"
    return upside, evidence, p1, p2, p3


# ---------------------------------------------------------------------------
# Momentum Score  (0-100, recomputed daily)
# ---------------------------------------------------------------------------
# On-chain (60%): holder growth 7d (15%), smart money net flow (20%),
#                 volume trend 14d (10%), team wallet behavior (10%),
#                 independent movement vs $VIRTUAL (5%)
# Social (40%): engagement velocity (15%), smart account mentions (15%),
#               sentiment trend (5%), cross-platform presence (5%)

def _score_momentum(agent_data: dict, ai_analysis: dict) -> tuple[float, dict]:
    """
    Compute Momentum Score (0-100) from on-chain and social signals.
    All missing data defaults to 50 (median).
    """
    evidence = {}
    on_chain: list[tuple[float, float]] = []
    social: list[tuple[float, float]]   = []

    # ── On-chain (60%) ────────────────────────────────────────────────────

    # Holder growth 7d (15%)
    holder_change = agent_data.get("holder_count_change_24h")
    if holder_change is not None:
        holders = max(_safe(agent_data.get("holder_count"), 1), 1)
        pct_24h = _safe(holder_change, 0) / holders * 100.0
        hg = _clamp(50.0 + pct_24h * 5.0)
        evidence["holder_growth_pct_24h"] = round(pct_24h, 2)
        evidence["holder_growth_source"] = "real"
    else:
        hg = 50.0
        evidence["holder_growth_source"] = "default"
    on_chain.append((hg, 0.15))
    evidence["momentum_holder_score"] = round(hg, 1)

    # Smart money net flow (20%)
    smart_flow = agent_data.get("smart_money_net_flow_14d")
    if smart_flow is not None:
        flow = _safe(smart_flow, 0)
        mcap = max(_safe(agent_data.get("market_cap"), 1), 1)
        flow_pct = flow / mcap * 100
        sm = _clamp(50.0 + flow_pct * 10.0)
        evidence["smart_money_source"] = "real"
    else:
        sm = 50.0
        evidence["smart_money_source"] = "default"
    on_chain.append((sm, 0.20))
    evidence["momentum_smart_money_score"] = round(sm, 1)

    # Volume trend 14d (10%)
    vol = _safe(agent_data.get("volume_24h"), -1)
    if vol >= 0:
        vt = _bm_log_score(vol, _BM_VOLUME, floor=5.0)
        evidence["volume_source"] = "real"
    else:
        vt = 50.0
        evidence["volume_source"] = "default"
    on_chain.append((vt, 0.10))
    evidence["momentum_volume_score"] = round(vt, 1)

    # Team wallet behavior (10%) — proxy: github activity
    commits = _safe(agent_data.get("github_commits_30d"), 0)
    if agent_data.get("github_commits_30d") is not None:
        tw_behavior = _clamp(min(commits / 30.0, 1.0) * 100.0)
        evidence["team_wallet_source"] = "github_proxy"
    else:
        tw_behavior = 50.0
        evidence["team_wallet_source"] = "default"
    on_chain.append((tw_behavior, 0.10))
    evidence["momentum_team_wallet_score"] = round(tw_behavior, 1)

    # Independent movement vs $VIRTUAL (5%) — placeholder, default 50
    on_chain.append((50.0, 0.05))
    evidence["virtual_independence_source"] = "default"

    # ── Social (40%) ──────────────────────────────────────────────────────

    # Engagement velocity (15%)
    tw_followers = _safe(agent_data.get("twitter_followers"), 0)
    tw_engagement = _safe(agent_data.get("twitter_engagement_rate"), 0)
    if tw_followers > 0:
        follower_part = min(50.0, math.log10(max(tw_followers, 1)) / math.log10(100_000) * 50.0)
        eng_part = min(50.0, tw_engagement * 5.0) if 0 < tw_engagement <= 10 else 0.0
        eng_vel = _clamp(follower_part + eng_part)
        evidence["engagement_source"] = "real"
    else:
        eng_vel = 50.0
        evidence["engagement_source"] = "default"
    social.append((eng_vel, 0.15))
    evidence["momentum_engagement_score"] = round(eng_vel, 1)

    # Smart account mentions (15%) — placeholder, default 50
    social.append((50.0, 0.15))
    evidence["smart_mentions_source"] = "default"

    # Sentiment trend (5%) — placeholder, default 50
    social.append((50.0, 0.05))
    evidence["sentiment_source"] = "default"

    # Cross-platform presence (5%)
    has_twitter  = bool(str(agent_data.get("linked_twitter") or "").strip())
    has_website  = bool(str(agent_data.get("linked_website") or "").strip())
    has_telegram = bool(str(agent_data.get("linked_telegram") or "").strip())
    platform_count = sum([has_twitter, has_website, has_telegram])
    cross_platform = _clamp(20.0 * platform_count + 40.0)  # 1 → 60, 2 → 80, 3 → 100
    social.append((cross_platform, 0.05))
    evidence["cross_platform_score"] = round(cross_platform, 1)
    evidence["cross_platform_source"] = "real"

    # ── Combine ───────────────────────────────────────────────────────────
    oc_total_w = sum(w for _, w in on_chain)
    oc_score   = sum(s * w for s, w in on_chain) / oc_total_w if oc_total_w > 0 else 50.0

    soc_total_w = sum(w for _, w in social)
    soc_score   = sum(s * w for s, w in social) / soc_total_w if soc_total_w > 0 else 50.0

    momentum = _clamp(round(0.60 * oc_score + 0.40 * soc_score, 1))
    evidence["on_chain_score"]  = round(oc_score, 1)
    evidence["social_score"]    = round(soc_score, 1)
    evidence["momentum_score"]  = momentum
    return momentum, evidence


# ---------------------------------------------------------------------------
# Risk Score  (0-100 — higher = SAFER, used as a multiplier)
# ---------------------------------------------------------------------------
# Liquidity Depth vs MCap     25%
# Holder Concentration top-10 20%
# Team Wallet Distribution    15%
# Contract Mutability         15%
# Wash Trading Signal         10%
# Time Since Last Update      10%
# Audit Status                 5%

def _score_risk(agent_data: dict, ai_analysis: dict) -> tuple[float, dict]:
    """
    Compute Risk Score (0-100). Higher score = lower risk (safer).
    Missing data defaults to 50 (median = neutral).
    """
    evidence = {}
    components: list[tuple[float, float]] = []

    # ── Liquidity Depth vs MCap (25%) ─────────────────────────────────────
    vol  = _safe(agent_data.get("volume_24h"), -1)
    mcap = _safe(agent_data.get("market_cap"), 0)
    if vol >= 0 and mcap > 0:
        vol_mcap_ratio = vol / mcap
        if vol_mcap_ratio < 0.001:       liq_score = 10.0   # severe
        elif vol_mcap_ratio < 0.01:      liq_score = 30.0
        elif vol_mcap_ratio < 0.05:      liq_score = 55.0
        elif vol_mcap_ratio < 0.10:      liq_score = 70.0
        elif vol_mcap_ratio < 2.0:       liq_score = 85.0   # healthy
        else:                            liq_score = 25.0   # wash trading signal
        evidence["liquidity_ratio"]  = round(vol_mcap_ratio, 4)
        evidence["liquidity_source"] = "real"
    else:
        liq_score = 50.0
        evidence["liquidity_source"] = "default"
    components.append((liq_score, 0.25))
    evidence["risk_liquidity_score"] = round(liq_score, 1)

    # ── Holder Concentration Top 10 (20%) ─────────────────────────────────
    top10 = _safe(agent_data.get("top_10_concentration"), -1)
    if top10 >= 0:
        if top10 > 60:       conc_score = 10.0  # severe
        elif top10 > 40:     conc_score = 35.0  # red flag
        elif top10 > 25:     conc_score = 60.0
        elif top10 > 15:     conc_score = 80.0
        else:                conc_score = 90.0
        evidence["top10_concentration"] = round(top10, 1)
        evidence["concentration_source"] = "real"
    else:
        conc_score = 50.0
        evidence["concentration_source"] = "default"
    components.append((conc_score, 0.20))
    evidence["risk_concentration_score"] = round(conc_score, 1)

    # ── Team Wallet Distribution (15%) ─────────────────────────────────────
    # Proxy: buy_sell_ratio — high sell ratio indicates team distribution
    buy_sell = _safe(agent_data.get("buy_sell_ratio"), -1)
    if buy_sell >= 0:
        if buy_sell < 0.3:       tw_dist = 15.0   # heavy selling
        elif buy_sell < 0.5:     tw_dist = 35.0
        elif buy_sell < 0.8:     tw_dist = 55.0
        elif buy_sell < 1.5:     tw_dist = 85.0   # balanced
        else:                    tw_dist = 70.0   # very high buys, could be artificial
        evidence["buy_sell_ratio"] = round(buy_sell, 2)
        evidence["team_wallet_source"] = "buy_sell_proxy"
    else:
        tw_dist = 50.0
        evidence["team_wallet_source"] = "default"
    components.append((tw_dist, 0.15))
    evidence["risk_team_wallet_score"] = round(tw_dist, 1)

    # ── Contract Mutability (15%) ──────────────────────────────────────────
    # Proxy: AI risk assessment
    ai_risk = ai_analysis.get("risk", {})
    contract_risk = ai_risk.get("contract_risk")
    if contract_risk is not None:
        # Expect "low", "medium", "high"
        cr_map = {"low": 90.0, "medium": 55.0, "high": 20.0}
        mutability_score = cr_map.get(str(contract_risk).lower(), 50.0)
        evidence["contract_risk_source"] = "ai_analysis"
    else:
        mutability_score = 50.0
        evidence["contract_risk_source"] = "default"
    components.append((mutability_score, 0.15))
    evidence["risk_mutability_score"] = round(mutability_score, 1)

    # ── Wash Trading Signal (10%) ──────────────────────────────────────────
    moralis_wash = agent_data.get("wash_score")
    if moralis_wash is not None:
        # Moralis wash_score is 0-100 where higher = more wash trading
        # Invert: high wash = low safety score
        wash_score = _clamp(100.0 - float(moralis_wash))
        evidence["wash_trading_source"] = "moralis"
    elif vol >= 0 and mcap > 0:
        vol_ratio = vol / mcap
        if vol_ratio > 2.0:      wash_score = 10.0   # very likely wash trading
        elif vol_ratio > 1.0:    wash_score = 30.0
        elif vol_ratio > 0.5:    wash_score = 60.0
        else:                    wash_score = 90.0   # healthy
        evidence["wash_trading_source"] = "vol_ratio"
    else:
        wash_score = 50.0
        evidence["wash_trading_source"] = "default"
    components.append((wash_score, 0.10))
    evidence["risk_wash_trading_score"] = round(wash_score, 1)

    # ── Time Since Last Update (10%) ──────────────────────────────────────
    last_update = (agent_data.get("github_last_commit") or
                   agent_data.get("last_analyzed") or
                   agent_data.get("updated_at"))
    days_stale = _days_since(last_update)
    if days_stale is not None:
        if days_stale > 90:      stale_score = 5.0    # death signal
        elif days_stale > 30:    stale_score = 30.0   # concerning
        elif days_stale > 14:    stale_score = 60.0
        elif days_stale > 7:     stale_score = 80.0
        else:                    stale_score = 95.0
        evidence["days_since_update"] = round(days_stale, 1)
        evidence["staleness_source"] = "real"
    else:
        stale_score = 50.0
        evidence["staleness_source"] = "default"
    components.append((stale_score, 0.10))
    evidence["risk_staleness_score"] = round(stale_score, 1)

    # ── Audit Status (5%) ─────────────────────────────────────────────────
    audit_status = ai_risk.get("audit_status")
    if audit_status is not None:
        audit_map = {"audited": 90.0, "partial": 55.0, "none": 20.0, "pending": 35.0}
        audit_score = audit_map.get(str(audit_status).lower(), 50.0)
        evidence["audit_source"] = "ai_analysis"
    else:
        audit_score = 50.0
        evidence["audit_source"] = "default"
    components.append((audit_score, 0.05))
    evidence["risk_audit_score"] = round(audit_score, 1)

    # ── Weighted average ──────────────────────────────────────────────────
    total_w = sum(w for _, w in components)
    risk = _clamp(round(sum(s * w for s, w in components) / total_w, 1)) if total_w > 0 else 50.0
    evidence["risk_score"] = risk
    return risk, evidence


# ---------------------------------------------------------------------------
# Main scoring entry point — score_agent()
# ---------------------------------------------------------------------------

def score_agent(agent_data: dict, ai_analysis: dict,
                category_agents: list | None = None) -> dict:
    """
    Full v1.1 scoring pipeline. Returns all sub-scores, composite, and evidence.

    Args:
        agent_data:      Agent row dict from DB (or ingestion).
        ai_analysis:     AI analysis dict (from analyze_agent or batch_triage).
        category_agents: Optional list of peer agents in same category (for
                         Phase 3 edge calculation). Each peer needs at minimum
                         {quality_score, composite_score, market_cap}.

    Returns a dict with:
        composite_score       float  [8, 95]
        quality_score         float  [0, 100]
        upside_score          float  [0, 100]
        momentum_score        float  [0, 100]
        risk_score            float  [0, 100]  higher = safer
        lifecycle_phase       int    1/2/3
        momentum_break_active bool
        phase1_upside         float
        phase2_upside         float
        phase3_upside         float
        tier_classification   str
        score_evidence        dict   {component: source/value}
        ... (all legacy fields for backward compat)
    """
    # ── Phase ──────────────────────────────────────────────────────────────
    phase = classify_phase(agent_data)
    momentum_break_active = False
    if phase == 3:
        momentum_break_active = check_momentum_break(agent_data, ai_analysis)

    # ── Four sub-scores ────────────────────────────────────────────────────
    quality, q_ev = _score_quality(agent_data, ai_analysis)
    upside, u_ev, p1, p2, p3 = _score_upside(
        agent_data, ai_analysis, phase, quality, momentum_break_active, category_agents
    )
    momentum, m_ev = _score_momentum(agent_data, ai_analysis)
    risk, r_ev = _score_risk(agent_data, ai_analysis)

    # ── Composite ─────────────────────────────────────────────────────────
    base = 0.30 * quality + 0.45 * upside + 0.25 * momentum

    # Phase-aware risk multiplier
    if phase == 1:
        risk_mult = 0.20 + 0.80 * (risk / 100.0)
    else:
        risk_mult = 0.30 + 0.70 * (risk / 100.0)

    composite = _clamp(base * risk_mult, 8.0, 95.0)
    composite = round(composite, 1)

    # ── Tier classification ────────────────────────────────────────────────
    classification = _classify_tier(composite)

    # ── Legacy sub-scores (kept for backward compat with factor_details) ──
    legacy = _calculate_legacy_scores(agent_data, ai_analysis)
    scores = legacy["scores"]

    # Inject new v1.1 scores into the scores dict
    scores["quality_score"]        = quality
    scores["upside_score"]         = upside
    scores["momentum_score"]       = momentum
    scores["risk_score"]           = risk
    scores["lifecycle_phase"]      = phase
    scores["momentum_break_active"] = momentum_break_active
    scores["phase1_upside"]        = p1
    scores["phase2_upside"]        = p2
    scores["phase3_upside"]        = p3

    # Combined evidence
    score_evidence = {**q_ev, **u_ev, **m_ev, **r_ev}
    scores["_score_evidence"] = score_evidence

    # Rebuild edge score with new composite
    _mcap_percentile = _bm_log_score(
        _safe(agent_data.get("market_cap"), 0), _BM_MCAP
    ) if _safe(agent_data.get("market_cap"), 0) > 0 else None
    if _mcap_percentile is not None:
        edge_score = round(composite - _mcap_percentile, 1)
        edge_label = ("Undervalued" if edge_score > 5 else
                      "Overvalued" if edge_score < -5 else "Fair Value")
    else:
        edge_score = None
        edge_label = None

    scores["_edge_score"] = edge_score
    scores["_edge_label"] = edge_label

    # Generate sub-score reason strings
    factor_reasons = _build_factor_reasons(agent_data, ai_analysis)
    factor_reasons["quality_score"]  = _build_quality_reason(q_ev)
    factor_reasons["upside_score"]   = _build_upside_reason(u_ev, phase, momentum_break_active)
    factor_reasons["momentum_score"] = _build_momentum_reason(m_ev)
    factor_reasons["risk_score"]     = _build_risk_reason(r_ev)
    scores["_factor_reasons"] = factor_reasons

    return {
        # New v1.1 primary fields
        "composite_score":         composite,
        "tier_classification":     classification,
        "quality_score":           quality,
        "upside_score":            upside,
        "momentum_score":          momentum,
        "risk_score":              risk,
        "lifecycle_phase":         phase,
        "momentum_break_active":   momentum_break_active,
        "phase1_upside":           p1,
        "phase2_upside":           p2,
        "phase3_upside":           p3,
        "score_evidence":          score_evidence,
        # Legacy fields (backward compat)
        "scores":                  scores,
        "tier_scores":             legacy["tier_scores"],
        "first_mover":             legacy["first_mover"],
        "score_narrative":         legacy["score_narrative"],
        "one_liner":               legacy["one_liner"],
        "top_helped":              legacy["top_helped"],
        "top_hurt":                legacy["top_hurt"],
        "doxx_tier_detail":        legacy["doxx_tier_detail"],
        "dead_flagged":            legacy["dead_flagged"],
        "strong_flagged":          legacy["strong_flagged"],
        "factors_scored":          legacy["factors_scored"],
        "factor_reasons":          factor_reasons,
        "score_modifiers":         legacy["score_modifiers"],
        "edge_score":              edge_score,
        "edge_label":              edge_label,
    }


# ---------------------------------------------------------------------------
# Reason string builders
# ---------------------------------------------------------------------------

def _build_quality_reason(ev: dict) -> str:
    parts = []
    if ev.get("idea_tam_source") == "ai_analysis":
        parts.append("Idea/TAM from AI analysis")
    else:
        parts.append("Idea/TAM from category lookup")
    ps = ev.get("product_stage", "unknown")
    if ps and ps != "unknown":
        parts.append(f"Product: {ps}")
    doxx = ev.get("team_doxx_tier")
    if doxx:
        parts.append(f"Doxx tier {doxx}")
    if ev.get("moat_source") == "real":
        parts.append(f"Moat {ev.get('moat_score', 50)}/100")
    return ". ".join(parts) if parts else "Quality score from category defaults"


def _build_upside_reason(ev: dict, phase: int, mb_active: bool) -> str:
    phase_label = f"Phase {phase}"
    if mb_active:
        phase_label = "Phase 2 (momentum break override)"
    key = {1: "phase1_upside_score", 2: "phase2_upside_score",
           3: "phase3_upside_score"}.get(2 if mb_active else phase)
    score = ev.get(key, "?")
    parts = [f"{phase_label}: score {score}"]
    if ev.get("divergence_penalty"):
        parts.append("divergence penalty applied")
    if "age_decay" in ev:
        parts.append(f"age decay ×{ev['age_decay']}")
    if "raw_edge" in ev:
        gap = ev["raw_edge"]
        parts.append(f"quality-vs-mcap gap {'+' if gap >= 0 else ''}{gap:.1f}")
    return ". ".join(parts)


def _build_momentum_reason(ev: dict) -> str:
    parts = []
    oc = ev.get("on_chain_score")
    sc = ev.get("social_score")
    if oc is not None:
        parts.append(f"On-chain {oc:.0f}/100")
    if sc is not None:
        parts.append(f"social {sc:.0f}/100")
    if ev.get("holder_growth_pct_24h") is not None:
        h = ev["holder_growth_pct_24h"]
        parts.append(f"holders {'+' if h >= 0 else ''}{h:.1f}%/24h")
    if ev.get("smart_money_source") == "real":
        parts.append("smart money data available")
    return ". ".join(parts) if parts else "Momentum from available signals"


def _build_risk_reason(ev: dict) -> str:
    parts = []
    liq = ev.get("risk_liquidity_score")
    conc = ev.get("risk_concentration_score")
    stale = ev.get("days_since_update")
    if liq is not None:
        parts.append(f"Liquidity {liq:.0f}/100")
    if conc is not None:
        top10 = ev.get("top10_concentration")
        if top10 is not None:
            parts.append(f"top-10 holds {top10:.0f}%")
    if stale is not None:
        parts.append(f"{stale:.0f}d since update")
    return ". ".join(parts) if parts else "Risk from available signals"


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
# doxx tier2 detail (UI breakdown only)
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
# Classification helpers
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
# Factor evidence strings (human-readable reason behind each legacy factor)
# ---------------------------------------------------------------------------

def _build_factor_reasons(agent_data: dict, ai_analysis: dict) -> dict:
    """Return a {factor_key: one_line_evidence_string} dict for UI display."""
    reasons: dict[str, str] = {}
    fm = ai_analysis.get("first_mover", {})

    # F_idea_market_fit
    cat = str(
        agent_data.get("category") or agent_data.get("agent_type") or
        ai_analysis.get("category") or ""
    ).strip() or "Unknown"
    ai_tam = ai_analysis.get("market", {}).get("tam_score")
    if fm.get("category_unique") is not None or fm.get("approach_novel") is not None:
        cu = fm.get("category_unique")
        an = fm.get("approach_novel")
        parts = []
        if cu is True:  parts.append("novel category")
        elif cu is False: parts.append("crowded category")
        if an is True:  parts.append("novel approach")
        elif an is False: parts.append("common approach")
        reasons["F_idea_market_fit"] = "AI: " + ", ".join(parts) if parts else f"Category: {cat}"
    elif ai_tam is not None:
        reasons["F_idea_market_fit"] = f"Category: {cat}, AI TAM score {ai_tam:.0f}/100"
    else:
        reasons["F_idea_market_fit"] = f"Category: {cat}"

    # F_moat
    days = _days_since(agent_data.get("creation_date") or agent_data.get("first_seen"))
    ai_def = fm.get("defensibility_score")
    stars = _safe(agent_data.get("github_stars"), 0)
    moat_parts = []
    if ai_def is not None:
        moat_parts.append(f"defensibility {ai_def:.0f}/100")
    if days is not None:
        moat_parts.append(f"{int(days)} days old")
    if stars > 0:
        moat_parts.append(f"{int(stars)} GitHub stars")
    reasons["F_moat"] = ", ".join(moat_parts) if moat_parts else "No moat signals available"

    # F_execution
    twitter = str(agent_data.get("linked_twitter") or "").strip()
    website = str(agent_data.get("linked_website") or "").strip()
    ai_status = str(ai_analysis.get("product", {}).get("status", "")).strip()
    exec_parts = []
    if website and len(website) > 5:
        exec_parts.append("website ✓")
    if twitter and len(twitter) > 5:
        followers = int(_safe(agent_data.get("twitter_followers"), 0))
        exec_parts.append(f"Twitter {followers:,} followers" if followers else "Twitter ✓")
    if ai_status:
        exec_parts.append(f"stage: {ai_status}")
    reasons["F_execution"] = ", ".join(exec_parts) if exec_parts else "No execution signals"

    # F_holders
    h = _safe(agent_data.get("holder_count"), -1)
    if h < 0:
        reasons["F_holders"] = "No holder data available"
    else:
        cmp = "above" if h >= _BM_HOLDERS else "below"
        reasons["F_holders"] = f"{int(h):,} holders vs {_BM_HOLDERS:,} benchmark — {cmp} median"

    # F_mcap
    m = _safe(agent_data.get("market_cap"), -1)
    if m < 0:
        reasons["F_mcap"] = "No market cap data available"
    elif m == 0:
        reasons["F_mcap"] = "Zero market cap"
    else:
        cmp = "above" if m >= _BM_MCAP else "below"
        mc_str = (f"${m/1_000_000:.1f}M" if m >= 1_000_000
                  else f"${m/1_000:.0f}K" if m >= 1_000 else f"${m:,.0f}")
        reasons["F_mcap"] = f"{mc_str} market cap vs ${_BM_MCAP/1_000_000:.1f}M benchmark — {cmp} median"

    # F_volume
    v = _safe(agent_data.get("volume_24h"), -1)
    if v < 0:
        reasons["F_volume"] = "No volume data available"
    elif v == 0:
        reasons["F_volume"] = "Zero trading volume — project may be inactive"
    else:
        cmp = "above" if v >= _BM_VOLUME else "below"
        v_str = (f"${v/1_000_000:.1f}M" if v >= 1_000_000
                 else f"${v/1_000:.0f}K" if v >= 1_000 else f"${v:,.0f}")
        reasons["F_volume"] = f"{v_str} daily volume vs ${_BM_VOLUME/1_000:.0f}K benchmark — {cmp} median"

    # F_efficiency
    vol  = _safe(agent_data.get("volume_24h"), -1)
    mcap = _safe(agent_data.get("market_cap"),  0)
    if vol < 0 or mcap <= 0:
        reasons["F_efficiency"] = "Insufficient data for efficiency calculation"
    else:
        r = vol / mcap
        if r > 2.0:   label = "likely wash trading"
        elif r > 0.1: label = "healthy activity"
        elif r > 0.01: label = "moderate trading"
        else:          label = "low activity"
        reasons["F_efficiency"] = f"Vol/MCap ratio {r:.2%} — {label}"

    # F_momentum
    change = agent_data.get("holder_count_change_24h")
    if change is None:
        reasons["F_momentum"] = "No holder change data available"
    else:
        c = _safe(change, 0)
        holders = max(_safe(agent_data.get("holder_count"), 1), 1)
        pct = c / holders * 100.0
        sign = "+" if c >= 0 else ""
        reasons["F_momentum"] = f"{sign}{int(c):,} holders in 24h ({sign}{pct:.1f}%)"

    return reasons


# ---------------------------------------------------------------------------
# Legacy factor computation (kept for backward compat + factor_details UI)
# ---------------------------------------------------------------------------

FACTORS = [
    (_score_quality,  1.00, "F_idea_market_fit", "idea"),
    (None,            1.00, "F_moat",            "moat"),
    (None,            1.00, "F_execution",        "execution"),
    (None,            0.35, "F_holders",         "market"),
    (None,            0.22, "F_mcap",            "market"),
    (None,            0.30, "F_volume",          "market"),
    (None,            0.08, "F_efficiency",      "market"),
    (None,            0.05, "F_momentum",        "market"),
]

_TIER_WEIGHTS = {
    "idea":      0.35,
    "moat":      0.20,
    "execution": 0.10,
    "market":    0.35,
}


def _f_idea_market_fit(agent: dict, ai: dict) -> float:
    """Combined idea × market-fit for legacy factor display."""
    q, _ = _score_quality(agent, ai)
    return q


def _f_moat(agent: dict, ai: dict) -> float | None:
    moat_signals = []
    fm = ai.get("first_mover", {})
    ai_def = fm.get("defensibility_score")
    if ai_def is not None:
        moat_signals.append((_clamp(float(ai_def)), 3.0))
    stars        = _safe(agent.get("github_stars"), 0)
    contributors = _safe(agent.get("github_contributors"), 0)
    commits      = _safe(agent.get("github_commits_30d"), 0)
    if stars > 0 or contributors > 0 or commits > 0:
        gh_score  = min(60.0, math.log10(max(stars, 1)) * 22.0)
        gh_score += min(20.0, contributors * 5.0)
        gh_score += min(20.0, commits * 1.0)
        moat_signals.append((_clamp(gh_score), 2.0))
    days = _days_since(agent.get("creation_date") or agent.get("first_seen"))
    if days is not None:
        age_score = (80.0 if days > 365 else 65.0 if days > 180 else
                     48.0 if days > 90  else 30.0 if days > 30  else 15.0)
        moat_signals.append((age_score, 1.0))
    partnership = ai.get("product", {}).get("partnership_score")
    if partnership is not None:
        moat_signals.append((_clamp(float(partnership)), 1.5))
    holders = _safe(agent.get("holder_count"), 0)
    if holders >= 1_000:
        moat_signals.append((_clamp(20.0 * math.log10(holders)), 2.0))
    if not moat_signals:
        return None
    w_sum   = sum(s * w for s, w in moat_signals)
    w_total = sum(w for _, w in moat_signals)
    return _clamp(round(w_sum / w_total, 1))


def _f_execution(agent: dict, ai: dict) -> float | None:
    points = 0.0
    max_pts = 0.0
    has_any = False

    website = str(agent.get("linked_website") or "").strip()
    max_pts += 35.0
    if website and len(website) > 5:
        has_any = True
        points += 35.0 if "github.com" not in website else 15.0

    twitter = str(agent.get("linked_twitter") or "").strip()
    max_pts += 30.0
    if twitter and len(twitter) > 5:
        has_any = True
        followers = _safe(agent.get("twitter_followers"), 0)
        if   followers > 50_000: points += 30.0
        elif followers > 10_000: points += 25.0
        elif followers > 1_000:  points += 20.0
        elif followers > 100:    points += 14.0
        else:                    points +=  8.0

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

    doxx_raw = agent.get("doxx_tier") or ai.get("team", {}).get("doxx_tier")
    if doxx_raw is not None:
        doxx_tier_val = int(doxx_raw)
        if doxx_tier_val in (1, 2):
            max_pts += 10.0
            has_any  = True
            points += 10.0 if doxx_tier_val == 1 else 6.0

    if not has_any:
        return None
    raw = (points / max_pts * 100.0) if max_pts > 0 else 0.0
    return _clamp(round(raw, 1))


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
    return _bm_log_score(v, _BM_VOLUME) if v > 0 else 1.0


def _f_efficiency(agent: dict, ai: dict) -> float | None:
    vol  = _safe(agent.get("volume_24h"), -1)
    mcap = _safe(agent.get("market_cap"),  0)
    if vol < 0 or mcap <= 0: return None
    r = vol / mcap
    if r > 2.0:   return 18.0
    if r > 0.5:   return 38.0
    if r > 0.1:   return 72.0
    if r > 0.05:  return 62.0
    if r > 0.01:  return 50.0
    if r > 0.001: return 32.0
    return 15.0


def _f_momentum_legacy(agent: dict, ai: dict) -> float | None:
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


_LEGACY_FACTORS = [
    (_f_idea_market_fit, 1.00, "F_idea_market_fit", "idea"),
    (_f_moat,            1.00, "F_moat",            "moat"),
    (_f_execution,       1.00, "F_execution",       "execution"),
    (_f_holders,         0.35, "F_holders",         "market"),
    (_f_mcap,            0.22, "F_mcap",            "market"),
    (_f_volume,          0.30, "F_volume",          "market"),
    (_f_efficiency,      0.08, "F_efficiency",      "market"),
    (_f_momentum_legacy, 0.05, "F_momentum",        "market"),
]


def _calculate_legacy_scores(agent_data: dict, ai_analysis: dict) -> dict:
    """
    Compute the legacy factor breakdown used for UI display and factor_details.
    Returns the same shape as the old calculate_composite_score() internal dict,
    WITHOUT the new composite (that is overridden by score_agent()).
    """
    scores         = {}
    factor_details = []
    tier_acc: dict[str, list] = {t: [] for t in _TIER_WEIGHTS}

    for fn, weight, name, tier in _LEGACY_FACTORS:
        try:
            if fn is None:
                raw = None
            elif name == "F_idea_market_fit":
                raw = fn(agent_data, ai_analysis)
            else:
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

    tier_scores: dict[str, float | None] = {}
    for tier_name, items in tier_acc.items():
        if items:
            w_sum   = sum(s * w for s, w in items)
            w_total = sum(w       for _, w in items)
            tier_scores[tier_name] = round(w_sum / w_total, 1)
        else:
            tier_scores[tier_name] = None

    comb_sum = 0.0
    comb_w   = 0.0
    for tier_name, tw in _TIER_WEIGHTS.items():
        ts = tier_scores.get(tier_name)
        if ts is not None:
            comb_sum += ts * tw
            comb_w   += tw

    raw_composite = (comb_sum / comb_w) if comb_w > 0 else 20.0
    composite = round(_clamp(8.0 + (raw_composite / 100.0) * 87.0, 8.0, 95.0), 1)

    # Volume vitality
    _vol = _safe(agent_data.get("volume_24h"), 0)
    score_modifiers: list[dict] = []
    _pre = composite
    if _vol == 0:        _vitality = 0.62
    elif _vol < 200:     _vitality = 0.80
    elif _vol < 1_000:   _vitality = 0.90
    elif _vol < 10_000:  _vitality = 1.00
    else:                _vitality = 1.06

    composite = round(_clamp(composite * _vitality, 8.0, 95.0), 1)
    if _vitality != 1.00:
        impact = round(composite - _pre, 1)
        if _vol == 0:
            score_modifiers.append({"label": "Zero volume penalty", "impact": impact})
        elif _vitality < 1.00:
            score_modifiers.append({"label": "Low activity penalty", "impact": impact})
        else:
            score_modifiers.append({"label": "Strong volume bonus", "impact": impact})

    _holders = _safe(agent_data.get("holder_count"), 0)
    _pre_cap = composite
    if _vol == 0 and _holders < 50:
        composite = min(composite, 15.0)
    elif _vol < 10 and _holders < 100:
        composite = min(composite, 20.0)
    if composite < _pre_cap:
        score_modifiers.append({"label": "Dead project cap applied",
                                 "impact": round(composite - _pre_cap, 1)})

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

    dead_flagged   = _is_dead_agent(agent_data)
    strong_flagged = _is_strong_investment(agent_data)
    tier_scores_display = {k: (v if v is not None else 0.0) for k, v in tier_scores.items()}
    one_liner = _build_one_liner(tier_scores_display, agent_data)
    narrative = _build_score_narrative(
        agent_data, ai_analysis, scores, composite, tier_scores_display
    )

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

    active = [fd for fd in factor_details if not fd["skipped"]]
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

    _mcap_pct = _f_mcap(agent_data, ai_analysis)
    if _mcap_pct is not None:
        edge_score = round(composite - _mcap_pct, 1)
        edge_label = ("Undervalued" if edge_score > 5 else
                      "Overvalued" if edge_score < -5 else "Fair Value")
    else:
        edge_score = None
        edge_label = None

    scores["_tier_scores"]      = tier_scores_display
    scores["_one_liner"]        = one_liner
    scores["_top_helped"]       = top_helped
    scores["_top_hurt"]         = top_hurt
    scores["_doxx_tier_detail"] = doxx_detail
    scores["_dead_flagged"]     = dead_flagged
    scores["_strong_flagged"]   = strong_flagged
    scores["_factors_scored"]   = factors_scored
    scores["_factor_reasons"]   = _build_factor_reasons(agent_data, ai_analysis)
    scores["_score_modifiers"]  = score_modifiers
    scores["_edge_score"]       = edge_score
    scores["_edge_label"]       = edge_label

    return {
        "composite_score":    composite,
        "tier_classification": _classify_tier(composite),
        "scores":              scores,
        "tier_scores":         tier_scores_display,
        "first_mover":         False,
        "score_narrative":     narrative,
        "one_liner":           one_liner,
        "top_helped":          top_helped,
        "top_hurt":            top_hurt,
        "doxx_tier_detail":    doxx_detail,
        "dead_flagged":        dead_flagged,
        "strong_flagged":      strong_flagged,
        "factors_scored":      factors_scored,
        "factor_reasons":      _build_factor_reasons(agent_data, ai_analysis),
        "score_modifiers":     score_modifiers,
        "edge_score":          edge_score,
        "edge_label":          edge_label,
    }


# ---------------------------------------------------------------------------
# calculate_composite_score — backward-compat entry point
# ---------------------------------------------------------------------------

def calculate_composite_score(agent_data: dict, ai_analysis: dict,
                               category_agents: list | None = None) -> dict:
    """
    Backward-compatible entry point. Delegates to score_agent() and returns
    the full v1.1 result. All callers in database.py and analyzer.py that
    previously used this function will now get the new composite scores.
    """
    return score_agent(agent_data, ai_analysis, category_agents)
