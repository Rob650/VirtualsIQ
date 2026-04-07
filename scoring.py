"""
VirtualsIQ — 24-Factor Weighted Scoring Engine

Tier 1 — Foundation / First Mover (30%)
Tier 2 — Traction / Team & Execution (28%)
Tier 3 — On-chain / Value Pool (24%)
Tier 4 — Narrative / Community (18%)

Missing data defaults to 50 (neutral), never penalized.
"""

import math
from datetime import datetime


NEUTRAL = 50.0  # Default score when data is absent

TIER_LABELS = {
    "first_mover": "Foundation",
    "team": "Traction",
    "value": "On-chain",
    "community": "Narrative",
}

FACTOR_LABELS = {
    "F1_category_uniqueness": "Category Uniqueness",
    "F2_approach_novelty": "Approach Novelty",
    "F3_cross_chain_originality": "Cross-Chain Originality",
    "F4_timing_advantage": "Timing Advantage",
    "F5_defensibility": "Defensibility / Moat",
    "F6_doxx_tier": "Team Visibility",
    "F7_track_record": "Track Record",
    "F8_code_activity": "Code Activity",
    "F9_shipping_cadence": "Shipping Cadence",
    "F10_product_status": "Product Status",
    "F11_partnerships": "Partnerships",
    "F12_wallet_behavior": "Wallet Behavior",
    "F13_tam": "Total Addressable Market",
    "F14_real_world_comparables": "Real-World Comparables",
    "F15_revenue_model": "Revenue Model",
    "F16_current_revenue": "Current Revenue",
    "F17_mcap_to_tam": "MCap / TAM Ratio",
    "F18_saturation": "Market Saturation",
    "F19_holder_distribution": "Holder Distribution",
    "F20_twitter_engagement": "Twitter Engagement",
    "F21_follower_growth": "Follower Growth",
    "F22_community_depth": "Community Depth",
    "F23_organic_signals": "Organic Signals",
    "F24_smart_money": "Smart Money",
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe(value, default=NEUTRAL) -> float:
    """Return numeric value or neutral default."""
    try:
        v = float(value)
        return v if not math.isnan(v) else default
    except (TypeError, ValueError):
        return default


def _clamp(v: float, lo=0.0, hi=100.0) -> float:
    return max(lo, min(hi, v))


def _days_since(date_str: str | None) -> float | None:
    """Return days since date_str (ISO or similar). None if unparseable."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str[:26], fmt[:len(date_str[:26])])
            return (datetime.utcnow() - dt).total_seconds() / 86400
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Tier 1 — Foundation / First Mover (30%)
# ---------------------------------------------------------------------------

def _f1_category_uniqueness(agent: dict, ai: dict) -> float:
    v = ai.get("first_mover", {}).get("category_unique")
    if v is True:
        return 100.0
    if v is False:
        return 20.0
    return NEUTRAL


def _f2_approach_novelty(agent: dict, ai: dict) -> float:
    v = ai.get("first_mover", {}).get("approach_novel")
    if v is True:
        return 90.0
    if v is False:
        return 30.0
    return NEUTRAL


def _f3_cross_chain_originality(agent: dict, ai: dict) -> float:
    v = ai.get("first_mover", {}).get("cross_chain_original")
    if v is True:
        return 85.0
    if v is False:
        return 35.0
    return NEUTRAL


def _f4_timing_advantage(agent: dict, ai: dict) -> float:
    days = ai.get("first_mover", {}).get("days_ahead_of_competitor")
    if days is None:
        return NEUTRAL
    days = float(days)
    if days > 180:
        return 100.0
    if days > 90:
        return 85.0
    if days > 30:
        return 65.0
    if days > 7:
        return 45.0
    return 20.0


def _f5_defensibility(agent: dict, ai: dict) -> float:
    v = ai.get("first_mover", {}).get("defensibility_score")
    if v is not None:
        return _clamp(float(v))
    github = _safe(agent.get("github_stars"), 0)
    if github > 500:
        return 80.0
    if github > 100:
        return 60.0
    return NEUTRAL


# ---------------------------------------------------------------------------
# Tier 2 — Traction / Team & Execution (28%)
# ---------------------------------------------------------------------------

def _f6_doxx_tier(agent: dict, ai: dict) -> float:
    """Dynamic doxx scoring with deeper Tier 2 analysis."""
    tier = int(agent.get("doxx_tier") or ai.get("team", {}).get("doxx_tier") or 3)

    if tier == 1:
        base = 100.0
    elif tier == 2:
        # Dynamic Tier 2 scoring — not just "has Twitter"
        base = 60.0
        twitter_age = _safe(agent.get("twitter_account_age"), 0)
        followers = _safe(agent.get("twitter_followers"), 0)
        engagement = _safe(agent.get("twitter_engagement_rate"), 0)
        creation_date = agent.get("creation_date")

        # Account age bonus: older accounts more credible
        if twitter_age > 365:  # over 1 year
            base += 10.0
        elif twitter_age > 180:
            base += 5.0

        # Follower quality: very low followers = suspicious
        if followers > 10000:
            base += 8.0
        elif followers > 1000:
            base += 4.0
        elif followers < 100:
            base -= 5.0

        # Engagement authenticity: suspiciously high or zero = red flag
        if 0.5 <= engagement <= 8.0:
            base += 5.0  # healthy range
        elif engagement > 15.0:
            base -= 5.0  # likely bot engagement
        elif engagement <= 0:
            base -= 3.0

        # Account existed before project? (Compare twitter age vs project age)
        if creation_date:
            project_days = _days_since(creation_date)
            if project_days and twitter_age > 0:
                if twitter_age > project_days + 90:
                    base += 5.0  # account predates project — more credible
                elif twitter_age < project_days - 30:
                    base -= 5.0  # account created after project launch — suspicious

        base = _clamp(base)
    else:
        base = 20.0

    # Penalize for team red flags
    team = ai.get("team", {})
    if team.get("red_flags") and len(team["red_flags"]) > 0:
        base = max(0, base - len(team["red_flags"]) * 10)
    return _clamp(base)


def _f7_track_record(agent: dict, ai: dict) -> float:
    v = ai.get("team", {}).get("track_record_score")
    if v is not None:
        score = _clamp(float(v))
        tier = int(agent.get("doxx_tier") or ai.get("team", {}).get("doxx_tier") or 3)
        if tier == 3:
            score = min(score, 40.0)
        return score
    return NEUTRAL


def _f8_code_activity(agent: dict, ai: dict) -> float:
    stars = _safe(agent.get("github_stars"), 0)
    commits = _safe(agent.get("github_commits_30d"), 0)
    contributors = _safe(agent.get("github_contributors"), 0)
    if stars == 0 and commits == 0:
        return NEUTRAL
    score = 0.0
    score += min(stars / 10, 30)
    score += min(commits / 2, 40)
    score += min(contributors * 5, 30)
    return _clamp(score)


def _f9_shipping_cadence(agent: dict, ai: dict) -> float:
    last_commit = agent.get("github_last_commit")
    days = _days_since(last_commit)
    if days is None:
        return NEUTRAL
    if days <= 7:
        return 100.0
    if days <= 30:
        return 75.0
    if days <= 90:
        return 50.0
    if days <= 180:
        return 25.0
    return 10.0


def _f10_product_status(agent: dict, ai: dict) -> float:
    status_map = {
        "live": 100.0, "production": 100.0,
        "beta": 70.0, "mainnet_beta": 70.0,
        "testnet": 40.0, "alpha": 40.0,
        "pre-product": 20.0, "pre_product": 20.0, "development": 20.0,
        "vaporware": 0.0, "concept": 0.0,
    }
    v = str(ai.get("product", {}).get("status", "")).lower().replace(" ", "_")
    if v in status_map:
        return status_map[v]
    if agent.get("status") == "Sentient":
        return 80.0
    return NEUTRAL


def _f11_partnerships(agent: dict, ai: dict) -> float:
    v = ai.get("product", {}).get("partnership_score")
    if v is not None:
        score = _clamp(float(v))
        product = ai.get("product", {})
        if product.get("technical_moat") and not product.get("red_flags"):
            score = min(100, score + 5)
        if product.get("red_flags") and len(product["red_flags"]) > 0:
            score = max(0, score - len(product["red_flags"]) * 8)
        return score
    return NEUTRAL


def _f12_wallet_behavior(agent: dict, ai: dict) -> float:
    v = ai.get("team", {}).get("wallet_behavior_score")
    return _clamp(float(v)) if v is not None else NEUTRAL


# ---------------------------------------------------------------------------
# Tier 3 — On-chain / Value Pool (24%)
# ---------------------------------------------------------------------------

def _f13_tam(agent: dict, ai: dict) -> float:
    v = ai.get("market", {}).get("tam_score")
    return _clamp(float(v)) if v is not None else NEUTRAL


def _f14_real_world_comparables(agent: dict, ai: dict) -> float:
    v = ai.get("market", {}).get("comparables_score")
    return _clamp(float(v)) if v is not None else NEUTRAL


def _f15_revenue_model(agent: dict, ai: dict) -> float:
    v = ai.get("market", {}).get("revenue_model_score")
    return _clamp(float(v)) if v is not None else NEUTRAL


def _f16_current_revenue(agent: dict, ai: dict) -> float:
    v = ai.get("market", {}).get("current_revenue_score")
    if v is not None:
        return _clamp(float(v))
    mcap = _safe(agent.get("market_cap"), 0)
    if mcap > 0:
        if mcap > 50_000_000:
            return 95.0
        if mcap > 10_000_000:
            return 82.0
        if mcap > 1_000_000:
            return 68.0
        if mcap > 100_000:
            return 48.0
        return 25.0
    vol = _safe(agent.get("volume_24h"), 0)
    if vol > 1_000_000:
        return 90.0
    if vol > 100_000:
        return 70.0
    if vol > 10_000:
        return 50.0
    if vol > 1_000:
        return 30.0
    return 20.0


def _f17_mcap_to_tam(agent: dict, ai: dict) -> float:
    v = ai.get("market", {}).get("mcap_tam_ratio")
    if v is not None:
        ratio = float(v)
        if ratio < 0.001:
            return 95.0
        if ratio < 0.01:
            return 80.0
        if ratio < 0.1:
            return 60.0
        if ratio < 0.5:
            return 35.0
        return 15.0
    mcap = _safe(agent.get("market_cap"), 0)
    if mcap > 0:
        if mcap < 100_000:
            return 90.0
        if mcap < 1_000_000:
            return 78.0
        if mcap < 10_000_000:
            return 62.0
        if mcap < 100_000_000:
            return 42.0
        return 22.0
    return NEUTRAL


def _f18_saturation(agent: dict, ai: dict) -> float:
    v = ai.get("market", {}).get("saturation_score")
    return _clamp(float(v)) if v is not None else NEUTRAL


# ---------------------------------------------------------------------------
# Tier 4 — Narrative / Community (18%)
# ---------------------------------------------------------------------------

def _f19_holder_distribution(agent: dict, ai: dict) -> float:
    holders = _safe(agent.get("holder_count"), 0)
    top10 = _safe(agent.get("top_10_concentration"), 100)
    if holders == 0:
        return NEUTRAL
    holder_score = min(math.log10(max(holders, 1)) * 10, 60)
    concentration_score = max(0, 40 - top10 / 2.5)
    return _clamp(holder_score + concentration_score)


def _f20_twitter_engagement(agent: dict, ai: dict) -> float:
    rate = _safe(agent.get("twitter_engagement_rate"), -1)
    if rate < 0:
        return NEUTRAL
    if rate >= 5.0:
        return 95.0
    if rate >= 2.0:
        return 75.0
    if rate >= 1.0:
        return 55.0
    if rate >= 0.5:
        return 35.0
    return 20.0


def _f21_follower_growth(agent: dict, ai: dict) -> float:
    v = ai.get("community", {}).get("follower_growth_score")
    if v is not None:
        return _clamp(float(v))
    followers = _safe(agent.get("twitter_followers"), 0)
    if followers > 50_000:
        return 90.0
    if followers > 10_000:
        return 70.0
    if followers > 1_000:
        return 50.0
    if followers > 100:
        return 30.0
    return 15.0


def _f22_community_depth(agent: dict, ai: dict) -> float:
    v = ai.get("community", {}).get("depth_score")
    return _clamp(float(v)) if v is not None else NEUTRAL


def _f23_organic_signals(agent: dict, ai: dict) -> float:
    v = ai.get("community", {}).get("organic_score")
    return _clamp(float(v)) if v is not None else NEUTRAL


def _f24_smart_money(agent: dict, ai: dict) -> float:
    v = ai.get("community", {}).get("smart_money_score")
    if v is not None:
        return _clamp(float(v))
    bsr = _safe(agent.get("buy_sell_ratio"), 1.0)
    vol = _safe(agent.get("volume_24h"), 0)
    if bsr > 1.5 and vol > 100_000:
        return 80.0
    if bsr > 1.2:
        return 60.0
    return NEUTRAL


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

FACTORS = [
    # (function, weight, name, tier)
    (_f1_category_uniqueness,  0.08, "F1_category_uniqueness",   "first_mover"),
    (_f2_approach_novelty,     0.07, "F2_approach_novelty",       "first_mover"),
    (_f3_cross_chain_originality, 0.06, "F3_cross_chain_originality", "first_mover"),
    (_f4_timing_advantage,     0.05, "F4_timing_advantage",       "first_mover"),
    (_f5_defensibility,        0.04, "F5_defensibility",          "first_mover"),

    (_f6_doxx_tier,            0.05, "F6_doxx_tier",              "team"),
    (_f7_track_record,         0.05, "F7_track_record",           "team"),
    (_f8_code_activity,        0.04, "F8_code_activity",          "team"),
    (_f9_shipping_cadence,     0.04, "F9_shipping_cadence",        "team"),
    (_f10_product_status,      0.04, "F10_product_status",         "team"),
    (_f11_partnerships,        0.03, "F11_partnerships",           "team"),
    (_f12_wallet_behavior,     0.03, "F12_wallet_behavior",        "team"),

    (_f13_tam,                 0.05, "F13_tam",                    "value"),
    (_f14_real_world_comparables, 0.05, "F14_real_world_comparables", "value"),
    (_f15_revenue_model,       0.04, "F15_revenue_model",          "value"),
    (_f16_current_revenue,     0.04, "F16_current_revenue",        "value"),
    (_f17_mcap_to_tam,         0.03, "F17_mcap_to_tam",            "value"),
    (_f18_saturation,          0.03, "F18_saturation",             "value"),

    (_f19_holder_distribution, 0.04, "F19_holder_distribution",   "community"),
    (_f20_twitter_engagement,  0.03, "F20_twitter_engagement",     "community"),
    (_f21_follower_growth,     0.03, "F21_follower_growth",        "community"),
    (_f22_community_depth,     0.03, "F22_community_depth",        "community"),
    (_f23_organic_signals,     0.03, "F23_organic_signals",        "community"),
    (_f24_smart_money,         0.02, "F24_smart_money",            "community"),
]


def _classify_tier(score: float) -> str:
    if score >= 85:
        return "Top Tier"
    if score >= 70:
        return "Strong"
    if score >= 50:
        return "Moderate"
    if score >= 30:
        return "Weak"
    return "Avoid"


def _build_one_liner(tier_scores: dict, agent_data: dict) -> str:
    """Build plain-English one-liner: 'Strong on X and Y, weak on Z'"""
    if not tier_scores:
        return ""
    sorted_tiers = sorted(tier_scores.items(), key=lambda x: x[1], reverse=True)
    strong = [TIER_LABELS.get(t, t) for t, s in sorted_tiers if s >= 60]
    weak = [TIER_LABELS.get(t, t) for t, s in sorted_tiers if s < 40]

    parts = []
    if strong:
        parts.append("Strong on " + " and ".join(strong[:2]))
    if weak:
        parts.append("weak on " + " and ".join(weak[:2]))

    if parts:
        return ", ".join(parts)
    # All moderate
    best = TIER_LABELS.get(sorted_tiers[0][0], sorted_tiers[0][0])
    worst = TIER_LABELS.get(sorted_tiers[-1][0], sorted_tiers[-1][0])
    return f"Balanced profile, strongest in {best}, watch {worst}"


def calculate_composite_score(agent_data: dict, ai_analysis: dict) -> dict:
    """
    Run all 24 factors and return composite score with breakdown.

    Returns:
        {
            "composite_score": float,
            "tier_classification": str,
            "scores": {factor_name: float, ...},
            "tier_scores": {tier_name: float, ...},
            "first_mover": bool,
            "score_narrative": str,
            "one_liner": str,
            "top_helped": [{"factor": str, "score": float, "label": str}, ...],
            "top_hurt": [{"factor": str, "score": float, "label": str}, ...],
            "doxx_tier_detail": {...},
        }
    """
    scores = {}
    weighted_sum = 0.0
    factor_details = []

    for fn, weight, name, _tier in FACTORS:
        try:
            raw = fn(agent_data, ai_analysis)
        except Exception:
            raw = NEUTRAL
        score = _clamp(raw)
        scores[name] = round(score, 1)
        # Track contribution = (score - 50) * weight (deviation from neutral)
        contribution = (score - NEUTRAL) * weight
        factor_details.append({
            "factor": name,
            "label": FACTOR_LABELS.get(name, name),
            "score": round(score, 1),
            "weight": weight,
            "tier": _tier,
            "contribution": round(contribution, 2),
        })
        weighted_sum += score * weight

    composite = _clamp(round(weighted_sum, 1))

    # SWOT-based adjustment
    swot = ai_analysis.get("swot", {})
    weakness_count = len(swot.get("weaknesses", []))
    strength_count = len(swot.get("strengths", []))
    threat_count = len(swot.get("threats", []))
    swot_adjust = (strength_count - weakness_count - threat_count * 0.5) * 0.5
    composite = _clamp(round(composite + swot_adjust, 1))

    # Technical bonus
    tech = ai_analysis.get("technical", {})
    if tech.get("open_source") is True:
        composite = _clamp(round(composite + 1.0, 1))
    if tech.get("audit_status") == "audited":
        composite = _clamp(round(composite + 1.5, 1))

    tier = _classify_tier(composite)

    # Tier-level rollups
    tier_buckets = {"first_mover": [], "team": [], "value": [], "community": []}
    for fn, weight, name, tier_name in FACTORS:
        tier_buckets[tier_name].append(scores[name])

    tier_scores = {
        k: round(sum(v) / len(v), 1) if v else NEUTRAL
        for k, v in tier_buckets.items()
    }

    # First mover determination
    first_mover = (
        scores.get("F1_category_uniqueness", NEUTRAL) >= 80 or
        scores.get("F2_approach_novelty", NEUTRAL) >= 80
    )

    # Top 3 helped (highest positive contribution) and top 3 hurt (most negative)
    sorted_by_contribution = sorted(factor_details, key=lambda x: x["contribution"], reverse=True)
    top_helped = [
        {"factor": f["factor"], "score": f["score"], "label": f["label"]}
        for f in sorted_by_contribution[:3]
        if f["contribution"] > 0
    ]
    top_hurt = [
        {"factor": f["factor"], "score": f["score"], "label": f["label"]}
        for f in sorted_by_contribution[-3:]
        if f["contribution"] < 0
    ]
    # Reverse hurt so worst is first
    top_hurt.reverse()

    # Plain English one-liner
    one_liner = _build_one_liner(tier_scores, agent_data)

    # Doxx tier detail
    doxx_tier_val = int(agent_data.get("doxx_tier") or ai_analysis.get("team", {}).get("doxx_tier") or 3)
    doxx_reasons = {
        1: "Fully verified identity — public team with verifiable credentials",
        2: "Social presence only — pseudonymous with active social accounts",
        3: "Anonymous — no verifiable identity found",
    }
    doxx_detail = {
        "tier": doxx_tier_val,
        "label": {1: "Full Doxx", 2: "Social Presence", 3: "Anonymous"}.get(doxx_tier_val, "Anonymous"),
        "score": scores.get("F6_doxx_tier", NEUTRAL),
        "reason": doxx_reasons.get(doxx_tier_val, "Unknown"),
    }

    # Build narrative
    narrative = _build_score_narrative(agent_data, ai_analysis, scores, composite, tier_scores)

    # Embed metadata into scores dict so it's available in scores_json on the frontend
    scores["_tier_scores"] = tier_scores
    scores["_one_liner"] = one_liner
    scores["_top_helped"] = top_helped
    scores["_top_hurt"] = top_hurt
    scores["_doxx_tier_detail"] = doxx_detail

    return {
        "composite_score": composite,
        "tier_classification": tier,
        "scores": scores,
        "tier_scores": tier_scores,
        "first_mover": first_mover,
        "score_narrative": narrative,
        "one_liner": one_liner,
        "top_helped": top_helped,
        "top_hurt": top_hurt,
        "doxx_tier_detail": doxx_detail,
    }


def _build_score_narrative(agent_data: dict, ai_analysis: dict, scores: dict,
                           composite: float, tier_scores: dict) -> str:
    """Build a human-readable narrative explaining the score."""
    name = agent_data.get("name", "This agent")
    parts = []

    team = ai_analysis.get("team", {})
    doxx_tier = int(agent_data.get("doxx_tier") or team.get("doxx_tier") or 3)
    if doxx_tier == 3:
        parts.append("Team is anonymous (no verifiable identity), limiting trust score")
    elif doxx_tier == 1:
        parts.append("Team is fully doxxed, boosting credibility")
    if team.get("red_flags"):
        parts.append(f"Team red flags: {', '.join(team['red_flags'][:2])}")

    product = ai_analysis.get("product", {})
    if product.get("status"):
        parts.append(f"Product status: {product['status']}")

    mcap = float(agent_data.get("market_cap") or 0)
    vol = float(agent_data.get("volume_24h") or 0)
    holders = int(agent_data.get("holder_count") or 0)
    if vol < 10000 and vol > 0:
        parts.append(f"Very low 24h volume (${vol:,.0f})")
    if holders < 100 and holders > 0:
        parts.append(f"Only {holders} holders")
    elif holders > 5000:
        parts.append(f"{holders:,} holders indicates solid distribution")

    strongest_tier = max(tier_scores, key=tier_scores.get) if tier_scores else None
    weakest_tier = min(tier_scores, key=tier_scores.get) if tier_scores else None
    if strongest_tier and weakest_tier and strongest_tier != weakest_tier:
        parts.append(
            f"Strongest: {TIER_LABELS.get(strongest_tier, strongest_tier)} "
            f"({tier_scores[strongest_tier]:.0f}). "
            f"Weakest: {TIER_LABELS.get(weakest_tier, weakest_tier)} "
            f"({tier_scores[weakest_tier]:.0f})"
        )

    return ". ".join(parts) + "." if parts else ""
