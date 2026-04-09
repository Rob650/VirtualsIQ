"""
VirtualsIQ — 24-Factor Weighted Scoring Engine

Tier 1 — Foundation / First Mover (30%)
Tier 2 — Traction / Team & Execution (28%)
Tier 3 — On-chain / Value Pool (24%)
Tier 4 — Narrative / Community (18%)

Missing data: factors with NO data return None and are SKIPPED.
Remaining factors are re-weighted proportionally so the composite
reflects real signal only — never averaged down by absent data.
"""

import math
from datetime import datetime


NEUTRAL = 50.0  # Score when data IS present but is genuinely average

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
# TAM values by vertical (in USD)
# ---------------------------------------------------------------------------

TAM_BY_CATEGORY = {
    "defi": 100_000_000_000,
    "infrastructure": 80_000_000_000,
    "gaming": 50_000_000_000,
    "social": 30_000_000_000,
    "community": 30_000_000_000,
    "entertainment": 20_000_000_000,
    "creative": 20_000_000_000,
    "productivity": 15_000_000_000,
    "tools": 15_000_000_000,
    "other": 25_000_000_000,
    "unknown": 25_000_000_000,
}


def _get_category_tam(agent: dict, ai: dict) -> float:
    """Return TAM in dollars based on agent category/vertical."""
    cat = (
        agent.get("category") or
        agent.get("agent_type") or
        ai.get("category") or
        "other"
    )
    cat = str(cat).lower().strip()
    for key, tam in TAM_BY_CATEGORY.items():
        if key in cat:
            return float(tam)
    return float(TAM_BY_CATEGORY["other"])


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

def _f1_category_uniqueness(agent: dict, ai: dict) -> float | None:
    v = ai.get("first_mover", {}).get("category_unique")
    if v is True:
        return 100.0
    if v is False:
        return 20.0
    return None  # no AI data — skip this factor


def _f2_approach_novelty(agent: dict, ai: dict) -> float | None:
    v = ai.get("first_mover", {}).get("approach_novel")
    if v is True:
        return 90.0
    if v is False:
        return 30.0
    return None


def _f3_cross_chain_originality(agent: dict, ai: dict) -> float | None:
    v = ai.get("first_mover", {}).get("cross_chain_original")
    if v is True:
        return 85.0
    if v is False:
        return 35.0
    return None


def _f4_timing_advantage(agent: dict, ai: dict) -> float | None:
    days = ai.get("first_mover", {}).get("days_ahead_of_competitor")
    if days is None:
        return None
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


def _f5_defensibility(agent: dict, ai: dict) -> float | None:
    v = ai.get("first_mover", {}).get("defensibility_score")
    if v is not None:
        return _clamp(float(v))
    github = _safe(agent.get("github_stars"), 0)
    if github > 0:
        if github > 500:
            return 80.0
        if github > 100:
            return 60.0
        return 40.0
    return None  # no AI and no github data


# ---------------------------------------------------------------------------
# Tier 2 — Traction / Team & Execution (28%)
# ---------------------------------------------------------------------------

def score_doxx_tier2(agent: dict) -> dict:
    """
    Dynamic Tier 2 (Social) doxx scoring.
    Returns a detailed sub-score object 0-100 with component breakdown.
    """
    twitter_age = _safe(agent.get("twitter_account_age"), 0)
    followers = _safe(agent.get("twitter_followers"), 0)
    engagement = _safe(agent.get("twitter_engagement_rate"), 0)
    creation_date = agent.get("creation_date")

    components = {}

    # 1. Account age sub-score (0-25)
    if twitter_age > 730:
        age_score = 25.0
    elif twitter_age > 365:
        age_score = 20.0
    elif twitter_age > 180:
        age_score = 14.0
    elif twitter_age > 30:
        age_score = 8.0
    else:
        age_score = 2.0
    components["account_age"] = round(age_score, 1)

    # 2. Follower quality sub-score (0-25)
    if followers > 50000:
        fq_score = 25.0
    elif followers > 10000:
        fq_score = 20.0
    elif followers > 5000:
        fq_score = 16.0
    elif followers > 1000:
        fq_score = 12.0
    elif followers > 100:
        fq_score = 6.0
    else:
        fq_score = 1.0
    if followers > 1000 and engagement > 15.0:
        fq_score = max(0, fq_score - 8.0)
    components["follower_quality"] = round(fq_score, 1)

    # 3. Engagement authenticity sub-score (0-25)
    if 1.0 <= engagement <= 5.0:
        eng_score = 25.0
    elif 0.5 <= engagement <= 8.0:
        eng_score = 18.0
    elif 0.1 <= engagement < 0.5:
        eng_score = 10.0
    elif engagement > 15.0:
        eng_score = 3.0
    elif engagement > 8.0:
        eng_score = 8.0
    else:
        eng_score = 2.0
    components["engagement_authenticity"] = round(eng_score, 1)

    # 4. Pre-project existence sub-score (0-25)
    pre_score = 12.5
    if creation_date and twitter_age > 0:
        project_days = _days_since(creation_date)
        if project_days:
            if twitter_age > project_days + 180:
                pre_score = 25.0
            elif twitter_age > project_days + 90:
                pre_score = 22.0
            elif twitter_age > project_days:
                pre_score = 16.0
            elif twitter_age > project_days - 30:
                pre_score = 10.0
            else:
                pre_score = 3.0
    components["pre_project_existence"] = round(pre_score, 1)

    total = _clamp(age_score + fq_score + eng_score + pre_score)

    return {
        "total_score": round(total, 1),
        "components": components,
        "twitter_age_days": int(twitter_age),
        "followers": int(followers),
        "engagement_rate": round(engagement, 2),
    }


def _f6_doxx_tier(agent: dict, ai: dict) -> float:
    """Dynamic doxx scoring — always returns a value (defaults to anonymous=20)."""
    tier = int(agent.get("doxx_tier") or ai.get("team", {}).get("doxx_tier") or 3)

    if tier == 1:
        base = 100.0
    elif tier == 2:
        tier2_result = score_doxx_tier2(agent)
        base = tier2_result["total_score"]
    else:
        base = 20.0

    team = ai.get("team", {})
    if team.get("red_flags") and len(team["red_flags"]) > 0:
        base = max(0, base - len(team["red_flags"]) * 10)
    return _clamp(base)


def _f7_track_record(agent: dict, ai: dict) -> float | None:
    v = ai.get("team", {}).get("track_record_score")
    if v is not None:
        score = _clamp(float(v))
        tier = int(agent.get("doxx_tier") or ai.get("team", {}).get("doxx_tier") or 3)
        if tier == 3:
            score = min(score, 40.0)
        return score
    return None


def _f8_code_activity(agent: dict, ai: dict) -> float | None:
    stars = _safe(agent.get("github_stars"), 0)
    commits = _safe(agent.get("github_commits_30d"), 0)
    contributors = _safe(agent.get("github_contributors"), 0)
    if stars == 0 and commits == 0:
        return None  # no github data at all
    score = 0.0
    score += min(stars / 10, 30)
    score += min(commits / 2, 40)
    score += min(contributors * 5, 30)
    return _clamp(score)


def _f9_shipping_cadence(agent: dict, ai: dict) -> float | None:
    last_commit = agent.get("github_last_commit")
    days = _days_since(last_commit)
    if days is None:
        return None
    if days <= 7:
        return 100.0
    if days <= 30:
        return 75.0
    if days <= 90:
        return 50.0
    if days <= 180:
        return 25.0
    return 10.0


def _f10_product_status(agent: dict, ai: dict) -> float | None:
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
    return None  # no AI data and not Sentient


def _f11_partnerships(agent: dict, ai: dict) -> float | None:
    v = ai.get("product", {}).get("partnership_score")
    if v is not None:
        score = _clamp(float(v))
        product = ai.get("product", {})
        if product.get("technical_moat") and not product.get("red_flags"):
            score = min(100, score + 5)
        if product.get("red_flags") and len(product["red_flags"]) > 0:
            score = max(0, score - len(product["red_flags"]) * 8)
        return score
    return None


def _f12_wallet_behavior(agent: dict, ai: dict) -> float | None:
    v = ai.get("team", {}).get("wallet_behavior_score")
    return _clamp(float(v)) if v is not None else None


# ---------------------------------------------------------------------------
# Tier 3 — On-chain / Value Pool (24%)
# ---------------------------------------------------------------------------

def _f13_tam(agent: dict, ai: dict) -> float:
    """Always returns a value — uses AI score or vertical-specific TAM."""
    v = ai.get("market", {}).get("tam_score")
    if v is not None:
        return _clamp(float(v))
    tam = _get_category_tam(agent, ai)
    if tam >= 80_000_000_000:
        return 95.0
    if tam >= 50_000_000_000:
        return 85.0
    if tam >= 30_000_000_000:
        return 75.0
    if tam >= 20_000_000_000:
        return 65.0
    if tam >= 15_000_000_000:
        return 60.0
    return 55.0


def _f14_real_world_comparables(agent: dict, ai: dict) -> float | None:
    v = ai.get("market", {}).get("comparables_score")
    return _clamp(float(v)) if v is not None else None


def _f15_revenue_model(agent: dict, ai: dict) -> float | None:
    v = ai.get("market", {}).get("revenue_model_score")
    return _clamp(float(v)) if v is not None else None


def _f16_current_revenue(agent: dict, ai: dict) -> float:
    """Always returns a value based on on-chain market data."""
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


def _f17_mcap_to_tam(agent: dict, ai: dict) -> float | None:
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
        tam = _get_category_tam(agent, ai)
        ratio = mcap / tam
        if ratio < 0.0001:
            return 95.0
        if ratio < 0.001:
            return 80.0
        if ratio < 0.01:
            return 65.0
        if ratio < 0.05:
            return 45.0
        if ratio < 0.1:
            return 30.0
        return 15.0
    return None  # no market cap and no AI data


def _f18_saturation(agent: dict, ai: dict) -> float | None:
    v = ai.get("market", {}).get("saturation_score")
    return _clamp(float(v)) if v is not None else None


# ---------------------------------------------------------------------------
# Tier 4 — Narrative / Community (18%)
# ---------------------------------------------------------------------------

def _f19_holder_distribution(agent: dict, ai: dict) -> float | None:
    holders = _safe(agent.get("holder_count"), 0)
    if holders == 0:
        return None  # no holder data
    top10 = _safe(agent.get("top_10_concentration"), 100)
    holder_score = min(math.log10(max(holders, 1)) * 10, 60)
    concentration_score = max(0, 40 - top10 / 2.5)
    return _clamp(holder_score + concentration_score)


def _f20_twitter_engagement(agent: dict, ai: dict) -> float | None:
    rate = _safe(agent.get("twitter_engagement_rate"), -1)
    if rate < 0:
        return None  # no engagement data
    if rate >= 5.0:
        return 95.0
    if rate >= 2.0:
        return 75.0
    if rate >= 1.0:
        return 55.0
    if rate >= 0.5:
        return 35.0
    return 20.0


def _f21_follower_growth(agent: dict, ai: dict) -> float | None:
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
    return None  # no follower data and no AI data


def _f22_community_depth(agent: dict, ai: dict) -> float | None:
    v = ai.get("community", {}).get("depth_score")
    return _clamp(float(v)) if v is not None else None


def _f23_organic_signals(agent: dict, ai: dict) -> float | None:
    v = ai.get("community", {}).get("organic_score")
    return _clamp(float(v)) if v is not None else None


def _f24_smart_money(agent: dict, ai: dict) -> float | None:
    v = ai.get("community", {}).get("smart_money_score")
    if v is not None:
        return _clamp(float(v))
    bsr = _safe(agent.get("buy_sell_ratio"), 1.0)
    vol = _safe(agent.get("volume_24h"), 0)
    if vol > 0:
        if bsr > 1.5 and vol > 100_000:
            return 80.0
        if bsr > 1.2:
            return 60.0
        return 40.0
    return None  # no volume data at all


# ---------------------------------------------------------------------------
# Score filters and distribution helpers
# ---------------------------------------------------------------------------

def _is_dead_agent(agent: dict) -> bool:
    """Return True if agent meets ANY dead/scam criteria (binary flag)."""
    holders = _safe(agent.get("holder_count"), 0)
    mcap = _safe(agent.get("market_cap"), 0)
    vol = _safe(agent.get("volume_24h"), 0)

    if mcap > 0 and mcap < 5_000:
        return True

    if holders > 0 and holders < 50 and mcap > 0 and mcap < 10_000:
        return True

    creation_date = agent.get("creation_date") or agent.get("first_seen")
    days = _days_since(creation_date)
    if days is not None and days > 90 and vol > 0 and vol < 1_000:
        return True

    holder_change = _safe(agent.get("holder_count_change_24h"), None)
    if holder_change is not None and holder_change < 0 and vol < 500:
        return True

    return False


def _is_strong_investment(agent: dict) -> bool:
    """Return True if agent meets ALL strong investment criteria."""
    holders = _safe(agent.get("holder_count"), 0)
    mcap = _safe(agent.get("market_cap"), 0)
    vol = _safe(agent.get("volume_24h"), 0)
    holder_change = _safe(agent.get("holder_count_change_24h"), 0)

    return (
        holders >= 2_000 and
        mcap >= 5_000_000 and
        vol >= 20_000 and
        holder_change >= 0
    )


def _widen_distribution(score: float) -> float:
    """Multiply deviation from 50 by 1.5x, then clamp to 5-100."""
    widened = 50.0 + (score - 50.0) * 1.5
    return _clamp(widened, 5.0, 100.0)


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
    best = TIER_LABELS.get(sorted_tiers[0][0], sorted_tiers[0][0])
    worst = TIER_LABELS.get(sorted_tiers[-1][0], sorted_tiers[-1][0])
    return f"Balanced profile, strongest in {best}, watch {worst}"


def calculate_composite_score(agent_data: dict, ai_analysis: dict) -> dict:
    """
    Run all 24 factors and return composite score with breakdown.

    Factors with no data return None and are skipped; the remaining
    factors are re-weighted proportionally so the composite reflects
    only real signal.

    Returns:
        {
            "composite_score": float,
            "tier_classification": str,
            "scores": {factor_name: float|None, ...},
            "tier_scores": {tier_name: float, ...},
            "first_mover": bool,
            "score_narrative": str,
            "one_liner": str,
            "top_helped": [...],
            "top_hurt": [...],
            "doxx_tier_detail": {...},
            "dead_flagged": bool,
            "strong_flagged": bool,
            "factors_scored": int,
        }
    """
    scores = {}
    weighted_sum = 0.0
    total_weight = 0.0
    factor_details = []

    for fn, weight, name, _tier in FACTORS:
        try:
            raw = fn(agent_data, ai_analysis)
        except Exception:
            raw = None

        if raw is None:
            # No data — skip this factor entirely
            scores[name] = None
            factor_details.append({
                "factor": name,
                "label": FACTOR_LABELS.get(name, name),
                "score": None,
                "weight": weight,
                "tier": _tier,
                "contribution": 0.0,
                "skipped": True,
            })
            continue

        score = _clamp(raw)
        scores[name] = round(score, 1)
        weighted_sum += score * weight
        total_weight += weight
        factor_details.append({
            "factor": name,
            "label": FACTOR_LABELS.get(name, name),
            "score": round(score, 1),
            "weight": weight,
            "tier": _tier,
            "contribution": 0.0,  # filled in after normalization
            "skipped": False,
        })

    # Renormalize: composite = weighted_avg of included factors
    if total_weight > 0:
        composite = _clamp(round(weighted_sum / total_weight, 1))
    else:
        composite = NEUTRAL

    # Back-fill contributions now that we have the composite
    for fd in factor_details:
        if not fd["skipped"]:
            fd["contribution"] = round((fd["score"] - NEUTRAL) * (fd["weight"] / total_weight), 2)

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

    # ---- Post-processing pipeline ----
    # Step 1: widen distribution (1.5x deviation from 50)
    composite = _widen_distribution(composite)

    # Step 2: apply dead/scam filter OR strong investment boost
    dead_flagged = _is_dead_agent(agent_data)
    strong_flagged = _is_strong_investment(agent_data)
    if dead_flagged:
        composite = min(round(composite * 0.3, 1), 25.0)
    elif strong_flagged:
        composite = min(round(composite * 1.2, 1), 100.0)

    # Step 3: final clamp 5-100
    composite = _clamp(round(composite, 1), 5.0, 100.0)

    tier = _classify_tier(composite)

    # Tier-level rollups — average of non-skipped factors per tier
    tier_buckets: dict[str, list[float]] = {"first_mover": [], "team": [], "value": [], "community": []}
    for fn, weight, name, tier_name in FACTORS:
        s = scores.get(name)
        if s is not None:
            tier_buckets[tier_name].append(s)

    tier_scores = {
        k: round(sum(v) / len(v), 1) if v else NEUTRAL
        for k, v in tier_buckets.items()
    }

    # First mover determination (treat None as not first-mover)
    first_mover = (
        (scores.get("F1_category_uniqueness") or 0) >= 80 or
        (scores.get("F2_approach_novelty") or 0) >= 80
    )

    # Top 3 helped / hurt (exclude skipped factors)
    active_details = [fd for fd in factor_details if not fd["skipped"]]
    sorted_by_contribution = sorted(active_details, key=lambda x: x["contribution"], reverse=True)
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
    top_hurt.reverse()

    one_liner = _build_one_liner(tier_scores, agent_data)

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

    narrative = _build_score_narrative(agent_data, ai_analysis, scores, composite, tier_scores)

    if doxx_tier_val == 2:
        doxx_detail["tier2_breakdown"] = score_doxx_tier2(agent_data)

    factors_scored = sum(1 for v in scores.values() if v is not None)

    scores["_tier_scores"] = tier_scores
    scores["_one_liner"] = one_liner
    scores["_top_helped"] = top_helped
    scores["_top_hurt"] = top_hurt
    scores["_doxx_tier_detail"] = doxx_detail
    scores["_dead_flagged"] = dead_flagged
    scores["_strong_flagged"] = strong_flagged
    scores["_factors_scored"] = factors_scored

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
        "dead_flagged": dead_flagged,
        "strong_flagged": strong_flagged,
        "factors_scored": factors_scored,
    }


def _build_score_narrative(agent_data: dict, ai_analysis: dict, scores: dict,
                           composite: float, tier_scores: dict) -> str:
    """Build a human-readable narrative explaining the score."""
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
