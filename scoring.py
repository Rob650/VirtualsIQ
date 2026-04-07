"""
VirtualsIQ — 24-Factor Weighted Scoring Engine

Tier 1 — First Mover Advantage (30%)
Tier 2 — Team & Execution (28%)
Tier 3 — Value Pool / TAM (24%)
Tier 4 — Community & Social (18%)

Missing data defaults to 50 (neutral), never penalized.
"""

import math
from datetime import datetime


NEUTRAL = 50.0  # Default score when data is absent


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
# Tier 1 — First Mover Advantage (30%)
# ---------------------------------------------------------------------------

def _f1_category_uniqueness(agent: dict, ai: dict) -> float:
    """Is this the only agent of its type? (8%)"""
    v = ai.get("first_mover", {}).get("category_unique")
    if v is True:
        return 100.0
    if v is False:
        return 20.0
    return NEUTRAL


def _f2_approach_novelty(agent: dict, ai: dict) -> float:
    """Unique method in same category? (7%)"""
    v = ai.get("first_mover", {}).get("approach_novel")
    if v is True:
        return 90.0
    if v is False:
        return 30.0
    return NEUTRAL


def _f3_cross_chain_originality(agent: dict, ai: dict) -> float:
    """Exists on other platforms? (6%)"""
    v = ai.get("first_mover", {}).get("cross_chain_original")
    if v is True:
        return 85.0
    if v is False:
        return 35.0
    return NEUTRAL


def _f4_timing_advantage(agent: dict, ai: dict) -> float:
    """Days ahead of nearest competitor (5%)"""
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
    """Can it be trivially cloned? (4%)"""
    v = ai.get("first_mover", {}).get("defensibility_score")
    if v is not None:
        return _clamp(float(v))
    # Use social/github as proxy
    github = _safe(agent.get("github_stars"), 0)
    if github > 500:
        return 80.0
    if github > 100:
        return 60.0
    return NEUTRAL


# ---------------------------------------------------------------------------
# Tier 2 — Team & Execution (28%)
# ---------------------------------------------------------------------------

def _f6_doxx_tier(agent: dict, ai: dict) -> float:
    """Team visibility: Tier 1=Full Doxx, Tier 2=Social, Tier 3=Anon (5%)"""
    tier = int(agent.get("doxx_tier") or ai.get("team", {}).get("doxx_tier") or 3)
    base = {1: 100.0, 2: 60.0, 3: 20.0}.get(tier, NEUTRAL)
    # If overview found team red flags, penalize further
    team = ai.get("team", {})
    if team.get("red_flags") and len(team["red_flags"]) > 0:
        base = max(0, base - len(team["red_flags"]) * 10)
    return _clamp(base)


def _f7_track_record(agent: dict, ai: dict) -> float:
    """Prior track record (5%)"""
    v = ai.get("team", {}).get("track_record_score")
    if v is not None:
        score = _clamp(float(v))
        # Cross-reference: if team is anonymous (tier 3), cap track record
        tier = int(agent.get("doxx_tier") or ai.get("team", {}).get("doxx_tier") or 3)
        if tier == 3:
            score = min(score, 40.0)
        return score
    return NEUTRAL


def _f8_code_activity(agent: dict, ai: dict) -> float:
    """GitHub stats (4%)"""
    stars = _safe(agent.get("github_stars"), 0)
    commits = _safe(agent.get("github_commits_30d"), 0)
    contributors = _safe(agent.get("github_contributors"), 0)

    if stars == 0 and commits == 0:
        return NEUTRAL

    score = 0.0
    score += min(stars / 10, 30)        # up to 30pts for stars
    score += min(commits / 2, 40)       # up to 40pts for commits
    score += min(contributors * 5, 30)  # up to 30pts for contributors
    return _clamp(score)


def _f9_shipping_cadence(agent: dict, ai: dict) -> float:
    """Shipping velocity (4%)"""
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
    """Product maturity: Live=100, Beta=70, Testnet=40, Pre-product=20, Vaporware=0 (4%)"""
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
    # Infer from agent status
    if agent.get("status") == "Sentient":
        return 80.0
    return NEUTRAL


def _f11_partnerships(agent: dict, ai: dict) -> float:
    """Partnership evidence (3%)"""
    v = ai.get("product", {}).get("partnership_score")
    if v is not None:
        score = _clamp(float(v))
        # Boost if product has no red flags and technical moat is described
        product = ai.get("product", {})
        if product.get("technical_moat") and not product.get("red_flags"):
            score = min(100, score + 5)
        # Penalize if product has red flags
        if product.get("red_flags") and len(product["red_flags"]) > 0:
            score = max(0, score - len(product["red_flags"]) * 8)
        return score
    return NEUTRAL


def _f12_wallet_behavior(agent: dict, ai: dict) -> float:
    """Team wallet behavior (3%)"""
    v = ai.get("team", {}).get("wallet_behavior_score")
    return _clamp(float(v)) if v is not None else NEUTRAL


# ---------------------------------------------------------------------------
# Tier 3 — Value Pool / TAM (24%)
# ---------------------------------------------------------------------------

def _f13_tam(agent: dict, ai: dict) -> float:
    """Total Addressable Market (5%)"""
    v = ai.get("market", {}).get("tam_score")
    return _clamp(float(v)) if v is not None else NEUTRAL


def _f14_real_world_comparables(agent: dict, ai: dict) -> float:
    """Real-world comparable valuations (5%)"""
    v = ai.get("market", {}).get("comparables_score")
    return _clamp(float(v)) if v is not None else NEUTRAL


def _f15_revenue_model(agent: dict, ai: dict) -> float:
    """Revenue model clarity (4%)"""
    v = ai.get("market", {}).get("revenue_model_score")
    return _clamp(float(v)) if v is not None else NEUTRAL


def _f16_current_revenue(agent: dict, ai: dict) -> float:
    """Current revenue evidence (4%)"""
    v = ai.get("market", {}).get("current_revenue_score")
    if v is not None:
        return _clamp(float(v))
    # Use market_cap as primary proxy (larger MC = more established, more traction)
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
    # Fall back to volume_24h when market_cap unavailable
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
    """MCap-to-TAM ratio (3%) — lower is better (more upside remaining)"""
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
    # Fallback: use raw market_cap as a proxy for how much TAM is already captured.
    # Smaller MC = more room to grow = higher score here.
    mcap = _safe(agent.get("market_cap"), 0)
    if mcap > 0:
        if mcap < 100_000:
            return 90.0   # tiny MC → massive upside potential
        if mcap < 1_000_000:
            return 78.0
        if mcap < 10_000_000:
            return 62.0
        if mcap < 100_000_000:
            return 42.0
        return 22.0       # >$100M → already captured meaningful share of TAM
    return NEUTRAL


def _f18_saturation(agent: dict, ai: dict) -> float:
    """Crypto vertical saturation — less saturated = higher score (3%)"""
    v = ai.get("market", {}).get("saturation_score")
    return _clamp(float(v)) if v is not None else NEUTRAL


# ---------------------------------------------------------------------------
# Tier 4 — Community & Social (18%)
# ---------------------------------------------------------------------------

def _f19_holder_distribution(agent: dict, ai: dict) -> float:
    """Holder distribution quality (4%)"""
    holders = _safe(agent.get("holder_count"), 0)
    top10 = _safe(agent.get("top_10_concentration"), 100)

    if holders == 0:
        return NEUTRAL

    # Log scale so more holders always improves the score with no early cap:
    #   100 → 20pts, 1K → 30pts, 10K → 40pts, 100K → 50pts, 1M → 60pts
    holder_score = min(math.log10(max(holders, 1)) * 10, 60)
    concentration_score = max(0, 40 - top10 / 2.5)  # up to 40pts (lower conc = better)
    return _clamp(holder_score + concentration_score)


def _f20_twitter_engagement(agent: dict, ai: dict) -> float:
    """Twitter engagement quality (3%)"""
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
    """Follower growth velocity (3%)"""
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
    """Community depth (3%)"""
    v = ai.get("community", {}).get("depth_score")
    return _clamp(float(v)) if v is not None else NEUTRAL


def _f23_organic_signals(agent: dict, ai: dict) -> float:
    """Organic vs paid signals (3%)"""
    v = ai.get("community", {}).get("organic_score")
    return _clamp(float(v)) if v is not None else NEUTRAL


def _f24_smart_money(agent: dict, ai: dict) -> float:
    """Smart money presence (2%)"""
    v = ai.get("community", {}).get("smart_money_score")
    if v is not None:
        return _clamp(float(v))
    # Use buy/sell ratio + volume as proxy
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


def _build_score_narrative(agent_data: dict, ai_analysis: dict, scores: dict,
                           composite: float, tier_scores: dict) -> str:
    """Build a human-readable narrative explaining WHY the score is what it is,
    citing specific findings from the AI overview."""
    name = agent_data.get("name", "This agent")
    parts = []

    # Team insight
    team = ai_analysis.get("team", {})
    doxx_tier = int(agent_data.get("doxx_tier") or team.get("doxx_tier") or 3)
    if doxx_tier == 3:
        parts.append(f"Team is anonymous (no verifiable identity), limiting trust score")
    elif doxx_tier == 1:
        parts.append(f"Team is fully doxxed, boosting credibility")
    if team.get("red_flags"):
        parts.append(f"Team red flags: {', '.join(team['red_flags'][:2])}")

    # Product insight
    product = ai_analysis.get("product", {})
    if product.get("status"):
        parts.append(f"Product status: {product['status']}")
    if product.get("red_flags"):
        parts.append(f"Product concerns: {', '.join(product['red_flags'][:2])}")

    # Market data insight
    mcap = float(agent_data.get("market_cap") or 0)
    vol = float(agent_data.get("volume_24h") or 0)
    holders = int(agent_data.get("holder_count") or 0)
    if vol < 10000 and vol > 0:
        parts.append(f"Very low 24h volume (${vol:,.0f}) indicates limited trading activity")
    if holders < 100 and holders > 0:
        parts.append(f"Only {holders} holders — very early or low adoption")
    elif holders > 5000:
        parts.append(f"{holders:,} holders indicates solid distribution")

    # SWOT insight
    swot = ai_analysis.get("swot", {})
    if swot.get("strengths"):
        parts.append(f"Key strength: {swot['strengths'][0]}")
    if swot.get("weaknesses"):
        parts.append(f"Key weakness: {swot['weaknesses'][0]}")

    # First mover insight
    fm = ai_analysis.get("first_mover", {})
    if fm.get("category_unique") is True:
        parts.append("First mover advantage: unique category positioning")
    elif fm.get("category_unique") is False:
        parts.append("Operates in a crowded category with established competitors")

    # Tier summary
    strongest_tier = max(tier_scores, key=tier_scores.get) if tier_scores else None
    weakest_tier = min(tier_scores, key=tier_scores.get) if tier_scores else None
    tier_labels = {"first_mover": "First Mover", "team": "Team & Execution",
                   "value": "Value Pool/TAM", "community": "Community"}
    if strongest_tier and weakest_tier and strongest_tier != weakest_tier:
        parts.append(
            f"Strongest dimension: {tier_labels.get(strongest_tier, strongest_tier)} "
            f"({tier_scores[strongest_tier]:.0f}). "
            f"Weakest: {tier_labels.get(weakest_tier, weakest_tier)} "
            f"({tier_scores[weakest_tier]:.0f})"
        )

    return ". ".join(parts) + "." if parts else ""


def _factor_explanation(name: str, score: float, agent_data: dict, ai_analysis: dict) -> str:
    """Generate a short explanation for why a factor got its score."""
    fm = ai_analysis.get("first_mover", {})
    team = ai_analysis.get("team", {})
    product = ai_analysis.get("product", {})
    market = ai_analysis.get("market", {})
    community = ai_analysis.get("community", {})
    v = round(score)

    explanations = {
        "F1_category_uniqueness": (
            "Novel category — no direct competitors identified" if fm.get("category_unique") is True
            else "Crowded category with established rivals" if fm.get("category_unique") is False
            else f"Category uniqueness: {v}/100"
        ),
        "F2_approach_novelty": (
            "Architecturally distinct approach" if fm.get("approach_novel") is True
            else "Methodology mirrors existing agents" if fm.get("approach_novel") is False
            else f"Approach novelty: {v}/100"
        ),
        "F3_cross_chain_originality": (
            "No cross-chain equivalent exists" if fm.get("cross_chain_original") is True
            else "Cross-chain equivalents already live" if fm.get("cross_chain_original") is False
            else f"Cross-chain originality: {v}/100"
        ),
        "F4_timing_advantage": (
            f"~{fm['days_ahead_of_competitor']}d ahead of nearest competitor"
            if fm.get("days_ahead_of_competitor")
            else f"Timing advantage: {v}/100"
        ),
        "F5_defensibility": (
            fm.get("analysis", "")[:120] if fm.get("analysis")
            else f"Defensibility score: {v}/100"
        ),
        "F6_doxx_tier": (
            team.get("doxx_description", "")
            or {1: "Fully doxxed team", 2: "Pseudonymous with social presence", 3: "Anonymous team"}.get(
                int(agent_data.get("doxx_tier") or team.get("doxx_tier") or 3), "Unknown"
            )
        ),
        "F7_track_record": (
            (team.get("team_summary", "") or "")[:130] or f"Track record: {v}/100"
        ),
        "F8_code_activity": (
            f"{agent_data.get('github_commits_30d', 0)} commits/30d, {agent_data.get('github_stars', 0)} stars"
            if agent_data.get("github_commits_30d") or agent_data.get("github_stars")
            else "No GitHub activity detected"
        ),
        "F9_shipping_cadence": (
            f"Product: {product.get('status', 'unknown')}" if product.get("status")
            else f"Shipping cadence: {v}/100"
        ),
        "F10_product_status": (
            (product.get("description", "") or "")[:120] or f"Product status: {v}/100"
        ),
        "F11_partnerships": (
            (product.get("technical_moat", "") or "")[:120] or f"Partnerships: {v}/100"
        ),
        "F12_wallet_behavior": (
            f"Wallet behavior: {v}/100" + (" — unusual activity" if v < 40 else " — healthy" if v >= 70 else "")
        ),
        "F13_tam": (
            (market.get("tam_description", "") or "")[:130] or f"TAM score: {v}/100"
        ),
        "F14_real_world_comparables": (
            f"Comparable to {market['real_world_comparable']}" if market.get("real_world_comparable")
            else f"Comparables: {v}/100"
        ),
        "F15_revenue_model": f"Revenue model clarity: {v}/100",
        "F16_current_revenue": (
            f"Revenue evidence: {v}/100" + (" — no confirmed revenue" if v < 35 else " — active revenue" if v >= 65 else "")
        ),
        "F17_mcap_to_tam": (
            f"MCap/TAM ratio: {market['mcap_tam_ratio']:.4f}" if market.get("mcap_tam_ratio") is not None
            else f"MCap/TAM: {v}/100"
        ),
        "F18_saturation": (
            (market.get("saturation_description", "") or "")[:120] or f"Saturation: {v}/100"
        ),
        "F19_holder_distribution": (
            f"{agent_data.get('holder_count', 0):,} holders — distribution score {v}/100"
            if agent_data.get("holder_count")
            else f"Holder distribution: {v}/100"
        ),
        "F20_twitter_engagement": (
            f"{agent_data.get('twitter_followers', 0):,} followers — engagement {v}/100"
            if agent_data.get("twitter_followers")
            else f"Twitter engagement: {v}/100"
        ),
        "F21_follower_growth": (
            f"Follower growth: {v}/100" + (" — accelerating" if v >= 65 else " — stagnant" if v < 40 else " — steady")
        ),
        "F22_community_depth": (
            (community.get("community_analysis", "") or "")[:120] or f"Community depth: {v}/100"
        ),
        "F23_organic_signals": (
            f"Organic signals: {v}/100" + (" — genuine engagement" if v >= 65 else " — bot signals" if v < 40 else "")
        ),
        "F24_smart_money": (
            f"Smart money: {v}/100" + (" — notable wallets accumulating" if v >= 65 else "")
        ),
    }
    return explanations.get(name, f"Score: {v}/100")


def calculate_composite_score(agent_data: dict, ai_analysis: dict) -> dict:
    """
    Run all 24 factors and return composite score with breakdown.
    Scores are INFORMED by the deep AI analysis findings.

    Returns:
        {
            "composite_score": float,
            "tier_classification": str,
            "scores": {factor_name: {"value": float, "explanation": str}, ...},
            "tier_scores": {tier_name: float, ...},
            "first_mover": bool,
            "score_narrative": str,
        }
    """
    scores = {}
    scores_flat = {}  # flat numeric scores for backward compat
    weighted_sum = 0.0

    for fn, weight, name, _tier in FACTORS:
        try:
            raw = fn(agent_data, ai_analysis)
        except Exception:
            raw = NEUTRAL
        score = _clamp(raw)
        explanation = _factor_explanation(name, score, agent_data, ai_analysis)
        scores[name] = round(score, 1)
        scores_flat[name] = round(score, 1)
        weighted_sum += score * weight

    composite = _clamp(round(weighted_sum, 1))

    # SWOT-based adjustment: if the analysis found critical weaknesses/threats,
    # apply a small penalty; if strong strengths, small boost
    swot = ai_analysis.get("swot", {})
    weakness_count = len(swot.get("weaknesses", []))
    strength_count = len(swot.get("strengths", []))
    threat_count = len(swot.get("threats", []))
    swot_adjust = (strength_count - weakness_count - threat_count * 0.5) * 0.5
    composite = _clamp(round(composite + swot_adjust, 1))

    # Technical bonus: if open source and audited, small boost
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

    # Build narrative citing actual findings
    narrative = _build_score_narrative(
        agent_data, ai_analysis, scores, composite, tier_scores
    )

    return {
        "composite_score": composite,
        "tier_classification": tier,
        "scores": scores,
        "tier_scores": tier_scores,
        "first_mover": first_mover,
        "score_narrative": narrative,
    }
