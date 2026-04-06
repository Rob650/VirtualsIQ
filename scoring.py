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
    return {1: 100.0, 2: 60.0, 3: 20.0}.get(tier, NEUTRAL)


def _f7_track_record(agent: dict, ai: dict) -> float:
    """Prior track record (5%)"""
    v = ai.get("team", {}).get("track_record_score")
    return _clamp(float(v)) if v is not None else NEUTRAL


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
    return _clamp(float(v)) if v is not None else NEUTRAL


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
        }
    """
    scores = {}
    weighted_sum = 0.0

    for fn, weight, name, _tier in FACTORS:
        try:
            raw = fn(agent_data, ai_analysis)
        except Exception:
            raw = NEUTRAL
        score = _clamp(raw)
        scores[name] = round(score, 1)
        weighted_sum += score * weight

    composite = _clamp(round(weighted_sum, 1))
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

    return {
        "composite_score": composite,
        "tier_classification": tier,
        "scores": scores,
        "tier_scores": tier_scores,
        "first_mover": first_mover,
    }
