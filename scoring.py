"""
VirtualsIQ — Data-Driven Scoring Engine v2

Built from analysis of top market-cap agents on the Virtuals Protocol.

Key findings from benchmark cohort (agents with >$5M MC):
  - ALL top agents are Sentient status
  - ALL top agents are anonymous (doxx_tier=3) — doxx is NOT predictive here
  - Holder count is the strongest differentiator: top agents avg 470K holders
  - Market cap itself is a strong momentum/validation signal
  - Volume (24h) reflects activity/liquidity
  - Social links present in ~37% of top agents (weak separator, still rewarded)
  - No top agent has website or telegram links (in current dataset)

Scoring philosophy:
  - "How similar is this agent to proven winners at $5M+ MC?"
  - Score = 0–100, heavily driven by on-chain fundamentals
  - Holder count + market cap + volume carry 55% of total weight
  - Moat/status, idea-market-fit, and execution round out the rest
  - Dead/scam filters applied post-composite
  - Distribution widened to fill the 0-100 range

Factor map (compatible with F-key naming for template FACTOR_LABELS):
  F1_holders          — Holder count (log-scale)
  F2_mcap             — Market cap (log-scale)
  F3_volume           — 24h trading volume (log-scale)
  F4_moat             — Sentient status + social presence + bio quality
  F5_idea_market_fit  — Category-based + AI analysis fit score
  F6_execution        — Social links + documentation + profile completeness
  F7_efficiency       — Volume/MCap turnover ratio
  F8_momentum         — Price change + holder growth signals
"""

import math
from datetime import datetime


NEUTRAL = 50.0

TIER_LABELS = {
    "traction": "Traction",
    "fundamentals": "Fundamentals",
    "market": "Market",
    "execution": "Execution",
}

FACTOR_LABELS = {
    "F1_holders":         "Holder Count",
    "F2_mcap":            "Market Cap",
    "F3_volume":          "24h Volume",
    "F4_moat":            "Moat / Status",
    "F5_idea_market_fit": "Idea × Market Fit",
    "F6_execution":       "Execution Quality",
    "F7_efficiency":      "Vol/MC Efficiency",
    "F8_momentum":        "Price Momentum",
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe(value, default=None):
    """Return float value or default."""
    try:
        v = float(value)
        return v if not math.isnan(v) else default
    except (TypeError, ValueError):
        return default


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _days_since(date_str):
    """Return days since date_str (ISO or similar). None if unparseable."""
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


# ---------------------------------------------------------------------------
# Factor scoring functions
# ---------------------------------------------------------------------------

def _f1_holders(agent: dict, ai: dict):
    """
    Holder count on log10 scale.
    Calibration from top agents:
      100 holders   -> ~5
      1,000 holders -> 30
      10,000        -> 51
      100,000       -> 73
      1,000,000     -> 95
    Formula: (log10(h) - 2) * (90 / 4) - 17.5, clamped 5-100
    Top agents (>$5M MC) avg 470K holders = score ~90
    Mid agents ($1-5M) avg 33K holders = score ~70
    """
    h = _safe(agent.get("holder_count"), 0)
    if h is None or h <= 0:
        return None
    # log10(100)=2 -> ~5, log10(1M)=6 -> ~95
    s = (math.log10(h) - 2.0) * 22.5
    return _clamp(s, 5.0, 100.0)


def _f2_mcap(agent: dict, ai: dict):
    """
    Market cap on log10 scale.
    Calibration:
      $10K     -> ~5
      $100K    -> 25
      $1M      -> 47
      $10M     -> 70
      $100M    -> 92
    Top agents avg $35M = score ~75
    """
    mc = _safe(agent.get("market_cap"), 0)
    if mc is None or mc <= 0:
        return None
    # log10(10K)=4 -> 5, log10(100M)=8 -> 92
    s = (math.log10(mc) - 4.0) * 21.75 + 5.0
    return _clamp(s, 5.0, 100.0)


def _f3_volume(agent: dict, ai: dict):
    """
    24h volume on log10 scale.
    Calibration:
      $0 / missing -> None (skip)
      $100         -> 15
      $1K          -> 28
      $10K         -> 42
      $100K        -> 56
      $1M          -> 70
    Top agents avg $35K = score ~48
    """
    vol = _safe(agent.get("volume_24h"), 0)
    if vol is None or vol <= 0:
        return None
    # log10($100)=2 -> 15, log10($1M)=6 -> 70
    s = (math.log10(vol) - 2.0) * 13.75 + 15.0
    return _clamp(s, 5.0, 100.0)


def _f4_moat(agent: dict, ai: dict) -> float:
    """
    Moat / protocol status.
    Sentient status is the dominant signal (all top agents are Sentient).
    Social presence adds secondary signal.
    Base (Prototype/Genesis): 20
    Sentient: +50 (so Sentient with nothing else = 70)
    Each social link (twitter/website/telegram): +7
    Biography present (>50 chars): +5
    AI defensibility score: can add up to +10 if available
    """
    score = 20.0

    status = str(agent.get("status", "")).lower()
    if "sentient" in status:
        score += 50.0
    elif "genesis" in status:
        score += 25.0

    # Social links
    for link_key in ("linked_twitter", "linked_website", "linked_telegram"):
        if agent.get(link_key):
            score += 7.0

    # Biography quality
    bio_len = len(agent.get("biography", "") or "")
    if bio_len > 500:
        score += 7.0
    elif bio_len > 200:
        score += 5.0
    elif bio_len > 50:
        score += 2.0

    # AI defensibility bonus (if available)
    ai_def = _safe(ai.get("first_mover", {}).get("defensibility_score"))
    if ai_def is not None:
        score += (ai_def - 50.0) * 0.1  # max ±5 adjustment

    return _clamp(score)


def _f5_idea_market_fit(agent: dict, ai: dict) -> float:
    """
    Idea × market fit.
    Uses AI analysis data if available, otherwise category heuristic.
    Top agents span Trading, DeFi, Infra, Gaming — all core categories.
    """
    # Try AI analysis sources
    ai_score = (
        _safe(ai.get("market", {}).get("tam_score")) or
        _safe(ai.get("market", {}).get("comparables_score"))
    )
    if ai_score is not None:
        # Blend AI score with base
        base = ai_score * 0.7 + 50.0 * 0.3
        return _clamp(base)

    # Category-based heuristic (from data: top agents are Trading, DeFi, Infra, Gaming)
    cat = str(
        agent.get("category") or
        agent.get("agent_type") or
        ai.get("market", {}).get("category") or
        "Other"
    ).lower().strip()

    # Categories seen in top agents get higher base scores
    cat_scores = {
        "trading": 68.0,
        "defi": 68.0,
        "infra": 70.0,
        "gaming": 68.0,
        "social": 60.0,
        "entertainment": 58.0,
        "nft": 55.0,
        "info": 57.0,
        "other": 58.0,
    }
    for key, val in cat_scores.items():
        if key in cat:
            return val
    return 58.0  # default


def _f6_execution(agent: dict, ai: dict) -> float:
    """
    Execution quality — social presence, documentation, profile completeness.
    Data shows only 37% of top agents have ANY social link.
    This factor differentiates within-tier: social presence = bonus.
    Base: 20
    Each social link: +20
    Bio >200 chars: +12
    Bio >50 chars: +5
    Twitter followers (if known): bonus up to +8
    """
    score = 20.0

    social_count = sum(
        1 for k in ("linked_twitter", "linked_website", "linked_telegram")
        if agent.get(k)
    )
    score += social_count * 20.0

    bio_len = len(agent.get("biography", "") or "")
    if bio_len > 500:
        score += 12.0
    elif bio_len > 200:
        score += 10.0
    elif bio_len > 50:
        score += 5.0

    # Twitter followers signal
    twf = _safe(agent.get("twitter_followers"), 0)
    if twf and twf > 50_000:
        score += 8.0
    elif twf and twf > 10_000:
        score += 5.0
    elif twf and twf > 1_000:
        score += 2.0

    return _clamp(score)


def _f7_efficiency(agent: dict, ai: dict):
    """
    Volume/MCap efficiency (turnover ratio).
    Healthy signal: 0.01%-5% daily turnover.
    Top agents range from 0.005% (Toshi, nearly frozen) to 2.2% (Keyboard Cat).
    Missing vol or MC -> skip.
    """
    mc = _safe(agent.get("market_cap"), 0)
    vol = _safe(agent.get("volume_24h"), 0)
    if not mc or not vol or mc <= 0 or vol <= 0:
        return None

    ratio = vol / mc  # 0.0001 to 0.10 for typical agents

    # Score: 0.001-0.01 = healthy low = 60, 0.01-0.05 = very healthy = 75
    # Very low (<0.0001) = stale = 20, Very high (>0.1) = thin/volatile = 40
    if ratio > 0.10:
        return 35.0  # extremely volatile or very thin — suspicious
    elif ratio > 0.05:
        return 55.0
    elif ratio >= 0.005:
        return 70.0  # healthy range: 0.5-5% daily turnover
    elif ratio >= 0.0005:
        return 55.0
    elif ratio >= 0.00005:
        return 35.0
    else:
        return 20.0  # nearly zero volume


def _f8_momentum(agent: dict, ai: dict):
    """
    Price momentum and holder growth.
    Uses price_change_24h and holder_count_change_24h.
    Missing both -> skip.
    """
    price_ch = _safe(agent.get("price_change_24h"))
    holder_ch = _safe(agent.get("holder_count_change_24h"))

    if price_ch is None and holder_ch is None:
        return None

    score = 50.0  # neutral baseline

    if price_ch is not None:
        # +5% price change -> +10 pts, -5% -> -10 pts (capped at ±20)
        score += _clamp(price_ch * 2.0, -20.0, 20.0)

    if holder_ch is not None:
        # Positive holder growth is a strong signal
        if holder_ch > 100:
            score += 15.0
        elif holder_ch > 10:
            score += 8.0
        elif holder_ch > 0:
            score += 3.0
        elif holder_ch < -50:
            score -= 15.0
        elif holder_ch < 0:
            score -= 5.0

    return _clamp(score)


# ---------------------------------------------------------------------------
# Dead / strong filters
# ---------------------------------------------------------------------------

def _is_dead_agent(agent: dict) -> bool:
    """Return True if agent meets any dead/scam criteria."""
    mc = _safe(agent.get("market_cap"), 0) or 0
    vol = _safe(agent.get("volume_24h"), 0) or 0
    holders = _safe(agent.get("holder_count"), 0) or 0

    if mc > 0 and mc < 5_000:
        return True

    if holders > 0 and holders < 50 and mc > 0 and mc < 10_000:
        return True

    creation_date = agent.get("creation_date") or agent.get("first_seen")
    days = _days_since(creation_date)
    if days is not None and days > 90 and vol > 0 and vol < 1_000:
        return True

    holder_change = _safe(agent.get("holder_count_change_24h"))
    if holder_change is not None and holder_change < 0 and vol < 500:
        return True

    return False


def _is_strong_investment(agent: dict) -> bool:
    """Return True if agent meets all strong investment criteria."""
    holders = _safe(agent.get("holder_count"), 0) or 0
    mc = _safe(agent.get("market_cap"), 0) or 0
    vol = _safe(agent.get("volume_24h"), 0) or 0
    holder_change = _safe(agent.get("holder_count_change_24h"), 0) or 0

    return (
        holders >= 50_000 and
        mc >= 5_000_000 and
        vol >= 10_000 and
        holder_change >= 0
    )


def _widen_distribution(score: float) -> float:
    """Widen scores: multiply deviation from 50 by 1.4x, clamp to 5-100."""
    widened = 50.0 + (score - 50.0) * 1.4
    return _clamp(widened, 5.0, 100.0)


# ---------------------------------------------------------------------------
# Factor registry
# ---------------------------------------------------------------------------

FACTORS = [
    # (function, weight, name, tier)
    (_f1_holders,         0.25, "F1_holders",         "traction"),
    (_f2_mcap,            0.20, "F2_mcap",             "fundamentals"),
    (_f3_volume,          0.15, "F3_volume",           "fundamentals"),
    (_f4_moat,            0.15, "F4_moat",             "market"),
    (_f5_idea_market_fit, 0.10, "F5_idea_market_fit",  "market"),
    (_f6_execution,       0.08, "F6_execution",        "execution"),
    (_f7_efficiency,      0.05, "F7_efficiency",       "fundamentals"),
    (_f8_momentum,        0.02, "F8_momentum",         "traction"),
]


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

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
    """Build plain-English one-liner from tier scores."""
    if not tier_scores:
        return ""
    sorted_tiers = sorted(tier_scores.items(), key=lambda x: x[1], reverse=True)
    strong = [TIER_LABELS.get(t, t) for t, s in sorted_tiers if s >= 65]
    weak = [TIER_LABELS.get(t, t) for t, s in sorted_tiers if s < 45]

    parts = []
    if strong:
        parts.append("Strong on " + " and ".join(strong[:2]))
    if weak:
        parts.append("weak on " + " and ".join(weak[:2]))

    if parts:
        return ", ".join(parts)
    best = TIER_LABELS.get(sorted_tiers[0][0], sorted_tiers[0][0])
    worst = TIER_LABELS.get(sorted_tiers[-1][0], sorted_tiers[-1][0])
    return f"Balanced profile; strongest in {best}, watch {worst}"


def _build_score_narrative(agent_data: dict, ai_analysis: dict, scores: dict,
                           composite: float, tier_scores: dict) -> str:
    """Build a human-readable narrative explaining the score."""
    parts = []

    status = agent_data.get("status", "")
    if status == "Sentient":
        parts.append("Sentient status achieved — live autonomous agent")
    elif status:
        parts.append(f"Status: {status}")

    holders = int(_safe(agent_data.get("holder_count"), 0) or 0)
    mc = float(_safe(agent_data.get("market_cap"), 0) or 0)
    vol = float(_safe(agent_data.get("volume_24h"), 0) or 0)

    if holders >= 100_000:
        parts.append(f"{holders:,} holders — strong community distribution")
    elif holders >= 10_000:
        parts.append(f"{holders:,} holders — growing distribution")
    elif holders > 0:
        parts.append(f"{holders:,} holders — early stage")

    if vol < 5_000 and vol > 0:
        parts.append(f"Very low 24h volume (${vol:,.0f})")
    elif vol >= 50_000:
        parts.append(f"Active 24h volume (${vol:,.0f})")

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


# ---------------------------------------------------------------------------
# Composite scoring — main entry point
# ---------------------------------------------------------------------------

def calculate_composite_score(agent_data: dict, ai_analysis: dict) -> dict:
    """
    Score an agent 0-100 based on how similar it is to proven top-MC winners.

    Factors with no data are skipped and remaining weights are renormalized.

    Returns:
        {
            "composite_score": float,
            "tier_classification": str,
            "scores": {factor_name: float|None, ..., _metadata...},
            "tier_scores": {tier_name: float},
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
    ai = ai_analysis or {}
    scores: dict = {}
    weighted_sum = 0.0
    total_weight = 0.0
    factor_details = []

    for fn, weight, name, tier_name in FACTORS:
        try:
            raw = fn(agent_data, ai)
        except Exception:
            raw = None

        if raw is None:
            scores[name] = None
            factor_details.append({
                "factor": name,
                "label": FACTOR_LABELS.get(name, name),
                "score": None,
                "weight": weight,
                "tier": tier_name,
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
            "tier": tier_name,
            "contribution": 0.0,
            "skipped": False,
        })

    # Renormalize over available factors
    if total_weight > 0:
        composite = _clamp(round(weighted_sum / total_weight, 1))
    else:
        composite = NEUTRAL

    # Back-fill contributions
    for fd in factor_details:
        if not fd["skipped"]:
            fd["contribution"] = round((fd["score"] - NEUTRAL) * (fd["weight"] / total_weight), 2)

    # SWOT adjustment from AI analysis (if available)
    swot = ai.get("swot", {})
    weakness_count = len(swot.get("weaknesses", []))
    strength_count = len(swot.get("strengths", []))
    threat_count = len(swot.get("threats", []))
    swot_adjust = (strength_count - weakness_count - threat_count * 0.5) * 0.5
    composite = _clamp(round(composite + swot_adjust, 1))

    # Technical open-source / audit bonus
    tech = ai.get("technical", {})
    if tech.get("open_source") is True:
        composite = _clamp(round(composite + 1.0, 1))
    if tech.get("audit_status") == "audited":
        composite = _clamp(round(composite + 1.5, 1))

    # Distribution widening (1.4x deviation from 50)
    composite = _widen_distribution(composite)

    # Dead / strong investment filters
    dead_flagged = _is_dead_agent(agent_data)
    strong_flagged = _is_strong_investment(agent_data)
    if dead_flagged:
        composite = min(round(composite * 0.3, 1), 20.0)
    elif strong_flagged:
        composite = min(round(composite * 1.15, 1), 100.0)

    # Final clamp 5-100
    composite = _clamp(round(composite, 1), 5.0, 100.0)

    tier = _classify_tier(composite)

    # Tier-level rollups
    tier_buckets: dict = {"traction": [], "fundamentals": [], "market": [], "execution": []}
    for fn, weight, name, tier_name in FACTORS:
        s = scores.get(name)
        if s is not None:
            if tier_name in tier_buckets:
                tier_buckets[tier_name].append(s)

    tier_scores = {
        k: round(sum(v) / len(v), 1) if v else NEUTRAL
        for k, v in tier_buckets.items()
    }

    # First mover — uses moat score as proxy
    first_mover = (scores.get("F4_moat") or 0) >= 80 and (scores.get("F1_holders") or 0) >= 70

    # Top 3 helped / hurt
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

    # Doxx detail (kept for UI compatibility — all agents currently anonymous)
    doxx_tier_val = int(agent_data.get("doxx_tier") or ai.get("team", {}).get("doxx_tier") or 3)
    doxx_detail = {
        "tier": doxx_tier_val,
        "label": {1: "Full Doxx", 2: "Social Presence", 3: "Anonymous"}.get(doxx_tier_val, "Anonymous"),
        "score": None,  # not scored — doxx not predictive in this ecosystem
        "reason": {
            1: "Fully verified identity — public team with verifiable credentials",
            2: "Social presence only — pseudonymous with active social accounts",
            3: "Anonymous — standard for Virtuals Protocol agents",
        }.get(doxx_tier_val, "Unknown"),
    }

    narrative = _build_score_narrative(agent_data, ai, scores, composite, tier_scores)

    factors_scored = sum(1 for v in scores.values() if v is not None)

    # Pack metadata into scores dict for DB storage
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


# ---------------------------------------------------------------------------
# Legacy compatibility shim
# ---------------------------------------------------------------------------

def score_doxx_tier2(agent: dict) -> dict:
    """
    Legacy compatibility — returns a minimal Tier 2 social sub-score.
    In v2 doxx tier is not scored (not predictive in Virtuals ecosystem).
    """
    twitter_age = _safe(agent.get("twitter_account_age"), 0) or 0
    followers = _safe(agent.get("twitter_followers"), 0) or 0
    engagement = _safe(agent.get("twitter_engagement_rate"), 0) or 0
    return {
        "total_score": 50.0,
        "components": {
            "account_age": round(min(twitter_age / 730 * 25, 25), 1),
            "follower_quality": round(min(followers / 50_000 * 25, 25), 1),
            "engagement_authenticity": 12.5,
            "pre_project_existence": 12.5,
        },
        "twitter_age_days": int(twitter_age),
        "followers": int(followers),
        "engagement_rate": round(engagement, 2),
    }
