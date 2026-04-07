"""
VirtualsIQ — Claude AI Analysis Layer
Produces structured 5-section overviews and scoring data for Virtuals Protocol agents.
"""

import json
import logging
import os

import anthropic
import httpx

from scoring import calculate_composite_score

logger = logging.getLogger(__name__)

MODEL_SONNET = "claude-sonnet-4-20250514"
MODEL_HAIKU = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Website content fetcher
# ---------------------------------------------------------------------------

async def _fetch_website_content(url: str, max_chars: int = 5000) -> str:
    if not url or url == "None" or not url.startswith("http"):
        return ""
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; VirtualsIQ/1.0)"
            })
            if resp.status_code != 200:
                return ""
            text = resp.text
            import re
            text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:max_chars]
    except Exception as e:
        logger.debug(f"Website fetch failed for {url}: {e}")
        return ""


async def _fetch_twitter_bio(twitter_url: str) -> str:
    if not twitter_url or twitter_url == "None":
        return ""
    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            resp = await client.get(twitter_url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; VirtualsIQ/1.0)"
            })
            if resp.status_code != 200:
                return ""
            import re
            match = re.search(r'<meta\s+(?:name|property)=["\'](?:description|og:description)["\']\s+content=["\']([^"\']+)', resp.text, re.IGNORECASE)
            if match:
                return match.group(1)[:500]
            return ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 5-Section Overview Prompt (THE MOST IMPORTANT CHANGE)
# ---------------------------------------------------------------------------

FIVE_SECTION_PROMPT = """You are VirtualsIQ, a senior intelligence analyst specializing in AI agent ecosystems on Virtuals Protocol. Your job is to produce comprehensive, in-depth analysis reports that investors and researchers rely on to make informed decisions. Be thorough, specific, and analytical — vague or generic answers are unacceptable.

Analyze this AI agent and return a structured overview in exactly 5 sections.

=== PROJECT DATA ===
Name: AGENT_NAME
Ticker: AGENT_TICKER
Category: AGENT_CATEGORY
Status: AGENT_STATUS
Biography: AGENT_BIO
Twitter: AGENT_TWITTER
Website: AGENT_WEBSITE
Telegram: AGENT_TELEGRAM

=== MARKET DATA ===
Price: $AGENT_PRICE | Market Cap: $AGENT_MCAP | 24h Volume: $AGENT_VOL
Price Change 24h: AGENT_CHANGE% | Liquidity: $AGENT_LIQUIDITY
Holders: AGENT_HOLDERS | Buy/Sell Ratio: AGENT_BSR

=== WEBSITE CONTENT ===
AGENT_WEBSITE_CONTENT

=== TWITTER BIO ===
AGENT_TWITTER_BIO

=== ADDITIONAL CONTEXT ===
Virtuals Protocol Page: https://app.virtuals.io/virtuals/AGENT_VIRTUAL_ID
AGENT_EXTRA_CONTEXT

=== INSTRUCTIONS ===
Return ONLY valid JSON with exactly this structure (no markdown, no code blocks):

{
  "what_it_does": "3-4 sentences. Explain in plain English what this agent does, who its target users are, what specific problem it solves, and how it differs from a simple chatbot or generic AI. Be concrete about its capabilities and use cases.",
  "who_is_behind_it": "3-4 sentences. Describe the team's background, anonymity level, verifiable credentials or past projects, and any wallet/on-chain behavior worth noting. If the team is anonymous, state that explicitly and explain what that means for risk. Call out any discrepancies between claimed and verifiable identity.",
  "what_is_notable": "3-4 sentences. Identify the most compelling and verifiable strengths — genuine first-mover advantages, real product traction, confirmed partnerships, unique technical moats, or standout community metrics. Be specific: name the partners, quote the metrics, explain the moat. Do not include hype without evidence.",
  "what_is_concerning": "3-4 sentences. NEVER EMPTY. Provide an honest, specific risk assessment covering the most significant red flags. Consider: team anonymity, thin liquidity, low volume, concentrated holder distribution, missing product, no GitHub activity, suspicious buy/sell patterns, competitor saturation, or regulatory exposure. Every agent has risks — identify the most material ones with specifics.",
  "recent_activity": "3-4 sentences. Describe what has changed in the last 7 days with specifics: price movement and magnitude, holder count changes, any new announcements, social activity shifts, development updates. If data shows no change or silence, state that explicitly and explain whether silence is normal or a warning sign for this stage of project.",

  "scoring_data": {
    "first_mover": {
      "category_unique": true,
      "approach_novel": true,
      "cross_chain_original": true,
      "days_ahead_of_competitor": 60,
      "defensibility_score": 50,
      "analysis": "Brief moat analysis"
    },
    "team": {
      "doxx_tier": 3,
      "doxx_description": "Anonymous / Social Presence / Full Doxx",
      "team_summary": "Who is behind this",
      "track_record_score": 50,
      "wallet_behavior_score": 50,
      "red_flags": []
    },
    "product": {
      "status": "live|beta|testnet|pre-product|vaporware",
      "description": "What has shipped",
      "partnership_score": 50,
      "technical_moat": "Technical advantages",
      "red_flags": []
    },
    "market": {
      "tam_score": 50,
      "tam_description": "Niche-specific TAM, NOT generic AI market size",
      "comparables_score": 50,
      "revenue_model_score": 50,
      "current_revenue_score": 50,
      "mcap_tam_ratio": 0.001,
      "saturation_score": 50,
      "saturation_description": "How saturated is this vertical"
    },
    "community": {
      "depth_score": 50,
      "organic_score": 50,
      "smart_money_score": 50,
      "follower_growth_score": 50,
      "community_analysis": "Community quality assessment"
    },
    "risk": {
      "overall_risk": "low|medium|high|extreme",
      "key_risks": ["specific risk 1", "specific risk 2"],
      "bull_case": "What needs to go right",
      "bear_case": "What could go wrong"
    },
    "technical": {
      "open_source": false,
      "audit_status": "audited|unaudited|unknown"
    },
    "swot": {
      "strengths": ["specific strength"],
      "weaknesses": ["specific weakness"],
      "opportunities": ["specific opportunity"],
      "threats": ["specific threat"]
    },
    "prediction": {
      "7d": {"probability_up": 50, "range_low": -20, "range_high": 20, "catalyst": "Short-term catalyst"},
      "30d": {"probability_up": 50, "range_low": -40, "range_high": 60, "catalyst": "30d outlook"},
      "90d": {"probability_up": 50, "range_low": -60, "range_high": 150, "catalyst": "3mo thesis"}
    }
  }
}

CRITICAL RULES:
1. The 5 overview sections must be consistent across ALL agents — same voice, same analytical depth.
2. "what_is_concerning" is NEVER empty. There is ALWAYS something to flag. If the agent looks clean, dig deeper: concentration risk, market saturation, execution risk, dependency risk.
3. Never invent facts. If unknown, say "Cannot be verified" but then explain WHAT that absence implies for risk or credibility.
4. Every score must reflect actual data — not defaults. Anonymous team = doxx_tier 3, low trust scores. Low volume = low liquidity scores. No GitHub = low technical scores.
5. TAM must be niche-specific (e.g. "AI trading bots on Base L2", "AI-generated music NFTs"), NOT generic "AI market is $XXX trillion" platitudes.
6. Specific data flags: volume < $10K = liquidity warning; holders < 100 = concentration risk; buy/sell ratio > 1.5 = momentum buy pressure; < 0.7 = sustained selling pressure.
7. All 5 text sections must be substantive. Avoid filler phrases like "this is an exciting project" or "the team is working hard". Every sentence must carry specific, verifiable information.
8. Scoring values must span the full 0-100 range based on evidence — do NOT default everything to 50. A project with no product should score 10-20 on product status. A fully doxxed team with track record scores 70-90 on team trust."""


BATCH_TRIAGE_PROMPT = """You are VirtualsIQ. Perform rapid triage analysis of these AI agents from Virtuals Protocol.

AGENTS:
AGENTS_LIST

For each agent, return a JSON array with lightweight assessments. Return ONLY valid JSON array:
[
  {
    "virtuals_id": "id",
    "first_mover": {"category_unique": true, "approach_novel": true, "cross_chain_original": true, "days_ahead_of_competitor": 60, "defensibility_score": 50},
    "team": {"doxx_tier": 3, "track_record_score": 50, "wallet_behavior_score": 50},
    "product": {"status": "pre-product", "partnership_score": 50},
    "market": {"tam_score": 50, "comparables_score": 50, "revenue_model_score": 50, "current_revenue_score": 50, "saturation_score": 50},
    "community": {"depth_score": 50, "organic_score": 50, "smart_money_score": 50, "follower_growth_score": 50},
    "risk": {"overall_risk": "medium", "key_risks": []},
    "prediction": {
      "7d": {"probability_up": 50, "range_low": -20, "range_high": 20},
      "30d": {"probability_up": 50, "range_low": -40, "range_high": 60},
      "90d": {"probability_up": 50, "range_low": -60, "range_high": 150}
    }
  }
]

Be concise. Focus on what differentiates each agent."""


def _build_prompt(agent: dict, website_content: str = "", twitter_bio: str = "") -> str:
    """Build the 5-section analysis prompt."""
    prompt = FIVE_SECTION_PROMPT
    prompt = prompt.replace("AGENT_NAME", str(agent.get("name", "Unknown")))
    prompt = prompt.replace("AGENT_TICKER", str(agent.get("ticker", "N/A")))
    prompt = prompt.replace("AGENT_CATEGORY", str(agent.get("agent_type", "Unknown")))
    prompt = prompt.replace("AGENT_STATUS", str(agent.get("status", "Prototype")))
    prompt = prompt.replace("AGENT_BIO", str(agent.get("biography", "No biography available"))[:1000])
    prompt = prompt.replace("AGENT_TWITTER_BIO", twitter_bio if twitter_bio else "Not available")
    prompt = prompt.replace("AGENT_TWITTER", str(agent.get("linked_twitter", "None")))
    prompt = prompt.replace("AGENT_WEBSITE_CONTENT", website_content if website_content else "Not available.")
    prompt = prompt.replace("AGENT_WEBSITE", str(agent.get("linked_website", "None")))
    prompt = prompt.replace("AGENT_TELEGRAM", str(agent.get("linked_telegram", "None")))
    prompt = prompt.replace("AGENT_PRICE", f"{agent.get('price_usd', 0)}")
    prompt = prompt.replace("AGENT_MCAP", f"{agent.get('market_cap', 0):,.0f}")
    prompt = prompt.replace("AGENT_VOL", f"{agent.get('volume_24h', 0):,.0f}")
    prompt = prompt.replace("AGENT_CHANGE", f"{agent.get('price_change_24h', 0):+.1f}")
    prompt = prompt.replace("AGENT_LIQUIDITY", f"{agent.get('liquidity_usd', 0):,.0f}")
    prompt = prompt.replace("AGENT_BSR", f"{agent.get('buy_sell_ratio', 'N/A')}")
    prompt = prompt.replace("AGENT_HOLDERS", str(agent.get("holder_count", 0)))
    prompt = prompt.replace("AGENT_VIRTUAL_ID", str(agent.get("virtuals_id", "")))

    extra = []
    if agent.get("contract_address"):
        extra.append(f"Contract Address: {agent['contract_address']}")
        extra.append(f"DexScreener: https://dexscreener.com/base/{agent['contract_address']}")
    if agent.get("twitter_followers"):
        extra.append(f"Twitter Followers: {agent['twitter_followers']:,}")
    if agent.get("twitter_engagement_rate"):
        extra.append(f"Twitter Engagement Rate: {agent['twitter_engagement_rate']}%")
    if agent.get("top_10_concentration"):
        extra.append(f"Top 10 Holder Concentration: {agent['top_10_concentration']}%")
    if agent.get("github_contributors"):
        extra.append(f"GitHub Contributors: {agent['github_contributors']}")
    if agent.get("github_last_commit"):
        extra.append(f"Last GitHub Commit: {agent['github_last_commit']}")
    prompt = prompt.replace("AGENT_EXTRA_CONTEXT", "\n".join(extra) if extra else "No additional context available.")

    return prompt


def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return anthropic.Anthropic(api_key=api_key)


def _parse_json_response(text: str) -> dict:
    """Robustly parse JSON from Claude's response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    logger.warning("Could not parse JSON from Claude response")
    return {}


def select_model(agent_data: dict, top_ids: set | None = None) -> str:
    """All agents analyzed with Haiku for cost efficiency."""
    return MODEL_HAIKU


def should_reanalyze(agent_data: dict, prev_snapshot: dict | None = None) -> bool:
    """Check if agent needs re-analysis: never analyzed, status change, MC move >30%, doxx tier change."""
    if not agent_data.get("last_analyzed"):
        return True

    # Check if analysis is stale (>7 days)
    from scoring import _days_since
    days = _days_since(agent_data.get("last_analyzed"))
    if days and days > 7:
        return True

    if prev_snapshot:
        # Status change (e.g. Prototype -> Sentient)
        old_status = prev_snapshot.get("status")
        new_status = agent_data.get("status")
        if old_status and new_status and old_status != new_status:
            logger.info(f"Re-analysis trigger: status change {old_status}->{new_status} for {agent_data.get('name')}")
            return True

        # Market cap moved >30% in 24h
        old_mc = float(prev_snapshot.get("market_cap") or 0)
        new_mc = float(agent_data.get("market_cap") or 0)
        if old_mc > 0 and new_mc > 0:
            pct_change = abs(new_mc - old_mc) / old_mc
            if pct_change > 0.30:
                logger.info(f"Re-analysis trigger: MC moved {pct_change:.0%} for {agent_data.get('name')}")
                return True

        # Doxx tier change
        old_doxx = prev_snapshot.get("doxx_tier")
        new_doxx = agent_data.get("doxx_tier")
        if old_doxx is not None and new_doxx is not None and int(old_doxx) != int(new_doxx):
            logger.info(f"Re-analysis trigger: doxx tier {old_doxx}->{new_doxx} for {agent_data.get('name')}")
            return True

    return False


async def analyze_agent(agent_data: dict, model: str = None, top_ids: set = None) -> dict:
    """
    Full analysis for a single agent.
    1. Fetch website + Twitter context
    2. Call Claude for 5-section overview + scoring data
    3. Calculate composite scores
    Returns combined result.
    """
    try:
        client = _get_client()
        chosen_model = model or select_model(agent_data, top_ids)

        website_content = await _fetch_website_content(
            agent_data.get("linked_website", "")
        )
        twitter_bio = await _fetch_twitter_bio(
            agent_data.get("linked_twitter", "")
        )

        logger.info(
            f"Analyzing {agent_data.get('name')} with {chosen_model} "
            f"(website={'yes' if website_content else 'no'}, "
            f"twitter_bio={'yes' if twitter_bio else 'no'})"
        )

        prompt = _build_prompt(agent_data, website_content, twitter_bio)

        message = client.messages.create(
            model=chosen_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = message.content[0].text
        parsed = _parse_json_response(raw)

        if not parsed:
            parsed = {}

        # Extract the 5-section overview
        overview = {
            "what_it_does": parsed.get("what_it_does", ""),
            "who_is_behind_it": parsed.get("who_is_behind_it", ""),
            "what_is_notable": parsed.get("what_is_notable", ""),
            "what_is_concerning": parsed.get("what_is_concerning", ""),
            "recent_activity": parsed.get("recent_activity", ""),
        }

        # Extract scoring data
        scoring_data = parsed.get("scoring_data", {})

        # Calculate composite scores using scoring data + on-chain data
        score_result = calculate_composite_score(agent_data, scoring_data)

        # Build the analysis_json for backward compat (scoring uses this)
        ai_analysis = scoring_data
        prediction_json = scoring_data.get("prediction", {})

        return {
            "overview": overview,
            "analysis": ai_analysis,
            "scores": score_result,
            "prediction": prediction_json,
        }

    except Exception as e:
        logger.error(f"Analysis failed for {agent_data.get('name')}: {e}")
        score_result = calculate_composite_score(agent_data, {})
        return {
            "overview": {},
            "analysis": {},
            "scores": score_result,
            "prediction": {},
        }


async def batch_triage(agents: list[dict]) -> list[dict]:
    """Lightweight batch analysis for multiple agents."""
    if not agents:
        return []

    agent_lines = []
    for a in agents:
        line = (
            f"ID:{a.get('virtuals_id')} | "
            f"Name:{a.get('name')} | "
            f"Type:{a.get('agent_type')} | "
            f"Status:{a.get('status')} | "
            f"MCap:${a.get('market_cap', 0):,.0f} | "
            f"Bio:{str(a.get('biography', ''))[:100]}"
        )
        agent_lines.append(line)

    agents_text = "\n".join(agent_lines)
    prompt = BATCH_TRIAGE_PROMPT.replace("AGENTS_LIST", agents_text)

    results = []
    try:
        client = _get_client()
        message = client.messages.create(
            model=MODEL_HAIKU,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        triage_results = json.loads(raw)
        triage_map = {str(r.get("virtuals_id", "")): r for r in triage_results}

        for agent in agents:
            vid = str(agent.get("virtuals_id", ""))
            ai_data = triage_map.get(vid, {})
            score_result = calculate_composite_score(agent, ai_data)
            results.append({
                "virtuals_id": vid,
                "analysis": ai_data,
                "scores": score_result,
            })

    except Exception as e:
        logger.error(f"Batch triage failed: {e}")
        for agent in agents:
            score_result = calculate_composite_score(agent, {})
            results.append({
                "virtuals_id": str(agent.get("virtuals_id", "")),
                "analysis": {},
                "scores": score_result,
            })

    return results
