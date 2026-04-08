"""
VirtualsIQ — Claude AI Analysis Layer
Produces structured 5-section deep-dive reports and scoring data for Virtuals Protocol agents.
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
# Deep-Dive Analysis Prompt (~1000 words per report)
# ---------------------------------------------------------------------------

FIVE_SECTION_PROMPT = """You are a senior crypto research analyst writing institutional-grade due diligence reports for a hedge fund. Your reports are thorough, specific, data-driven, and approximately 1000 words. Vague or generic language is unacceptable. Every claim must be grounded in the data provided or explicitly flagged as unverifiable.

Analyze this AI agent from Virtuals Protocol and return a structured deep-dive report in exactly 5 sections.

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
Return ONLY valid JSON with exactly this structure (no markdown, no code blocks).

Each of the 5 text sections must be 2-3 detailed paragraphs (200 words minimum per section). Reference the specific data points provided above — cite actual numbers, URLs, holder counts, volume figures, and market cap. Do not write generic filler. Every paragraph must add analytical value a fund manager could act on.

{
  "what_it_does": "PARAGRAPH 1: Describe the agent's core function in precise technical terms — what it does, how it works, and what underlying technology or protocol it leverages. Name the specific capabilities visible from the website content or biography. PARAGRAPH 2: Identify the target user segment and specific use case this agent addresses. Explain how its workflow differs from a general-purpose LLM or chatbot and what proprietary value it creates. Cite any specific features, integrations, or APIs mentioned in the data. PARAGRAPH 3: Assess the product completeness based on status (AGENT_STATUS) and available evidence. If the product is live, describe what is actually deployed. If pre-product, explain the gap between claimed capability and what is verifiable today.",

  "who_is_behind_it": "PARAGRAPH 1: Describe everything known or inferable about the team — named founders, social handles, organizational structure, and any verifiable credentials or prior projects. If the team is anonymous, state that explicitly and cross-reference against the Twitter URL (AGENT_TWITTER) and website (AGENT_WEBSITE) for any identifying information. PARAGRAPH 2: Analyze on-chain and operational behavior as a proxy for team credibility — holder concentration, liquidity provision decisions, launch mechanics, and whether the project's stated roadmap aligns with on-chain activity. Reference specific data: AGENT_HOLDERS holders, liquidity of $AGENT_LIQUIDITY, buy/sell ratio of AGENT_BSR. PARAGRAPH 3: Provide a frank assessment of team risk. Anonymous teams on Virtuals Protocol are common but not uniform in risk — distinguish between pseudonymous builders with track records versus fully anonymous with no verifiable history. State what due diligence steps a fund would need to take before deploying capital.",

  "what_is_notable": "PARAGRAPH 1: Identify the single most compelling differentiator this agent has — whether that is a genuine first-mover position in its category (AGENT_CATEGORY), a confirmed technical moat, a verifiable partnership, or standout on-chain metrics. Cite the specific data that supports this claim. Do not include unverifiable hype. PARAGRAPH 2: Analyze the market traction evidence available. Discuss holder count (AGENT_HOLDERS), trading volume ($AGENT_VOL 24h), market cap ($AGENT_MCAP), and buy/sell ratio (AGENT_BSR) in the context of comparable Virtuals Protocol agents. Flag whether these metrics suggest genuine organic interest or manufactured activity. PARAGRAPH 3: If notable partnerships, integrations, or ecosystem positioning are referenced in the website content or biography, analyze their significance. If none exist, identify what strategic advantages this project has that are defensible without partnerships — and assess how durable those advantages are against well-funded competitors entering the same niche.",

  "what_is_concerning": "PARAGRAPH 1: Lead with the most material risk identified from the quantitative data. Assess liquidity ($AGENT_LIQUIDITY) — if under $50K this represents significant slippage risk for any meaningful position. Assess volume ($AGENT_VOL) — if under $10K/day the market is illiquid. Assess holder distribution (AGENT_HOLDERS holders) — if under 200 holders the project is highly concentrated and vulnerable to coordinated selling. Assess buy/sell ratio (AGENT_BSR) — ratios above 1.5 signal speculative momentum; below 0.7 signal sustained distribution. PARAGRAPH 2: Assess product and execution risk. If the agent is pre-product or vaporware, calculate the funding gap between current market cap ($AGENT_MCAP) and what a working product in this category actually costs to build. If the product is live, assess whether the technical moat is defensible or easily replicated. Identify specific execution risks: team bandwidth, dependency on Virtuals Protocol infrastructure, regulatory exposure of the use case. PARAGRAPH 3: Assess market and competitive risk. How saturated is the AGENT_CATEGORY vertical on Virtuals Protocol and across the broader AI agent ecosystem? Name the most direct competitors and explain how this agent differentiates. Flag any red flags in the biography or website content — inconsistencies, overstatements, missing information that should be present for a legitimate project at this stage.",

  "recent_activity": "PARAGRAPH 1: Analyze the 24-hour price action in context. A AGENT_CHANGE% move with $AGENT_VOL volume and $AGENT_MCAP market cap implies a specific volume-to-mcap ratio — calculate it and assess whether the price move is credible or potentially manipulated. Cross-reference with buy/sell ratio (AGENT_BSR) to determine whether buying or selling pressure drove the move. PARAGRAPH 2: Assess social and development activity signals. If a Twitter URL (AGENT_TWITTER) is provided, note whether the account appears active or dormant based on bio content. If a website (AGENT_WEBSITE) is provided, assess whether the website content reflects recent updates or appears stale. If GitHub data is available, comment on commit frequency and contributor count — no recent commits for a product claiming active development is a significant warning sign. PARAGRAPH 3: Provide a synthesis of recent trajectory. Based on all available signals — price, volume, holders, social activity, product status — characterize whether this project is in an accumulation phase, distribution phase, stagnation, or active growth. Explain what a material change in any one of these indicators would signal about the project's health.",

  "scoring_data": {
    "first_mover": {
      "category_unique": true,
      "approach_novel": true,
      "cross_chain_original": true,
      "days_ahead_of_competitor": 60,
      "defensibility_score": 50,
      "analysis": "Specific moat analysis referencing the category and competitors"
    },
    "team": {
      "doxx_tier": 3,
      "doxx_description": "Anonymous / Social Presence / Full Doxx",
      "team_summary": "Who is behind this and what is verifiable",
      "track_record_score": 50,
      "wallet_behavior_score": 50,
      "red_flags": []
    },
    "product": {
      "status": "live|beta|testnet|pre-product|vaporware",
      "description": "What has actually shipped",
      "partnership_score": 50,
      "technical_moat": "Specific technical advantages or lack thereof",
      "red_flags": []
    },
    "market": {
      "tam_score": 50,
      "tam_description": "Niche-specific TAM — e.g. AI trading bots on Base L2, NOT generic AI market",
      "comparables_score": 50,
      "revenue_model_score": 50,
      "current_revenue_score": 50,
      "mcap_tam_ratio": 0.001,
      "saturation_score": 50,
      "saturation_description": "Specific assessment of vertical saturation"
    },
    "community": {
      "depth_score": 50,
      "organic_score": 50,
      "smart_money_score": 50,
      "follower_growth_score": 50,
      "community_analysis": "Specific community quality assessment with data"
    },
    "risk": {
      "overall_risk": "low|medium|high|extreme",
      "key_risks": ["specific risk 1 with data", "specific risk 2 with data"],
      "bull_case": "Specific conditions required for upside with metrics",
      "bear_case": "Specific failure modes with data-backed reasoning"
    },
    "technical": {
      "open_source": false,
      "audit_status": "audited|unaudited|unknown"
    },
    "swot": {
      "strengths": ["specific strength backed by data"],
      "weaknesses": ["specific weakness backed by data"],
      "opportunities": ["specific opportunity with market context"],
      "threats": ["specific threat with named competitors or risks"]
    }
  }
}

CRITICAL RULES:
1. All 5 text sections combined must total approximately 1000 words. Each section is 200+ words minimum.
2. "what_is_concerning" is NEVER empty. Flag real risks with specific numbers from the data provided.
3. Never invent facts. If data is missing, write "Cannot be verified" and explain what that absence implies for risk.
4. Every score must reflect actual data — not defaults. Anonymous team = doxx_tier 3, low trust scores. Volume < $10K = 10-20 on liquidity scores. No GitHub = 10-20 on technical scores.
5. TAM must be niche-specific (e.g. "AI trading bots on Base L2"), NEVER "the AI market is $XXX trillion".
6. Data thresholds: volume < $10K = liquidity warning; holders < 100 = extreme concentration risk; holders < 500 = high concentration risk; buy/sell ratio > 1.5 = momentum buying; < 0.7 = sustained selling pressure; liquidity < $50K = illiquid.
7. Scores must span the full 0-100 range based on evidence. Do NOT default everything to 50.
8. Reference specific URLs, numbers, and identifiers from the data throughout all 5 sections."""


BATCH_TRIAGE_PROMPT = """You are a senior crypto research analyst performing rapid triage on AI agents from Virtuals Protocol for a hedge fund screening process.

AGENTS:
AGENTS_LIST

For each agent, return a JSON array with assessments. Return ONLY valid JSON array:
[
  {
    "virtuals_id": "id",
    "first_mover": {"category_unique": true, "approach_novel": true, "cross_chain_original": true, "days_ahead_of_competitor": 60, "defensibility_score": 50},
    "team": {"doxx_tier": 3, "track_record_score": 50, "wallet_behavior_score": 50},
    "product": {"status": "pre-product", "partnership_score": 50},
    "market": {"tam_score": 50, "comparables_score": 50, "revenue_model_score": 50, "current_revenue_score": 50, "saturation_score": 50},
    "community": {"depth_score": 50, "organic_score": 50, "smart_money_score": 50, "follower_growth_score": 50},
    "risk": {"overall_risk": "medium", "key_risks": []}
  }
]

Score based on actual data. Do not default everything to 50. Focus on differentiation between agents."""


def _build_prompt(agent: dict, website_content: str = "", twitter_bio: str = "") -> str:
    """Build the deep-dive analysis prompt."""
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
    2. Call Claude for 5-section deep-dive report + scoring data
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
            max_tokens=6000,
            system="You are a senior crypto research analyst writing institutional-grade due diligence reports for a hedge fund. Your reports are thorough, specific, data-driven, and approximately 1000 words.",
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

        # Extract scoring data (no prediction)
        scoring_data = parsed.get("scoring_data", {})

        # Calculate composite scores using scoring data + on-chain data
        score_result = calculate_composite_score(agent_data, scoring_data)

        ai_analysis = scoring_data

        return {
            "overview": overview,
            "analysis": ai_analysis,
            "scores": score_result,
        }

    except Exception as e:
        logger.error(f"Analysis failed for {agent_data.get('name')}: {e}")
        score_result = calculate_composite_score(agent_data, {})
        return {
            "overview": {},
            "analysis": {},
            "scores": score_result,
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
