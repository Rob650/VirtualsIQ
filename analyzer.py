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

FIVE_SECTION_PROMPT = """You are a crypto research analyst at a fund that specializes in finding early-stage gems in the AI agent ecosystem. Your job is to write engaging, insight-driven research reports that help investors understand what makes a project interesting and worth watching. Your tone is enthusiastic and exploratory — you love this space and are genuinely looking for the next breakout project. You note risks briefly and factually, but your energy is on what's promising.

Analyze this AI agent from Virtuals Protocol and return a structured report in exactly 5 sections (~1000 words total).

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

WORD COUNT TARGETS: what_it_does=500 words (half the report) | who_is_behind_it=125 words | what_is_notable=125 words | risks_to_monitor=125 words | market_opportunity=125 words. Total ~1000 words.

{
  "what_it_does": "This is the most important section — spend 400-500 words here. Cover ALL of the following:\n\nPARAGRAPH 1 — Core product: What does this agent actually do? Describe the product or service in detail. What problem does it solve? Who is the target user? What makes the interaction model unique compared to a generic chatbot or LLM? If a website URL (AGENT_WEBSITE) is provided and website content is available, describe what the site shows — features, demos, pricing, use cases, integrations. If website content mentions a tech stack, describe it.\n\nPARAGRAPH 2 — Twitter presence: The agent's Twitter is AGENT_TWITTER. Based on the Twitter bio content provided, describe what the account posts about, what personality or tone it projects, what topics it engages with, and any links or handles mentioned in the bio. If the bio references a Linktree, personal site, or Discord, describe what those destinations suggest about the project's community and scope. Note any follower count signals if present.\n\nPARAGRAPH 3 — How it works: Describe the technology or mechanism behind the agent as specifically as possible. What platform or infrastructure does it run on? Does it use any specific AI models, APIs, blockchains, or data sources mentioned anywhere in the data? How does the $AGENT_TICKER token connect to the agent's operation — is it a governance token, access key, revenue share, or pure speculation?\n\nPARAGRAPH 4 — Product status and roadmap: Is this agent currently live and usable? What is the evidence — website demo, Twitter activity, Virtuals Protocol status (AGENT_STATUS)? If it is pre-launch, what has the team said about the timeline? What would a user actually experience if they interacted with this agent today?",

  "who_is_behind_it": "1-2 paragraphs, ~125 words. Focus on the people, their background, and what they've built. Who are the founders — named individuals with LinkedIn/Twitter profiles, pseudonymous builders with a track record, or anonymous? If named, describe their prior experience, previous projects, and any relevant domain expertise visible from the website or Twitter bio. If pseudonymous but active, describe their presence and credibility signals. If anonymous, note it in one sentence and move on. Focus most of this section on: what prior projects or credentials are visible, whether the team has shipped products before, what their communication style suggests about professionalism and commitment, and any advisors or backers mentioned anywhere in the data. Briefly note holder count (AGENT_HOLDERS) as a signal of community size.",

  "what_is_notable": "1-2 paragraphs, ~125 words. Lead with the most exciting thing about this project from a fundamentals perspective. Focus on: what problem this solves that is genuinely valuable, any partnerships or integrations with other protocols or platforms, any shipped milestones or upcoming roadmap items mentioned in the data, ecosystem positioning within Virtuals Protocol or the broader AI agent space, community momentum signals (follower growth, active Discord/Telegram, organic mentions). If the market cap ($AGENT_MCAP) looks small relative to the project's ambition or comparable projects, note the upside optionality. Be specific — cite actual features, partner names, or product milestones. Avoid generic statements.",

  "risks_to_monitor": "1 paragraph, ~125 words. Note the 2-3 most important risks briefly and factually — no catastrophizing, no dwelling. Focus on FUNDAMENTAL risks: team anonymity if present, pre-launch product risk, market saturation in the AGENT_CATEGORY vertical, or missing documentation. You may mention thin liquidity ($AGENT_LIQUIDITY) or low volume ($AGENT_VOL) in a single clause if they are genuinely severe (under $50K liquidity or under $5K volume). Frame each risk as something to watch, and end with what positive development would resolve it.",

  "market_opportunity": "1-2 paragraphs, ~125 words. Paint the bull case for the broader market this project operates in, then anchor the project within it. Cover: (1) The macro narrative — what is the total addressable market for this category (AGENT_CATEGORY) within crypto and beyond? Name the specific verticals (e.g. AI agent infrastructure, Web3 influencer marketing, autonomous DeFi). Reference real market trends or growth signals if visible in the website or Twitter content. (2) Key competitors and comparables — who else is building in this space on Virtuals Protocol or adjacent ecosystems? How does AGENT_NAME differentiate or outposition them? Name specific projects or categories if possible. (3) Where does this project fit in the opportunity — early entrant with room to grow, or late to a crowded market? What would success look like at 10x or 100x the current traction? Keep the tone bullish and forward-looking, grounded in specific details from the data.",

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
1. Hit the word count targets: what_it_does must be 400-500 words. Other sections as specified.
2. Never invent facts. If data is missing, say "not available" briefly and move on — do not dwell on absences.
3. Every score must reflect actual data — not defaults. Anonymous team = doxx_tier 3. Volume < $10K = low liquidity scores. No GitHub = low technical scores.
4. TAM must be niche-specific (e.g. "AI influencer agents on Base L2"), NEVER "the AI market is $XXX trillion".
5. Scores must span the full 0-100 range. Do NOT default everything to 50.
6. The overall tone is optimistic and exploratory. Risks are noted briefly, not dwelt upon."""


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
            max_tokens=8192,
            system="You are a senior crypto research analyst at an institutional fund. Write thorough, data-driven due diligence reports of approximately 1000 words total. Reference specific data points, URLs, and metrics provided. Never use filler phrases. Every sentence must add new information.",
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
            "risks_to_monitor": parsed.get("risks_to_monitor", ""),
            "market_opportunity": parsed.get("market_opportunity", ""),
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
