"""
VirtualsIQ — Claude AI Analysis Layer
Produces structured research reports and predictions for Virtuals Protocol agents
"""

import json
import logging
import os

import anthropic
import httpx

from scoring import calculate_composite_score

logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-sonnet-4-20250514"

# ---------------------------------------------------------------------------
# Website content fetcher
# ---------------------------------------------------------------------------

async def _fetch_website_content(url: str, max_chars: int = 5000) -> str:
    """Fetch and extract text content from a project's website."""
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
            # Strip HTML tags for a rough text extraction
            import re
            # Remove script/style blocks
            text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', text)
            # Collapse whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:max_chars]
    except Exception as e:
        logger.debug(f"Website fetch failed for {url}: {e}")
        return ""


async def _fetch_twitter_bio(twitter_url: str) -> str:
    """Try to extract Twitter bio/description from a profile URL."""
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
            # Try to find meta description which often contains the bio
            match = re.search(r'<meta\s+(?:name|property)=["\'](?:description|og:description)["\']\s+content=["\']([^"\']+)', resp.text, re.IGNORECASE)
            if match:
                return match.group(1)[:500]
            return ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Prompt templates (using .replace() to avoid KeyError with JSON curly braces)
# ---------------------------------------------------------------------------

DEEP_ANALYSIS_PROMPT = """You are VirtualsIQ, an elite intelligence analyst for the Virtuals Protocol ecosystem.
You have been given comprehensive data about an AI agent project. Produce a DEEP, PROJECT-SPECIFIC analysis.

=== PROJECT DATA ===
Name: AGENT_NAME
Ticker: AGENT_TICKER
Category: AGENT_CATEGORY
Status: AGENT_STATUS
Biography: AGENT_BIO
Twitter: AGENT_TWITTER
Website: AGENT_WEBSITE
Telegram: AGENT_TELEGRAM

=== MARKET DATA (from DexScreener) ===
Current Price: $AGENT_PRICE
Market Cap: $AGENT_MCAP
24h Volume: $AGENT_VOL
Price Change 24h: AGENT_CHANGE%
Liquidity (USD): $AGENT_LIQUIDITY
Buy/Sell Ratio: AGENT_BSR
Holder Count: AGENT_HOLDERS

=== DEVELOPMENT DATA ===
GitHub Stars: AGENT_GITHUB_STARS
GitHub Commits (30d): AGENT_GITHUB_COMMITS

=== WEBSITE CONTENT ===
AGENT_WEBSITE_CONTENT

=== TWITTER BIO ===
AGENT_TWITTER_BIO

=== ADDITIONAL CONTEXT ===
Virtuals Protocol Page: https://app.virtuals.io/virtuals/AGENT_VIRTUAL_ID
AGENT_EXTRA_CONTEXT

=== ANALYSIS INSTRUCTIONS ===
You MUST produce an exhaustive, project-specific report. Do NOT give generic AI industry analysis.
Research and reason deeply about THIS specific project based on ALL the data above.

For each section, be SPECIFIC to this project:
- Reference actual data points (holder count, volume, mcap, bio details)
- If the team is anonymous, say so explicitly and score accordingly
- If there's no website or GitHub, flag that as a risk
- Compare to REAL competitors in the same niche on Virtuals Protocol
- The TAM should be specific to their niche (e.g., "AI trading bots on Base L2"), NOT generic "AI market"

Return ONLY valid JSON in this exact structure (no markdown, no code blocks):
{
  "overview": {
    "summary": "3-5 sentence comprehensive summary of what this project does, its core product/service, and current state. Be specific about what it actually builds or offers.",
    "what_it_is": "Detailed explanation of the project's core product, service, or protocol. What problem does it solve? How does it work?",
    "value_proposition": "Core value proposition — what makes someone want to use or invest in this specific agent",
    "niche": "The specific sub-niche this operates in (e.g., 'AI-powered DeFi yield optimization on Base' not just 'DeFi')",
    "competitive_advantages": "What makes this project unique vs direct competitors. Moat analysis — network effects, proprietary tech, data advantages, brand, partnerships",
    "stage": "concept|pre-product|testnet|beta|live"
  },
  "swot": {
    "strengths": ["strength 1 specific to THIS project", "strength 2", "strength 3"],
    "weaknesses": ["weakness 1 specific to THIS project", "weakness 2", "weakness 3"],
    "opportunities": ["opportunity 1", "opportunity 2", "opportunity 3"],
    "threats": ["threat 1", "threat 2", "threat 3"]
  },
  "technical": {
    "architecture_summary": "What is known about the technical architecture, AI models used, blockchain integration, smart contracts",
    "tech_stack_score": 50,
    "ai_model_details": "What AI/ML models or frameworks does this agent use? Is it a wrapper or proprietary?",
    "blockchain_integration": "How does it integrate with Virtuals Protocol and Base chain? Token utility?",
    "open_source": true,
    "audit_status": "audited|unaudited|unknown"
  },
  "team": {
    "doxx_tier": 1,
    "doxx_description": "Full Doxx / Social Presence / Anonymous — be specific about what is known",
    "team_summary": "Detailed summary: who is behind this project, their backgrounds, prior projects, credibility. If anonymous, state that clearly.",
    "track_record_score": 50,
    "wallet_behavior_score": 50,
    "red_flags": []
  },
  "product": {
    "status": "live|beta|testnet|pre-product|vaporware",
    "description": "What has actually been shipped and is usable today. Be brutally honest.",
    "partnership_score": 50,
    "technical_moat": "Description of technical advantages or lack thereof — specific to this project",
    "red_flags": []
  },
  "first_mover": {
    "category_unique": true,
    "approach_novel": true,
    "cross_chain_original": true,
    "days_ahead_of_competitor": 90,
    "defensibility_score": 60,
    "analysis": "Detailed first mover analysis — who are the direct competitors on Virtuals and other platforms? How far ahead or behind is this project?"
  },
  "market": {
    "tam_score": 50,
    "tam_description": "SPECIFIC TAM for this project's niche. E.g., 'The on-chain AI trading bot market on EVM L2s is estimated at $X-YB' — NOT generic 'the AI industry is worth $500B'",
    "tam_size_estimate": "$500M-2B",
    "growth_trajectory": "Description of market growth specific to their vertical",
    "real_world_comparable": "Bloomberg Terminal / Stripe / etc — what Web2 company is the closest analogy?",
    "comparables_score": 50,
    "competitor_landscape": "Name specific competitors and compare. Who else does this on Virtuals? On Solana? On other chains?",
    "revenue_model_score": 50,
    "current_revenue_score": 50,
    "mcap_tam_ratio": 0.001,
    "saturation_score": 50,
    "saturation_description": "How saturated is this specific vertical? How many similar projects exist?"
  },
  "community": {
    "depth_score": 50,
    "organic_score": 50,
    "smart_money_score": 50,
    "follower_growth_score": 50,
    "community_analysis": "Detailed narrative on community quality — are holders organic? Is there real engagement or bot activity? Reference actual holder count and volume data."
  },
  "risk": {
    "overall_risk": "low|medium|high|extreme",
    "key_risks": ["risk 1 specific to this project", "risk 2", "risk 3"],
    "bull_case": "Specific bull case: what needs to go right for this project to 5-10x. Reference actual catalysts.",
    "bear_case": "Specific bear case: what could cause this to lose 80%+ of value. Be honest."
  },
  "prediction": {
    "7d": {
      "probability_up": 55,
      "range_low": -20,
      "range_high": 30,
      "catalyst": "Specific short-term catalyst for THIS project"
    },
    "30d": {
      "probability_up": 55,
      "range_low": -40,
      "range_high": 80,
      "catalyst": "30 day outlook specific to this project"
    },
    "90d": {
      "probability_up": 55,
      "range_low": -60,
      "range_high": 200,
      "catalyst": "3 month thesis specific to this project"
    }
  },
  "intelligence_notes": "Additional insights, patterns, alpha, or observations. What would a smart investor want to know that isn't captured above?"
}

CRITICAL RULES:
1. Every score MUST reflect the actual data. If team is anonymous (no Twitter, no website), doxx_tier=3 and track_record_score should be LOW (10-25).
2. If volume is under $10K, that's a red flag. If holder count is under 100, flag it.
3. If there's no GitHub, open_source=false and code_activity should be noted as absent.
4. Do NOT hallucinate partnerships or features not evident from the data.
5. For unknown fields, use neutral scores (50) but note the uncertainty.
6. TAM must be NICHE-SPECIFIC, not generic AI industry numbers."""


RESEARCH_REPORT_PROMPT = """You are VirtualsIQ, writing a professional analyst research report for a Virtuals Protocol AI agent.

=== PROJECT DATA ===
Name: AGENT_NAME
Ticker: AGENT_TICKER
Category: AGENT_CATEGORY
Status: AGENT_STATUS
Biography: AGENT_BIO
Website: AGENT_WEBSITE
Twitter: AGENT_TWITTER

=== MARKET DATA ===
Price: $AGENT_PRICE | Market Cap: $AGENT_MCAP | 24h Volume: $AGENT_VOL
Price Change 24h: AGENT_CHANGE% | Liquidity: $AGENT_LIQUIDITY
Holders: AGENT_HOLDERS | Buy/Sell Ratio: AGENT_BSR

=== WEBSITE CONTENT ===
AGENT_WEBSITE_CONTENT

=== TWITTER BIO ===
AGENT_TWITTER_BIO

=== AI ANALYSIS FINDINGS ===
AGENT_ANALYSIS_JSON

=== VIQ SCORE ===
Composite: AGENT_VIQ_SCORE / 100 (AGENT_VIQ_TIER)

=== INSTRUCTIONS ===
Write a ~1000 word professional research report about this specific project. This should read like a Wall Street analyst report adapted for crypto/AI.

Structure the report with these sections (use markdown headers ##):

## Executive Summary
2-3 sentences on what this is and the investment thesis.

## Product & Technology
What they built, how it works, tech stack if known. Be specific to THIS project. If info is limited, say so honestly.

## Team & Credibility
Who is behind it. If anonymous, discuss implications. Prior work if known.

## Competitive Landscape
Direct competitors on Virtuals Protocol and other chains. What differentiates this project.

## On-Chain Metrics Analysis
Analyze the actual holder count, volume, liquidity, and price action. What do these numbers tell us?

## SWOT Analysis
Formatted as bullet points under Strengths, Weaknesses, Opportunities, Threats.

## Market Opportunity
TAM specific to their niche. NOT generic AI market size. Real opportunity analysis.

## Risk Assessment
Key risks ranked by severity. Be honest and specific.

## Investment Thesis
Bull case, bear case, and base case. What would need to happen for each scenario.

CRITICAL: Be SPECIFIC to this project. Reference actual data points. If data is missing, say "data not available" rather than fabricating. Do NOT pad with generic AI industry commentary. Every sentence should be about THIS project."""


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


def _build_analysis_prompt(agent: dict, website_content: str = "", twitter_bio: str = "") -> str:
    """Build the deep analysis prompt using .replace() — safe with JSON curly braces."""
    prompt = DEEP_ANALYSIS_PROMPT
    prompt = prompt.replace("AGENT_NAME", str(agent.get("name", "Unknown")))
    prompt = prompt.replace("AGENT_TICKER", str(agent.get("ticker", "N/A")))
    prompt = prompt.replace("AGENT_CATEGORY", str(agent.get("agent_type", "Unknown")))
    prompt = prompt.replace("AGENT_STATUS", str(agent.get("status", "Prototype")))
    prompt = prompt.replace("AGENT_BIO", str(agent.get("biography", "No biography available"))[:1000])
    prompt = prompt.replace("AGENT_TWITTER_BIO", twitter_bio if twitter_bio else "Not available")
    prompt = prompt.replace("AGENT_TWITTER", str(agent.get("linked_twitter", "None")))
    prompt = prompt.replace("AGENT_WEBSITE_CONTENT", website_content if website_content else "Not available — no website content could be retrieved.")
    prompt = prompt.replace("AGENT_WEBSITE", str(agent.get("linked_website", "None")))
    prompt = prompt.replace("AGENT_TELEGRAM", str(agent.get("linked_telegram", "None")))
    prompt = prompt.replace("AGENT_PRICE", f"{agent.get('price_usd', 0)}")
    prompt = prompt.replace("AGENT_MCAP", f"{agent.get('market_cap', 0):,.0f}")
    prompt = prompt.replace("AGENT_VOL", f"{agent.get('volume_24h', 0):,.0f}")
    prompt = prompt.replace("AGENT_CHANGE", f"{agent.get('price_change_24h', 0):+.1f}")
    prompt = prompt.replace("AGENT_LIQUIDITY", f"{agent.get('liquidity_usd', 0):,.0f}")
    prompt = prompt.replace("AGENT_BSR", f"{agent.get('buy_sell_ratio', 'N/A')}")
    prompt = prompt.replace("AGENT_HOLDERS", str(agent.get("holder_count", 0)))
    prompt = prompt.replace("AGENT_GITHUB_STARS", str(agent.get("github_stars", 0)))
    prompt = prompt.replace("AGENT_GITHUB_COMMITS", str(agent.get("github_commits_30d", 0)))
    prompt = prompt.replace("AGENT_VIRTUAL_ID", str(agent.get("virtuals_id", "")))

    # Build extra context from any additional data we have
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


def _build_report_prompt(agent: dict, ai_analysis: dict, score_result: dict,
                         website_content: str = "", twitter_bio: str = "") -> str:
    """Build the research report prompt."""
    prompt = RESEARCH_REPORT_PROMPT
    prompt = prompt.replace("AGENT_NAME", str(agent.get("name", "Unknown")))
    prompt = prompt.replace("AGENT_TICKER", str(agent.get("ticker", "N/A")))
    prompt = prompt.replace("AGENT_CATEGORY", str(agent.get("agent_type", "Unknown")))
    prompt = prompt.replace("AGENT_STATUS", str(agent.get("status", "Prototype")))
    prompt = prompt.replace("AGENT_BIO", str(agent.get("biography", "No biography available"))[:1000])
    prompt = prompt.replace("AGENT_TWITTER_BIO", twitter_bio if twitter_bio else "Not available")
    prompt = prompt.replace("AGENT_TWITTER", str(agent.get("linked_twitter", "None")))
    prompt = prompt.replace("AGENT_WEBSITE_CONTENT", website_content if website_content else "Not available.")
    prompt = prompt.replace("AGENT_WEBSITE", str(agent.get("linked_website", "None")))
    prompt = prompt.replace("AGENT_PRICE", f"{agent.get('price_usd', 0)}")
    prompt = prompt.replace("AGENT_MCAP", f"{agent.get('market_cap', 0):,.0f}")
    prompt = prompt.replace("AGENT_VOL", f"{agent.get('volume_24h', 0):,.0f}")
    prompt = prompt.replace("AGENT_CHANGE", f"{agent.get('price_change_24h', 0):+.1f}")
    prompt = prompt.replace("AGENT_LIQUIDITY", f"{agent.get('liquidity_usd', 0):,.0f}")
    prompt = prompt.replace("AGENT_BSR", f"{agent.get('buy_sell_ratio', 'N/A')}")
    prompt = prompt.replace("AGENT_HOLDERS", str(agent.get("holder_count", 0)))

    # Truncate analysis JSON for context
    analysis_str = json.dumps(ai_analysis, indent=2)
    if len(analysis_str) > 4000:
        analysis_str = analysis_str[:4000] + "\n... (truncated)"
    prompt = prompt.replace("AGENT_ANALYSIS_JSON", analysis_str)

    prompt = prompt.replace("AGENT_VIQ_SCORE", str(score_result.get("composite_score", 50)))
    prompt = prompt.replace("AGENT_VIQ_TIER", score_result.get("tier_classification", "Moderate"))

    return prompt


def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return anthropic.Anthropic(api_key=api_key)


def _parse_json_response(text: str) -> dict:
    """Robustly parse JSON from Claude's response."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    logger.warning("Could not parse JSON from Claude response")
    return {}


async def analyze_agent(agent_data: dict) -> dict:
    """
    Full deep analysis for a single agent.
    1. Fetch website content and Twitter bio for richer context
    2. Call Claude for structured JSON analysis
    3. Calculate composite scores
    4. Call Claude again for ~1000 word narrative research report
    Returns combined result with scores, analysis, and research report.
    """
    try:
        client = _get_client()

        # Step 1: Gather additional context in parallel-ish fashion
        website_content = await _fetch_website_content(
            agent_data.get("linked_website", "")
        )
        twitter_bio = await _fetch_twitter_bio(
            agent_data.get("linked_twitter", "")
        )

        logger.info(
            f"Gathered context for {agent_data.get('name')}: "
            f"website={'yes' if website_content else 'no'}, "
            f"twitter_bio={'yes' if twitter_bio else 'no'}"
        )

        # Step 2: Structured JSON analysis
        prompt = _build_analysis_prompt(agent_data, website_content, twitter_bio)

        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = message.content[0].text
        ai_analysis = _parse_json_response(raw)

        if not ai_analysis:
            ai_analysis = {}

        # Step 3: Calculate composite scores using AI analysis + on-chain data
        score_result = calculate_composite_score(agent_data, ai_analysis)

        # Step 4: Generate narrative research report
        report_text = ""
        try:
            report_prompt = _build_report_prompt(
                agent_data, ai_analysis, score_result,
                website_content, twitter_bio
            )
            report_message = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=3000,
                messages=[{"role": "user", "content": report_prompt}]
            )
            report_text = report_message.content[0].text.strip()
            logger.info(
                f"Research report generated for {agent_data.get('name')}: "
                f"{len(report_text)} chars"
            )
        except Exception as e:
            logger.error(f"Report generation failed for {agent_data.get('name')}: {e}")
            report_text = ""

        # Store the report in the analysis dict
        ai_analysis["research_report"] = report_text

        return {
            "analysis": ai_analysis,
            "scores": score_result,
        }

    except Exception as e:
        logger.error(f"Analysis failed for {agent_data.get('name')}: {e}")
        # Return neutral scores on failure
        score_result = calculate_composite_score(agent_data, {})
        return {
            "analysis": {},
            "scores": score_result,
        }


async def batch_triage(agents: list[dict]) -> list[dict]:
    """
    Lightweight batch analysis for multiple agents.
    Returns list of (agent_id, ai_analysis, score_result) dicts.
    """
    if not agents:
        return []

    # Build agent summaries for batch prompt
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
            model=CLAUDE_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        triage_results = json.loads(raw)

        # Build a lookup by virtuals_id
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
        # Fall back to scoring without AI analysis
        for agent in agents:
            score_result = calculate_composite_score(agent, {})
            results.append({
                "virtuals_id": str(agent.get("virtuals_id", "")),
                "analysis": {},
                "scores": score_result,
            })

    return results
