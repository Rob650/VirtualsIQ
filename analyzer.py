"""
VirtualsIQ — Claude AI Analysis Layer
Produces structured research reports and predictions for Virtuals Protocol agents
"""

import json
import logging
import os

import anthropic

from scoring import calculate_composite_score

logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Prompt templates (using .replace() to avoid KeyError with JSON curly braces)
# ---------------------------------------------------------------------------

DEEP_ANALYSIS_PROMPT = """You are VirtualsIQ, an elite intelligence analyst for the Virtuals Protocol ecosystem.

Analyze this AI agent and return a structured JSON intelligence report.

AGENT DATA:
Name: AGENT_NAME
Ticker: AGENT_TICKER
Category: AGENT_CATEGORY
Status: AGENT_STATUS
Biography: AGENT_BIO
Twitter: AGENT_TWITTER
Website: AGENT_WEBSITE
Telegram: AGENT_TELEGRAM
Market Cap: $AGENT_MCAP
24h Volume: $AGENT_VOL
Price Change 24h: AGENT_CHANGE%
Holder Count: AGENT_HOLDERS
GitHub Stars: AGENT_GITHUB_STARS
GitHub Commits (30d): AGENT_GITHUB_COMMITS

Return ONLY valid JSON in this exact structure (no markdown, no code blocks):
{
  "overview": {
    "summary": "2-3 sentence executive summary of what this agent does",
    "value_proposition": "core value proposition in one sentence",
    "stage": "concept|pre-product|testnet|beta|live"
  },
  "team": {
    "doxx_tier": 1,
    "doxx_description": "Full Doxx / Social Presence / Anonymous",
    "team_summary": "what is known about the team",
    "track_record_score": 50,
    "wallet_behavior_score": 50,
    "red_flags": []
  },
  "product": {
    "status": "live|beta|testnet|pre-product|vaporware",
    "description": "what has actually been shipped",
    "partnership_score": 50,
    "technical_moat": "description of technical advantages or lack thereof",
    "red_flags": []
  },
  "first_mover": {
    "category_unique": true,
    "approach_novel": true,
    "cross_chain_original": true,
    "days_ahead_of_competitor": 90,
    "defensibility_score": 60,
    "analysis": "first mover analysis narrative"
  },
  "market": {
    "tam_score": 50,
    "tam_description": "description of addressable market",
    "real_world_comparable": "Bloomberg Terminal / Stripe / etc",
    "comparables_score": 50,
    "revenue_model_score": 50,
    "current_revenue_score": 50,
    "mcap_tam_ratio": 0.001,
    "saturation_score": 50,
    "saturation_description": "how saturated is this vertical"
  },
  "community": {
    "depth_score": 50,
    "organic_score": 50,
    "smart_money_score": 50,
    "follower_growth_score": 50,
    "community_analysis": "narrative on community quality"
  },
  "risk": {
    "overall_risk": "low|medium|high|extreme",
    "key_risks": ["risk 1", "risk 2", "risk 3"],
    "bull_case": "what needs to go right",
    "bear_case": "what could go wrong"
  },
  "prediction": {
    "7d": {
      "probability_up": 55,
      "range_low": -20,
      "range_high": 30,
      "catalyst": "what could drive this"
    },
    "30d": {
      "probability_up": 55,
      "range_low": -40,
      "range_high": 80,
      "catalyst": "30 day outlook"
    },
    "90d": {
      "probability_up": 55,
      "range_low": -60,
      "range_high": 200,
      "catalyst": "3 month thesis"
    }
  },
  "intelligence_notes": "additional insights, patterns, or observations not captured above"
}

Base your analysis on the data provided. For unknown fields, use neutral scores (50). Be honest about uncertainty."""


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


def _build_analysis_prompt(agent: dict) -> str:
    """Build the deep analysis prompt using .replace() — safe with JSON curly braces."""
    prompt = DEEP_ANALYSIS_PROMPT
    prompt = prompt.replace("AGENT_NAME", str(agent.get("name", "Unknown")))
    prompt = prompt.replace("AGENT_TICKER", str(agent.get("ticker", "N/A")))
    prompt = prompt.replace("AGENT_CATEGORY", str(agent.get("agent_type", "Unknown")))
    prompt = prompt.replace("AGENT_STATUS", str(agent.get("status", "Prototype")))
    prompt = prompt.replace("AGENT_BIO", str(agent.get("biography", "No biography available"))[:500])
    prompt = prompt.replace("AGENT_TWITTER", str(agent.get("linked_twitter", "None")))
    prompt = prompt.replace("AGENT_WEBSITE", str(agent.get("linked_website", "None")))
    prompt = prompt.replace("AGENT_TELEGRAM", str(agent.get("linked_telegram", "None")))
    prompt = prompt.replace("AGENT_MCAP", f"{agent.get('market_cap', 0):,.0f}")
    prompt = prompt.replace("AGENT_VOL", f"{agent.get('volume_24h', 0):,.0f}")
    prompt = prompt.replace("AGENT_CHANGE", f"{agent.get('price_change_24h', 0):+.1f}")
    prompt = prompt.replace("AGENT_HOLDERS", str(agent.get("holder_count", 0)))
    prompt = prompt.replace("AGENT_GITHUB_STARS", str(agent.get("github_stars", 0)))
    prompt = prompt.replace("AGENT_GITHUB_COMMITS", str(agent.get("github_commits_30d", 0)))
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
    Returns combined result with scores, analysis, and predictions.
    """
    try:
        client = _get_client()
        prompt = _build_analysis_prompt(agent_data)

        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = message.content[0].text
        ai_analysis = _parse_json_response(raw)

        if not ai_analysis:
            ai_analysis = {}

        # Calculate composite scores using AI analysis + on-chain data
        score_result = calculate_composite_score(agent_data, ai_analysis)

        return {
            "analysis": ai_analysis,
            "scores": score_result,
        }

    except Exception as e:
        logger.error(f"Analysis failed for {agent_data.get('name')}: {type(e).__name__}: {e}", exc_info=True)
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
