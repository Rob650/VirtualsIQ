"""
VirtualsIQ — Virtuals Protocol data ingestion
Fetches agents from Virtuals API and enriches with DexScreener market data
"""

import asyncio
import logging
from datetime import datetime

import httpx

from database import upsert_agent, get_existing_ids

logger = logging.getLogger(__name__)

VIRTUALS_API = "https://api2.virtuals.io/api/virtuals"
DEXSCREENER_API = "https://api.dexscreener.com/latest/dex/tokens"

HEADERS = {
    "User-Agent": "VirtualsIQ/1.0 (intelligence terminal; contact@virtualsiq.com)",
    "Accept": "application/json",
}

# Map Virtuals status codes to human-readable labels
STATUS_MAP = {
    5: "Sentient",
    4: "Prototype",
    3: "Prototype",
    2: "Prototype",
    1: "Prototype",
}

AGENT_TYPE_MAP = {
    "trading": "Trading",
    "defi": "DeFi",
    "information": "Information",
    "entertainment": "Entertainment",
    "social": "Social",
    "infrastructure": "Infrastructure",
    "gaming": "Gaming",
    "creative": "Creative",
    "research": "Research",
    "security": "Security",
    "data": "Data",
    "governance": "Governance",
}


def normalize_agent_type(raw: str) -> str:
    if not raw:
        return "Information"
    key = raw.lower().strip()
    return AGENT_TYPE_MAP.get(key, raw.title())


async def _fetch_page(client: httpx.AsyncClient, page: int, page_size: int = 100) -> dict:
    """Fetch a single page of agents from Virtuals API with retry."""
    url = VIRTUALS_API
    params = {
        "filters[status]": 5,
        "pagination[page]": page,
        "pagination[pageSize]": page_size,
        "sort[0]": "volume24h:desc",
    }

    for attempt in range(3):
        try:
            resp = await client.get(url, params=params, headers=HEADERS, timeout=30.0)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error page {page} attempt {attempt+1}: {e}")
        except httpx.RequestError as e:
            logger.warning(f"Request error page {page} attempt {attempt+1}: {e}")

        if attempt < 2:
            await asyncio.sleep(2 ** attempt)

    return {}


def _parse_agent(item: dict) -> dict:
    """Convert raw Virtuals API item to our agent schema."""
    # Use `or item` so a null "attributes" value falls back to the item itself
    attrs = item.get("attributes") or item

    # Handle nested structures from Virtuals API
    virtual_id = str(item.get("id") or attrs.get("id") or "")
    name = attrs.get("name") or "Unknown"
    ticker = attrs.get("symbol") or attrs.get("ticker") or ""
    contract = attrs.get("tokenAddress") or attrs.get("contractAddress") or ""
    status_code = attrs.get("status") or 4
    status = STATUS_MAP.get(status_code, "Prototype")

    # Category / type
    category = attrs.get("category") or attrs.get("agentType") or ""
    if isinstance(category, dict):
        category = category.get("name") or ""
    agent_type = normalize_agent_type(category)

    # Social links — guard against null socials
    socials = attrs.get("socials") or {}
    if isinstance(socials, list):
        socials = {s.get("type", ""): s.get("url", "") for s in socials if isinstance(s, dict)}

    linked_twitter = (
        socials.get("twitter", "") or
        attrs.get("twitter", "") or
        attrs.get("linkedTwitter", "")
    ) or ""
    linked_website = (
        socials.get("website", "") or
        attrs.get("website", "") or
        attrs.get("linkedWebsite", "")
    ) or ""
    linked_telegram = (
        socials.get("telegram", "") or
        attrs.get("telegram", "") or
        attrs.get("linkedTelegram", "")
    ) or ""

    # Market data (may be enriched later via DexScreener)
    market_cap = float(attrs.get("marketCap") or 0)
    price_usd = float(attrs.get("currentPrice") or attrs.get("priceUsd") or 0)

    # Image
    image_obj = attrs.get("image") or {}
    if isinstance(image_obj, dict):
        image_url = image_obj.get("url") or ""
    else:
        image_url = str(image_obj or "")

    # Creation date
    created_at_raw = attrs.get("createdAt") or ""
    creator_wallet = attrs.get("creatorWallet") or attrs.get("creator") or ""
    biography = attrs.get("description") or attrs.get("bio") or attrs.get("biography") or ""

    return {
        "virtuals_id": virtual_id,
        "name": name,
        "ticker": ticker,
        "contract_address": contract,
        "status": status,
        "agent_type": agent_type,
        "biography": biography,
        "creation_date": created_at_raw,
        "linked_twitter": linked_twitter,
        "linked_website": linked_website,
        "linked_telegram": linked_telegram,
        "creator_wallet": creator_wallet,
        "image_url": image_url,
        "market_cap": market_cap,
        "volume_24h": 0.0,
        "volume_6h": 0.0,
        "price_usd": price_usd,
        "price_change_24h": 0.0,
        "liquidity_usd": 0.0,
        "tx_count_24h": 0,
        "buy_sell_ratio": 1.0,
        "holder_count": int(attrs.get("holderCount") or attrs.get("holders") or 0),
        "top_10_concentration": 0.0,
        "twitter_followers": 0,
        "twitter_engagement_rate": 0.0,
        "twitter_account_age": 0,
        "github_stars": 0,
        "github_commits_30d": 0,
        "github_contributors": 0,
        "github_last_commit": None,
        "composite_score": 50.0,
        "tier_classification": "Moderate",
        "scores_json": "{}",
        "analysis_json": "{}",
        "prediction_json": "{}",
        "first_mover": 0,
        "doxx_tier": 3,
        "last_scanned": None,
        "updated_at": datetime.utcnow().isoformat(),
    }


async def fetch_all_agents(max_pages: int = 400) -> list[dict]:
    """
    Paginate through the full Virtuals API and return all agents.
    Virtuals has 38k+ agents; at 100/page that's ~387 pages.
    """
    agents = []
    async with httpx.AsyncClient() as client:
        # Fetch first page to get total count
        first = await _fetch_page(client, 1, 100)
        if not first:
            logger.error("Failed to fetch first page from Virtuals API")
            return []

        items = first.get("data", [])
        agents.extend([_parse_agent(item) for item in items])

        pagination = first.get("meta", {}).get("pagination", {})
        total_pages = min(
            pagination.get("pageCount", 1),
            max_pages
        )
        logger.info(f"Virtuals API: {pagination.get('total', '?')} total agents, {total_pages} pages")

        # Fetch remaining pages concurrently in batches of 10
        page = 2
        while page <= total_pages:
            batch = range(page, min(page + 10, total_pages + 1))
            tasks = [_fetch_page(client, p, 100) for p in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Page fetch error: {result}")
                    continue
                items = result.get("data", [])
                agents.extend([_parse_agent(item) for item in items])

            page += 10
            # Brief pause between batches to be a good API citizen
            await asyncio.sleep(0.5)

    logger.info(f"Fetched {len(agents)} total agents from Virtuals Protocol")
    return agents


async def fetch_dexscreener_data(contract_address: str) -> dict:
    """
    Enrich agent with DexScreener market data.
    Returns price, volume, liquidity, tx counts.
    """
    if not contract_address:
        return {}

    url = f"{DEXSCREENER_API}/{contract_address}"
    for attempt in range(3):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=HEADERS, timeout=15.0)
                resp.raise_for_status()
                data = resp.json()

            pairs = data.get("pairs", [])
            if not pairs:
                return {}

            # Use the pair with the highest liquidity
            best_pair = max(pairs, key=lambda p: float((p.get("liquidity") or {}).get("usd", 0) or 0))

            volume = best_pair.get("volume") or {}
            txns = best_pair.get("txns") or {}
            h24 = txns.get("h24") or {}
            buys = int(h24.get("buys", 0) or 0)
            sells = int(h24.get("sells", 0) or 0)
            buy_sell_ratio = (buys / sells) if sells > 0 else (1.0 if buys == 0 else 2.0)

            price_change = best_pair.get("priceChange") or {}

            return {
                "price_usd": float(best_pair.get("priceUsd", 0) or 0),
                "price_change_24h": float(price_change.get("h24", 0) or 0),
                "volume_24h": float(volume.get("h24", 0) or 0),
                "volume_6h": float(volume.get("h6", 0) or 0),
                "liquidity_usd": float((best_pair.get("liquidity") or {}).get("usd", 0) or 0),
                "tx_count_24h": buys + sells,
                "buy_sell_ratio": round(buy_sell_ratio, 2),
                "market_cap": float(best_pair.get("marketCap") or best_pair.get("fdv") or 0),
            }

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                await asyncio.sleep(2 ** attempt)
            else:
                logger.debug(f"DexScreener {contract_address}: {e}")
                return {}
        except Exception as e:
            logger.debug(f"DexScreener {contract_address}: {e}")
            if attempt < 2:
                await asyncio.sleep(1)

    return {}


async def detect_new_agents(existing_ids: set) -> list[dict]:
    """
    Fetch ALL current agents and return those not yet in the database.
    Full pagination ensures we don't miss new launches anywhere in the list.
    """
    current = await fetch_all_agents()  # Full scan — no page cap
    new_agents = [a for a in current if a["virtuals_id"] not in existing_ids]
    if new_agents:
        logger.info(f"Detected {len(new_agents)} new agents")
    return current  # Return all so the daily scan can update market data for everyone


async def preload_all_agents():
    """
    Fetch and store ALL agents by market cap (38,700+).
    Enriches top 200 with DexScreener data.
    Paginates until the API returns no more results.
    """
    logger.info("Preloading ALL agents from Virtuals Protocol (this may take a while)...")

    agents = await fetch_all_agents()  # No page cap — fetches everything

    # Sort by market cap so most important agents are processed first
    agents.sort(key=lambda a: a.get("market_cap", 0), reverse=True)
    logger.info(f"Fetched {len(agents)} agents total, beginning DexScreener enrichment...")

    # Enrich top 200 with DexScreener (rate-limit friendly)
    top_200 = [a for a in agents[:200] if a.get("contract_address")]
    logger.info(f"Enriching top {len(top_200)} agents with DexScreener data...")

    semaphore = asyncio.Semaphore(5)  # Max 5 concurrent DexScreener calls

    async def enrich(agent: dict):
        async with semaphore:
            dex = await fetch_dexscreener_data(agent["contract_address"])
            if dex:
                agent.update(dex)
            await asyncio.sleep(0.2)  # Rate limit courtesy
        return agent

    enriched = await asyncio.gather(*[enrich(a) for a in top_200], return_exceptions=True)
    for i, result in enumerate(enriched):
        if not isinstance(result, Exception):
            agents[i] = result

    # Store all agents in DB with progress logging every 100
    stored = 0
    for agent in agents:
        try:
            await upsert_agent(agent)
            stored += 1
            if stored % 100 == 0:
                logger.info(f"Stored {stored}/{len(agents)} agents...")
        except Exception as e:
            logger.warning(f"Failed to store agent {agent.get('virtuals_id')}: {e}")

    logger.info(f"Preload complete: {stored}/{len(agents)} agents stored in database")
    return stored
