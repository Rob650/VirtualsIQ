"""
VirtualsIQ — Virtuals Protocol data ingestion
Fetches agents from Virtuals API and enriches with DexScreener market data
"""

import asyncio
import logging
from datetime import datetime

import aiosqlite
import httpx

from database import upsert_agent, get_existing_ids, bulk_upsert_agents, update_market_data, DB_PATH

logger = logging.getLogger(__name__)

VIRTUALS_API = "https://api2.virtuals.io/api/virtuals"
DEXSCREENER_API = "https://api.dexscreener.com/latest/dex/tokens"

HEADERS = {
    "User-Agent": "VirtualsIQ/1.0 (intelligence terminal; contact@virtualsiq.com)",
    "Accept": "application/json",
}

# Map Virtuals status codes AND strings to human-readable labels
STATUS_MAP = {
    5: "Sentient",
    4: "Prototype",
    3: "Prototype",
    2: "Prototype",
    1: "Prototype",
    "AVAILABLE": "Sentient",
    "BONDING": "Prototype",
    "available": "Sentient",
    "bonding": "Prototype",
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
    # Virtuals API returns flat items (no "attributes" wrapper)
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

    # Social links — handle both uppercase and lowercase keys
    socials = attrs.get("socials") or {}
    if isinstance(socials, list):
        socials = {s.get("type", ""): s.get("url", "") for s in socials if isinstance(s, dict)}

    # Check both cases for social links, plus verified links
    verified = socials.get("VERIFIED_LINKS") or socials.get("verified_links") or {}

    linked_twitter = (
        socials.get("TWITTER", "") or
        socials.get("twitter", "") or
        verified.get("TWITTER", "") or
        attrs.get("twitter", "") or
        attrs.get("linkedTwitter", "")
    ) or ""
    linked_website = (
        (socials.get("USER_LINKS") or {}).get("WEBSITE", "") or
        socials.get("WEBSITE", "") or
        socials.get("website", "") or
        attrs.get("website", "") or
        attrs.get("linkedWebsite", "")
    ) or ""
    linked_telegram = (
        socials.get("TELEGRAM", "") or
        socials.get("telegram", "") or
        verified.get("TELEGRAM", "") or
        attrs.get("telegram", "") or
        attrs.get("linkedTelegram", "")
    ) or ""

    # Market data — use actual API field names
    market_cap = float(attrs.get("mcapInVirtual") or attrs.get("marketCap") or 0)
    price_usd = float(attrs.get("currentPrice") or attrs.get("priceUsd") or 0)

    # Image
    image_obj = attrs.get("image") or {}
    if isinstance(image_obj, dict):
        image_url = image_obj.get("url") or ""
    else:
        image_url = str(image_obj or "")

    # Creation date
    created_at_raw = attrs.get("createdAt") or ""

    # Creator wallet — CRITICAL: avoid taking the full creator object (dict)
    raw_creator = attrs.get("creatorWallet") or attrs.get("walletAddress") or ""
    if isinstance(raw_creator, dict):
        creator_wallet = ""
    else:
        creator_wallet = str(raw_creator)

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
        "volume_24h": float(attrs.get("volume24h") or 0),
        "volume_6h": 0.0,
        "price_usd": price_usd,
        "price_change_24h": float(attrs.get("priceChangePercent24h") or 0),
        "liquidity_usd": float(attrs.get("liquidityUsd") or 0),
        "tx_count_24h": 0,
        "buy_sell_ratio": 1.0,
        "holder_count": int(attrs.get("holderCount") or attrs.get("holders") or 0),
        "top_10_concentration": float(attrs.get("top10HolderPercentage") or 0),
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
        logger.info(f"First page: {len(items)} items received")
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
    Fetch ALL agents from Virtuals API and immediately save them to the database.
    DexScreener enrichment is NOT done here — call enrich_top_agents_dexscreener()
    separately after this completes so agents appear in the dashboard right away.
    """
    logger.info("Preloading ALL agents from Virtuals Protocol (this may take a while)...")

    agents = await fetch_all_agents()  # No page cap — fetches everything

    if not agents:
        logger.error("No agents fetched from Virtuals API — aborting preload")
        return 0

    # Sort by market cap so most important agents get lowest DB row IDs
    agents.sort(key=lambda a: a.get("market_cap", 0), reverse=True)
    logger.info(f"Fetched {len(agents)} agents, saving to database in bulk...")

    # Log a sample agent for debugging
    if agents:
        sample = agents[0]
        logger.info(f"Sample agent: id={sample.get('virtuals_id')}, name={sample.get('name')}, "
                     f"mcap={sample.get('market_cap')}, wallet={sample.get('creator_wallet')[:20] if sample.get('creator_wallet') else 'None'}...")

    stored = await bulk_upsert_agents(agents)

    logger.info(f"Preload complete: {stored}/{len(agents)} agents stored in database")
    return stored


async def enrich_top_agents_dexscreener(top_n: int = 100):
    """
    Enrich the top N agents (by market cap) with DexScreener market data.
    Call this AFTER preload_all_agents() so agents are already visible in the dashboard.
    Stays well within DexScreener rate limits by capping at top_n and pacing requests.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT virtuals_id, contract_address FROM agents
               WHERE contract_address IS NOT NULL AND contract_address != ''
               ORDER BY market_cap DESC LIMIT ?""",
            (top_n,)
        ) as cur:
            rows = [dict(r) for r in await cur.fetchall()]

    if not rows:
        logger.info("No agents with contract addresses found for DexScreener enrichment")
        return

    logger.info(f"Enriching top {len(rows)} agents with DexScreener data...")

    semaphore = asyncio.Semaphore(3)  # Conservative concurrency to avoid 429s

    async def enrich_one(row: dict):
        async with semaphore:
            try:
                dex = await fetch_dexscreener_data(row["contract_address"])
                if dex:
                    await update_market_data(row["virtuals_id"], dex)
            except Exception as e:
                logger.debug(f"DexScreener enrich failed for {row['virtuals_id']}: {e}")
            await asyncio.sleep(0.3)  # ~3 req/s sustained

    await asyncio.gather(*[enrich_one(r) for r in rows], return_exceptions=True)
    logger.info(f"DexScreener enrichment complete for top {len(rows)} agents")
