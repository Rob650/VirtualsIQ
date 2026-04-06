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

# Map Virtuals status codes/strings to human-readable labels
STATUS_MAP = {
    5: "Sentient",
    4: "Prototype",
    3: "Prototype",
    2: "Prototype",
    1: "Prototype",
    "AVAILABLE": "Sentient",
    "BONDING": "Prototype",
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

    # Social links — guard against null socials; API returns UPPERCASE keys
    socials = attrs.get("socials") or {}
    if isinstance(socials, list):
        socials = {s.get("type", ""): s.get("url", "") for s in socials if isinstance(s, dict)}

    linked_twitter = (
        socials.get("TWITTER") or socials.get("twitter") or
        attrs.get("twitter") or attrs.get("linkedTwitter") or ""
    )
    linked_website = (
        socials.get("WEBSITE") or socials.get("website") or
        attrs.get("website") or attrs.get("linkedWebsite") or ""
    )
    linked_telegram = (
        socials.get("TELEGRAM") or socials.get("telegram") or
        attrs.get("telegram") or attrs.get("linkedTelegram") or ""
    )

    # Market data — use Virtuals API fields first, fall back to legacy names
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
    # `creator` is a full JSON object in the API response — extract wallet from walletAddress instead
    _creator_wallet = attrs.get("walletAddress") or attrs.get("creatorWallet") or ""
    creator_wallet = _creator_wallet if isinstance(_creator_wallet, str) else ""
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


async def fetch_dexscreener_batch(addresses: list[str]) -> dict[str, dict]:
    """
    Fetch DexScreener data for up to 30 contract addresses in a single request.
    DexScreener supports comma-separated token addresses (max 30 per call).
    Returns a dict mapping contract_address (lowercase) -> market data dict.
    """
    if not addresses:
        return {}

    joined = ",".join(addresses[:30])
    url = f"{DEXSCREENER_API}/{joined}"

    for attempt in range(3):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=HEADERS, timeout=20.0)
                resp.raise_for_status()
                data = resp.json()

            pairs = data.get("pairs") or []
            result: dict[str, dict] = {}

            for pair in pairs:
                base_addr = (pair.get("baseToken") or {}).get("address", "").lower()
                if not base_addr:
                    continue

                liq = float((pair.get("liquidity") or {}).get("usd", 0) or 0)
                existing = result.get(base_addr)
                # Keep the pair with highest liquidity for each token
                if existing is not None and liq <= existing.get("_liq", 0):
                    continue

                volume = pair.get("volume") or {}
                txns = pair.get("txns") or {}
                h24 = txns.get("h24") or {}
                buys = int(h24.get("buys", 0) or 0)
                sells = int(h24.get("sells", 0) or 0)
                buy_sell_ratio = (buys / sells) if sells > 0 else (1.0 if buys == 0 else 2.0)
                price_change = pair.get("priceChange") or {}

                result[base_addr] = {
                    "_liq": liq,
                    "price_usd": float(pair.get("priceUsd", 0) or 0),
                    "price_change_24h": float(price_change.get("h24", 0) or 0),
                    "volume_24h": float(volume.get("h24", 0) or 0),
                    "volume_6h": float(volume.get("h6", 0) or 0),
                    "liquidity_usd": liq,
                    "tx_count_24h": buys + sells,
                    "buy_sell_ratio": round(buy_sell_ratio, 2),
                    "market_cap": float(pair.get("marketCap") or pair.get("fdv") or 0),
                }

            # Remove internal tracking key
            for addr in result:
                result[addr].pop("_liq", None)

            return result

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                await asyncio.sleep(2 ** (attempt + 1))
            else:
                logger.debug(f"DexScreener batch error: {e}")
                return {}
        except Exception as e:
            logger.debug(f"DexScreener batch error: {e}")
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

    # Sort by market cap so most important agents get lowest DB row IDs
    agents.sort(key=lambda a: a.get("market_cap", 0), reverse=True)
    logger.info(f"Fetched {len(agents)} agents, saving to database in bulk...")

    stored = await bulk_upsert_agents(agents)

    logger.info(f"Preload complete: {stored}/{len(agents)} agents stored in database")
    return stored


async def enrich_top_agents_dexscreener(top_n: int = None):
    """
    Enrich ALL agents with contract addresses with DexScreener market data.
    Uses batched requests (30 addresses per call) to minimise API traffic.
    Pass top_n to limit to the top N agents by market cap (useful for quick refreshes).
    """
    query = """SELECT virtuals_id, contract_address FROM agents
               WHERE contract_address IS NOT NULL AND contract_address != ''
               ORDER BY market_cap DESC"""
    params: tuple = ()
    if top_n:
        query += " LIMIT ?"
        params = (top_n,)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(query, params) as cur:
            rows = [dict(r) for r in await cur.fetchall()]

    if not rows:
        logger.info("No agents with contract addresses found for DexScreener enrichment")
        return

    logger.info(f"Enriching {len(rows)} agents with DexScreener data (batched, 30 per request)...")

    BATCH_SIZE = 30
    enriched = 0
    total_batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        addresses = [r["contract_address"] for r in batch]
        addr_to_vid = {r["contract_address"].lower(): r["virtuals_id"] for r in batch}
        batch_num = i // BATCH_SIZE + 1

        try:
            dex_data = await fetch_dexscreener_batch(addresses)
            for addr, data in dex_data.items():
                vid = addr_to_vid.get(addr)
                if vid and data:
                    await update_market_data(vid, data)
                    enriched += 1
        except Exception as e:
            logger.debug(f"DexScreener batch {batch_num} failed: {e}")

        # Log progress every 50 batches (~1500 agents)
        if batch_num % 50 == 0:
            logger.info(f"DexScreener enrichment: {batch_num}/{total_batches} batches done ({enriched} enriched so far)")

        # ~2 batch requests/sec — well within DexScreener free tier limits
        if i + BATCH_SIZE < len(rows):
            await asyncio.sleep(0.5)

    logger.info(f"DexScreener enrichment complete: {enriched}/{len(rows)} agents enriched")
