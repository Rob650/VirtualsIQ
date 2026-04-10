"""
VirtualsIQ — Virtuals Protocol data ingestion
Fetches agents from Virtuals API and enriches with DexScreener market data
"""

import asyncio
import logging
from datetime import datetime

import httpx

from database import upsert_agent, get_existing_ids, bulk_upsert_agents, update_market_data, update_holder_count, _db, DB_PATH

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

# Contracts where the Virtuals API mcapInVirtual field is known to be wrong.
# For these, we zero out the Virtuals-sourced MC so DexScreener can supply the
# real value (or it stays 0 rather than showing a wildly inflated number).
_VIRTUALS_MC_SKIP_CONTRACTS: set[str] = {
    "0xb7f98fc88ee269642092275b49e3cc8ca006852a",  # 365love — Virtuals API returns ~$497M, clearly wrong
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


# Raw Virtuals internal type codes that carry no user-facing meaning
_RAW_VIRTUALS_CODES = {
    "acp_launch", "ip mirror", "ip_mirror", "x_launch", "functional",
    "acp launch", "ipmirror",
}

# Keyword → category mapping used when the raw type is not in AGENT_TYPE_MAP
_KW_CATS = [
    ("DeFi",          ["defi","swap","lend","yield","liquidity","amm","vault","stake","borrow","dex","pool","bridge","finance"]),
    ("Trading",       ["trade","signal","bot","alpha","snipe","copy","arbitrage","hedge","leverage","margin"]),
    ("Gaming",        ["game","play","quest","battle","arena","rpg","metaverse","world","land"]),
    ("Social",        ["social","chat","community","dao","governance","vote","forum","message","friend","connect"]),
    ("Entertainment", ["entertainment","music","art","meme","fun","comedy","video","stream","creator","content"]),
    ("Infra",         ["infra","protocol","sdk","api","oracle","node","chain","layer","bridge","index","analytics"]),
    ("NFT",           ["nft","collectible","pfp","mint","collection","generative"]),
    ("Info",          ["info","news","research","learn","education","wiki","guide","report","insight","intelligence"]),
]


def _infer_type_from_text(text: str) -> str:
    t = text.lower()
    for cat, kws in _KW_CATS:
        if any(kw in t for kw in kws):
            return cat
    return "Other"


def normalize_agent_type(raw: str, name: str = "", biography: str = "") -> str:
    if not raw:
        key = ""
    else:
        key = raw.lower().strip()

    # If it maps cleanly, use the map
    if key in AGENT_TYPE_MAP:
        return AGENT_TYPE_MAP[key]

    # If it's a raw internal Virtuals code, infer from name + biography
    if key in _RAW_VIRTUALS_CODES or not raw:
        text = f"{name} {biography}"
        return _infer_type_from_text(text)

    # Fallback: title-case the raw value (handles human-readable types we don't explicitly map)
    return raw.title()


async def _fetch_page(
    client: httpx.AsyncClient,
    page: int,
    page_size: int = 100,
    status_filter=None,
    sort: str = "volume24h:desc",
) -> dict:
    """Fetch a single page of agents from Virtuals API with retry."""
    url = VIRTUALS_API
    params = {
        "pagination[page]": page,
        "pagination[pageSize]": page_size,
        "sort[0]": sort,
    }
    if status_filter is not None:
        params["filters[status]"] = status_filter

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
    attrs = item.get("attributes") or item

    virtual_id = str(item.get("id") or attrs.get("id") or "")
    name = attrs.get("name") or "Unknown"
    ticker = attrs.get("symbol") or attrs.get("ticker") or ""
    contract = attrs.get("tokenAddress") or attrs.get("contractAddress") or ""
    status_code = attrs.get("status") or 4
    status = STATUS_MAP.get(status_code, "Prototype")

    biography = attrs.get("description") or attrs.get("bio") or attrs.get("biography") or ""
    category = attrs.get("category") or attrs.get("agentType") or attrs.get("role") or ""
    if isinstance(category, dict):
        category = category.get("name") or ""
    agent_type = normalize_agent_type(category, name=name, biography=biography)

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

    # Use DexScreener as the authoritative MC source for contracts in the skip set;
    # the Virtuals API field mcapInVirtual can be wildly wrong for some tokens.
    contract_lower = contract.lower()
    if contract_lower in _VIRTUALS_MC_SKIP_CONTRACTS:
        market_cap = 0.0  # will be populated (or kept 0) by DexScreener enrichment
    else:
        market_cap = float(attrs.get("mcapInVirtual") or attrs.get("marketCap") or 0)
    price_usd = float(attrs.get("currentPrice") or attrs.get("priceUsd") or 0)

    image_obj = attrs.get("image") or {}
    if isinstance(image_obj, dict):
        image_url = image_obj.get("url") or ""
    else:
        image_url = str(image_obj or "")

    created_at_raw = attrs.get("createdAt") or ""
    _creator_wallet = attrs.get("walletAddress") or attrs.get("creatorWallet") or ""
    creator_wallet = _creator_wallet if isinstance(_creator_wallet, str) else ""

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
        "overview_json": "{}",
        "first_mover": 0,
        "doxx_tier": 3,
        "last_scanned": None,
        "updated_at": datetime.utcnow().isoformat(),
    }


async def fetch_all_agents(max_pages: int = 500) -> list[dict]:
    """Paginate through the full Virtuals API and return all agents."""
    agents = []
    async with httpx.AsyncClient() as client:
        first = await _fetch_page(client, 1, 100, status_filter=None)
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

        page = 2
        while page <= total_pages:
            batch = range(page, min(page + 10, total_pages + 1))
            tasks = [_fetch_page(client, p, 100, status_filter=None) for p in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Page fetch error: {result}")
                    continue
                items = result.get("data", [])
                agents.extend([_parse_agent(item) for item in items])

            page += 10
            await asyncio.sleep(0.5)

    logger.info(f"Fetched {len(agents)} total agents from Virtuals Protocol")
    return agents


async def fetch_dexscreener_data(contract_address: str) -> dict:
    """Enrich agent with DexScreener market data."""
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
    """Fetch ALL current agents and return those not yet in the database."""
    current = await fetch_all_agents()
    new_agents = [a for a in current if a["virtuals_id"] not in existing_ids]
    if new_agents:
        logger.info(f"Detected {len(new_agents)} new agents")
    return current


async def preload_all_agents(on_batch=None):
    """Fetch ALL agents from Virtuals API and save progressively."""
    logger.info("Preloading ALL agents from Virtuals Protocol (progressive)...")
    total_stored = 0

    async with httpx.AsyncClient() as client:
        first = await _fetch_page(client, 1, 100, status_filter=None)
        if not first:
            logger.error("Failed to fetch first page from Virtuals API")
            return 0

        items = first.get("data", [])
        first_batch = [_parse_agent(item) for item in items]
        first_batch.sort(key=lambda a: a.get("market_cap", 0), reverse=True)

        stored = await bulk_upsert_agents(first_batch)
        total_stored += stored
        if on_batch:
            on_batch(total_stored)
        logger.info(f"First page saved: {stored} agents now visible in dashboard")

        pagination = first.get("meta", {}).get("pagination", {})
        total_pages = min(pagination.get("pageCount", 1), 500)
        logger.info(f"Virtuals API: {pagination.get('total', '?')} total agents, {total_pages} pages to fetch")

        page = 2
        while page <= total_pages:
            batch_range = range(page, min(page + 10, total_pages + 1))
            tasks = [_fetch_page(client, p, 100, status_filter=None) for p in batch_range]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            batch_agents = []
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Page fetch error: {result}")
                    continue
                batch_agents.extend([_parse_agent(item) for item in result.get("data", [])])

            if batch_agents:
                stored = await bulk_upsert_agents(batch_agents)
                total_stored += stored
                if on_batch:
                    on_batch(total_stored)

            page += 10
            await asyncio.sleep(0.5)

    logger.info(f"Preload complete: {total_stored} agents stored in database")
    return total_stored


async def enrich_top_agents_dexscreener(top_n: int = 100):
    """Enrich the top N agents (by market cap) with DexScreener market data.

    Agents in _VIRTUALS_MC_SKIP_CONTRACTS are always included regardless of
    their stored market_cap (which may be 0 after the Virtuals MC was zeroed out).
    """
    async with _db() as db:
        top_rows = await db.fetch_all(
            """SELECT virtuals_id, contract_address FROM agents
               WHERE contract_address IS NOT NULL AND contract_address != ''
               ORDER BY market_cap DESC LIMIT ?""",
            (top_n,)
        )
        # Always include the MC-override contracts so they get a DexScreener update
        override_rows = []
        for contract_addr in _VIRTUALS_MC_SKIP_CONTRACTS:
            row = await db.fetch_one(
                "SELECT virtuals_id, contract_address FROM agents WHERE LOWER(contract_address) = ?",
                (contract_addr,)
            )
            if row:
                override_rows.append(row)

    # Merge, deduplicating by virtuals_id
    seen = {r["virtuals_id"] for r in top_rows}
    rows = list(top_rows) + [r for r in override_rows if r["virtuals_id"] not in seen]

    if not rows:
        logger.info("No agents with contract addresses found for DexScreener enrichment")
        return

    logger.info(f"Enriching top {len(rows)} agents with DexScreener data...")

    semaphore = asyncio.Semaphore(3)

    async def enrich_one(row: dict):
        async with semaphore:
            try:
                dex = await fetch_dexscreener_data(row["contract_address"])
                if dex:
                    await update_market_data(row["virtuals_id"], dex)
            except Exception as e:
                logger.debug(f"DexScreener enrich failed for {row['virtuals_id']}: {e}")
            await asyncio.sleep(0.3)

    await asyncio.gather(*[enrich_one(r) for r in rows], return_exceptions=True)
    logger.info(f"DexScreener enrichment complete for top {len(rows)} agents")


async def refresh_holder_counts(limit: int = 100):
    """Refresh holder counts for top agents by re-fetching from Virtuals API."""
    async with _db() as db:
        rows = await db.fetch_all(
            "SELECT virtuals_id FROM agents ORDER BY market_cap DESC LIMIT ?",
            (limit,)
        )

    if not rows:
        return 0

    updated = 0
    async with httpx.AsyncClient() as client:
        for row in rows:
            try:
                resp = await client.get(
                    f"{VIRTUALS_API}/{row['virtuals_id']}",
                    headers=HEADERS, timeout=15.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    attrs = data.get("data", {}).get("attributes", data.get("data", {}))
                    holder_count = int(attrs.get("holderCount") or attrs.get("holders") or 0)
                    if holder_count > 0:
                        await update_holder_count(row["virtuals_id"], holder_count)
                        updated += 1
                await asyncio.sleep(0.2)
            except Exception as e:
                logger.debug(f"Holder refresh failed for {row['virtuals_id']}: {e}")

    logger.info(f"Holder count refresh: updated {updated}/{len(rows)} agents")
    return updated


async def scan_newest_agents(pages: int = 5) -> int:
    """Quick scan: fetch newest agents and upsert new ones."""
    existing_ids = await get_existing_ids()
    new_count = 0

    async with httpx.AsyncClient() as client:
        for page in range(1, pages + 1):
            result = await _fetch_page(client, page, 100, status_filter=None, sort="createdAt:desc")
            if not result:
                break

            items = result.get("data", [])
            if not items:
                break

            agents = [_parse_agent(item) for item in items]
            new_agents = [a for a in agents if a["virtuals_id"] not in existing_ids]

            if new_agents:
                await bulk_upsert_agents(new_agents)
                for a in new_agents:
                    existing_ids.add(a["virtuals_id"])
                new_count += len(new_agents)
                logger.info(f"scan_newest_agents page {page}: {len(new_agents)} new agents upserted")
            else:
                logger.info(f"scan_newest_agents: no new agents on page {page}, stopping early")
                break

    logger.info(f"scan_newest_agents complete: {new_count} new agents found")
    return new_count


async def get_api_total_count() -> int:
    """Fetch page 1 with pageSize=1 to retrieve the total agent count."""
    async with httpx.AsyncClient() as client:
        result = await _fetch_page(client, 1, 1, status_filter=None)
    total = result.get("meta", {}).get("pagination", {}).get("total", 0)
    return int(total)
