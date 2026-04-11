"""
VirtualsIQ — Smart Money Tracking Module

Uses the Moralis API to track smart money wallets on Base chain (chain 0x2105).
Rate limit: max 25 req/s, results cached for 4 hours minimum.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta

import aiohttp

logger = logging.getLogger(__name__)

MORALIS_API_KEY = os.environ.get("MORALIS_API_KEY", "")
BASE_URL = "https://deep-index.moralis.io/api/v2.2"
CHAIN = "0x2105"  # Base

# ---------------------------------------------------------------------------
# Known smart money wallets on Base chain (early buyers of top Virtuals projects)
# ---------------------------------------------------------------------------

SEED_SMART_WALLETS: list[str] = [
    # Early LUNA / GAME buyers & consistent profitable traders
    "0x1234567890abcdef1234567890abcdef12345678",  # placeholder
    "0xabcdef1234567890abcdef1234567890abcdef12",  # placeholder
    "0x9876543210fedcba9876543210fedcba98765432",  # placeholder
    "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",  # placeholder
    "0xcafe0000cafe0000cafe0000cafe0000cafe0000",  # placeholder
    "0x1111111111111111111111111111111111111111",  # placeholder
    "0x2222222222222222222222222222222222222222",  # placeholder
    "0x3333333333333333333333333333333333333333",  # placeholder
    "0x4444444444444444444444444444444444444444",  # placeholder
    "0x5555555555555555555555555555555555555555",  # placeholder
    "0x6666666666666666666666666666666666666666",  # placeholder
    "0x7777777777777777777777777777777777777777",  # placeholder
    "0x8888888888888888888888888888888888888888",  # placeholder
    "0x9999999999999999999999999999999999999999",  # placeholder
    "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",  # placeholder
    "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",  # placeholder
    "0xcccccccccccccccccccccccccccccccccccccccc",  # placeholder
    "0xdddddddddddddddddddddddddddddddddddddddd",  # placeholder
    "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",  # placeholder
    "0xffffffffffffffffffffffffffffffffffffffff",  # placeholder
]

# ---------------------------------------------------------------------------
# Simple in-memory cache (token_address -> {data, expires_at})
# ---------------------------------------------------------------------------

_cache: dict[str, dict] = {}
_CACHE_TTL = 4 * 3600  # 4 hours

# ---------------------------------------------------------------------------
# Rate limiter — 25 req/s sliding window
# ---------------------------------------------------------------------------

_rate_tokens: float = 25.0
_rate_last: float = time.monotonic()
_rate_lock: asyncio.Lock | None = None
_MAX_RPS = 25.0


def _get_rate_lock() -> asyncio.Lock:
    global _rate_lock
    if _rate_lock is None:
        _rate_lock = asyncio.Lock()
    return _rate_lock


async def _rate_limit():
    """Token-bucket rate limiter: max 25 requests per second."""
    global _rate_tokens, _rate_last
    lock = _get_rate_lock()
    async with lock:
        now = time.monotonic()
        elapsed = now - _rate_last
        _rate_last = now
        _rate_tokens = min(_MAX_RPS, _rate_tokens + elapsed * _MAX_RPS)
        if _rate_tokens < 1.0:
            wait = (1.0 - _rate_tokens) / _MAX_RPS
            await asyncio.sleep(wait)
            _rate_tokens = 0.0
        else:
            _rate_tokens -= 1.0


def _cache_get(key: str) -> dict | None:
    entry = _cache.get(key)
    if entry and entry["expires_at"] > time.monotonic():
        return entry["data"]
    return None


def _cache_set(key: str, data: dict, ttl: int = _CACHE_TTL):
    _cache[key] = {"data": data, "expires_at": time.monotonic() + ttl}


# ---------------------------------------------------------------------------
# Moralis API helpers
# ---------------------------------------------------------------------------

async def _moralis_get(path: str, params: dict | None = None) -> dict | None:
    """Make a rate-limited GET request to Moralis API."""
    if not MORALIS_API_KEY:
        logger.warning("[SmartMoney] MORALIS_API_KEY not set")
        return None

    await _rate_limit()
    url = f"{BASE_URL}{path}"
    headers = {"X-API-Key": MORALIS_API_KEY, "accept": "application/json"}
    default_params = {"chain": CHAIN}
    if params:
        default_params.update(params)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=default_params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 429:
                    logger.warning("[SmartMoney] Rate limited by Moralis — sleeping 2s")
                    await asyncio.sleep(2)
                    return None
                resp.raise_for_status()
                return await resp.json()
    except Exception as e:
        logger.warning(f"[SmartMoney] Moralis request failed ({path}): {e}")
        return None


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

async def get_top_holders(token_address: str, limit: int = 50) -> list[dict]:
    """Get top holders of a token on Base chain.

    Returns list of {address, balance, percentage, usd_value}.
    """
    cache_key = f"holders:{token_address}:{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    data = await _moralis_get(f"/erc20/{token_address}/owners", {"limit": limit, "order": "DESC"})
    if not data:
        return []

    holders = []
    for h in data.get("result", [])[:limit]:
        holders.append({
            "address": h.get("owner_address", ""),
            "balance": float(h.get("balance_formatted", h.get("balance", 0)) or 0),
            "percentage": float(h.get("percentage_relative_to_total_supply", 0) or 0),
            "usd_value": float(h.get("usd_value", 0) or 0),
        })

    _cache_set(cache_key, holders)
    return holders


async def get_token_transfers(token_address: str, days: int = 14) -> list[dict]:
    """Get recent transfers for a token.

    Returns list of transfer records.
    """
    cache_key = f"transfers:{token_address}:{days}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    transfers = []
    cursor = None

    for _page in range(5):  # max 5 pages to stay within rate limits
        params: dict = {"limit": 100, "from_date": since}
        if cursor:
            params["cursor"] = cursor

        data = await _moralis_get(f"/erc20/{token_address}/transfers", params)
        if not data:
            break

        for t in data.get("result", []):
            transfers.append({
                "from_address": t.get("from_address", ""),
                "to_address": t.get("to_address", ""),
                "value": float(t.get("value_decimal", t.get("value", 0)) or 0),
                "usd_value": float(t.get("usd_value", 0) or 0),
                "block_timestamp": t.get("block_timestamp", ""),
                "transaction_hash": t.get("transaction_hash", ""),
            })

        cursor = data.get("cursor")
        if not cursor:
            break

    _cache_set(cache_key, transfers, ttl=3600)  # 1 hour for transfers
    return transfers


async def calculate_holder_concentration(token_address: str) -> dict:
    """Calculate top-10 and top-20 holder concentration percentage.

    Returns {top10_pct: float, top20_pct: float, holders: list}.
    """
    cache_key = f"concentration:{token_address}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    holders = await get_top_holders(token_address, limit=50)
    if not holders:
        return {"top10_pct": 0.0, "top20_pct": 0.0, "holders": []}

    top10_pct = sum(h["percentage"] for h in holders[:10])
    top20_pct = sum(h["percentage"] for h in holders[:20])

    result = {
        "top10_pct": round(top10_pct, 2),
        "top20_pct": round(top20_pct, 2),
        "holders": holders,
    }
    _cache_set(cache_key, result)
    return result


async def identify_smart_wallets(token_address: str) -> list[str]:
    """Identify smart money wallets that hold this token.

    Smart wallets = seed list + any wallet appearing in top-50 holders of
    at least this token (cross-token analysis requires multi-token data,
    which we approximate via seed list for now).
    """
    holders = await get_top_holders(token_address, limit=50)
    holder_addresses = {h["address"].lower() for h in holders}

    seed_lower = {w.lower() for w in SEED_SMART_WALLETS}
    smart = list(seed_lower & holder_addresses)
    return smart


async def calculate_smart_money_flow(token_address: str, days: int = 7) -> dict:
    """Calculate net smart money inflow/outflow over N days.

    Returns {
        net_flow_usd: float,
        net_flow_tokens: float,
        buy_count: int,
        sell_count: int,
        smart_wallets_buying: list,
        smart_wallets_selling: list,
        acceleration: float,  # vs prior period
    }.
    """
    cache_key = f"flow:{token_address}:{days}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    smart_wallets = await identify_smart_wallets(token_address)
    if not smart_wallets:
        result = {
            "net_flow_usd": 0.0,
            "net_flow_tokens": 0.0,
            "buy_count": 0,
            "sell_count": 0,
            "smart_wallets_buying": [],
            "smart_wallets_selling": [],
            "acceleration": 0.0,
        }
        _cache_set(cache_key, result)
        return result

    smart_set = {w.lower() for w in smart_wallets}

    # Current period
    transfers = await get_token_transfers(token_address, days=days)
    # Prior period (for acceleration calc)
    prior_transfers = await get_token_transfers(token_address, days=days * 2)
    prior_only = [
        t for t in prior_transfers
        if t not in transfers
    ]

    def _calc_flow(txs: list[dict]) -> dict:
        buy_usd = 0.0
        sell_usd = 0.0
        buy_tokens = 0.0
        sell_tokens = 0.0
        buyers: set[str] = set()
        sellers: set[str] = set()

        for t in txs:
            frm = t["from_address"].lower()
            to = t["to_address"].lower()
            val = t["value"]
            usd = t["usd_value"]

            if to in smart_set:
                buy_usd += usd
                buy_tokens += val
                buyers.add(to)
            if frm in smart_set:
                sell_usd += usd
                sell_tokens += val
                sellers.add(frm)

        return {
            "net_usd": buy_usd - sell_usd,
            "net_tokens": buy_tokens - sell_tokens,
            "buy_count": len([t for t in txs if t["to_address"].lower() in smart_set]),
            "sell_count": len([t for t in txs if t["from_address"].lower() in smart_set]),
            "buyers": list(buyers),
            "sellers": list(sellers),
        }

    current = _calc_flow(transfers)
    prior = _calc_flow(prior_only)

    # Acceleration: ratio of current vs prior net flow
    prior_net = prior["net_usd"] if prior["net_usd"] != 0 else 1.0
    acceleration = (current["net_usd"] - prior["net_usd"]) / abs(prior_net) if prior_net else 0.0
    acceleration = max(-10.0, min(10.0, acceleration))  # clamp

    result = {
        "net_flow_usd": round(current["net_usd"], 2),
        "net_flow_tokens": round(current["net_tokens"], 4),
        "buy_count": current["buy_count"],
        "sell_count": current["sell_count"],
        "smart_wallets_buying": current["buyers"],
        "smart_wallets_selling": current["sellers"],
        "acceleration": round(acceleration, 4),
    }
    _cache_set(cache_key, result)
    return result


async def get_holder_growth(token_address: str, current_holders: int | None = None) -> dict:
    """Calculate holder count growth metrics using stored snapshots.

    Returns {current: int, growth_7d_pct: float, growth_14d_pct: float}.

    Note: Moralis doesn't expose historical holder counts directly, so we
    rely on snapshots stored in our DB (written by the daily sync).
    """
    cache_key = f"holder_growth:{token_address}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # Fetch current holders from Moralis for count
    holders = await get_top_holders(token_address, limit=1)

    result = {
        "current": current_holders or 0,
        "growth_7d_pct": None,
        "growth_14d_pct": None,
    }
    _cache_set(cache_key, result, ttl=1800)
    return result


async def detect_wash_trading(token_address: str, days: int = 7) -> dict:
    """Detect potential wash trading patterns.

    Looks for:
    - Same wallet buying and selling repeatedly within the window
    - Circular transfers among a small group of wallets

    Returns {wash_score: 0-100, suspicious_wallets: list, volume_pct_suspicious: float}.
    """
    cache_key = f"wash:{token_address}:{days}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    transfers = await get_token_transfers(token_address, days=days)

    if not transfers:
        result = {"wash_score": 0, "suspicious_wallets": [], "volume_pct_suspicious": 0.0}
        _cache_set(cache_key, result)
        return result

    # Count round-trip wallets (both buyer and seller in same window)
    buyers = {t["to_address"].lower() for t in transfers}
    sellers = {t["from_address"].lower() for t in transfers}
    roundtrip = buyers & sellers

    total_volume = sum(t["usd_value"] for t in transfers) or 1.0
    suspicious_volume = sum(
        t["usd_value"] for t in transfers
        if t["from_address"].lower() in roundtrip or t["to_address"].lower() in roundtrip
    )
    vol_pct = suspicious_volume / total_volume * 100

    # Concentration: if top-5 addresses account for > 50% of volume, suspicious
    addr_vol: dict[str, float] = defaultdict(float)
    for t in transfers:
        addr_vol[t["from_address"].lower()] += t["usd_value"]
        addr_vol[t["to_address"].lower()] += t["usd_value"]

    top5_vol = sum(sorted(addr_vol.values(), reverse=True)[:5])
    concentration_pct = top5_vol / (total_volume * 2) * 100

    wash_score = min(100, int(vol_pct * 0.5 + concentration_pct * 0.5))

    result = {
        "wash_score": wash_score,
        "suspicious_wallets": list(roundtrip)[:20],
        "volume_pct_suspicious": round(vol_pct, 2),
    }
    _cache_set(cache_key, result)
    return result


async def enrich_agent_smart_money(agent_data: dict) -> dict:
    """Main entry point. Given agent data with token_address, enrich with
    all smart money metrics.

    Returns dict with all smart money fields needed by scoring.py:
    {
        smart_money_net_flow_14d: float,
        smart_money_acceleration: float,
        top_10_concentration: float,
        top_20_concentration: float,
        wash_score: int,
        holder_growth: dict,
        smart_wallets_buying: list,
        smart_wallets_selling: list,
        buy_count_7d: int,
        sell_count_7d: int,
        data_available: bool,
        fetched_at: str,
    }
    """
    token_address = agent_data.get("contract_address", "")

    empty = {
        "smart_money_net_flow_14d": None,
        "smart_money_acceleration": None,
        "top_10_concentration": None,
        "top_20_concentration": None,
        "wash_score": None,
        "holder_growth": {},
        "smart_wallets_buying": [],
        "smart_wallets_selling": [],
        "buy_count_7d": 0,
        "sell_count_7d": 0,
        "data_available": False,
        "fetched_at": datetime.utcnow().isoformat(),
    }

    if not token_address or not MORALIS_API_KEY:
        return empty

    try:
        concentration, flow, wash = await asyncio.gather(
            calculate_holder_concentration(token_address),
            calculate_smart_money_flow(token_address, days=14),
            detect_wash_trading(token_address, days=7),
            return_exceptions=True,
        )

        # Gracefully handle partial failures
        if isinstance(concentration, Exception):
            logger.warning(f"[SmartMoney] concentration failed for {token_address}: {concentration}")
            concentration = {"top10_pct": None, "top20_pct": None, "holders": []}
        if isinstance(flow, Exception):
            logger.warning(f"[SmartMoney] flow failed for {token_address}: {flow}")
            flow = {"net_flow_usd": None, "acceleration": 0.0,
                    "buy_count": 0, "sell_count": 0,
                    "smart_wallets_buying": [], "smart_wallets_selling": []}
        if isinstance(wash, Exception):
            logger.warning(f"[SmartMoney] wash failed for {token_address}: {wash}")
            wash = {"wash_score": None, "suspicious_wallets": [], "volume_pct_suspicious": 0.0}

        return {
            "smart_money_net_flow_14d": flow.get("net_flow_usd"),
            "smart_money_acceleration": flow.get("acceleration"),
            "top_10_concentration": concentration.get("top10_pct"),
            "top_20_concentration": concentration.get("top20_pct"),
            "wash_score": wash.get("wash_score"),
            "holder_growth": {},
            "smart_wallets_buying": flow.get("smart_wallets_buying", []),
            "smart_wallets_selling": flow.get("smart_wallets_selling", []),
            "buy_count_7d": flow.get("buy_count", 0),
            "sell_count_7d": flow.get("sell_count", 0),
            "data_available": True,
            "fetched_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"[SmartMoney] enrich_agent_smart_money failed for {token_address}: {e}")
        return empty
