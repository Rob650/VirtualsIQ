"""
VirtualsIQ — FastAPI Application
Bloomberg Terminal for the Virtuals Protocol ecosystem
"""

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from analyzer import analyze_agent, batch_triage
from database import (
    bulk_score_agents,
    get_agent_detail,
    get_agent_holders,
    get_agents,
    get_agents_for_backfill,
    get_agents_needing_reanalysis,
    get_category_summary,
    get_existing_ids,
    get_holders_last_updated,
    get_stats,
    get_top_agent_ids,
    get_trending_agents,
    get_trending_strip,
    init_db,
    search_agents,
    update_agent_category,
    update_agent_scores,
    upsert_agent,
    upsert_agent_holders,
)
from virtuals_ingestion import (
    detect_new_agents,
    enrich_top_agents_dexscreener,
    fetch_dexscreener_data,
    get_api_total_count,
    preload_all_agents,
    refresh_holder_counts,
    scan_newest_agents,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category inference helpers
# ---------------------------------------------------------------------------

GENERIC_CATEGORIES = {"IP", "Unknown", "", None}

CAT_KEYWORDS = {
    "DeFi": ["defi", "swap", "lend", "yield", "liquidity", "amm", "vault", "stake", "borrow", "finance", "dex", "pool", "bridge"],
    "Gaming": ["game", "play", "quest", "battle", "arena", "nft game", "rpg", "metaverse", "world", "land"],
    "Social": ["social", "chat", "community", "dao", "governance", "vote", "forum", "message", "friend", "connect"],
    "Trading": ["trade", "signal", "bot", "alpha", "snipe", "copy", "arbitrage", "hedge", "leverage", "margin"],
    "Infra": ["infra", "protocol", "sdk", "api", "oracle", "node", "chain", "layer", "bridge", "index", "data", "analytics"],
    "Info": ["info", "news", "research", "learn", "education", "wiki", "guide", "report", "insight", "intelligence"],
    "Entertainment": ["entertainment", "music", "art", "meme", "fun", "comedy", "video", "stream", "creator", "content"],
    "NFT": ["nft", "collectible", "pfp", "mint", "collection", "generative", "art"],
}


def _infer_category(agent: dict) -> str:
    current = agent.get("agent_type", "")
    if current and current not in GENERIC_CATEGORIES:
        return current

    text = " ".join([
        str(agent.get("name", "")),
        str(agent.get("biography", "")),
        str(agent.get("ticker", "")),
    ]).lower()

    for cat, keywords in CAT_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return cat
    return current or "Other"


# ---------------------------------------------------------------------------
# Enrichment state tracker
# ---------------------------------------------------------------------------

enrichment_state: dict = {
    "status": "idle",
    "agents_loaded": 0,
    "started_at": None,
    "completed_at": None,
    "api_total": 0,
    "last_new_scan": None,
    "new_agents_found": 0,
    "last_price_refresh": None,
    "last_holder_refresh": None,
    "last_description_refresh": None,
}

# ---------------------------------------------------------------------------
# In-memory job tracker
# ---------------------------------------------------------------------------

jobs: dict[str, dict] = {}


def create_job(agent_id: str) -> str:
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "job_id": job_id,
        "agent_id": agent_id,
        "status": "queued",
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "error": None,
    }
    return job_id


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------

async def _run_analysis_job(job_id: str, virtuals_id: str):
    """Background: fetch agent, analyze, store scores."""
    jobs[job_id]["status"] = "running"
    try:
        agent = await get_agent_detail(virtuals_id)
        if not agent:
            raise ValueError(f"Agent {virtuals_id} not found in database")

        if agent.get("contract_address"):
            dex = await fetch_dexscreener_data(agent["contract_address"])
            if dex:
                agent.update(dex)
                await upsert_agent(agent)

        # Get top IDs for model selection
        top_ids = set(await get_top_agent_ids(50))

        result = await analyze_agent(agent, top_ids=top_ids)
        scores = result["scores"]
        analysis = result["analysis"]
        overview = result.get("overview", {})
        prediction_json = result.get("prediction", {})

        # Infer category
        inferred_category = _infer_category(agent)
        if inferred_category != agent.get("agent_type"):
            await update_agent_category(virtuals_id, inferred_category)

        await update_agent_scores(
            virtuals_id=virtuals_id,
            composite_score=scores["composite_score"],
            tier=scores["tier_classification"],
            scores_json=scores["scores"],
            analysis_json=analysis,
            prediction_json=prediction_json,
            overview_json=overview,
            first_mover=scores["first_mover"],
            doxx_tier=int(analysis.get("team", {}).get("doxx_tier", 3)),
        )

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        logger.info(f"Analysis complete for {virtuals_id}: score={scores['composite_score']}")

    except Exception as e:
        logger.error(f"Analysis job {job_id} failed: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()


async def _auto_analyze_all(force: bool = False):
    """
    Background task: run AI analysis on agents that need it.
    - Top 100 by market cap: full analyze_agent, 1 at a time with 3s sleep between
    - Remaining: batch_triage in batches of 10, with 5s sleep between batches
    - Skips agents analyzed within 7 days unless force=True.
    - Covers up to 50000 agents per run.
    """
    try:
        logger.info(f"Auto-analysis starting (force={force})...")
        enrichment_state["status"] = "analyzing"

        top_ids_list = await get_top_agent_ids(100)
        top_ids_set = set(top_ids_list)

        # Semaphore: 1 at a time — server stays responsive
        sem = asyncio.Semaphore(1)

        async def _analyze_one(virtuals_id: str, force: bool = False):
            async with sem:
                try:
                    agent = await get_agent_detail(virtuals_id)
                    if not agent:
                        return
                    # Skip if analyzed within the last 7 days (unless forced)
                    if not force:
                        last = agent.get("last_analyzed")
                        if last:
                            try:
                                age = (datetime.utcnow() - datetime.fromisoformat(last)).total_seconds()
                                if age < 7 * 24 * 3600:
                                    logger.info(f"Skipping {agent.get('name')} — analyzed {age/3600:.1f}h ago")
                                    return
                            except Exception:
                                pass
                    result = await analyze_agent(agent, top_ids=top_ids_set)
                    scores = result["scores"]
                    analysis = result["analysis"]
                    overview = result.get("overview", {})
                    prediction_json = result.get("prediction", {})

                    inferred_category = _infer_category(agent)
                    if inferred_category != agent.get("agent_type"):
                        await update_agent_category(virtuals_id, inferred_category)

                    await update_agent_scores(
                        virtuals_id=virtuals_id,
                        composite_score=scores["composite_score"],
                        tier=scores["tier_classification"],
                        scores_json=scores["scores"],
                        analysis_json=analysis,
                        prediction_json=prediction_json,
                        overview_json=overview,
                        first_mover=scores["first_mover"],
                        doxx_tier=int(analysis.get("team", {}).get("doxx_tier", 3)),
                    )
                    logger.info(f"Auto-analyzed {agent.get('name')}: score={scores['composite_score']}")
                    await asyncio.sleep(3)
                except Exception as e:
                    logger.error(f"Auto-analysis failed for {virtuals_id}: {e}")

        # Full analysis for top 100, one at a time
        for vid in top_ids_list:
            await _analyze_one(vid, force=force)
        logger.info(f"Full analysis complete for top {len(top_ids_list)} agents")

        # Batch triage for remaining agents — up to 50000, batches of 10
        remaining = await get_agents_needing_reanalysis(limit=50000)
        remaining = [a for a in remaining if str(a["virtuals_id"]) not in top_ids_set]

        for i in range(0, len(remaining), 10):
            batch = remaining[i:i + 10]
            batch_by_vid = {str(a.get("virtuals_id", "")): a for a in batch}
            try:
                results = await batch_triage(batch)
                for r in results:
                    vid = r["virtuals_id"]
                    s = r["scores"]
                    agent_data = batch_by_vid.get(vid, {})
                    key_risks = r["analysis"].get("risk", {}).get("key_risks", [])
                    team_data = r["analysis"].get("team", {})
                    market_data = r["analysis"].get("market", {})
                    product_info = r["analysis"].get("product", {})
                    doxx_tier = int(team_data.get("doxx_tier", 3))
                    doxx_label = {1: "a fully doxxed team", 2: "a pseudonymous team with social presence", 3: "an anonymous team"}[doxx_tier]
                    agent_name = agent_data.get("name", "This agent")
                    agent_type = agent_data.get("agent_type") or "AI Agent"
                    holder_count = agent_data.get("holder_count") or 0
                    mcap = agent_data.get("market_cap") or 0
                    product_status = product_info.get("status", "unknown")
                    basic_overview = {
                        "what_it_does": agent_data.get("biography", ""),
                        "who_is_behind_it": team_data.get("team_summary", "") or (
                            f"{agent_name} is backed by {doxx_label}. "
                            f"The project has {holder_count:,} token holders. "
                            f"Full team background and credential analysis is pending a comprehensive deep-dive."
                        ),
                        "what_is_notable": (
                            f"Product status: {product_status}. "
                            f"{agent_name} is active in the {agent_type} vertical on Virtuals Protocol"
                            f"{f' with a market cap of ${mcap:,.0f}' if mcap else ''}. "
                            f"Detailed intelligence notes including partnerships and milestones are pending full analysis."
                        ),
                        "risks_to_monitor": (
                            ". ".join(key_risks) + "." if key_risks else
                            r["analysis"].get("risk", {}).get("bear_case", "") or
                            f"Risk factors for {agent_name} are pending full analysis. "
                            f"Key areas to investigate include team transparency, liquidity depth, and product execution."
                        ),
                        "market_opportunity": market_data.get("tam_description", "") or (
                            f"{agent_name} operates in the {agent_type} vertical on Virtuals Protocol. "
                            f"Full market opportunity analysis including TAM, competitive landscape, "
                            f"and growth projections is pending a comprehensive research pass."
                        ),
                    }
                    try:
                        await update_agent_scores(
                            virtuals_id=vid,
                            composite_score=s["composite_score"],
                            tier=s["tier_classification"],
                            scores_json=s["scores"],
                            analysis_json=r["analysis"],
                            prediction_json={},
                            overview_json=basic_overview,
                            first_mover=s["first_mover"],
                            doxx_tier=int(r["analysis"].get("team", {}).get("doxx_tier", 3)),
                        )
                    except Exception as e:
                        logger.error(f"Score update failed for {vid}: {e}")
                logger.info(f"Batch triage: {min(i + 10, len(remaining))}/{len(remaining)} agents")
            except Exception as e:
                logger.error(f"Batch triage failed for batch {i}: {e}")
            await asyncio.sleep(5)

        enrichment_state["status"] = "complete"
        logger.info("Auto-analysis complete for all agents")
    except Exception as e:
        logger.error(f"Auto-analyze-all failed: {e}")
        enrichment_state["status"] = "complete"


# ---------------------------------------------------------------------------
# Tiered refresh loops
# ---------------------------------------------------------------------------

async def _price_refresh_loop():
    """Tier 1: Price/MC/24h change every 3 minutes for top 100 agents."""
    while True:
        try:
            await asyncio.sleep(3 * 60)
            logger.info("Price refresh: enriching top 100 via DexScreener...")
            await enrich_top_agents_dexscreener(top_n=100)
            enrichment_state["last_price_refresh"] = datetime.utcnow().isoformat()
            logger.info("Price refresh complete")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Price refresh error: {e}")
            await asyncio.sleep(60)


async def _holder_refresh_loop():
    """Tier 2: Holder count every 30 minutes for top 100 agents."""
    while True:
        try:
            await asyncio.sleep(30 * 60)
            logger.info("Holder count refresh starting...")
            await refresh_holder_counts(limit=100)
            enrichment_state["last_holder_refresh"] = datetime.utcnow().isoformat()
            logger.info("Holder count refresh complete")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Holder refresh error: {e}")
            await asyncio.sleep(300)


async def _new_agent_scan_loop():
    """Tier 3: New agent discovery every hour."""
    while True:
        try:
            await asyncio.sleep(60 * 60)
            logger.info("Hourly new-agent scan...")
            new_count = await scan_newest_agents(pages=5)
            enrichment_state["last_new_scan"] = datetime.utcnow().isoformat()
            enrichment_state["new_agents_found"] = new_count
            if new_count > 0:
                logger.info(f"New-agent scan: {new_count} new agents — re-scoring...")
                await bulk_score_agents()
            enrichment_state["api_total"] = await get_api_total_count()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"New-agent scan error: {e}")
            await asyncio.sleep(300)


async def _daily_refresh_loop():
    """Tier 4: Description/team/socials full refresh every 24 hours."""
    while True:
        try:
            await asyncio.sleep(24 * 3600)
            logger.info("Starting daily full refresh...")
            count = await preload_all_agents()
            logger.info(f"Daily fetch: {count} agents refreshed, enriching DexScreener...")
            await enrich_top_agents_dexscreener(top_n=100)
            score_count = await bulk_score_agents()
            logger.info(f"Daily auto-scored {score_count} agents — queuing AI re-analysis")
            enrichment_state["last_description_refresh"] = datetime.utcnow().isoformat()
            asyncio.create_task(_auto_analyze_all())
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Daily refresh error: {e}")
            await asyncio.sleep(3600)


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("VirtualsIQ starting up...")
    await init_db()

    async def _preload():
        enrichment_state["status"] = "preloading"
        enrichment_state["started_at"] = datetime.utcnow().isoformat()
        enrichment_state["api_total"] = await get_api_total_count()

        def _on_batch(n: int):
            enrichment_state["agents_loaded"] = n

        try:
            count = await preload_all_agents(on_batch=_on_batch)
            enrichment_state["agents_loaded"] = count
            logger.info(f"Startup preload complete: {count} agents")

            enrichment_state["status"] = "enriching"
            await enrich_top_agents_dexscreener(top_n=100)
            enrichment_state["last_price_refresh"] = datetime.utcnow().isoformat()

            enrichment_state["status"] = "scoring"
            try:
                score_count = await bulk_score_agents()
                logger.info(f"Auto-scored {score_count} agents (on-chain only, AI analysis queued)")
            except Exception as score_err:
                logger.error(f"bulk_score_agents failed: {score_err}", exc_info=True)

            enrichment_state["completed_at"] = datetime.utcnow().isoformat()
            # Kick off full AI analysis in background — throttled (1 at a time, 3s between)
            asyncio.create_task(_auto_analyze_all())
        except Exception as e:
            logger.error(f"Startup preload failed: {e}", exc_info=True)
            enrichment_state["status"] = "error"

    preload_task = asyncio.create_task(_preload())

    # Tiered background loops
    price_task = asyncio.create_task(_price_refresh_loop())
    holder_task = asyncio.create_task(_holder_refresh_loop())
    new_agent_task = asyncio.create_task(_new_agent_scan_loop())
    daily_task = asyncio.create_task(_daily_refresh_loop())

    yield

    for task in [preload_task, price_task, holder_task, new_agent_task, daily_task]:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    logger.info("VirtualsIQ shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VirtualsIQ",
    description="AI-powered intelligence terminal for Virtuals Protocol",
    version="2.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/agent/{virtuals_id}", response_class=HTMLResponse)
async def agent_detail_page(request: Request, virtuals_id: str):
    """Full detail page for an agent (not a modal)."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "detail_agent_id": virtuals_id,
    })


@app.get("/category/{category}", response_class=HTMLResponse)
async def category_page(request: Request, category: str):
    """Category landing page."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "category_page": category,
    })


@app.get("/api/agents")
async def list_agents(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    category: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    doxx_tier: Optional[int] = Query(None, ge=1, le=3),
    sort: Optional[str] = Query("market_cap"),
    search: Optional[str] = Query(None),
):
    valid_sorts = ["market_cap", "composite_score", "price_change_24h", "newest", "holders"]
    if sort not in valid_sorts:
        sort = "market_cap"

    result = await get_agents(
        page=page,
        page_size=page_size,
        category=category,
        status=status,
        doxx_tier=doxx_tier,
        sort=sort,
        search=search,
    )
    return result


@app.get("/api/agent/{virtuals_id}")
async def agent_detail(virtuals_id: str):
    agent = await get_agent_detail(virtuals_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@app.get("/api/trending/{feed}")
async def trending_feed(feed: str, limit: int = Query(20, ge=1, le=50)):
    valid_feeds = ["hot", "top-scored", "new", "first-movers", "smart-money"]
    if feed not in valid_feeds:
        raise HTTPException(status_code=400, detail=f"Invalid feed. Choose from: {valid_feeds}")
    agents = await get_trending_agents(feed=feed, limit=limit)
    return {"feed": feed, "agents": agents}


@app.get("/api/trending-strip")
async def trending_strip():
    """Top movers, new launches, score changes for homepage strip."""
    data = await get_trending_strip()
    return data


@app.get("/api/category-summary/{category}")
async def category_summary(category: str):
    """Category landing page data."""
    data = await get_category_summary(category)
    return data


@app.post("/api/analyze/{virtuals_id}")
async def trigger_analysis(virtuals_id: str):
    agent = await get_agent_detail(virtuals_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    job_id = create_job(virtuals_id)
    asyncio.create_task(_run_analysis_job(job_id, virtuals_id))

    return {
        "job_id": job_id,
        "agent_id": virtuals_id,
        "status": "queued",
        "message": f"Analysis queued. Poll /api/job/{job_id} for status.",
    }


@app.get("/api/job/{job_id}")
async def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/stats")
async def ecosystem_stats():
    stats = await get_stats()
    return stats


@app.get("/api/search")
async def search_endpoint(q: str = Query("", min_length=1)):
    """Search agents by name or ticker, return top 10 matches."""
    results = await search_agents(q, limit=10)
    return {"query": q, "results": results, "count": len(results)}


@app.get("/api/agent/{virtuals_id}/on-chain")
async def agent_on_chain_signals(virtuals_id: str):
    """On-chain signal data for an agent. Placeholder for future enrichment."""
    agent = await get_agent_detail(virtuals_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Whale concentration: derive from holder data if available
    holder_count = agent.get("holder_count", 0) or 0
    top_10_conc = agent.get("top_10_concentration", 0) or 0
    whale_concentration = {
        "top_10_percent": top_10_conc,
        "holder_count": holder_count,
        "concentration_risk": (
            "high" if top_10_conc > 80 else
            "medium" if top_10_conc > 50 else
            "low"
        ) if top_10_conc > 0 else "unknown",
    }

    return {
        "virtuals_id": virtuals_id,
        "whale_concentration": whale_concentration,
        "smart_money_flow": {
            "status": "coming_soon",
            "description": "Smart money wallet tracking — available in a future update",
        },
        "team_wallet_activity": {
            "status": "coming_soon",
            "description": "Creator/team wallet movement alerts — available in a future update",
            "creator_wallet": agent.get("creator_wallet", ""),
        },
        "buy_sell_ratio": agent.get("buy_sell_ratio", 1.0),
        "tx_count_24h": agent.get("tx_count_24h", 0),
    }


@app.get("/api/agent/{virtuals_id}/holders")
async def agent_holders(virtuals_id: str):
    """Smart Money: top token holders for an agent, fetched from Moralis and cached."""
    import httpx
    from datetime import timedelta

    agent = await get_agent_detail(virtuals_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    contract_address = agent.get("contract_address", "")
    moralis_key = os.environ.get("MORALIS_API_KEY", "")

    # Check cache — refresh at most once per hour
    last_updated = await get_holders_last_updated(virtuals_id)
    cache_valid = False
    if last_updated:
        try:
            age = datetime.utcnow() - datetime.fromisoformat(last_updated)
            cache_valid = age < timedelta(hours=1)
        except Exception:
            cache_valid = False

    if cache_valid:
        holders = await get_agent_holders(virtuals_id)
        return {"virtuals_id": virtuals_id, "holders": holders, "source": "cache", "contract_address": contract_address}

    # No Moralis key — return graceful fallback
    if not moralis_key or not contract_address:
        cached = await get_agent_holders(virtuals_id)
        return {
            "virtuals_id": virtuals_id,
            "holders": cached,
            "source": "no_api_key" if not moralis_key else "no_contract",
            "contract_address": contract_address,
            "message": "Connect Moralis API for live holder data" if not moralis_key else "No token contract address available",
        }

    # Fetch from Moralis
    try:
        url = f"https://deep-index.moralis.io/api/v2.2/erc20/{contract_address}/owners"
        params = {"chain": "0x2105", "limit": 20, "order": "DESC"}
        headers = {"X-API-Key": moralis_key, "accept": "application/json"}
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        raw_holders = data.get("result", [])
        market_cap = agent.get("market_cap", 0) or 0

        holders = []
        for i, h in enumerate(raw_holders[:20]):
            balance = float(h.get("balance_formatted", h.get("balance", 0)) or 0)
            percentage = float(h.get("percentage_relative_to_total_supply", 0) or 0)
            balance_usd = float(h.get("usd_value", 0) or 0)
            if balance_usd == 0 and market_cap > 0 and percentage > 0:
                balance_usd = market_cap * percentage / 100

            labels = []
            if i < 10:
                labels.append("top10")
            if balance_usd > 500_000:
                labels.append("whale")

            holders.append({
                "wallet_address": h.get("owner_address", ""),
                "balance": balance,
                "balance_usd": balance_usd,
                "percentage": percentage,
                "rank": i + 1,
                "labels": labels,
            })

        await upsert_agent_holders(virtuals_id, holders)
        return {"virtuals_id": virtuals_id, "holders": holders, "source": "moralis", "contract_address": contract_address}

    except Exception as e:
        logger.warning(f"[Holders] Moralis fetch failed for {virtuals_id}: {e}")
        cached = await get_agent_holders(virtuals_id)
        return {
            "virtuals_id": virtuals_id,
            "holders": cached,
            "source": "cache_fallback",
            "contract_address": contract_address,
            "message": "Live data temporarily unavailable",
        }


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/sync/health")
async def sync_health():
    stats = await get_stats()
    db_count = stats.get("total_agents", 0)
    api_total = enrichment_state.get("api_total", 0)
    missing = max(0, api_total - db_count) if api_total else 0
    coverage_percent = round((db_count / api_total * 100), 2) if api_total else None
    return {
        "api_total": api_total,
        "db_count": db_count,
        "missing": missing,
        "coverage_percent": coverage_percent,
        "is_synced": missing <= 10,
        "last_full_preload": enrichment_state.get("completed_at"),
        "last_new_scan": enrichment_state.get("last_new_scan"),
        "new_agents_found_last_scan": enrichment_state.get("new_agents_found", 0),
        "status": enrichment_state.get("status"),
    }


@app.get("/api/status")
async def system_status():
    stats = await get_stats()
    return {
        **enrichment_state,
        "total_agents_in_db": stats.get("total_agents", 0),
    }


@app.post("/api/admin/write-overview")
async def write_overview(payload: dict):
    """
    Direct-write pre-generated overview_json for an agent.
    Accepts {virtuals_id: str, overview_json: dict}.
    """
    virtuals_id = str(payload.get("virtuals_id", "")).strip()
    overview_json = payload.get("overview_json", {})
    if not virtuals_id or not overview_json:
        raise HTTPException(status_code=400, detail="virtuals_id and overview_json required")
    agent = await get_agent_detail(virtuals_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {virtuals_id} not found")
    from database import update_overview_only
    await update_overview_only(virtuals_id, overview_json)
    return {"status": "ok", "virtuals_id": virtuals_id, "agent": agent.get("name")}


@app.post("/api/admin/refresh-analysis")
async def admin_refresh_analysis(force: bool = Query(False)):
    """
    Manually trigger a full AI re-analysis cycle.
    - force=false (default): only re-analyze agents older than 7 days
    - force=true: re-analyze ALL agents regardless of last_analyzed
    """
    if enrichment_state.get("status") == "analyzing":
        return {"status": "already_running", "message": "Analysis already in progress"}
    asyncio.create_task(_auto_analyze_all(force=force))
    return {
        "status": "queued",
        "force": force,
        "message": f"Analysis {'(forced, all agents)' if force else '(stale only, >7 days)'} queued",
    }



@app.post("/api/admin/backfill-categories")
async def backfill_categories():
    """Infer and fill agent_type for agents that have a generic/missing category."""
    rows = await get_agents_for_backfill()

    updated = 0
    MEANINGFUL_CATS = set(CAT_KEYWORDS.keys())
    for agent in rows:
        agent_for_inference = {**agent, "agent_type": ""}
        new_cat = _infer_category(agent_for_inference)
        if new_cat and new_cat in MEANINGFUL_CATS:
            await update_agent_category(agent["virtuals_id"], new_cat)
            updated += 1

    return {"status": "ok", "updated": updated, "total_checked": len(rows)}


@app.post("/api/admin/trigger-preload")
async def trigger_preload():
    """Trigger a full re-sync of all agents from the Virtuals Protocol API."""
    status = enrichment_state.get("status")
    if status == "preloading":
        return {"status": "already_running", "message": "Preload already in progress"}

    async def _do_preload():
        enrichment_state["status"] = "preloading"
        try:
            count = await preload_all_agents()
            enrichment_state["status"] = "complete"
            enrichment_state["completed_at"] = datetime.utcnow().isoformat()
            enrichment_state["last_preload_count"] = count
            logger.info(f"Triggered preload complete: {count} agents")
        except Exception as e:
            logger.error(f"Triggered preload failed: {e}")
            enrichment_state["status"] = "error"

    asyncio.create_task(_do_preload())
    return {"status": "queued", "message": "Full preload of all Virtuals agents triggered in background"}


@app.post("/api/admin/rescore-all")
async def admin_rescore_all():
    """Trigger bulk re-scoring of all agents from on-chain data (no AI calls)."""
    async def _do_rescore():
        prev_status = enrichment_state.get("status")
        enrichment_state["status"] = "scoring"
        try:
            count = await bulk_score_agents()
            logger.info(f"Manual rescore complete: {count} agents scored")
        except Exception as e:
            logger.error(f"Manual rescore failed: {e}", exc_info=True)
        finally:
            enrichment_state["status"] = prev_status or "complete"

    asyncio.create_task(_do_rescore())
    return {"status": "queued", "message": "Bulk re-scoring of all agents triggered in background"}


# ---------------------------------------------------------------------------
# Data-driven overview builder (no AI calls — uses structured DB fields only)
# ---------------------------------------------------------------------------

_VERTICAL_TAM = {
    "defi":           ("DeFi (decentralized finance)", "$100B+"),
    "trading":        ("AI-driven trading and market intelligence", "$100B+"),
    "finance":        ("financial services and fintech", "$90B+"),
    "infrastructure": ("blockchain infrastructure and tooling", "$80B+"),
    "infra":          ("blockchain infrastructure and tooling", "$80B+"),
    "protocol":       ("protocol infrastructure", "$80B+"),
    "security":       ("blockchain security and auditing", "$70B+"),
    "gaming":         ("Web3 gaming and interactive entertainment", "$50B+"),
    "game":           ("Web3 gaming and interactive entertainment", "$50B+"),
    "social":         ("social media and community platforms", "$30B+"),
    "community":      ("community and social networking", "$30B+"),
    "entertainment":  ("digital entertainment and media", "$20B+"),
    "creative":       ("creative tools and generative media", "$20B+"),
    "art":            ("digital art and creative expression", "$20B+"),
    "nft":            ("NFTs and digital collectibles", "$20B+"),
    "productivity":   ("productivity and automation tools", "$15B+"),
    "tools":          ("developer tools and utilities", "$15B+"),
    "analytics":      ("data analytics and market intelligence", "$25B+"),
    "data":           ("data services and analytics", "$25B+"),
    "meme":           ("meme culture and viral social tokens", "$5B+"),
}

_STATUS_LABELS = {
    "sentient": "Sentient",
    "genesis":  "Genesis",
    "prototype": "Prototype",
}


def _fmt_mcap(v: float) -> str:
    """Format market cap as $XM or $XK etc."""
    if v >= 1_000_000_000:
        return f"${v / 1_000_000_000:.1f}B"
    if v >= 1_000_000:
        return f"${v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"${v / 1_000:.0f}K"
    return f"${v:,.0f}"


def _median(lst: list) -> float:
    s = sorted(lst)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _build_data_driven_overview(agent: dict, category_peers: list = None) -> dict:
    """
    Compose a deeply analytical 5-section overview using data ratios and
    category comparisons. No AI API calls — uses only values already in the DB.
    Returns a dict with the 5 sections.
    """
    name         = agent.get("name") or "This agent"
    ticker       = agent.get("ticker") or ""
    biography    = (agent.get("biography") or "").strip()
    agent_type   = (agent.get("agent_type") or "AI Agent").strip()
    status_raw   = (agent.get("status") or "Prototype").lower()
    status_lbl   = _STATUS_LABELS.get(status_raw, status_raw.title())
    doxx_tier    = int(agent.get("doxx_tier") or 3)
    holder_count = int(agent.get("holder_count") or 0)
    market_cap   = float(agent.get("market_cap") or 0)
    volume_24h   = float(agent.get("volume_24h") or 0)
    top10_conc   = float(agent.get("top_10_concentration") or 0)
    tw_followers = int(agent.get("twitter_followers") or 0)
    tw_handle    = (agent.get("linked_twitter") or "").strip()
    website      = (agent.get("linked_website") or "").strip()
    telegram     = (agent.get("linked_telegram") or "").strip()

    ticker_str   = f" (${ticker})" if ticker else ""
    vertical_key = agent_type.lower()
    tam_label, tam_size = _VERTICAL_TAM.get(vertical_key, (f"{agent_type} AI agents", "billions"))

    # ── compute key ratios ────────────────────────────────────────────────
    turnover_rate    = volume_24h / market_cap if market_cap > 0 else 0.0   # daily liquidity health
    value_per_holder = market_cap / holder_count if holder_count > 0 else 0.0
    social_ratio     = tw_followers / holder_count if holder_count > 0 else 0.0

    # ── category peer comparison ──────────────────────────────────────────
    cat_stats: dict = {}
    if category_peers:
        valid_peers = [p for p in category_peers if p.get("virtuals_id") != agent.get("virtuals_id")]
        cat_mcaps    = [float(p.get("market_cap") or 0) for p in valid_peers if p.get("market_cap")]
        cat_holders  = [int(p.get("holder_count") or 0) for p in valid_peers if p.get("holder_count")]
        cat_volumes  = [float(p.get("volume_24h") or 0) for p in valid_peers if p.get("volume_24h")]
        cat_tw       = [int(p.get("twitter_followers") or 0) for p in valid_peers if p.get("twitter_followers")]
        if cat_mcaps:
            cat_stats["median_mcap"]  = _median(cat_mcaps)
            cat_stats["top_mcap"]     = max(cat_mcaps)
            sorted_desc = sorted(cat_mcaps + ([market_cap] if market_cap else []), reverse=True)
            cat_stats["mcap_rank"]    = sorted_desc.index(market_cap) + 1 if market_cap in sorted_desc else None
            cat_stats["cat_size"]     = len(sorted_desc)
        if cat_holders:
            cat_stats["median_holders"] = _median(cat_holders)
        if cat_volumes:
            cat_stats["median_volume"]  = _median(cat_volumes)
        if cat_tw:
            cat_stats["median_tw"]      = _median(cat_tw)

    # ── helpers ───────────────────────────────────────────────────────────
    def _turnover_label(rate: float) -> str:
        if rate == 0:
            return "near-zero daily turnover"
        if rate >= 0.20:
            return f"{rate:.1%} daily turnover — exceptionally high trading velocity"
        if rate >= 0.05:
            return f"{rate:.1%} daily turnover — healthy trading activity"
        if rate >= 0.01:
            return f"{rate:.1%} daily turnover — moderate liquidity"
        return f"{rate:.2%} daily turnover — very thin liquidity relative to market cap"

    def _holder_quality(holders: int, mcap: float) -> str:
        if holders == 0:
            return "no on-chain holder data available"
        vph = mcap / holders if mcap > 0 else 0
        if vph > 50_000:
            return f"{holders:,} holders with an average position of {_fmt_mcap(vph)} — large, concentrated positions typical of institutional or whale-dominated distribution"
        if vph > 10_000:
            return f"{holders:,} holders averaging {_fmt_mcap(vph)} per wallet — mid-size positions suggesting a mix of serious investors and early community"
        if vph > 1_000:
            return f"{holders:,} holders averaging {_fmt_mcap(vph)} per wallet — broad distribution with healthy retail participation"
        return f"{holders:,} holders averaging {_fmt_mcap(vph)} per wallet — very wide distribution, typical of high-visibility or meme-driven projects"

    def _social_quality(followers: int, holders: int) -> str:
        if followers == 0 and holders == 0:
            return "no social or holder data available"
        if followers == 0:
            return "no Twitter/X following data found"
        if holders == 0:
            return f"{followers:,} Twitter/X followers with no holder count available for comparison"
        ratio = followers / holders
        if ratio >= 10:
            return f"{followers:,} Twitter/X followers versus {holders:,} token holders — a {ratio:.1f}x social-to-holder ratio indicating massive brand awareness well beyond the token-holding community"
        if ratio >= 3:
            return f"{followers:,} Twitter/X followers versus {holders:,} token holders — a {ratio:.1f}x ratio showing strong community interest that significantly outpaces on-chain participation"
        if ratio >= 1:
            return f"{followers:,} Twitter/X followers versus {holders:,} token holders — a {ratio:.1f}x social-to-holder ratio, relatively balanced"
        return f"{followers:,} Twitter/X followers versus {holders:,} token holders — holder base exceeds social following, suggesting on-chain distribution precedes social traction"

    # ── what_it_does ──────────────────────────────────────────────────────
    if biography and len(biography) > 80:
        # Build a rich wrapper around the biography with product-specific context
        bio_excerpt = biography if len(biography) <= 600 else biography[:597] + "..."
        wid_parts = []

        # Opening: what it is
        wid_parts.append(
            f"{name}{ticker_str} is an AI agent on the Virtuals Protocol, operating in the {agent_type} vertical "
            f"at {status_lbl} status on the Base blockchain. "
            f"The agent describes its own mission as follows: {bio_excerpt}"
        )

        # Mechanism paragraph derived from biography keywords
        bio_lower = biography.lower()
        mechanism_hints = []
        if any(w in bio_lower for w in ["trade", "trading", "market", "alpha", "signal", "portfolio"]):
            mechanism_hints.append(
                f"The core product is designed around market intelligence and autonomous trading capabilities — "
                f"areas where AI agents on Base can offer execution speed and data synthesis that human traders cannot match at scale."
            )
        if any(w in bio_lower for w in ["defi", "yield", "liquidity", "swap", "vault", "stake", "amm", "dex", "lending"]):
            mechanism_hints.append(
                f"The DeFi integration layer suggests {name} is targeting on-chain financial automation — "
                f"an area that benefits from Virtuals Protocol's permissionless deployment model."
            )
        if any(w in bio_lower for w in ["social", "content", "community", "influencer", "post", "tweet", "audience"]):
            mechanism_hints.append(
                f"The social media and content angle positions {name} at the intersection of AI-native community building "
                f"and autonomous digital presence — a rapidly emerging category within the Virtuals ecosystem."
            )
        if any(w in bio_lower for w in ["game", "gaming", "npc", "player", "quest", "world", "rpg"]):
            mechanism_hints.append(
                f"Gaming as a vertical makes {name} a natural fit for the autonomous agent model — "
                f"AI-driven NPCs, in-game advisors, and autonomous companions represent one of the clearest near-term use cases for on-chain AI."
            )
        if any(w in bio_lower for w in ["analys", "research", "data", "insight", "intelligence", "report"]):
            mechanism_hints.append(
                f"The research and analytics angle puts {name} in the information-to-action pipeline — "
                f"surfacing intelligence that helps users make better decisions faster than would be possible manually."
            )

        for hint in mechanism_hints[:2]:
            wid_parts.append(hint)

        # Token mechanics
        if ticker:
            wid_parts.append(
                f"The ${ticker} token serves as the native coordination layer for {name}'s ecosystem. "
                f"Within the Virtuals Protocol framework, tokens enable holders to access the agent's outputs, "
                f"participate in governance, and share in the value the agent generates over time."
            )

        # Status and deployment maturity
        if status_raw == "sentient":
            wid_parts.append(
                f"{name} has achieved Sentient status on Virtuals Protocol — a significant milestone "
                f"confirming the agent is live, operational, and has met the protocol's autonomy thresholds "
                f"for independent on-chain operation."
            )
        elif status_raw == "genesis":
            wid_parts.append(
                f"{name} operates at Genesis status — the highest tier within Virtuals Protocol, "
                f"signifying full deployment maturity and protocol-verified autonomy capabilities."
            )
        else:
            wid_parts.append(
                f"As a Prototype-stage agent, {name} is in active development. "
                f"The core capabilities are being built and tested ahead of the Sentient status milestone."
            )

        # Social and community context
        if tw_handle or website:
            channels = []
            if tw_handle:
                handle_display = tw_handle.split("/")[-1].lstrip("@") if "/" in tw_handle else tw_handle.lstrip("@")
                followers_note = f" ({tw_followers:,} followers)" if tw_followers > 0 else ""
                channels.append(f"Twitter/X (@{handle_display}{followers_note})")
            if website:
                channels.append(f"a project website ({website})")
            if telegram:
                channels.append("Telegram")
            wid_parts.append(
                f"The project maintains an active public presence across {', '.join(channels)}, "
                f"providing the community and investors with ongoing updates and product visibility."
            )

        what_it_does = "\n\n".join(wid_parts)
    elif biography:
        what_it_does = (
            f"{name}{ticker_str} is a {status_lbl}-stage AI agent on the Virtuals Protocol in the {agent_type} vertical. "
            f"{biography}"
        )
    else:
        what_it_does = (
            f"{name}{ticker_str} is an AI agent operating on the Virtuals Protocol, "
            f"classified in the {agent_type} vertical at {status_lbl} stage on the Base blockchain. "
            f"No detailed biography or product description has been publicly disclosed at this time. "
            f"The agent's social channels and on-chain presence are the primary sources of information while "
            f"documentation is pending."
        )

    # ── who_is_behind_it ──────────────────────────────────────────────────
    who_parts = []

    # Team identity
    if doxx_tier == 1:
        who_parts.append(
            f"The team behind {name} is publicly identified — verifiable founder identities have been linked to the project, "
            f"providing a meaningful level of accountability uncommon among early-stage Virtuals agents."
        )
    elif doxx_tier == 2:
        who_parts.append(
            f"The team behind {name} operates pseudonymously but maintains a consistent and traceable social presence. "
            f"While founder identities are not fully doxxed, the pseudonymous track record provides partial accountability."
        )
    else:
        who_parts.append(
            f"The team behind {name} is fully anonymous — no verified founder identities have been publicly linked to the project. "
            f"Full anonymity is common among Virtuals Protocol builders, but it means community trust must be earned through execution."
        )

    # Community size as proxy for team capability
    if holder_count > 0:
        hq = _holder_quality(holder_count, market_cap)
        who_parts.append(
            f"The on-chain community profile shows {hq}. "
            + (f"Top-10 holders control {top10_conc:.0f}% of supply — {'a highly concentrated ownership structure that warrants monitoring' if top10_conc > 70 else 'a relatively healthy distribution with no extreme whale dominance' if top10_conc < 40 else 'moderate concentration typical of early-stage projects'}."
               if top10_conc > 0 else "")
        )

    # Social channels
    if tw_handle:
        handle_display = tw_handle.split("/")[-1].lstrip("@") if "/" in tw_handle else tw_handle.lstrip("@")
        sq = _social_quality(tw_followers, holder_count)
        who_parts.append(
            f"The project's Twitter/X presence (@{handle_display}) shows {sq}. "
            f"{'A large follower base relative to holders signals strong brand awareness and potential for future on-chain growth.' if social_ratio >= 3 else 'Active social communication is the primary window into team development cadence and community health.' if tw_followers > 0 else ''}"
        )

    if website:
        who_parts.append(f"A project website ({website}) provides additional discoverability and serves as the team's primary documentation surface.")

    if telegram:
        who_parts.append(f"The Telegram community channel offers direct access to the core team and early adopters.")

    if not tw_handle and not website and not telegram:
        who_parts.append(
            f"The project currently has no verified social channels listed, making independent due diligence challenging. "
            f"This is a higher-risk profile common at the earliest stages of Virtuals deployment."
        )

    who_is_behind_it = " ".join(who_parts)

    # ── what_is_notable ───────────────────────────────────────────────────
    notable_parts = []

    # Lead with the most data-driven insight
    if cat_stats and market_cap > 0:
        median_mcap = cat_stats.get("median_mcap", 0)
        top_mcap    = cat_stats.get("top_mcap", 0)
        mcap_rank   = cat_stats.get("mcap_rank")
        cat_size    = cat_stats.get("cat_size", 0)

        if mcap_rank and cat_size:
            if mcap_rank == 1:
                notable_parts.append(
                    f"{name} is the largest agent by market cap in the {agent_type} vertical on Virtuals Protocol "
                    f"at {_fmt_mcap(market_cap)}, leading a category of {cat_size} agents."
                )
            elif mcap_rank <= 3:
                notable_parts.append(
                    f"{name} ranks #{mcap_rank} by market cap in the {agent_type} vertical at {_fmt_mcap(market_cap)}, "
                    f"placing it in the top tier of {cat_size} agents in this category."
                )
            else:
                pct_of_leader = (market_cap / top_mcap * 100) if top_mcap > 0 else 0
                notable_parts.append(
                    f"{name} ranks #{mcap_rank} of {cat_size} agents in the {agent_type} vertical at {_fmt_mcap(market_cap)} — "
                    f"{pct_of_leader:.0f}% of the category leader's market cap, "
                    f"{'well within striking distance' if pct_of_leader >= 30 else 'with substantial upside if it can close the gap'}."
                )

        if median_mcap > 0 and market_cap > 0:
            ratio_to_median = market_cap / median_mcap
            if ratio_to_median >= 2:
                notable_parts.append(
                    f"At {ratio_to_median:.1f}x the category median market cap of {_fmt_mcap(median_mcap)}, "
                    f"{name} is a category standout, not a median player."
                )
            elif ratio_to_median < 0.5:
                notable_parts.append(
                    f"With a market cap {ratio_to_median:.1f}x the category median ({_fmt_mcap(median_mcap)}), "
                    f"{name} is trading at a significant discount to peers, which could represent either a value opportunity or reflect real differentiation challenges."
                )
    elif market_cap > 0:
        notable_parts.append(
            f"{name}{ticker_str} has established a market cap of {_fmt_mcap(market_cap)}, "
            f"reflecting real on-chain capital commitment from its holder community."
        )

    # Turnover ratio
    if turnover_rate > 0:
        notable_parts.append(
            f"Daily trading metrics show {_turnover_label(turnover_rate)} "
            f"(${volume_24h:,.0f} volume against {_fmt_mcap(market_cap)} market cap). "
            + ("High turnover confirms deep market liquidity and active speculative interest." if turnover_rate >= 0.05
               else "Moderate turnover is consistent with a maturing holder base holding for the longer term." if turnover_rate >= 0.01
               else "Low turnover suggests most holders are not actively trading — either strong conviction or thin market depth.")
        )
    elif volume_24h == 0 and market_cap > 0:
        notable_parts.append(
            f"No 24-hour trading volume has been recorded for {name} at this time, "
            f"which may reflect data latency or very low trading activity on the current measurement window."
        )

    # Status milestone
    if status_raw == "sentient":
        notable_parts.append(
            f"Sentient status on Virtuals Protocol is a meaningful on-chain milestone, confirming {name} is live "
            f"and operational as an autonomous agent — not merely a concept or prototype."
        )
    elif status_raw == "genesis":
        notable_parts.append(
            f"Genesis status is the highest tier on Virtuals Protocol, distinguishing {name} as a fully mature, "
            f"protocol-verified autonomous agent."
        )

    # Twitter standout
    if tw_followers > 0 and cat_stats.get("median_tw", 0) > 0:
        tw_vs_median = tw_followers / cat_stats["median_tw"]
        if tw_vs_median >= 2:
            notable_parts.append(
                f"With {tw_followers:,} Twitter/X followers — {tw_vs_median:.1f}x the {agent_type} category median — "
                f"{name} has exceptional social reach relative to its peers."
            )
    elif tw_followers > 10_000:
        notable_parts.append(
            f"A Twitter/X following of {tw_followers:,} represents strong social brand visibility for an AI agent in this category."
        )

    if not notable_parts:
        notable_parts.append(
            f"{name}{ticker_str} is positioned in the {agent_type} vertical on Virtuals Protocol at {status_lbl} stage. "
            f"The agent is building its on-chain presence with foundational social and distribution infrastructure in place."
        )

    what_is_notable = " ".join(notable_parts)

    # ── risks_to_monitor ──────────────────────────────────────────────────
    risks = []

    # Team anonymity
    if doxx_tier == 3:
        risks.append(
            f"Full team anonymity is {name}'s primary trust risk. "
            f"No verified identities are publicly linked — accountability depends entirely on the team's future execution record. "
            f"This risk resolves if founders choose to doxx or if on-chain product delivery builds a long enough public track record."
        )
    elif doxx_tier == 2:
        risks.append(
            f"The pseudonymous team structure introduces moderate trust uncertainty. "
            f"Accountability is partial — social presence exists but identities are not fully verifiable. "
            f"This mitigates as delivery cadence accumulates."
        )

    # Liquidity risk
    if market_cap > 0 and turnover_rate < 0.005:
        risks.append(
            f"Liquidity risk: daily turnover of {turnover_rate:.3%} against a {_fmt_mcap(market_cap)} market cap "
            f"(${volume_24h:,.0f} 24h volume) suggests very thin trading depth. "
            f"This becomes meaningful if large holders attempt to exit — resolves with broader distribution and sustained trading activity."
        )
    elif volume_24h == 0 and market_cap > 0:
        risks.append(
            f"Zero recorded 24-hour trading volume is a flag worth monitoring — it may reflect data latency, "
            f"but persistent zero volume against a {_fmt_mcap(market_cap)} market cap would indicate liquidity has dried up."
        )

    # Concentration risk
    if holder_count > 0 and holder_count < 100:
        risks.append(
            f"Extreme concentration risk: only {holder_count:,} token holder{'s' if holder_count != 1 else ''} "
            f"means even modest selling by a single large wallet could cause outsized price impact. "
            f"Watch for holder growth as the primary signal of distribution improvement."
        )
    elif top10_conc > 80:
        risks.append(
            f"Top-10 holders control {top10_conc:.0f}% of supply — a highly concentrated ownership structure. "
            f"Coordinated selling by whales could destabilize the market. "
            f"Broader distribution and time will reduce this risk."
        )
    elif top10_conc > 60:
        risks.append(
            f"Top-10 holder concentration at {top10_conc:.0f}% is elevated. "
            f"This is not unusual for early-stage projects, but it is a dynamic to watch as distribution evolves."
        )

    # Website gap
    if not website:
        risks.append(
            f"The absence of a project website limits independent product verification and makes onboarding new users harder. "
            f"A dedicated web presence would strengthen credibility and provide a home for documentation."
        )

    # Prototype stage risk
    if status_raw == "prototype":
        risks.append(
            f"{name} is at Prototype stage — the core product has not yet been verified at the level required for "
            f"Sentient or Genesis status. Execution risk is elevated until the product ships and gains on-chain traction."
        )

    # Competition
    if cat_stats.get("cat_size", 0) > 5:
        risks.append(
            f"The {agent_type} vertical has {cat_stats['cat_size']} agents competing on Virtuals Protocol. "
            f"Differentiation must be clear and defensible to avoid being crowded out as the category matures."
        )

    if not risks:
        risks.append(
            f"{name} operates in the competitive {agent_type} vertical on Virtuals Protocol. "
            f"Primary risks include market saturation in the category, execution pace relative to peers, "
            f"and sustaining community engagement as the broader Virtuals ecosystem continues to grow."
        )

    risks_to_monitor = " ".join(risks[:4])  # Cap at 4 risk points for readability

    # ── market_opportunity ────────────────────────────────────────────────
    mkt_parts = []

    mkt_parts.append(
        f"{name} operates in {tam_label} — a sector with an estimated addressable market of {tam_size}. "
        f"The Virtuals Protocol ecosystem on Base is one of the most active launchpads for autonomous AI agents on-chain, "
        f"giving projects direct access to a crypto-native, AI-literate audience."
    )

    # Upside framing using market cap
    if market_cap > 0 and cat_stats.get("top_mcap", 0) > 0:
        top_mcap = cat_stats["top_mcap"]
        x2 = market_cap * 2
        x5 = market_cap * 5
        x10 = market_cap * 10
        pct_of_leader = market_cap / top_mcap * 100
        if pct_of_leader < 100:
            mkt_parts.append(
                f"The category leader in the {agent_type} vertical commands {_fmt_mcap(top_mcap)}. "
                f"{name}'s current {_fmt_mcap(market_cap)} implies {pct_of_leader:.0f}% of leader market cap — "
                f"a 2x would put it at {_fmt_mcap(x2)}, a 5x at {_fmt_mcap(x5)}, and a 10x at {_fmt_mcap(x10)}. "
                f"{'Reaching leader parity would require a' + f' {top_mcap/market_cap:.1f}x move.' if top_mcap/market_cap >= 2 else 'It is already within striking distance of category leadership.'}"
            )
        else:
            mkt_parts.append(
                f"{name} is the category leader in the {agent_type} vertical at {_fmt_mcap(market_cap)}. "
                f"Sustaining this position requires continued product delivery and community engagement as challengers emerge."
            )
    elif market_cap > 0:
        mkt_parts.append(
            f"At {_fmt_mcap(market_cap)}, {name} is in the early-stage range for the {agent_type} vertical, "
            f"representing meaningful upside potential if the product gains traction within the Virtuals ecosystem."
        )

    # Vertical narrative tailwind
    if agent_type.lower() in ("trading", "defi", "finance"):
        mkt_parts.append(
            f"AI-powered on-chain finance is one of the highest-conviction narratives in Web3. "
            f"Autonomous agents that can trade, execute strategies, or surface alpha 24/7 address the most fundamental "
            f"limitation of human traders — sleep, emotion, and processing speed. "
            f"Institutional attention to this category continues to grow."
        )
    elif agent_type.lower() in ("gaming", "game"):
        mkt_parts.append(
            f"AI companions and autonomous game characters represent one of the most natural near-term use cases "
            f"for on-chain AI agents. The Web3 gaming sector is expanding rapidly, and the demand for AI-driven "
            f"in-game experiences is still in early innings."
        )
    elif agent_type.lower() in ("social", "community", "entertainment"):
        mkt_parts.append(
            f"Autonomous social AI agents — capable of building and engaging communities without human operators — "
            f"are an emerging category with strong narrative tailwinds as AI-native content creation scales globally."
        )
    elif agent_type.lower() in ("infrastructure", "infra", "protocol", "tools"):
        mkt_parts.append(
            f"Infrastructure agents sit at the base layer of the AI agent stack — "
            f"the tools and primitives that all other agents depend on. "
            f"Infrastructure plays in crypto historically accrue disproportionate value as the ecosystem above them scales."
        )
    else:
        mkt_parts.append(
            f"As the Virtuals Protocol ecosystem expands, agents with clear vertical focus and active deployment "
            f"status are best positioned to capture the protocol-level growth tailwinds that benefit the whole ecosystem."
        )

    market_opportunity = " ".join(mkt_parts)

    return {
        "what_it_does":       what_it_does,
        "who_is_behind_it":   who_is_behind_it,
        "what_is_notable":    what_is_notable,
        "risks_to_monitor":   risks_to_monitor,
        "market_opportunity": market_opportunity,
    }


def _is_template_overview(ov: dict) -> bool:
    """Detect overviews generated by the old shallow template (not real analysis)."""
    wid = (ov.get("what_it_does") or "").lower()
    who = (ov.get("who_is_behind_it") or "").lower()
    notable = (ov.get("what_is_notable") or "").lower()
    # Old template patterns
    return (
        "is a sentient ai agent operating within the virtuals protocol ecosystem, positioned in the" in wid
        or "is pending a comprehensive" in wid
        or "pending full analysis" in wid
        or "is an ai agent operating on the virtuals protocol, classified in the" in wid
        # who section old pattern
        or "is currently anonymous — no verified identity has been publicly confirmed" in who
        # notable section old pattern
        or "scoring data supports the view that this agent is operating in a large total addressable market" in notable
    )


@app.post("/api/admin/rebuild-pending-overviews")
async def rebuild_pending_overviews(dry_run: bool = Query(False)):
    """
    Scan all agents whose overview_json has placeholder 'pending' text OR was
    generated by the old shallow template, and rebuild using the richer
    data-driven generator. No AI API calls.
    """
    async def _do_rebuild():
        from database import get_all_agents, update_overview_only
        logger.info("rebuild-pending-overviews: loading all agents…")
        agents = await get_all_agents()
        logger.info(f"rebuild-pending-overviews: {len(agents)} agents loaded")

        # Build category peer map for comparison stats
        from collections import defaultdict
        category_map: dict = defaultdict(list)
        for a in agents:
            cat = (a.get("agent_type") or "Other").strip().lower()
            category_map[cat].append(a)

        updated = 0
        skipped = 0
        for agent in agents:
            ov = agent.get("overview_json") or {}
            if isinstance(ov, str):
                try:
                    ov = json.loads(ov)
                except Exception:
                    ov = {}
            full_text = json.dumps(ov).lower()
            needs_rebuild = (
                "pending full analysis" in full_text
                or "pending a comprehensive" in full_text
                or _is_template_overview(ov)
            )
            if not needs_rebuild:
                skipped += 1
                continue
            cat = (agent.get("agent_type") or "Other").strip().lower()
            peers = category_map.get(cat, [])
            new_ov = _build_data_driven_overview({**agent, "overview_json": ov}, category_peers=peers)
            if not dry_run:
                try:
                    await update_overview_only(str(agent.get("virtuals_id", "")), new_ov)
                    updated += 1
                except Exception as e:
                    logger.error(f"rebuild-overview failed for {agent.get('virtuals_id')}: {e}")
            else:
                updated += 1
        logger.info(f"rebuild-pending-overviews: done — {updated} updated, {skipped} skipped")

    asyncio.create_task(_do_rebuild())
    return {
        "status": "queued",
        "dry_run": dry_run,
        "message": "Scanning all agents for pending/template overviews and rebuilding. Check server logs for progress.",
    }


@app.post("/api/admin/rebuild-all-overviews")
async def rebuild_all_overviews(dry_run: bool = Query(False)):
    """
    Force-rebuild ALL agent overviews using the richer data-driven generator.
    Overwrites even overviews that already exist. No AI API calls.
    """
    async def _do_rebuild_all():
        from database import get_all_agents, update_overview_only
        from collections import defaultdict
        logger.info("rebuild-all-overviews: loading all agents…")
        agents = await get_all_agents()
        logger.info(f"rebuild-all-overviews: {len(agents)} agents to process")

        category_map: dict = defaultdict(list)
        for a in agents:
            cat = (a.get("agent_type") or "Other").strip().lower()
            category_map[cat].append(a)

        updated = 0
        errors = 0
        for agent in agents:
            cat = (agent.get("agent_type") or "Other").strip().lower()
            peers = category_map.get(cat, [])
            new_ov = _build_data_driven_overview(agent, category_peers=peers)
            if not dry_run:
                try:
                    await update_overview_only(str(agent.get("virtuals_id", "")), new_ov)
                    updated += 1
                except Exception as e:
                    logger.error(f"rebuild-all-overviews failed for {agent.get('virtuals_id')}: {e}")
                    errors += 1
            else:
                updated += 1
        logger.info(f"rebuild-all-overviews: done — {updated} updated, {errors} errors")

    asyncio.create_task(_do_rebuild_all())
    return {
        "status": "queued",
        "dry_run": dry_run,
        "message": "Rebuilding ALL agent overviews with enhanced data-driven analysis. Check server logs for progress.",
    }
