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
from typing import Any, Dict, Optional

import aiosqlite
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
    get_agents,
    get_agents_needing_reanalysis,
    get_category_summary,
    get_existing_ids,
    get_stats,
    get_top_agent_ids,
    get_trending_agents,
    get_trending_strip,
    init_db,
    search_agents,
    update_agent_scores,
    upsert_agent,
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
            import aiosqlite
            async with aiosqlite.connect("virtualsiq.db") as db:
                await db.execute(
                    "UPDATE agents SET agent_type=? WHERE virtuals_id=?",
                    (inferred_category, virtuals_id)
                )
                await db.commit()

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
                        import aiosqlite
                        async with aiosqlite.connect("virtualsiq.db") as db:
                            await db.execute(
                                "UPDATE agents SET agent_type=? WHERE virtuals_id=?",
                                (inferred_category, virtuals_id)
                            )
                            await db.commit()

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
            score_count = await bulk_score_agents()
            logger.info(f"Auto-scored {score_count} agents (on-chain only, AI analysis queued)")

            enrichment_state["completed_at"] = datetime.utcnow().isoformat()
            # Kick off full AI analysis in background — throttled (1 at a time, 3s between)
            asyncio.create_task(_auto_analyze_all())
        except Exception as e:
            logger.error(f"Startup preload failed: {e}")
            enrichment_state["status"] = "complete"

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


class OverviewRequest(BaseModel):
    virtuals_id: str
    overview_json: Dict[str, Any]


@app.post("/api/admin/write-overview")
async def write_overview(req: OverviewRequest):
    """Write / update the overview_json blob for a single agent."""
    async with aiosqlite.connect("virtualsiq.db") as db:
        await db.execute(
            "UPDATE agents SET overview_json = ? WHERE virtuals_id = ?",
            (json.dumps(req.overview_json), req.virtuals_id),
        )
        await db.commit()
        async with db.execute(
            "SELECT changes()"
        ) as cur:
            rows_changed = (await cur.fetchone())[0]
    if rows_changed == 0:
        raise HTTPException(status_code=404, detail=f"Agent {req.virtuals_id} not found")
    return {"status": "ok", "virtuals_id": req.virtuals_id}


@app.post("/api/admin/backfill-categories")
async def backfill_categories():
    """Infer and fill agent_type for agents that have a generic/missing category."""
    async with aiosqlite.connect("virtualsiq.db") as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT virtuals_id, name, biography, agent_type FROM agents
               WHERE agent_type IS NULL OR agent_type = '' OR agent_type IN ('Unknown', 'IP', 'Information')"""
        ) as cur:
            rows = [dict(r) for r in await cur.fetchall()]

    updated = 0
    async with aiosqlite.connect("virtualsiq.db") as db:
        for agent in rows:
            new_cat = _infer_category(agent)
            if new_cat and new_cat not in GENERIC_CATEGORIES:
                await db.execute(
                    "UPDATE agents SET agent_type = ? WHERE virtuals_id = ?",
                    (new_cat, agent["virtuals_id"]),
                )
                updated += 1
        await db.commit()

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
