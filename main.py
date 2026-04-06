"""
VirtualsIQ — FastAPI Application
Bloomberg Terminal for the Virtuals Protocol ecosystem
"""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from analyzer import analyze_agent
from database import (
    bulk_score_agents,
    get_agent_detail,
    get_agents,
    get_existing_ids,
    get_stats,
    get_trending_agents,
    init_db,
    update_agent_scores,
    upsert_agent,
)
from virtuals_ingestion import detect_new_agents, enrich_top_agents_dexscreener, fetch_dexscreener_data, preload_all_agents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

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

        # Enrich with fresh DexScreener data
        if agent.get("contract_address"):
            dex = await fetch_dexscreener_data(agent["contract_address"])
            if dex:
                agent.update(dex)
                await upsert_agent(agent)

        result = await analyze_agent(agent)
        scores = result["scores"]
        analysis = result["analysis"]

        prediction_json = analysis.get("prediction", {})

        await update_agent_scores(
            virtuals_id=virtuals_id,
            composite_score=scores["composite_score"],
            tier=scores["tier_classification"],
            scores_json=scores["scores"],
            analysis_json=analysis,
            prediction_json=prediction_json,
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


async def _daily_scan_loop():
    """Periodic background scan: fetch all agents to detect new launches and refresh market data."""
    while True:
        try:
            await asyncio.sleep(24 * 3600)  # 24 hour interval
            logger.info("Starting daily full refresh...")
            count = await preload_all_agents()
            logger.info(f"Daily Virtuals fetch complete: {count} agents refreshed, starting DexScreener enrichment...")
            await enrich_top_agents_dexscreener(top_n=100)
            score_count = await bulk_score_agents()
            logger.info(f"Daily auto-scored {score_count} agents")
            logger.info("Daily scan complete")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Daily scan error: {e}")
            await asyncio.sleep(3600)  # retry in 1h on error


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("VirtualsIQ starting up...")
    await init_db()

    # Preload ALL agents in background (don't block startup)
    async def _preload():
        try:
            count = await preload_all_agents()
            logger.info(f"Startup preload complete: {count} agents — dashboard is now live")
            # Enrich top 100 with DexScreener data after agents are already visible
            await enrich_top_agents_dexscreener(top_n=100)
            logger.info("DexScreener enrichment complete")
            # Score all agents using on-chain data
            score_count = await bulk_score_agents()
            logger.info(f"Auto-scored {score_count} agents")
        except Exception as e:
            logger.error(f"Startup preload failed: {e}")

    preload_task = asyncio.create_task(_preload())
    scan_task = asyncio.create_task(_daily_scan_loop())

    yield

    # Cleanup
    preload_task.cancel()
    scan_task.cancel()
    try:
        await preload_task
        await scan_task
    except asyncio.CancelledError:
        pass
    logger.info("VirtualsIQ shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VirtualsIQ",
    description="AI-powered intelligence terminal for Virtuals Protocol",
    version="1.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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


@app.post("/api/analyze/{virtuals_id}")
async def trigger_analysis(virtuals_id: str, background_tasks: BackgroundTasks):
    agent = await get_agent_detail(virtuals_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    job_id = create_job(virtuals_id)
    background_tasks.add_task(_run_analysis_job, job_id, virtuals_id)

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


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0", "timestamp": datetime.utcnow().isoformat()}
