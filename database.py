"""
VirtualsIQ — Database layer
SQLite with aiosqlite for async access
"""

import json
import logging
import os
import aiosqlite
from datetime import datetime

logger = logging.getLogger(__name__)

# Use Railway persistent volume if available, otherwise local
_vol = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "")
DB_PATH = os.path.join(_vol, "virtualsiq.db") if _vol else "virtualsiq.db"


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")

        # Main agents table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                virtuals_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                ticker TEXT,
                contract_address TEXT,
                status TEXT DEFAULT 'Prototype',
                agent_type TEXT,
                biography TEXT,
                creation_date TEXT,
                linked_twitter TEXT,
                linked_website TEXT,
                linked_telegram TEXT,
                creator_wallet TEXT,
                image_url TEXT,

                -- Market data
                market_cap REAL DEFAULT 0,
                volume_24h REAL DEFAULT 0,
                volume_6h REAL DEFAULT 0,
                price_usd REAL DEFAULT 0,
                price_change_24h REAL DEFAULT 0,
                liquidity_usd REAL DEFAULT 0,
                tx_count_24h INTEGER DEFAULT 0,
                buy_sell_ratio REAL DEFAULT 1.0,
                holder_count INTEGER DEFAULT 0,
                top_10_concentration REAL DEFAULT 0,

                -- Social / off-chain
                twitter_followers INTEGER DEFAULT 0,
                twitter_engagement_rate REAL DEFAULT 0,
                twitter_account_age INTEGER DEFAULT 0,
                github_stars INTEGER DEFAULT 0,
                github_commits_30d INTEGER DEFAULT 0,
                github_contributors INTEGER DEFAULT 0,
                github_last_commit TEXT,

                -- VIQ Intelligence
                composite_score REAL DEFAULT 50,
                tier_classification TEXT DEFAULT 'Moderate',
                scores_json TEXT DEFAULT '{}',
                analysis_json TEXT DEFAULT '{}',
                prediction_json TEXT DEFAULT '{}',
                overview_json TEXT DEFAULT '{}',
                first_mover INTEGER DEFAULT 0,
                doxx_tier INTEGER DEFAULT 3,

                last_analyzed TEXT,
                last_scanned TEXT,
                last_price_refresh TEXT,
                last_holder_refresh TEXT,
                last_description_refresh TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # Score history for tracking changes over time
        await db.execute("""
            CREATE TABLE IF NOT EXISTS score_history (
                score_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                composite_score REAL,
                scores_json TEXT DEFAULT '{}',
                recorded_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (agent_id) REFERENCES agents(virtuals_id)
            )
        """)

        # Predictions table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                predicted_at TEXT DEFAULT (datetime('now')),
                horizon TEXT NOT NULL,
                probability REAL,
                range_low REAL,
                range_high REAL,
                actual_return REAL,
                resolved_at TEXT,
                FOREIGN KEY (agent_id) REFERENCES agents(virtuals_id)
            )
        """)

        # Indexes for common queries
        await db.execute("CREATE INDEX IF NOT EXISTS idx_agents_market_cap ON agents(market_cap DESC)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_agents_composite_score ON agents(composite_score DESC)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_agents_agent_type ON agents(agent_type)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_agents_first_mover ON agents(first_mover)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_score_history_agent ON score_history(agent_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_predictions_agent ON predictions(agent_id)")

        # Migrate: add new columns if missing (safe for existing DBs)
        for col, dtype, default in [
            ("overview_json", "TEXT", "'{}'"),
            ("last_analyzed", "TEXT", "NULL"),
            ("last_price_refresh", "TEXT", "NULL"),
            ("last_holder_refresh", "TEXT", "NULL"),
            ("last_description_refresh", "TEXT", "NULL"),
        ]:
            try:
                await db.execute(f"ALTER TABLE agents ADD COLUMN {col} {dtype} DEFAULT {default}")
            except Exception:
                pass  # column already exists

        await db.commit()
    print("[DB] Schema initialized")


async def upsert_agent(agent: dict):
    """Insert or update an agent record."""
    # Serialize any dict/list values for JSON columns before passing to SQLite
    safe = dict(agent)
    for f in ("scores_json", "analysis_json", "prediction_json", "overview_json"):
        if isinstance(safe.get(f), (dict, list)):
            safe[f] = json.dumps(safe[f])
    agent = safe
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO agents (
                virtuals_id, name, ticker, contract_address, status, agent_type,
                biography, creation_date, linked_twitter, linked_website,
                linked_telegram, creator_wallet, image_url,
                market_cap, volume_24h, volume_6h, price_usd, price_change_24h,
                liquidity_usd, tx_count_24h, buy_sell_ratio, holder_count,
                top_10_concentration, twitter_followers, twitter_engagement_rate,
                twitter_account_age, github_stars, github_commits_30d,
                github_contributors, github_last_commit,
                composite_score, tier_classification, scores_json,
                analysis_json, prediction_json, overview_json, first_mover, doxx_tier,
                last_scanned, updated_at
            ) VALUES (
                :virtuals_id, :name, :ticker, :contract_address, :status, :agent_type,
                :biography, :creation_date, :linked_twitter, :linked_website,
                :linked_telegram, :creator_wallet, :image_url,
                :market_cap, :volume_24h, :volume_6h, :price_usd, :price_change_24h,
                :liquidity_usd, :tx_count_24h, :buy_sell_ratio, :holder_count,
                :top_10_concentration, :twitter_followers, :twitter_engagement_rate,
                :twitter_account_age, :github_stars, :github_commits_30d,
                :github_contributors, :github_last_commit,
                :composite_score, :tier_classification, :scores_json,
                :analysis_json, :prediction_json, :overview_json, :first_mover, :doxx_tier,
                :last_scanned, :updated_at
            )
            ON CONFLICT(virtuals_id) DO UPDATE SET
                name=excluded.name,
                ticker=excluded.ticker,
                contract_address=excluded.contract_address,
                status=excluded.status,
                agent_type=excluded.agent_type,
                biography=excluded.biography,
                creation_date=excluded.creation_date,
                linked_twitter=excluded.linked_twitter,
                linked_website=excluded.linked_website,
                linked_telegram=excluded.linked_telegram,
                creator_wallet=excluded.creator_wallet,
                image_url=excluded.image_url,
                market_cap=excluded.market_cap,
                volume_24h=excluded.volume_24h,
                volume_6h=excluded.volume_6h,
                price_usd=excluded.price_usd,
                price_change_24h=excluded.price_change_24h,
                liquidity_usd=excluded.liquidity_usd,
                tx_count_24h=excluded.tx_count_24h,
                buy_sell_ratio=excluded.buy_sell_ratio,
                holder_count=excluded.holder_count,
                top_10_concentration=excluded.top_10_concentration,
                twitter_followers=excluded.twitter_followers,
                twitter_engagement_rate=excluded.twitter_engagement_rate,
                last_scanned=excluded.last_scanned,
                updated_at=excluded.updated_at
        """, agent)
        await db.commit()


async def update_agent_scores(virtuals_id: str, composite_score: float,
                               tier: str, scores_json: dict,
                               analysis_json: dict, prediction_json: dict,
                               overview_json: dict,
                               first_mover: bool, doxx_tier: int):
    """Update intelligence scores for an agent and record history."""
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            UPDATE agents SET
                composite_score=?, tier_classification=?, scores_json=?,
                analysis_json=?, prediction_json=?, overview_json=?,
                first_mover=?, doxx_tier=?,
                last_analyzed=?, last_scanned=?, updated_at=?
            WHERE virtuals_id=?
        """, (
            composite_score, tier,
            json.dumps(scores_json), json.dumps(analysis_json),
            json.dumps(prediction_json), json.dumps(overview_json),
            1 if first_mover else 0, doxx_tier,
            now, now, now, virtuals_id
        ))

        # Record score history snapshot
        await db.execute("""
            INSERT INTO score_history (agent_id, composite_score, scores_json, recorded_at)
            VALUES (?, ?, ?, ?)
        """, (virtuals_id, composite_score, json.dumps(scores_json), now))

        await db.commit()


async def get_agents(
    page: int = 1,
    page_size: int = 25,
    category: str = None,
    status: str = None,
    doxx_tier: int = None,
    sort: str = "market_cap",
    search: str = None
) -> dict:
    """Paginated agent query with filters."""
    offset = (page - 1) * page_size
    where_clauses = []
    params = []

    if category:
        where_clauses.append("agent_type = ?")
        params.append(category)
    if status:
        where_clauses.append("status = ?")
        params.append(status)
    if doxx_tier is not None:
        where_clauses.append("doxx_tier = ?")
        params.append(doxx_tier)
    if search:
        where_clauses.append("(name LIKE ? OR ticker LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%"])

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sort_map = {
        "market_cap": "market_cap DESC",
        "composite_score": "composite_score DESC",
        "price_change_24h": "price_change_24h DESC",
        "newest": "created_at DESC",
        "holders": "holder_count DESC",
    }
    order_sql = sort_map.get(sort, "market_cap DESC")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            f"SELECT COUNT(*) FROM agents {where_sql}", params
        ) as cur:
            row = await cur.fetchone()
            total = row[0]

        async with db.execute(
            f"""SELECT virtuals_id, name, ticker, contract_address, status,
                       agent_type, image_url, market_cap, price_usd,
                       price_change_24h, volume_24h, holder_count,
                       composite_score, tier_classification, scores_json,
                       first_mover, doxx_tier, created_at, updated_at,
                       linked_twitter, linked_website, linked_telegram,
                       overview_json, last_analyzed
                FROM agents {where_sql}
                ORDER BY {order_sql}
                LIMIT ? OFFSET ?""",
            params + [page_size, offset]
        ) as cur:
            rows = await cur.fetchall()

    agents = []
    for row in rows:
        d = dict(row)
        for f in ("scores_json", "overview_json"):
            try:
                d[f] = json.loads(d.get(f) or "{}")
            except Exception:
                d[f] = {}
        agents.append(d)

    return {
        "agents": agents,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


async def get_agent_detail(virtuals_id: str) -> dict | None:
    """Full agent detail including AI analysis."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM agents WHERE virtuals_id = ?", (virtuals_id,)
        ) as cur:
            row = await cur.fetchone()

    if not row:
        return None

    d = dict(row)
    for field in ("scores_json", "analysis_json", "prediction_json", "overview_json"):
        try:
            d[field] = json.loads(d.get(field) or "{}")
        except Exception:
            d[field] = {}
    return d


async def get_trending_agents(feed: str, limit: int = 20) -> list:
    """Return agents for a trending feed."""
    feed_queries = {
        "hot": """
            SELECT * FROM agents
            WHERE volume_24h > 0
            ORDER BY (volume_24h / NULLIF(market_cap, 0)) DESC
            LIMIT ?
        """,
        "top-scored": """
            SELECT * FROM agents
            ORDER BY composite_score DESC
            LIMIT ?
        """,
        "new": """
            SELECT * FROM agents
            ORDER BY created_at DESC
            LIMIT ?
        """,
        "first-movers": """
            SELECT * FROM agents
            WHERE first_mover = 1
            ORDER BY composite_score DESC
            LIMIT ?
        """,
        "smart-money": """
            SELECT * FROM agents
            WHERE holder_count > 100
            ORDER BY (composite_score * 0.5 + (price_change_24h * 0.5)) DESC
            LIMIT ?
        """,
    }
    query = feed_queries.get(feed, feed_queries["top-scored"])
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(query, (limit,)) as cur:
            rows = await cur.fetchall()

    agents = []
    for row in rows:
        d = dict(row)
        for f in ("scores_json", "overview_json"):
            try:
                d[f] = json.loads(d.get(f) or "{}")
            except Exception:
                d[f] = {}
        agents.append(d)
    return agents


async def get_trending_strip() -> dict:
    """Return data for the trending strip: top movers, new launches, score changes."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Top 3 movers (biggest absolute 24h change)
        async with db.execute("""
            SELECT virtuals_id, name, ticker, image_url, price_change_24h,
                   market_cap, composite_score
            FROM agents WHERE price_change_24h IS NOT NULL
            ORDER BY ABS(price_change_24h) DESC LIMIT 3
        """) as cur:
            movers = [dict(r) for r in await cur.fetchall()]

        # Top 3 newest launches
        async with db.execute("""
            SELECT virtuals_id, name, ticker, image_url, price_change_24h,
                   market_cap, composite_score, created_at
            FROM agents ORDER BY created_at DESC LIMIT 3
        """) as cur:
            new_launches = [dict(r) for r in await cur.fetchall()]

        # Top 3 score changes (agents with recent score history deltas)
        async with db.execute("""
            SELECT a.virtuals_id, a.name, a.ticker, a.image_url,
                   a.composite_score, a.market_cap,
                   (a.composite_score - COALESCE(
                     (SELECT sh.composite_score FROM score_history sh
                      WHERE sh.agent_id = a.virtuals_id
                      ORDER BY sh.recorded_at DESC LIMIT 1 OFFSET 1), a.composite_score
                   )) as score_delta
            FROM agents a
            WHERE a.composite_score > 0
            ORDER BY ABS(a.composite_score - COALESCE(
              (SELECT sh.composite_score FROM score_history sh
               WHERE sh.agent_id = a.virtuals_id
               ORDER BY sh.recorded_at DESC LIMIT 1 OFFSET 1), a.composite_score
            )) DESC
            LIMIT 3
        """) as cur:
            score_changes = [dict(r) for r in await cur.fetchall()]

    return {
        "movers": movers,
        "new_launches": new_launches,
        "score_changes": score_changes,
    }


async def get_category_summary(category: str) -> dict:
    """Return summary data for a category landing page."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Top 5 by VIQ
        async with db.execute("""
            SELECT virtuals_id, name, ticker, image_url, composite_score,
                   market_cap, price_change_24h, holder_count
            FROM agents WHERE agent_type = ?
            ORDER BY composite_score DESC LIMIT 5
        """, (category,)) as cur:
            top_by_viq = [dict(r) for r in await cur.fetchall()]

        # Biggest movers 24h
        async with db.execute("""
            SELECT virtuals_id, name, ticker, image_url, price_change_24h,
                   market_cap, composite_score
            FROM agents WHERE agent_type = ? AND price_change_24h IS NOT NULL
            ORDER BY ABS(price_change_24h) DESC LIMIT 5
        """, (category,)) as cur:
            biggest_movers = [dict(r) for r in await cur.fetchall()]

        # New launches
        async with db.execute("""
            SELECT virtuals_id, name, ticker, image_url, created_at,
                   market_cap, composite_score
            FROM agents WHERE agent_type = ?
            ORDER BY created_at DESC LIMIT 5
        """, (category,)) as cur:
            new_launches = [dict(r) for r in await cur.fetchall()]

        # Category stats
        async with db.execute("""
            SELECT COUNT(*) as total,
                   AVG(composite_score) as avg_score,
                   COALESCE(SUM(market_cap), 0) as total_mcap,
                   COALESCE(SUM(volume_24h), 0) as total_volume
            FROM agents WHERE agent_type = ?
        """, (category,)) as cur:
            stats = dict(await cur.fetchone())

    # Build a data-driven AI summary for the category page
    total = stats.get("total", 0)
    avg_score = stats.get("avg_score")
    total_mcap = stats.get("total_mcap", 0)
    top_name = top_by_viq[0]["name"] if top_by_viq else None
    top_score = top_by_viq[0].get("composite_score") if top_by_viq else None
    mover_name = biggest_movers[0]["name"] if biggest_movers else None
    mover_chg = biggest_movers[0].get("price_change_24h") if biggest_movers else None

    summary_parts = []
    if total:
        summary_parts.append(f"The {category} category contains {total} agents")
        if total_mcap:
            summary_parts[-1] += f" with a combined market cap of ${total_mcap:,.0f}"
        summary_parts[-1] += "."
    if avg_score:
        summary_parts.append(f"The average VIQ score is {avg_score:.1f}.")
    if top_name and top_score:
        summary_parts.append(f"{top_name} leads with a VIQ score of {top_score:.0f}.")
    if mover_name and mover_chg and abs(mover_chg) > 1:
        direction = "up" if mover_chg > 0 else "down"
        summary_parts.append(f"{mover_name} is the biggest mover, {direction} {abs(mover_chg):.1f}% in 24h.")

    ai_summary = " ".join(summary_parts) if summary_parts else None

    return {
        "category": category,
        "stats": stats,
        "top_by_viq": top_by_viq,
        "biggest_movers": biggest_movers,
        "new_launches": new_launches,
        "ai_summary": ai_summary,
    }


async def search_agents(query: str, limit: int = 10) -> list[dict]:
    """Search agents by name or ticker, return top matches."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT virtuals_id, name, ticker, image_url, agent_type,
                      market_cap, price_usd, price_change_24h, composite_score,
                      tier_classification, holder_count
               FROM agents
               WHERE name LIKE ? OR ticker LIKE ?
               ORDER BY market_cap DESC
               LIMIT ?""",
            (f"%{query}%", f"%{query}%", limit)
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_agents_needing_reanalysis(limit: int = 10) -> list[dict]:
    """Find agents that need re-analysis: never analyzed, status change, big MC move."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT * FROM agents
            WHERE last_analyzed IS NULL
               OR last_analyzed < datetime('now', '-7 days')
            ORDER BY market_cap DESC
            LIMIT ?
        """, (limit,)) as cur:
            rows = await cur.fetchall()

    agents = []
    for row in rows:
        d = dict(row)
        for field in ("scores_json", "analysis_json", "prediction_json", "overview_json"):
            try:
                d[field] = json.loads(d.get(field) or "{}")
            except Exception:
                d[field] = {}
        agents.append(d)
    return agents


async def get_stats() -> dict:
    """Ecosystem-wide stats."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        async with db.execute("""
            SELECT
                COUNT(*) as total_agents,
                AVG(composite_score) as average_score,
                COALESCE(SUM(volume_24h), 0) as total_volume_24h,
                COALESCE(SUM(market_cap), 0) as total_market_cap,
                COUNT(CASE WHEN status='Sentient' THEN 1 END) as sentient_count,
                COUNT(CASE WHEN status='Prototype' THEN 1 END) as prototype_count,
                COUNT(CASE WHEN first_mover=1 THEN 1 END) as first_mover_count,
                COUNT(CASE WHEN tier_classification='Top Tier' THEN 1 END) as top_tier_count
            FROM agents
        """) as cur:
            stats_row = dict(await cur.fetchone())

        async with db.execute("""
            SELECT agent_type, COUNT(*) as count
            FROM agents
            WHERE agent_type IS NOT NULL
            GROUP BY agent_type
            ORDER BY count DESC
        """) as cur:
            categories = [dict(r) for r in await cur.fetchall()]

    return {
        **stats_row,
        "category_breakdown": categories,
    }


async def get_all_agents() -> list[dict]:
    """Return all agents for batch AI analysis."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM agents") as cur:
            rows = await cur.fetchall()
    agents = []
    for row in rows:
        d = dict(row)
        for field in ("scores_json", "analysis_json", "prediction_json", "overview_json"):
            try:
                d[field] = json.loads(d.get(field) or "{}")
            except Exception:
                d[field] = {}
        agents.append(d)
    return agents


async def get_existing_ids() -> set:
    """Return set of all known virtuals_ids."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT virtuals_id FROM agents") as cur:
            rows = await cur.fetchall()
    return {row[0] for row in rows}


UPSERT_COLS = [
    "virtuals_id", "name", "ticker", "contract_address", "status", "agent_type",
    "biography", "creation_date", "linked_twitter", "linked_website",
    "linked_telegram", "creator_wallet", "image_url",
    "market_cap", "volume_24h", "volume_6h", "price_usd", "price_change_24h",
    "liquidity_usd", "tx_count_24h", "buy_sell_ratio", "holder_count",
    "top_10_concentration", "twitter_followers", "twitter_engagement_rate",
    "twitter_account_age", "github_stars", "github_commits_30d",
    "github_contributors", "github_last_commit",
    "composite_score", "tier_classification", "scores_json",
    "analysis_json", "prediction_json", "overview_json", "first_mover", "doxx_tier",
    "last_scanned", "updated_at",
]

UPSERT_SQL = f"""
    INSERT INTO agents ({', '.join(UPSERT_COLS)})
    VALUES ({', '.join('?' for _ in UPSERT_COLS)})
    ON CONFLICT(virtuals_id) DO UPDATE SET
        name=excluded.name,
        ticker=excluded.ticker,
        contract_address=excluded.contract_address,
        status=excluded.status,
        agent_type=excluded.agent_type,
        biography=excluded.biography,
        creation_date=excluded.creation_date,
        linked_twitter=excluded.linked_twitter,
        linked_website=excluded.linked_website,
        linked_telegram=excluded.linked_telegram,
        creator_wallet=excluded.creator_wallet,
        image_url=excluded.image_url,
        market_cap=excluded.market_cap,
        holder_count=excluded.holder_count,
        updated_at=excluded.updated_at
"""


def _dict_to_tuple(agent: dict) -> tuple:
    """Convert agent dict to positional tuple matching UPSERT_COLS order.
    Auto-serializes any dict/list values to JSON strings for SQLite."""
    result = []
    for col in UPSERT_COLS:
        val = agent.get(col)
        if isinstance(val, (dict, list)):
            val = json.dumps(val)
        result.append(val)
    return tuple(result)


async def bulk_upsert_agents(agents: list[dict], batch_size: int = 500) -> int:
    """Insert or update agents in committed batches to avoid transaction size limits."""
    if not agents:
        return 0
    stored = 0
    for i in range(0, len(agents), batch_size):
        batch_dicts = agents[i:i + batch_size]
        batch_tuples = [_dict_to_tuple(a) for a in batch_dicts]
        batch_num = i // batch_size + 1
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute("PRAGMA journal_mode=WAL")
                await db.executemany(UPSERT_SQL, batch_tuples)
                await db.commit()
            stored += len(batch_tuples)
            logger.info(f"Batch {batch_num}: saved {len(batch_tuples)} agents (total so far: {stored})")
        except Exception as e:
            logger.error(f"Batch {batch_num} executemany failed: {e!r} - retrying row-by-row")
            for j, agent_dict in enumerate(batch_dicts):
                try:
                    async with aiosqlite.connect(DB_PATH) as db:
                        await db.execute("PRAGMA journal_mode=WAL")
                        await db.execute(UPSERT_SQL, _dict_to_tuple(agent_dict))
                        await db.commit()
                    stored += 1
                except Exception as row_err:
                    logger.error(
                        f"Row {i + j} failed (virtuals_id={agent_dict.get('virtuals_id')!r}): {row_err!r}"
                    )
    logger.info(f"bulk_upsert_agents complete: {stored}/{len(agents)} stored")
    return stored


async def bulk_score_agents() -> int:
    """Score all agents using on-chain data. Returns count scored."""
    from scoring import calculate_composite_score

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM agents") as cur:
            rows = await cur.fetchall()

    scored = 0
    batch = []
    for row in rows:
        agent = dict(row)
        for f in ("scores_json", "analysis_json", "prediction_json", "overview_json"):
            try:
                agent[f] = json.loads(agent.get(f) or "{}")
            except Exception:
                agent[f] = {}

        result = calculate_composite_score(agent, agent.get("analysis_json", {}))

        batch.append((
            result["composite_score"],
            result["tier_classification"],
            json.dumps(result["scores"]),
            1 if result["first_mover"] else 0,
            agent["virtuals_id"]
        ))
        scored += 1

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        for i in range(0, len(batch), 500):
            chunk = batch[i:i + 500]
            await db.executemany(
                "UPDATE agents SET composite_score=?, tier_classification=?, scores_json=?, first_mover=?, updated_at=datetime('now') WHERE virtuals_id=?",
                chunk
            )
        await db.commit()

    return scored


async def update_market_data(virtuals_id: str, data: dict):
    """Update only DexScreener market data fields without overwriting Virtuals API data."""
    now = datetime.utcnow().isoformat()
    dex_mcap = data.get("market_cap", 0) or 0
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            UPDATE agents SET
                price_usd=?,
                price_change_24h=?,
                volume_24h=?,
                volume_6h=?,
                liquidity_usd=?,
                tx_count_24h=?,
                buy_sell_ratio=?,
                market_cap=CASE WHEN ? > 0 THEN ? ELSE market_cap END,
                last_price_refresh=?,
                updated_at=?
            WHERE virtuals_id=?
        """, (
            data.get("price_usd", 0),
            data.get("price_change_24h", 0),
            data.get("volume_24h", 0),
            data.get("volume_6h", 0),
            data.get("liquidity_usd", 0),
            data.get("tx_count_24h", 0),
            data.get("buy_sell_ratio", 1.0),
            dex_mcap, dex_mcap,
            now, now,
            virtuals_id,
        ))
        await db.commit()


async def update_holder_count(virtuals_id: str, holder_count: int):
    """Update holder count only."""
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE agents SET holder_count=?, last_holder_refresh=?, updated_at=? WHERE virtuals_id=?",
            (holder_count, now, now, virtuals_id)
        )
        await db.commit()


async def get_top_agent_ids(limit: int = 50) -> list[str]:
    """Return top N agent IDs by market cap for priority analysis."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT virtuals_id FROM agents ORDER BY market_cap DESC LIMIT ?",
            (limit,)
        ) as cur:
            rows = await cur.fetchall()
    return [row[0] for row in rows]
