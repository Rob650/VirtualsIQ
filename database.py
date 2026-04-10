"""
VirtualsIQ — Database layer
Supports PostgreSQL (asyncpg) when DATABASE_URL is set, falls back to SQLite.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import aiosqlite

logger = logging.getLogger(__name__)

# ── Connection config ──────────────────────────────────────────────────────────

# Railway provides DATABASE_URL (sometimes as postgres://, asyncpg needs postgresql://)
_raw_url = os.environ.get("DATABASE_URL", "")
DATABASE_URL = _raw_url.replace("postgres://", "postgresql://", 1) if _raw_url else ""
USE_PG = DATABASE_URL.startswith("postgresql://")

# SQLite fallback path (used only when DATABASE_URL is not set)
_vol = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "")
DB_PATH = os.path.join(_vol, "virtualsiq.db") if _vol else "virtualsiq.db"

_pool = None  # asyncpg connection pool, set by init_db()


# ── SQL conversion helpers ─────────────────────────────────────────────────────

def _pg_sql(sql: str) -> str:
    """Convert SQLite-style ? placeholders and syntax to PostgreSQL."""
    # Replace positional ? with $1, $2, ...
    result = []
    n = 0
    for ch in sql:
        if ch == "?":
            n += 1
            result.append(f"${n}")
        else:
            result.append(ch)
    sql = "".join(result)
    # DateTime function replacements
    sql = sql.replace("datetime('now', '-7 days')", "NOW() - INTERVAL '7 days'")
    sql = sql.replace("datetime('now', '-30 days')", "NOW() - INTERVAL '30 days'")
    sql = sql.replace("datetime('now')", "NOW()")
    return sql


# ── Unified connection wrapper ─────────────────────────────────────────────────

class _Conn:
    """Thin wrapper that provides a uniform interface over asyncpg and aiosqlite."""

    def __init__(self, conn, is_pg: bool):
        self._c = conn
        self._pg = is_pg

    async def execute(self, sql: str, params=()):
        if self._pg:
            await self._c.execute(_pg_sql(sql), *params)
        else:
            await self._c.execute(sql, params)

    async def executemany(self, sql: str, params_list):
        if self._pg:
            await self._c.executemany(_pg_sql(sql), params_list)
        else:
            await self._c.executemany(sql, params_list)

    async def fetch_all(self, sql: str, params=()) -> list:
        if self._pg:
            rows = await self._c.fetch(_pg_sql(sql), *params)
            return [dict(r) for r in rows]
        else:
            async with self._c.execute(sql, params) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    async def fetch_one(self, sql: str, params=()):
        if self._pg:
            row = await self._c.fetchrow(_pg_sql(sql), *params)
            return dict(row) if row else None
        else:
            async with self._c.execute(sql, params) as cur:
                row = await cur.fetchone()
                return dict(row) if row else None

    async def fetch_val(self, sql: str, params=()):
        if self._pg:
            return await self._c.fetchval(_pg_sql(sql), *params)
        else:
            async with self._c.execute(sql, params) as cur:
                row = await cur.fetchone()
                return row[0] if row else None

    async def commit(self):
        if not self._pg:
            await self._c.commit()


@asynccontextmanager
async def _db():
    """Acquire a DB connection (asyncpg pool or aiosqlite)."""
    if USE_PG:
        async with _pool.acquire() as conn:
            async with conn.transaction():
                yield _Conn(conn, True)
    else:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA foreign_keys=ON")
            yield _Conn(db, False)


async def get_db() -> aiosqlite.Connection:
    """Legacy helper kept for compatibility — use _db() internally."""
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


# ── Schema init ────────────────────────────────────────────────────────────────

_CREATE_AGENTS_PG = """
CREATE TABLE IF NOT EXISTS agents (
    id BIGSERIAL PRIMARY KEY,
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

    market_cap DOUBLE PRECISION DEFAULT 0,
    volume_24h DOUBLE PRECISION DEFAULT 0,
    volume_6h DOUBLE PRECISION DEFAULT 0,
    price_usd DOUBLE PRECISION DEFAULT 0,
    price_change_24h DOUBLE PRECISION DEFAULT 0,
    liquidity_usd DOUBLE PRECISION DEFAULT 0,
    tx_count_24h INTEGER DEFAULT 0,
    buy_sell_ratio DOUBLE PRECISION DEFAULT 1.0,
    holder_count INTEGER DEFAULT 0,
    top_10_concentration DOUBLE PRECISION DEFAULT 0,

    twitter_followers INTEGER DEFAULT 0,
    twitter_engagement_rate DOUBLE PRECISION DEFAULT 0,
    twitter_account_age INTEGER DEFAULT 0,
    github_stars INTEGER DEFAULT 0,
    github_commits_30d INTEGER DEFAULT 0,
    github_contributors INTEGER DEFAULT 0,
    github_last_commit TEXT,

    composite_score DOUBLE PRECISION DEFAULT 50,
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
    created_at TEXT DEFAULT to_char(NOW() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS'),
    updated_at TEXT DEFAULT to_char(NOW() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS')
)
"""

_CREATE_AGENTS_SQLITE = """
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

    twitter_followers INTEGER DEFAULT 0,
    twitter_engagement_rate REAL DEFAULT 0,
    twitter_account_age INTEGER DEFAULT 0,
    github_stars INTEGER DEFAULT 0,
    github_commits_30d INTEGER DEFAULT 0,
    github_contributors INTEGER DEFAULT 0,
    github_last_commit TEXT,

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
"""


async def init_db():
    global _pool

    if USE_PG:
        import asyncpg
        logger.info(f"[DB] Connecting to PostgreSQL...")
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)

        async with _pool.acquire() as conn:
            await conn.execute(_CREATE_AGENTS_PG)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS score_history (
                    score_id BIGSERIAL PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    composite_score DOUBLE PRECISION,
                    scores_json TEXT DEFAULT '{}',
                    recorded_at TEXT DEFAULT to_char(NOW() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS'),
                    FOREIGN KEY (agent_id) REFERENCES agents(virtuals_id)
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id BIGSERIAL PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    predicted_at TEXT DEFAULT to_char(NOW() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS'),
                    horizon TEXT NOT NULL,
                    probability DOUBLE PRECISION,
                    range_low DOUBLE PRECISION,
                    range_high DOUBLE PRECISION,
                    actual_return DOUBLE PRECISION,
                    resolved_at TEXT,
                    FOREIGN KEY (agent_id) REFERENCES agents(virtuals_id)
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_holders (
                    id BIGSERIAL PRIMARY KEY,
                    virtuals_id TEXT NOT NULL,
                    wallet_address TEXT NOT NULL,
                    balance DOUBLE PRECISION DEFAULT 0,
                    balance_usd DOUBLE PRECISION DEFAULT 0,
                    percentage DOUBLE PRECISION DEFAULT 0,
                    rank INTEGER DEFAULT 0,
                    labels JSONB DEFAULT '[]',
                    last_updated TEXT,
                    UNIQUE(virtuals_id, wallet_address)
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS score_snapshots (
                    id BIGSERIAL PRIMARY KEY,
                    virtuals_id TEXT NOT NULL,
                    composite_score REAL,
                    edge_score REAL,
                    market_cap REAL,
                    scores_json JSONB DEFAULT '{}',
                    snapshot_date DATE NOT NULL DEFAULT CURRENT_DATE,
                    UNIQUE(virtuals_id, snapshot_date)
                )
            """)

            # Indexes
            for idx_sql in [
                "CREATE INDEX IF NOT EXISTS idx_agents_market_cap ON agents(market_cap DESC)",
                "CREATE INDEX IF NOT EXISTS idx_agents_composite_score ON agents(composite_score DESC)",
                "CREATE INDEX IF NOT EXISTS idx_agents_agent_type ON agents(agent_type)",
                "CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status)",
                "CREATE INDEX IF NOT EXISTS idx_agents_first_mover ON agents(first_mover)",
                "CREATE INDEX IF NOT EXISTS idx_score_history_agent ON score_history(agent_id)",
                "CREATE INDEX IF NOT EXISTS idx_predictions_agent ON predictions(agent_id)",
                "CREATE INDEX IF NOT EXISTS idx_score_snapshots_vid ON score_snapshots(virtuals_id)",
                "CREATE INDEX IF NOT EXISTS idx_score_snapshots_date ON score_snapshots(snapshot_date DESC)",
            ]:
                await conn.execute(idx_sql)

            # Migrations — add missing columns if needed
            for col, dtype, default in [
                ("overview_json", "TEXT", "'{}'"),
                ("last_analyzed", "TEXT", "NULL"),
                ("last_price_refresh", "TEXT", "NULL"),
                ("last_holder_refresh", "TEXT", "NULL"),
                ("last_description_refresh", "TEXT", "NULL"),
            ]:
                try:
                    await conn.execute(
                        f"ALTER TABLE agents ADD COLUMN IF NOT EXISTS {col} {dtype} DEFAULT {default}"
                    )
                except Exception:
                    pass

        logger.info("[DB] PostgreSQL schema initialized")

    else:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA foreign_keys=ON")

            await db.execute(_CREATE_AGENTS_SQLITE)

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

            await db.execute("""
                CREATE TABLE IF NOT EXISTS agent_holders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    virtuals_id TEXT NOT NULL,
                    wallet_address TEXT NOT NULL,
                    balance REAL DEFAULT 0,
                    balance_usd REAL DEFAULT 0,
                    percentage REAL DEFAULT 0,
                    rank INTEGER DEFAULT 0,
                    labels TEXT DEFAULT '[]',
                    last_updated TEXT,
                    UNIQUE(virtuals_id, wallet_address)
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS score_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    virtuals_id TEXT NOT NULL,
                    composite_score REAL,
                    edge_score REAL,
                    market_cap REAL,
                    scores_json TEXT DEFAULT '{}',
                    snapshot_date TEXT NOT NULL DEFAULT (date('now')),
                    UNIQUE(virtuals_id, snapshot_date)
                )
            """)

            for idx_sql in [
                "CREATE INDEX IF NOT EXISTS idx_agents_market_cap ON agents(market_cap DESC)",
                "CREATE INDEX IF NOT EXISTS idx_agents_composite_score ON agents(composite_score DESC)",
                "CREATE INDEX IF NOT EXISTS idx_agents_agent_type ON agents(agent_type)",
                "CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status)",
                "CREATE INDEX IF NOT EXISTS idx_agents_first_mover ON agents(first_mover)",
                "CREATE INDEX IF NOT EXISTS idx_score_history_agent ON score_history(agent_id)",
                "CREATE INDEX IF NOT EXISTS idx_predictions_agent ON predictions(agent_id)",
                "CREATE INDEX IF NOT EXISTS idx_score_snapshots_vid ON score_snapshots(virtuals_id)",
                "CREATE INDEX IF NOT EXISTS idx_score_snapshots_date ON score_snapshots(snapshot_date DESC)",
            ]:
                await db.execute(idx_sql)

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
                    pass

            await db.commit()

        logger.info("[DB] SQLite schema initialized")

    print(f"[DB] Schema initialized ({'PostgreSQL' if USE_PG else 'SQLite'})")


# ── Agent CRUD ─────────────────────────────────────────────────────────────────

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
        agent_type=CASE WHEN excluded.agent_type IS NOT NULL AND excluded.agent_type != '' AND excluded.agent_type != 'Other' THEN excluded.agent_type ELSE COALESCE(agents.agent_type, excluded.agent_type) END,
        biography=CASE WHEN excluded.biography IS NOT NULL AND excluded.biography != '' THEN excluded.biography ELSE COALESCE(agents.biography, excluded.biography) END,
        creation_date=excluded.creation_date,
        linked_twitter=CASE WHEN excluded.linked_twitter IS NOT NULL AND excluded.linked_twitter != '' THEN excluded.linked_twitter ELSE COALESCE(agents.linked_twitter, excluded.linked_twitter) END,
        linked_website=CASE WHEN excluded.linked_website IS NOT NULL AND excluded.linked_website != '' THEN excluded.linked_website ELSE COALESCE(agents.linked_website, excluded.linked_website) END,
        linked_telegram=CASE WHEN excluded.linked_telegram IS NOT NULL AND excluded.linked_telegram != '' THEN excluded.linked_telegram ELSE COALESCE(agents.linked_telegram, excluded.linked_telegram) END,
        creator_wallet=excluded.creator_wallet,
        image_url=excluded.image_url,
        market_cap=excluded.market_cap,
        holder_count=excluded.holder_count,
        updated_at=excluded.updated_at
"""


def _dict_to_tuple(agent: dict) -> tuple:
    """Convert agent dict to positional tuple matching UPSERT_COLS order."""
    result = []
    for col in UPSERT_COLS:
        val = agent.get(col)
        if isinstance(val, (dict, list)):
            val = json.dumps(val)
        result.append(val)
    return tuple(result)


async def upsert_agent(agent: dict):
    """Insert or update an agent record."""
    safe = dict(agent)
    for f in ("scores_json", "analysis_json", "prediction_json", "overview_json"):
        if isinstance(safe.get(f), (dict, list)):
            safe[f] = json.dumps(safe[f])
    await bulk_upsert_agents([safe])


async def update_agent_scores(virtuals_id: str, composite_score: float,
                               tier: str, scores_json: dict,
                               analysis_json: dict, prediction_json: dict,
                               overview_json: dict,
                               first_mover: bool, doxx_tier: int):
    """Update intelligence scores for an agent and record history."""
    now = datetime.utcnow().isoformat()
    async with _db() as db:
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

    async with _db() as db:
        total = await db.fetch_val(
            f"SELECT COUNT(*) FROM agents {where_sql}", tuple(params)
        )

        rows = await db.fetch_all(
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
            tuple(params) + (page_size, offset)
        )

    agents = []
    for d in rows:
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
    async with _db() as db:
        row = await db.fetch_one(
            "SELECT * FROM agents WHERE virtuals_id = ?", (virtuals_id,)
        )

    if not row:
        return None

    for field in ("scores_json", "analysis_json", "prediction_json", "overview_json"):
        try:
            row[field] = json.loads(row.get(field) or "{}")
        except Exception:
            row[field] = {}
    return row


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
    async with _db() as db:
        rows = await db.fetch_all(query, (limit,))

    agents = []
    for d in rows:
        for f in ("scores_json", "overview_json"):
            try:
                d[f] = json.loads(d.get(f) or "{}")
            except Exception:
                d[f] = {}
        agents.append(d)
    return agents


async def get_trending_strip() -> dict:
    """Return data for the trending strip."""
    async with _db() as db:
        movers = await db.fetch_all("""
            SELECT virtuals_id, name, ticker, image_url, price_change_24h,
                   market_cap, composite_score
            FROM agents WHERE price_change_24h IS NOT NULL
            ORDER BY ABS(price_change_24h) DESC LIMIT 3
        """)

        new_launches = await db.fetch_all("""
            SELECT virtuals_id, name, ticker, image_url, price_change_24h,
                   market_cap, composite_score, created_at
            FROM agents ORDER BY created_at DESC LIMIT 3
        """)

        score_changes = await db.fetch_all("""
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
        """)

    return {
        "movers": movers,
        "new_launches": new_launches,
        "score_changes": score_changes,
    }


async def get_category_summary(category: str) -> dict:
    """Return summary data for a category landing page."""
    async with _db() as db:
        top_by_viq = await db.fetch_all("""
            SELECT virtuals_id, name, ticker, image_url, composite_score,
                   market_cap, price_change_24h, holder_count
            FROM agents WHERE agent_type = ?
            ORDER BY composite_score DESC LIMIT 5
        """, (category,))

        biggest_movers = await db.fetch_all("""
            SELECT virtuals_id, name, ticker, image_url, price_change_24h,
                   market_cap, composite_score
            FROM agents WHERE agent_type = ? AND price_change_24h IS NOT NULL
            ORDER BY ABS(price_change_24h) DESC LIMIT 5
        """, (category,))

        new_launches = await db.fetch_all("""
            SELECT virtuals_id, name, ticker, image_url, created_at,
                   market_cap, composite_score
            FROM agents WHERE agent_type = ?
            ORDER BY created_at DESC LIMIT 5
        """, (category,))

        stats = await db.fetch_one("""
            SELECT COUNT(*) as total,
                   AVG(composite_score) as avg_score,
                   COALESCE(SUM(market_cap), 0) as total_mcap,
                   COALESCE(SUM(volume_24h), 0) as total_volume
            FROM agents WHERE agent_type = ?
        """, (category,))

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


async def search_agents(query: str, limit: int = 10) -> list:
    """Search agents by name or ticker."""
    async with _db() as db:
        rows = await db.fetch_all(
            """SELECT virtuals_id, name, ticker, image_url, agent_type,
                      market_cap, price_usd, price_change_24h, composite_score,
                      tier_classification, holder_count
               FROM agents
               WHERE name LIKE ? OR ticker LIKE ?
               ORDER BY market_cap DESC
               LIMIT ?""",
            (f"%{query}%", f"%{query}%", limit)
        )
    return rows


async def get_agents_needing_reanalysis(limit: int = 10) -> list:
    """Find agents that need re-analysis: never analyzed or stale (>7 days)."""
    # Use Python datetime to avoid SQLite/PG syntax differences
    week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
    async with _db() as db:
        rows = await db.fetch_all("""
            SELECT * FROM agents
            WHERE last_analyzed IS NULL
               OR last_analyzed < ?
            ORDER BY market_cap DESC
            LIMIT ?
        """, (week_ago, limit))

    agents = []
    for d in rows:
        for field in ("scores_json", "analysis_json", "prediction_json", "overview_json"):
            try:
                d[field] = json.loads(d.get(field) or "{}")
            except Exception:
                d[field] = {}
        agents.append(d)
    return agents


async def get_stats() -> dict:
    """Ecosystem-wide stats."""
    async with _db() as db:
        stats_row = await db.fetch_one("""
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
        """)

        categories = await db.fetch_all("""
            SELECT agent_type, COUNT(*) as count
            FROM agents
            WHERE agent_type IS NOT NULL
            GROUP BY agent_type
            ORDER BY count DESC
        """)

    return {
        **stats_row,
        "category_breakdown": categories,
    }


async def get_all_agents() -> list:
    """Return all agents for batch AI analysis."""
    async with _db() as db:
        rows = await db.fetch_all("SELECT * FROM agents")

    agents = []
    for d in rows:
        for field in ("scores_json", "analysis_json", "prediction_json", "overview_json"):
            try:
                d[field] = json.loads(d.get(field) or "{}")
            except Exception:
                d[field] = {}
        agents.append(d)
    return agents


async def get_existing_ids() -> set:
    """Return set of all known virtuals_ids."""
    async with _db() as db:
        rows = await db.fetch_all("SELECT virtuals_id FROM agents")
    return {row["virtuals_id"] for row in rows}


async def bulk_upsert_agents(agents: list, batch_size: int = 500) -> int:
    """Insert or update agents in committed batches."""
    if not agents:
        return 0
    stored = 0
    for i in range(0, len(agents), batch_size):
        batch_dicts = agents[i:i + batch_size]
        batch_tuples = [_dict_to_tuple(a) for a in batch_dicts]
        batch_num = i // batch_size + 1
        try:
            async with _db() as db:
                await db.executemany(UPSERT_SQL, batch_tuples)
                await db.commit()
            stored += len(batch_tuples)
            logger.info(f"Batch {batch_num}: saved {len(batch_tuples)} agents (total so far: {stored})")
        except Exception as e:
            logger.error(f"Batch {batch_num} executemany failed: {e!r} - retrying row-by-row")
            for j, agent_dict in enumerate(batch_dicts):
                try:
                    async with _db() as db:
                        await db.executemany(UPSERT_SQL, [_dict_to_tuple(agent_dict)])
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

    async with _db() as db:
        rows = await db.fetch_all("SELECT * FROM agents")

    scored = 0
    batch = []
    for agent in rows:
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

    async with _db() as db:
        for i in range(0, len(batch), 500):
            chunk = batch[i:i + 500]
            await db.executemany(
                "UPDATE agents SET composite_score=?, tier_classification=?, scores_json=?, first_mover=?, updated_at=? WHERE virtuals_id=?",
                [(r[0], r[1], r[2], r[3], datetime.utcnow().isoformat(), r[4]) for r in chunk]
            )
        await db.commit()

    return scored


async def update_market_data(virtuals_id: str, data: dict):
    """Update only DexScreener market data fields."""
    now = datetime.utcnow().isoformat()
    dex_mcap = data.get("market_cap", 0) or 0
    async with _db() as db:
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


async def update_overview_only(virtuals_id: str, overview_json: dict):
    """Write a pre-generated overview_json without touching any other fields."""
    now = datetime.utcnow().isoformat()
    async with _db() as db:
        await db.execute(
            "UPDATE agents SET overview_json=?, last_analyzed=?, updated_at=? WHERE virtuals_id=?",
            (json.dumps(overview_json), now, now, virtuals_id)
        )
        await db.commit()


async def update_holder_count(virtuals_id: str, holder_count: int):
    """Update holder count only."""
    now = datetime.utcnow().isoformat()
    async with _db() as db:
        await db.execute(
            "UPDATE agents SET holder_count=?, last_holder_refresh=?, updated_at=? WHERE virtuals_id=?",
            (holder_count, now, now, virtuals_id)
        )
        await db.commit()


async def get_top_agent_ids(limit: int = 50) -> list:
    """Return top N agent IDs by market cap."""
    async with _db() as db:
        rows = await db.fetch_all(
            "SELECT virtuals_id FROM agents ORDER BY market_cap DESC LIMIT ?",
            (limit,)
        )
    return [row["virtuals_id"] for row in rows]


async def update_agent_category(virtuals_id: str, agent_type: str):
    """Update just the agent_type field for a single agent."""
    async with _db() as db:
        await db.execute(
            "UPDATE agents SET agent_type=? WHERE virtuals_id=?",
            (agent_type, virtuals_id)
        )
        await db.commit()


async def get_agents_for_backfill() -> list:
    """Return agents with generic/missing categories for backfill."""
    async with _db() as db:
        rows = await db.fetch_all(
            """SELECT virtuals_id, name, biography, agent_type FROM agents
               WHERE agent_type IS NULL OR agent_type = ''
                  OR agent_type IN ('Unknown', 'IP', 'Information',
                                    'Acp_Launch', 'Ip Mirror', 'X_Launch',
                                    'Functional', 'Ip_Mirror', 'ACP_LAUNCH')"""
        )
    return rows


# ── Smart Money / Holder Snapshots ────────────────────────────────────────────

async def get_agent_holders(virtuals_id: str) -> list:
    """Return cached holder list for an agent, or empty list if none."""
    async with _db() as db:
        rows = await db.fetch_all(
            "SELECT wallet_address, balance, balance_usd, percentage, rank, labels, last_updated FROM agent_holders WHERE virtuals_id=? ORDER BY rank ASC",
            (virtuals_id,)
        )
    result = []
    for r in rows:
        labels = r.get("labels") or "[]"
        if isinstance(labels, str):
            try:
                labels = json.loads(labels)
            except Exception:
                labels = []
        result.append({
            "wallet_address": r["wallet_address"],
            "balance": r.get("balance", 0),
            "balance_usd": r.get("balance_usd", 0),
            "percentage": r.get("percentage", 0),
            "rank": r.get("rank", 0),
            "labels": labels,
            "last_updated": r.get("last_updated"),
        })
    return result


async def get_holders_last_updated(virtuals_id: str):
    """Return the last_updated timestamp for holder data, or None."""
    async with _db() as db:
        val = await db.fetch_val(
            "SELECT MAX(last_updated) FROM agent_holders WHERE virtuals_id=?",
            (virtuals_id,)
        )
    return val


async def upsert_agent_holders(virtuals_id: str, holders: list):
    """Store/replace holder snapshot for an agent."""
    now = datetime.utcnow().isoformat()
    async with _db() as db:
        # Clear existing holders for this agent before inserting fresh set
        await db.execute("DELETE FROM agent_holders WHERE virtuals_id=?", (virtuals_id,))
        for h in holders:
            labels = json.dumps(h.get("labels", []))
            await db.execute(
                """INSERT INTO agent_holders
                   (virtuals_id, wallet_address, balance, balance_usd, percentage, rank, labels, last_updated)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(virtuals_id, wallet_address) DO UPDATE SET
                     balance=excluded.balance, balance_usd=excluded.balance_usd,
                     percentage=excluded.percentage, rank=excluded.rank,
                     labels=excluded.labels, last_updated=excluded.last_updated
                """,
                (
                    virtuals_id,
                    h.get("wallet_address", ""),
                    h.get("balance", 0),
                    h.get("balance_usd", 0),
                    h.get("percentage", 0),
                    h.get("rank", 0),
                    labels,
                    now,
                )
            )
        await db.commit()


# ── Score Snapshots ───────────────────────────────────────────────────────────

async def take_score_snapshot(virtuals_id: str, composite_score: float,
                               edge_score: float, market_cap: float,
                               scores_json: dict):
    """Upsert today's score snapshot for an agent."""
    today = datetime.utcnow().date().isoformat()
    async with _db() as db:
        await db.execute("""
            INSERT INTO score_snapshots (virtuals_id, composite_score, edge_score, market_cap, scores_json, snapshot_date)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(virtuals_id, snapshot_date) DO UPDATE SET
                composite_score=excluded.composite_score,
                edge_score=excluded.edge_score,
                market_cap=excluded.market_cap,
                scores_json=excluded.scores_json
        """, (virtuals_id, composite_score, edge_score, market_cap,
              json.dumps(scores_json) if isinstance(scores_json, dict) else (scores_json or "{}"),
              today))
        await db.commit()


async def get_score_history(virtuals_id: str, days: int = 30) -> list:
    """Return last N days of score snapshots for an agent, oldest-first."""
    async with _db() as db:
        rows = await db.fetch_all("""
            SELECT snapshot_date, composite_score, edge_score, market_cap, scores_json
            FROM score_snapshots
            WHERE virtuals_id=?
            ORDER BY snapshot_date ASC
            LIMIT ?
        """, (virtuals_id, days))
    result = []
    for r in rows:
        sj = r.get("scores_json") or "{}"
        if isinstance(sj, str):
            try:
                sj = json.loads(sj)
            except Exception:
                sj = {}
        result.append({
            "date": r["snapshot_date"],
            "composite_score": r.get("composite_score"),
            "edge_score": r.get("edge_score"),
            "market_cap": r.get("market_cap"),
            "scores_json": sj,
        })
    return result


async def get_all_agents_for_snapshot() -> list:
    """Return minimal agent data needed for daily snapshots."""
    async with _db() as db:
        rows = await db.fetch_all(
            "SELECT virtuals_id, composite_score, market_cap, scores_json FROM agents"
        )
    result = []
    for r in rows:
        sj = r.get("scores_json") or "{}"
        if isinstance(sj, str):
            try:
                sj = json.loads(sj)
            except Exception:
                sj = {}
        result.append({
            "virtuals_id": r["virtuals_id"],
            "composite_score": r.get("composite_score") or 0,
            "market_cap": r.get("market_cap") or 0,
            "scores_json": sj,
        })
    return result


async def get_agent_comparables(virtuals_id: str, agent_type: str,
                                 current_score: float, limit: int = 5) -> list:
    """Find agents in the same category with similar scores, excluding the given agent."""
    async with _db() as db:
        rows = await db.fetch_all("""
            SELECT virtuals_id, name, ticker, image_url, composite_score,
                   tier_classification, market_cap, agent_type,
                   ABS(composite_score - ?) AS score_diff
            FROM agents
            WHERE virtuals_id != ? AND agent_type = ?
              AND composite_score IS NOT NULL
            ORDER BY score_diff ASC
            LIMIT ?
        """, (current_score, virtuals_id, agent_type, limit))
    return [dict(r) for r in rows]


async def get_backtest_data(days_ago: int = 30) -> list:
    """
    Return rows pairing each agent's score N days ago with current market cap,
    for backtest correlation analysis.
    """
    import datetime as _dt
    cutoff = (_dt.datetime.utcnow().date() - _dt.timedelta(days=days_ago)).isoformat()
    async with _db() as db:
        rows = await db.fetch_all("""
            SELECT s.virtuals_id, s.composite_score AS score_at_snapshot,
                   s.market_cap AS mcap_at_snapshot, s.snapshot_date,
                   a.market_cap AS current_mcap, a.name, a.ticker
            FROM score_snapshots s
            JOIN agents a ON a.virtuals_id = s.virtuals_id
            WHERE s.snapshot_date <= ? AND s.composite_score IS NOT NULL
              AND s.market_cap > 0 AND a.market_cap > 0
            ORDER BY s.snapshot_date DESC
        """, (cutoff,))
    # Keep only the earliest (oldest) snapshot per agent
    seen = {}
    for r in rows:
        vid = r["virtuals_id"]
        if vid not in seen:
            seen[vid] = dict(r)
    return list(seen.values())
