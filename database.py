"""
VirtualsIQ — Database layer
SQLite with aiosqlite for async access
"""

import json
import aiosqlite
from datetime import datetime

DB_PATH = "virtualsiq.db"


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
                first_mover INTEGER DEFAULT 0,
                doxx_tier INTEGER DEFAULT 3,

                last_scanned TEXT,
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

        await db.commit()
    print("[DB] Schema initialized")


async def upsert_agent(agent: dict):
    """Insert or update an agent record."""
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
                analysis_json, prediction_json, first_mover, doxx_tier,
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
                :analysis_json, :prediction_json, :first_mover, :doxx_tier,
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
                               first_mover: bool, doxx_tier: int):
    """Update intelligence scores for an agent and record history."""
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            UPDATE agents SET
                composite_score=?, tier_classification=?, scores_json=?,
                analysis_json=?, prediction_json=?, first_mover=?, doxx_tier=?,
                last_scanned=?, updated_at=?
            WHERE virtuals_id=?
        """, (
            composite_score, tier,
            json.dumps(scores_json), json.dumps(analysis_json),
            json.dumps(prediction_json),
            1 if first_mover else 0, doxx_tier,
            now, now, virtuals_id
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
                       first_mover, doxx_tier, created_at
                FROM agents {where_sql}
                ORDER BY {order_sql}
                LIMIT ? OFFSET ?""",
            params + [page_size, offset]
        ) as cur:
            rows = await cur.fetchall()

    agents = []
    for row in rows:
        d = dict(row)
        try:
            d["scores_json"] = json.loads(d.get("scores_json") or "{}")
        except Exception:
            d["scores_json"] = {}
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
    for field in ("scores_json", "analysis_json", "prediction_json"):
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
        try:
            d["scores_json"] = json.loads(d.get("scores_json") or "{}")
        except Exception:
            d["scores_json"] = {}
        agents.append(d)
    return agents


async def get_stats() -> dict:
    """Ecosystem-wide stats."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        async with db.execute("""
            SELECT
                COUNT(*) as total_agents,
                AVG(composite_score) as avg_score,
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


async def get_existing_ids() -> set:
    """Return set of all known virtuals_ids."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT virtuals_id FROM agents") as cur:
            rows = await cur.fetchall()
    return {row[0] for row in rows}


UPSERT_SQL = """
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
        analysis_json, prediction_json, first_mover, doxx_tier,
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
        :analysis_json, :prediction_json, :first_mover, :doxx_tier,
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
        holder_count=excluded.holder_count,
        updated_at=excluded.updated_at
"""


async def bulk_upsert_agents(agents: list[dict], batch_size: int = 500) -> int:
    """Insert or update agents in committed batches to avoid transaction size limits."""
    if not agents:
        return 0
    stored = 0
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        for i in range(0, len(agents), batch_size):
            batch = agents[i:i + batch_size]
            try:
                await db.executemany(UPSERT_SQL, batch)
                await db.commit()
                stored += len(batch)
            except Exception as e:
                await db.rollback()
                # Fall back to one-by-one so a single bad row doesn't lose the batch
                for agent in batch:
                    try:
                        await db.execute(UPSERT_SQL, agent)
                        await db.commit()
                        stored += 1
                    except Exception:
                        await db.rollback()
    return stored


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
            now,
            virtuals_id,
        ))
        await db.commit()
