"""
Database connection utilities for AWM system.
Provides async PostgreSQL/TimescaleDB connection management.
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import asyncpg
from asyncpg import Pool, Connection
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL/TimescaleDB connections."""
    
    def __init__(self):
        self.pool: Optional[Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        
    async def initialize(self):
        """Initialize database connections."""
        # PostgreSQL/TimescaleDB connection
        db_config = {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "database": os.getenv("POSTGRES_DB", "trading_db"),
            "user": os.getenv("POSTGRES_USER", "trading_user"),
            "password": os.getenv("POSTGRES_PASSWORD", ""),
            "min_size": int(os.getenv("DB_POOL_SIZE", "5")),
            "max_size": int(os.getenv("DB_POOL_SIZE", "10")),
            "command_timeout": int(os.getenv("SERVICE_TIMEOUT", "60"))
        }
        
        try:
            self.pool = await asyncpg.create_pool(**db_config)
            logger.info("PostgreSQL connection pool created successfully")
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL connection pool: {e}")
            raise
        
        # Redis connection
        redis_config = {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", "6379")),
            "db": int(os.getenv("REDIS_DB", "0")),
            "decode_responses": True
        }
        
        redis_password = os.getenv("REDIS_PASSWORD")
        if redis_password:
            redis_config["password"] = redis_password
        
        try:
            self.redis_client = redis.Redis(**redis_config)
            await self.redis_client.ping()
            logger.info("Redis connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def close(self):
        """Close all database connections."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")
        
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a PostgreSQL connection from the pool."""
        if not self.pool:
            raise RuntimeError("Database not initialized")
        
        async with self.pool.acquire() as connection:
            yield connection
    
    async def execute_query(
        self,
        query: str,
        *args,
        fetch: str = "none"
    ) -> Optional[List[Dict[str, Any]]]:
        """Execute a query and return results."""
        async with self.get_connection() as conn:
            if fetch == "all":
                rows = await conn.fetch(query, *args)
                return [dict(row) for row in rows]
            elif fetch == "one":
                row = await conn.fetchrow(query, *args)
                return dict(row) if row else None
            else:
                await conn.execute(query, *args)
                return None
    
    async def execute_transaction(self, queries: List[tuple]) -> None:
        """Execute multiple queries in a transaction."""
        async with self.get_connection() as conn:
            async with conn.transaction():
                for query, args in queries:
                    await conn.execute(query, *args)
    
    # Redis operations
    async def cache_set(
        self,
        key: str,
        value: str,
        expire: Optional[int] = None
    ) -> None:
        """Set a value in Redis cache."""
        if not self.redis_client:
            raise RuntimeError("Redis not initialized")
        
        await self.redis_client.set(key, value, ex=expire)
    
    async def cache_get(self, key: str) -> Optional[str]:
        """Get a value from Redis cache."""
        if not self.redis_client:
            raise RuntimeError("Redis not initialized")
        
        return await self.redis_client.get(key)
    
    async def cache_delete(self, key: str) -> None:
        """Delete a key from Redis cache."""
        if not self.redis_client:
            raise RuntimeError("Redis not initialized")
        
        await self.redis_client.delete(key)
    
    async def cache_exists(self, key: str) -> bool:
        """Check if a key exists in Redis cache."""
        if not self.redis_client:
            raise RuntimeError("Redis not initialized")
        
        return bool(await self.redis_client.exists(key))
    
    async def publish_message(self, channel: str, message: str) -> None:
        """Publish a message to Redis channel."""
        if not self.redis_client:
            raise RuntimeError("Redis not initialized")
        
        await self.redis_client.publish(channel, message)
    
    async def subscribe_channel(self, channel: str):
        """Subscribe to a Redis channel."""
        if not self.redis_client:
            raise RuntimeError("Redis not initialized")
        
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe(channel)
        return pubsub


# Global database manager instance
db_manager = DatabaseManager()


async def init_database():
    """Initialize the global database manager."""
    await db_manager.initialize()


async def close_database():
    """Close the global database manager."""
    await db_manager.close()


@asynccontextmanager
async def get_db_connection():
    """Get a database connection (convenience function)."""
    async with db_manager.get_connection() as conn:
        yield conn


# Common database operations
async def get_instrument_by_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    """Get instrument details by symbol."""
    query = "SELECT * FROM instruments WHERE symbol = $1 AND is_active = true"
    return await db_manager.execute_query(query, symbol, fetch="one")


async def get_portfolio_by_id(portfolio_id: str) -> Optional[Dict[str, Any]]:
    """Get portfolio details by ID."""
    query = "SELECT * FROM portfolios WHERE id = $1 AND is_active = true"
    return await db_manager.execute_query(query, portfolio_id, fetch="one")


async def get_latest_market_data(instrument_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Get latest market data for an instrument."""
    query = """
        SELECT * FROM market_data 
        WHERE instrument_id = $1 
        ORDER BY time DESC 
        LIMIT $2
    """
    return await db_manager.execute_query(query, instrument_id, limit, fetch="all")


async def insert_audit_log(
    action: str,
    table_name: str,
    record_id: Optional[str] = None,
    old_values: Optional[Dict] = None,
    new_values: Optional[Dict] = None,
    user_id: Optional[str] = None
) -> None:
    """Insert an audit log entry."""
    query = """
        INSERT INTO audit_log (action, table_name, record_id, old_values, new_values, user_id)
        VALUES ($1, $2, $3, $4, $5, $6)
    """
    await db_manager.execute_query(
        query,
        action,
        table_name,
        record_id,
        old_values,
        new_values,
        user_id
    )
