"""
Market Data MCP Server for AWM system.
Provides real-time and historical market data through MCP protocol with Zerodha integration.
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from decimal import Decimal
import json
import redis.asyncio as redis

# Add the project root to Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from shared.mcp_client.server import MCPServer, validate_required_fields, sanitize_input
from shared.database.connection import init_database, close_database, db_manager
from shared.models.trading import MarketData, Instrument, dict_to_instrument
from shared.zerodha import ZerodhaAuthService, format_indian_symbol, parse_zerodha_symbol

logger = logging.getLogger(__name__)


class MarketDataServer(MCPServer):
    """Market Data MCP Server implementation with Zerodha integration."""

    def __init__(self):
        host = os.getenv("MARKET_DATA_SERVER_HOST", "0.0.0.0")
        port = int(os.getenv("MARKET_DATA_SERVER_PORT", "8001"))
        super().__init__("market_data_server", host, port)

        # Zerodha integration
        self.zerodha_auth = ZerodhaAuthService()
        self.zerodha_client = None

        # Redis for caching
        self.redis_client = None
        self.cache_ttl = int(os.getenv("MARKET_DATA_CACHE_TTL", "60"))  # 60 seconds

        # Data quality tracking
        self.data_quality_stats = {
            "total_requests": 0,
            "zerodha_requests": 0,
            "cache_hits": 0,
            "errors": 0,
            "last_update": None
        }

        # Register handlers
        self.register_handlers()

        # Start background tasks
        asyncio.create_task(self._initialize_zerodha_client())
        asyncio.create_task(self._data_quality_monitor())

    async def _initialize_zerodha_client(self):
        """Initialize Zerodha client connection."""
        try:
            if await self.zerodha_auth.is_authenticated():
                self.zerodha_client = await self.zerodha_auth.get_authenticated_client()
                logger.info("Zerodha client initialized successfully")
            else:
                logger.warning("Zerodha not authenticated - using database only")
        except Exception as e:
            logger.error(f"Failed to initialize Zerodha client: {e}")

    async def _get_redis_client(self):
        """Get Redis client for caching."""
        if not self.redis_client:
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "redis"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                decode_responses=True
            )
        return self.redis_client

    async def _data_quality_monitor(self):
        """Background task to monitor data quality."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                # Log data quality stats
                logger.info(f"Data quality stats: {self.data_quality_stats}")

                # Reset counters periodically
                if self.data_quality_stats["total_requests"] > 10000:
                    self.data_quality_stats = {
                        "total_requests": 0,
                        "zerodha_requests": 0,
                        "cache_hits": 0,
                        "errors": 0,
                        "last_update": datetime.now().isoformat()
                    }

            except Exception as e:
                logger.error(f"Error in data quality monitor: {e}")

    def register_handlers(self):
        """Register all MCP handlers."""
        
        @self.handler("get_price_history")
        async def get_price_history(content: Dict[str, Any]) -> Dict[str, Any]:
            """Get historical price data for an instrument."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["symbol"])
            
            self.data_quality_stats["total_requests"] += 1

            symbol = content["symbol"]
            days = content.get("days", 30)
            interval = content.get("interval", "day")  # minute, day, week, month

            try:
                # Try to get data from Zerodha first
                if self.zerodha_client:
                    zerodha_data = await self._get_zerodha_historical_data(symbol, days, interval)
                    if zerodha_data:
                        self.data_quality_stats["zerodha_requests"] += 1
                        return zerodha_data

                # Fallback to database
                return await self._get_database_historical_data(symbol, days, interval)

            except Exception as e:
                self.data_quality_stats["errors"] += 1
                logger.error(f"Error getting price history for {symbol}: {e}")
                return {"error": f"Failed to get price history: {str(e)}"}
        
        @self.handler("get_current_quote")
        async def get_current_quote(content: Dict[str, Any]) -> Dict[str, Any]:
            """Get current quote for an instrument."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["symbol"])

            self.data_quality_stats["total_requests"] += 1

            symbol = content["symbol"]

            try:
                # Check cache first
                cached_quote = await self._get_cached_quote(symbol)
                if cached_quote:
                    self.data_quality_stats["cache_hits"] += 1
                    return cached_quote

                # Try Zerodha API
                if self.zerodha_client:
                    zerodha_quote = await self._get_zerodha_quote(symbol)
                    if zerodha_quote:
                        self.data_quality_stats["zerodha_requests"] += 1
                        # Cache the result
                        await self._cache_quote(symbol, zerodha_quote)
                        return zerodha_quote

                # Fallback to database
                return await self._get_database_quote(symbol)

            except Exception as e:
                self.data_quality_stats["errors"] += 1
                logger.error(f"Error getting quote for {symbol}: {e}")
                return {"error": f"Failed to get quote: {str(e)}"}
        
        @self.handler("scan_market")
        async def scan_market(content: Dict[str, Any]) -> Dict[str, Any]:
            """Scan market for instruments matching criteria."""
            content = await sanitize_input(content)
            
            # Build filter criteria
            filters = []
            params = []
            param_count = 0
            
            if "instrument_type" in content:
                param_count += 1
                filters.append(f"i.instrument_type = ${param_count}")
                params.append(content["instrument_type"])
            
            if "exchange" in content:
                param_count += 1
                filters.append(f"i.exchange = ${param_count}")
                params.append(content["exchange"])
            
            if "min_volume" in content:
                param_count += 1
                filters.append(f"md.volume >= ${param_count}")
                params.append(content["min_volume"])
            
            if "min_price" in content:
                param_count += 1
                filters.append(f"md.close_price >= ${param_count}")
                params.append(content["min_price"])
            
            if "max_price" in content:
                param_count += 1
                filters.append(f"md.close_price <= ${param_count}")
                params.append(content["max_price"])
            
            # Build query
            query = """
                SELECT DISTINCT ON (i.id) 
                    i.symbol, i.name, i.instrument_type, i.exchange,
                    md.close_price, md.volume, md.time
                FROM instruments i
                LEFT JOIN market_data md ON i.id = md.instrument_id
                WHERE i.is_active = true
            """
            
            if filters:
                query += " AND " + " AND ".join(filters)
            
            query += """
                ORDER BY i.id, md.time DESC
                LIMIT ${}
            """.format(param_count + 1)
            params.append(content.get("limit", 50))
            
            rows = await db_manager.execute_query(query, *params, fetch="all")
            
            # Format response
            instruments = []
            for row in rows:
                instruments.append({
                    "symbol": row["symbol"],
                    "name": row["name"],
                    "type": row["instrument_type"],
                    "exchange": row["exchange"],
                    "price": float(row["close_price"]) if row["close_price"] else None,
                    "volume": row["volume"] if row["volume"] else 0,
                    "last_updated": row["time"].isoformat() if row["time"] else None
                })
            
            return {
                "instruments": instruments,
                "count": len(instruments)
            }
        
        @self.handler("get_instruments")
        async def get_instruments(content: Dict[str, Any]) -> Dict[str, Any]:
            """Get list of available instruments."""
            content = await sanitize_input(content)
            
            query = """
                SELECT id, symbol, name, instrument_type, exchange, segment, lot_size, tick_size
                FROM instruments 
                WHERE is_active = true
                ORDER BY symbol
            """
            
            limit = content.get("limit", 100)
            query += f" LIMIT {limit}"
            
            rows = await db_manager.execute_query(query, fetch="all")
            
            instruments = []
            for row in rows:
                instruments.append({
                    "id": str(row["id"]),
                    "symbol": row["symbol"],
                    "name": row["name"],
                    "type": row["instrument_type"],
                    "exchange": row["exchange"],
                    "segment": row["segment"],
                    "lot_size": row["lot_size"],
                    "tick_size": float(row["tick_size"])
                })
            
            return {
                "instruments": instruments,
                "count": len(instruments)
            }
        
        @self.handler("ingest_market_data")
        async def ingest_market_data(content: Dict[str, Any]) -> Dict[str, Any]:
            """Ingest market data (for data providers)."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["symbol", "data"])
            
            symbol = content["symbol"]
            data_points = content["data"]
            
            # Get instrument
            instrument = await self._get_instrument_by_symbol(symbol)
            if not instrument:
                return {"error": f"Instrument {symbol} not found"}
            
            # Prepare data for insertion
            insert_queries = []
            for data_point in data_points:
                query = """
                    INSERT INTO market_data 
                    (time, instrument_id, open_price, high_price, low_price, close_price, volume, turnover)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (time, instrument_id) DO UPDATE SET
                        open_price = EXCLUDED.open_price,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price,
                        close_price = EXCLUDED.close_price,
                        volume = EXCLUDED.volume,
                        turnover = EXCLUDED.turnover
                """
                params = (
                    data_point["timestamp"],
                    instrument["id"],
                    Decimal(str(data_point["open"])),
                    Decimal(str(data_point["high"])),
                    Decimal(str(data_point["low"])),
                    Decimal(str(data_point["close"])),
                    data_point.get("volume", 0),
                    Decimal(str(data_point.get("turnover", 0)))
                )
                insert_queries.append((query, params))
            
            # Execute in transaction
            await db_manager.execute_transaction(insert_queries)
            
            return {
                "symbol": symbol,
                "inserted": len(data_points),
                "status": "success"
            }
    
    async def _get_instrument_by_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get instrument by symbol."""
        query = "SELECT * FROM instruments WHERE symbol = $1 AND is_active = true"
        return await db_manager.execute_query(query, symbol, fetch="one")

    async def _get_zerodha_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current quote from Zerodha API."""
        try:
            # Format symbol for Zerodha
            zerodha_symbol = format_indian_symbol(symbol)

            # Get quote from Zerodha
            quote_data = await self.zerodha_client.get_quote([zerodha_symbol])

            if zerodha_symbol in quote_data:
                quote = quote_data[zerodha_symbol]

                return {
                    "symbol": symbol,
                    "price": quote.get("last_price", 0),
                    "open": quote.get("ohlc", {}).get("open", 0),
                    "high": quote.get("ohlc", {}).get("high", 0),
                    "low": quote.get("ohlc", {}).get("low", 0),
                    "close": quote.get("ohlc", {}).get("close", 0),
                    "volume": quote.get("volume", 0),
                    "change": quote.get("net_change", 0),
                    "change_percent": quote.get("change", 0),
                    "timestamp": quote.get("timestamp", datetime.now().isoformat()),
                    "source": "zerodha"
                }

            return None

        except Exception as e:
            logger.error(f"Error getting Zerodha quote for {symbol}: {e}")
            return None

    async def _get_database_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current quote from database (fallback)."""
        try:
            instrument = await self._get_instrument_by_symbol(symbol)
            if not instrument:
                return {"error": f"Instrument {symbol} not found"}

            # Get latest market data
            query = """
                SELECT time, open_price, high_price, low_price, close_price, volume
                FROM market_data
                WHERE instrument_id = $1
                ORDER BY time DESC
                LIMIT 1
            """

            row = await db_manager.execute_query(query, instrument["id"], fetch="one")

            if not row:
                return {"error": f"No market data found for {symbol}"}

            return {
                "symbol": symbol,
                "price": float(row["close_price"]),
                "open": float(row["open_price"]),
                "high": float(row["high_price"]),
                "low": float(row["low_price"]),
                "close": float(row["close_price"]),
                "volume": row["volume"],
                "timestamp": row["time"].isoformat(),
                "source": "database"
            }

        except Exception as e:
            logger.error(f"Error getting database quote for {symbol}: {e}")
            return {"error": f"Failed to get quote from database: {str(e)}"}

    async def _cache_quote(self, symbol: str, quote_data: Dict[str, Any]) -> None:
        """Cache quote data in Redis."""
        try:
            redis_client = await self._get_redis_client()
            cache_key = f"quote:{symbol}"

            await redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(quote_data, default=str)
            )

        except Exception as e:
            logger.error(f"Error caching quote for {symbol}: {e}")

    async def _get_cached_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached quote data from Redis."""
        try:
            redis_client = await self._get_redis_client()
            cache_key = f"quote:{symbol}"

            cached_data = await redis_client.get(cache_key)
            if cached_data:
                quote_data = json.loads(cached_data)
                quote_data["source"] = "cache"
                return quote_data

            return None

        except Exception as e:
            logger.error(f"Error getting cached quote for {symbol}: {e}")
            return None

    async def _get_zerodha_historical_data(self, symbol: str, days: int, interval: str) -> Optional[Dict[str, Any]]:
        """Get historical data from Zerodha API."""
        try:
            # Format symbol for Zerodha
            zerodha_symbol = format_indian_symbol(symbol)

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Map interval to Zerodha format
            zerodha_interval = {
                "minute": "minute",
                "day": "day",
                "week": "week",
                "month": "month"
            }.get(interval, "day")

            # Note: Zerodha historical data API requires additional setup
            # For now, return None to use database fallback
            # In production, implement proper historical data fetching
            logger.info(f"Zerodha historical data not implemented yet for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Error getting Zerodha historical data for {symbol}: {e}")
            return None

    async def _get_database_historical_data(self, symbol: str, days: int, interval: str) -> Dict[str, Any]:
        """Get historical data from database."""
        try:
            # Get instrument
            instrument = await self._get_instrument_by_symbol(symbol)
            if not instrument:
                return {"error": f"Instrument {symbol} not found"}

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Query historical data
            query = """
                SELECT time, open_price, high_price, low_price, close_price, volume, turnover
                FROM market_data
                WHERE instrument_id = $1
                AND time >= $2 AND time <= $3
                ORDER BY time ASC
            """

            rows = await db_manager.execute_query(
                query,
                instrument["id"],
                start_date,
                end_date,
                fetch="all"
            )

            # Format response
            price_data = []
            for row in rows:
                price_data.append({
                    "timestamp": row["time"].isoformat(),
                    "open": float(row["open_price"]),
                    "high": float(row["high_price"]),
                    "low": float(row["low_price"]),
                    "close": float(row["close_price"]),
                    "volume": row["volume"],
                    "turnover": float(row["turnover"])
                })

            return {
                "symbol": symbol,
                "interval": interval,
                "data": price_data,
                "count": len(price_data),
                "source": "database"
            }

        except Exception as e:
            logger.error(f"Error getting database historical data for {symbol}: {e}")
            return {"error": f"Failed to get historical data: {str(e)}"}


async def main():
    """Main function to run the Market Data MCP Server."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize database
    await init_database()
    
    try:
        # Create and start server
        server = MarketDataServer()
        logger.info("Starting Market Data MCP Server...")
        await server.start()
    finally:
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
