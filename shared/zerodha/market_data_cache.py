"""
Market Data Caching Strategy for AWM System.
Implements Redis-based caching with appropriate TTL settings for different data types.
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from decimal import Decimal
import redis.asyncio as redis

from .utils import validate_trading_hours

logger = logging.getLogger(__name__)


class MarketDataCache:
    """
    Redis-based caching system for market data with intelligent TTL management.
    Optimized for Indian market trading hours and data freshness requirements.
    """
    
    def __init__(self, redis_host: str = "redis", redis_port: int = 6379, redis_db: int = 0):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_client = None
        
        # Cache TTL settings (in seconds)
        self.ttl_settings = {
            # Real-time data (during market hours)
            "quote_live": 5,           # 5 seconds for live quotes
            "tick_live": 1,            # 1 second for tick data
            "depth_live": 3,           # 3 seconds for market depth
            
            # Real-time data (after market hours)
            "quote_offline": 300,      # 5 minutes when market closed
            "tick_offline": 600,       # 10 minutes when market closed
            "depth_offline": 600,      # 10 minutes when market closed
            
            # Historical data
            "historical_intraday": 60,     # 1 minute for intraday historical
            "historical_daily": 3600,      # 1 hour for daily historical
            "historical_weekly": 86400,    # 24 hours for weekly/monthly
            
            # Static data
            "instruments": 86400,      # 24 hours for instrument list
            "holidays": 604800,        # 1 week for market holidays
            "margins": 300,            # 5 minutes for margin data
            
            # Analysis data
            "technical_indicators": 300,   # 5 minutes for technical analysis
            "news_sentiment": 900,         # 15 minutes for news sentiment
            
            # System data
            "health_check": 60,        # 1 minute for health checks
            "rate_limit": 1,           # 1 second for rate limiting
        }
        
        # Cache key prefixes
        self.key_prefixes = {
            "quote": "md:quote:",
            "tick": "md:tick:",
            "depth": "md:depth:",
            "historical": "md:hist:",
            "instruments": "md:inst:",
            "margins": "md:margin:",
            "technical": "md:tech:",
            "news": "md:news:",
            "health": "sys:health:",
            "rate_limit": "sys:rate:"
        }
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
    
    async def get_redis_client(self) -> redis.Redis:
        """Get Redis client connection."""
        if not self.redis_client:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=True
            )
        return self.redis_client
    
    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached quote data."""
        try:
            redis_client = await self.get_redis_client()
            key = f"{self.key_prefixes['quote']}{symbol}"
            
            cached_data = await redis_client.get(key)
            if cached_data:
                self.stats["hits"] += 1
                data = json.loads(cached_data)
                data["cache_hit"] = True
                return data
            
            self.stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached quote for {symbol}: {e}")
            self.stats["errors"] += 1
            return None
    
    async def set_quote(self, symbol: str, quote_data: Dict[str, Any]) -> bool:
        """Cache quote data with appropriate TTL."""
        try:
            redis_client = await self.get_redis_client()
            key = f"{self.key_prefixes['quote']}{symbol}"
            
            # Determine TTL based on market hours
            trading_status = validate_trading_hours()
            if trading_status["is_trading_hours"]:
                ttl = self.ttl_settings["quote_live"]
            else:
                ttl = self.ttl_settings["quote_offline"]
            
            # Add cache metadata
            cache_data = {
                **quote_data,
                "cached_at": datetime.now().isoformat(),
                "ttl": ttl,
                "cache_hit": False
            }
            
            await redis_client.setex(key, ttl, json.dumps(cache_data, default=str))
            self.stats["sets"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error caching quote for {symbol}: {e}")
            self.stats["errors"] += 1
            return False
    
    async def get_historical_data(self, symbol: str, interval: str, days: int) -> Optional[Dict[str, Any]]:
        """Get cached historical data."""
        try:
            redis_client = await self.get_redis_client()
            
            # Create cache key based on parameters
            cache_key = self._generate_historical_key(symbol, interval, days)
            
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                self.stats["hits"] += 1
                data = json.loads(cached_data)
                data["cache_hit"] = True
                return data
            
            self.stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached historical data for {symbol}: {e}")
            self.stats["errors"] += 1
            return None
    
    async def set_historical_data(self, symbol: str, interval: str, days: int, data: Dict[str, Any]) -> bool:
        """Cache historical data with appropriate TTL."""
        try:
            redis_client = await self.get_redis_client()
            
            # Create cache key
            cache_key = self._generate_historical_key(symbol, interval, days)
            
            # Determine TTL based on interval
            if interval in ["minute", "5minute", "15minute"]:
                ttl = self.ttl_settings["historical_intraday"]
            elif interval == "day":
                ttl = self.ttl_settings["historical_daily"]
            else:
                ttl = self.ttl_settings["historical_weekly"]
            
            # Add cache metadata
            cache_data = {
                **data,
                "cached_at": datetime.now().isoformat(),
                "ttl": ttl,
                "cache_hit": False
            }
            
            await redis_client.setex(cache_key, ttl, json.dumps(cache_data, default=str))
            self.stats["sets"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error caching historical data for {symbol}: {e}")
            self.stats["errors"] += 1
            return False
    
    def _generate_historical_key(self, symbol: str, interval: str, days: int) -> str:
        """Generate cache key for historical data."""
        # Create a hash of parameters for consistent key generation
        params = f"{symbol}:{interval}:{days}"
        param_hash = hashlib.md5(params.encode()).hexdigest()[:8]
        return f"{self.key_prefixes['historical']}{symbol}:{param_hash}"
    
    async def get_instruments(self, exchange: str = None) -> Optional[List[Dict[str, Any]]]:
        """Get cached instruments list."""
        try:
            redis_client = await self.get_redis_client()
            key = f"{self.key_prefixes['instruments']}{exchange or 'all'}"
            
            cached_data = await redis_client.get(key)
            if cached_data:
                self.stats["hits"] += 1
                return json.loads(cached_data)
            
            self.stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached instruments: {e}")
            self.stats["errors"] += 1
            return None
    
    async def set_instruments(self, instruments: List[Dict[str, Any]], exchange: str = None) -> bool:
        """Cache instruments list."""
        try:
            redis_client = await self.get_redis_client()
            key = f"{self.key_prefixes['instruments']}{exchange or 'all'}"
            
            ttl = self.ttl_settings["instruments"]
            await redis_client.setex(key, ttl, json.dumps(instruments, default=str))
            self.stats["sets"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error caching instruments: {e}")
            self.stats["errors"] += 1
            return False
    
    async def get_margins(self, user_id: str = None) -> Optional[Dict[str, Any]]:
        """Get cached margin data."""
        try:
            redis_client = await self.get_redis_client()
            key = f"{self.key_prefixes['margins']}{user_id or 'default'}"
            
            cached_data = await redis_client.get(key)
            if cached_data:
                self.stats["hits"] += 1
                return json.loads(cached_data)
            
            self.stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached margins: {e}")
            self.stats["errors"] += 1
            return None
    
    async def set_margins(self, margins: Dict[str, Any], user_id: str = None) -> bool:
        """Cache margin data."""
        try:
            redis_client = await self.get_redis_client()
            key = f"{self.key_prefixes['margins']}{user_id or 'default'}"
            
            ttl = self.ttl_settings["margins"]
            cache_data = {
                **margins,
                "cached_at": datetime.now().isoformat()
            }
            
            await redis_client.setex(key, ttl, json.dumps(cache_data, default=str))
            self.stats["sets"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error caching margins: {e}")
            self.stats["errors"] += 1
            return False
    
    async def invalidate_symbol(self, symbol: str) -> bool:
        """Invalidate all cached data for a symbol."""
        try:
            redis_client = await self.get_redis_client()
            
            # Find all keys for this symbol
            patterns = [
                f"{self.key_prefixes['quote']}{symbol}",
                f"{self.key_prefixes['tick']}{symbol}*",
                f"{self.key_prefixes['depth']}{symbol}",
                f"{self.key_prefixes['historical']}{symbol}*",
                f"{self.key_prefixes['technical']}{symbol}*"
            ]
            
            deleted_count = 0
            for pattern in patterns:
                keys = await redis_client.keys(pattern)
                if keys:
                    deleted_count += await redis_client.delete(*keys)
            
            self.stats["deletes"] += deleted_count
            logger.info(f"Invalidated {deleted_count} cache entries for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error invalidating cache for {symbol}: {e}")
            self.stats["errors"] += 1
            return False
    
    async def clear_expired_cache(self) -> int:
        """Clear expired cache entries (Redis handles this automatically, but useful for stats)."""
        try:
            redis_client = await self.get_redis_client()
            
            # Get memory info
            info = await redis_client.info("memory")
            
            logger.info(f"Redis memory usage: {info.get('used_memory_human', 'unknown')}")
            return 0
            
        except Exception as e:
            logger.error(f"Error checking cache status: {e}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            redis_client = await self.get_redis_client()
            
            # Get Redis info
            info = await redis_client.info()
            
            # Calculate hit ratio
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_ratio = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "cache_stats": self.stats,
                "hit_ratio_percent": round(hit_ratio, 2),
                "total_requests": total_requests,
                "redis_info": {
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory_human": info.get("used_memory_human", "unknown"),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                    "expired_keys": info.get("expired_keys", 0)
                },
                "ttl_settings": self.ttl_settings
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> bool:
        """Check cache health."""
        try:
            redis_client = await self.get_redis_client()
            
            # Test basic operations
            test_key = "health_check_test"
            await redis_client.set(test_key, "test", ex=5)
            result = await redis_client.get(test_key)
            await redis_client.delete(test_key)
            
            return result == "test"
            
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return False
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None


# Global cache instance
market_data_cache = MarketDataCache()
