"""
Configuration settings for LLM Market Intelligence MCP Server.
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """LLM model configuration."""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    model_path: str = "/app/models"
    max_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.9
    gpu_memory_fraction: float = 0.8
    tensor_parallel_size: int = 1
    dtype: str = "float16"
    trust_remote_code: bool = False


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8007
    workers: int = 1
    timeout: int = 300
    max_concurrent_requests: int = 10


@dataclass
class MarketConfig:
    """Market-specific configuration."""
    market_timezone: str = "Asia/Kolkata"
    trading_hours_start: str = "09:15"
    trading_hours_end: str = "15:30"
    currency: str = "INR"
    exchanges: List[str] = None
    
    def __post_init__(self):
        if self.exchanges is None:
            self.exchanges = ["NSE", "BSE"]


@dataclass
class NewsConfig:
    """News and data source configuration."""
    news_sources: List[str] = None
    sentiment_confidence_threshold: float = 0.7
    news_cache_ttl: int = 300  # 5 minutes
    max_news_articles: int = 50
    
    def __post_init__(self):
        if self.news_sources is None:
            self.news_sources = [
                "economictimes.indiatimes.com",
                "moneycontrol.com",
                "business-standard.com",
                "livemint.com"
            ]


@dataclass
class CacheConfig:
    """Caching configuration."""
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl: int = 300  # 5 minutes
    sentiment_cache_ttl: int = 1800  # 30 minutes
    regime_cache_ttl: int = 900  # 15 minutes


class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.llm = LLMConfig(
            model_name=os.getenv("LLM_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2"),
            model_path=os.getenv("LLM_MODEL_PATH", "/app/models"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            gpu_memory_fraction=float(os.getenv("LLM_GPU_MEMORY_FRACTION", "0.8")),
            tensor_parallel_size=int(os.getenv("LLM_TENSOR_PARALLEL_SIZE", "1"))
        )
        
        self.server = ServerConfig(
            host=os.getenv("LLM_MARKET_INTELLIGENCE_HOST", "0.0.0.0"),
            port=int(os.getenv("LLM_MARKET_INTELLIGENCE_PORT", "8007")),
            timeout=int(os.getenv("LLM_REQUEST_TIMEOUT", "300")),
            max_concurrent_requests=int(os.getenv("LLM_MAX_CONCURRENT_REQUESTS", "10"))
        )
        
        self.market = MarketConfig()
        
        self.news = NewsConfig(
            sentiment_confidence_threshold=float(os.getenv("SENTIMENT_CONFIDENCE_THRESHOLD", "0.7")),
            max_news_articles=int(os.getenv("MAX_NEWS_ARTICLES", "50"))
        )
        
        self.cache = CacheConfig(
            redis_host=os.getenv("REDIS_HOST", "redis"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            cache_ttl=int(os.getenv("CACHE_TTL", "300"))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "llm": self.llm.__dict__,
            "server": self.server.__dict__,
            "market": self.market.__dict__,
            "news": self.news.__dict__,
            "cache": self.cache.__dict__
        }


# Global configuration instance
config = Config()
