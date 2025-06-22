"""
Market Data Quality Monitoring for AWM System.
Monitors data quality, latency, feed health, and generates alerts for issues.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import deque
import statistics

from .client import ZerodhaClient
from .auth import ZerodhaAuthService
from .market_data_cache import MarketDataCache
from .utils import validate_trading_hours

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetric:
    """Data quality metric structure."""
    timestamp: datetime
    metric_type: str
    symbol: str
    value: float
    threshold: float
    status: str  # "good", "warning", "critical"
    details: Dict[str, Any]


@dataclass
class LatencyMetric:
    """Latency measurement structure."""
    timestamp: datetime
    operation: str
    latency_ms: float
    target_ms: float
    status: str


class DataQualityMonitor:
    """
    Comprehensive data quality monitoring system for market data feeds.
    Tracks latency, data freshness, completeness, and accuracy.
    """
    
    def __init__(self, cache: MarketDataCache = None):
        self.auth_service = ZerodhaAuthService()
        self.cache = cache or MarketDataCache()
        
        # Monitoring configuration
        self.config = {
            "latency_thresholds": {
                "quote_warning_ms": 200,
                "quote_critical_ms": 500,
                "historical_warning_ms": 1000,
                "historical_critical_ms": 3000,
                "websocket_warning_ms": 100,
                "websocket_critical_ms": 300
            },
            "freshness_thresholds": {
                "quote_warning_seconds": 10,
                "quote_critical_seconds": 30,
                "tick_warning_seconds": 5,
                "tick_critical_seconds": 15
            },
            "completeness_thresholds": {
                "missing_data_warning_pct": 5,
                "missing_data_critical_pct": 15,
                "stale_data_warning_pct": 10,
                "stale_data_critical_pct": 25
            }
        }
        
        # Monitoring state
        self.metrics_history = deque(maxlen=1000)
        self.latency_history = deque(maxlen=1000)
        self.alerts = deque(maxlen=100)
        
        # Test symbols for monitoring
        self.test_symbols = ["NSE:RELIANCE", "NSE:TCS", "NSE:INFY", "NSE:HDFCBANK"]
        
        # Statistics
        self.stats = {
            "total_checks": 0,
            "quality_issues": 0,
            "latency_issues": 0,
            "data_gaps": 0,
            "last_check": None
        }
        
        # Monitoring tasks
        self.monitoring_tasks = []
        self.is_monitoring = False
    
    async def start_monitoring(self):
        """Start continuous data quality monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return
        
        self.is_monitoring = True
        logger.info("Starting data quality monitoring")
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._latency_monitor()),
            asyncio.create_task(self._freshness_monitor()),
            asyncio.create_task(self._completeness_monitor()),
            asyncio.create_task(self._feed_health_monitor()),
            asyncio.create_task(self._alert_processor())
        ]
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
    
    async def stop_monitoring(self):
        """Stop data quality monitoring."""
        self.is_monitoring = False
        
        for task in self.monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()
        
        logger.info("Data quality monitoring stopped")
    
    async def _latency_monitor(self):
        """Monitor API response latencies."""
        while self.is_monitoring:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Test quote latency
                await self._test_quote_latency()
                
                # Test historical data latency
                await self._test_historical_latency()
                
            except Exception as e:
                logger.error(f"Error in latency monitor: {e}")
                await asyncio.sleep(60)
    
    async def _test_quote_latency(self):
        """Test quote API latency."""
        try:
            client = await self.auth_service.get_authenticated_client()
            
            for symbol in self.test_symbols:
                start_time = time.time()
                
                try:
                    await client.get_quote([symbol])
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # Determine status
                    if latency_ms > self.config["latency_thresholds"]["quote_critical_ms"]:
                        status = "critical"
                    elif latency_ms > self.config["latency_thresholds"]["quote_warning_ms"]:
                        status = "warning"
                    else:
                        status = "good"
                    
                    # Record metric
                    metric = LatencyMetric(
                        timestamp=datetime.now(),
                        operation=f"quote_{symbol}",
                        latency_ms=latency_ms,
                        target_ms=self.config["latency_thresholds"]["quote_warning_ms"],
                        status=status
                    )
                    
                    self.latency_history.append(metric)
                    
                    if status != "good":
                        await self._create_alert(
                            "latency",
                            f"High quote latency for {symbol}: {latency_ms:.1f}ms",
                            status,
                            {"symbol": symbol, "latency_ms": latency_ms}
                        )
                    
                except Exception as e:
                    logger.error(f"Error testing quote latency for {symbol}: {e}")
                    await self._create_alert(
                        "api_error",
                        f"Quote API error for {symbol}: {str(e)}",
                        "critical",
                        {"symbol": symbol, "error": str(e)}
                    )
                
                # Small delay between symbols
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in quote latency test: {e}")
    
    async def _test_historical_latency(self):
        """Test historical data API latency."""
        try:
            # Test with one symbol to avoid rate limits
            symbol = self.test_symbols[0]
            start_time = time.time()
            
            # This would test historical data API when implemented
            # For now, just simulate
            await asyncio.sleep(0.1)  # Simulate API call
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Record metric
            metric = LatencyMetric(
                timestamp=datetime.now(),
                operation=f"historical_{symbol}",
                latency_ms=latency_ms,
                target_ms=self.config["latency_thresholds"]["historical_warning_ms"],
                status="good"
            )
            
            self.latency_history.append(metric)
            
        except Exception as e:
            logger.error(f"Error in historical latency test: {e}")
    
    async def _freshness_monitor(self):
        """Monitor data freshness."""
        while self.is_monitoring:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check cached data freshness
                await self._check_cache_freshness()
                
            except Exception as e:
                logger.error(f"Error in freshness monitor: {e}")
                await asyncio.sleep(120)
    
    async def _check_cache_freshness(self):
        """Check freshness of cached data."""
        try:
            for symbol in self.test_symbols:
                # Check quote freshness
                quote = await self.cache.get_quote(symbol)
                if quote:
                    cached_at = datetime.fromisoformat(quote.get("cached_at", ""))
                    age_seconds = (datetime.now() - cached_at).total_seconds()
                    
                    # Determine status
                    if age_seconds > self.config["freshness_thresholds"]["quote_critical_seconds"]:
                        status = "critical"
                    elif age_seconds > self.config["freshness_thresholds"]["quote_warning_seconds"]:
                        status = "warning"
                    else:
                        status = "good"
                    
                    # Record metric
                    metric = DataQualityMetric(
                        timestamp=datetime.now(),
                        metric_type="freshness",
                        symbol=symbol,
                        value=age_seconds,
                        threshold=self.config["freshness_thresholds"]["quote_warning_seconds"],
                        status=status,
                        details={"data_type": "quote", "age_seconds": age_seconds}
                    )
                    
                    self.metrics_history.append(metric)
                    
                    if status != "good":
                        await self._create_alert(
                            "freshness",
                            f"Stale quote data for {symbol}: {age_seconds:.1f}s old",
                            status,
                            {"symbol": symbol, "age_seconds": age_seconds}
                        )
                
        except Exception as e:
            logger.error(f"Error checking cache freshness: {e}")
    
    async def _completeness_monitor(self):
        """Monitor data completeness."""
        while self.is_monitoring:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Check for missing data
                await self._check_data_completeness()
                
            except Exception as e:
                logger.error(f"Error in completeness monitor: {e}")
                await asyncio.sleep(600)
    
    async def _check_data_completeness(self):
        """Check for missing or incomplete data."""
        try:
            # Check if we have data for all test symbols
            missing_symbols = []
            
            for symbol in self.test_symbols:
                quote = await self.cache.get_quote(symbol)
                if not quote:
                    missing_symbols.append(symbol)
            
            if missing_symbols:
                missing_pct = (len(missing_symbols) / len(self.test_symbols)) * 100
                
                if missing_pct > self.config["completeness_thresholds"]["missing_data_critical_pct"]:
                    status = "critical"
                elif missing_pct > self.config["completeness_thresholds"]["missing_data_warning_pct"]:
                    status = "warning"
                else:
                    status = "good"
                
                if status != "good":
                    await self._create_alert(
                        "completeness",
                        f"Missing data for {len(missing_symbols)} symbols ({missing_pct:.1f}%)",
                        status,
                        {"missing_symbols": missing_symbols, "missing_pct": missing_pct}
                    )
            
        except Exception as e:
            logger.error(f"Error checking data completeness: {e}")
    
    async def _feed_health_monitor(self):
        """Monitor overall feed health."""
        while self.is_monitoring:
            try:
                await asyncio.sleep(120)  # Check every 2 minutes
                
                # Check trading hours
                trading_status = validate_trading_hours()
                
                if trading_status["is_trading_hours"]:
                    # During trading hours, expect fresh data
                    await self._check_trading_hours_health()
                else:
                    # Outside trading hours, different expectations
                    await self._check_non_trading_hours_health()
                
            except Exception as e:
                logger.error(f"Error in feed health monitor: {e}")
                await asyncio.sleep(240)
    
    async def _check_trading_hours_health(self):
        """Check feed health during trading hours."""
        try:
            # During trading hours, we expect:
            # 1. Low latency
            # 2. Fresh data
            # 3. Complete data coverage
            
            recent_latencies = [
                m.latency_ms for m in list(self.latency_history)[-10:]
                if m.timestamp > datetime.now() - timedelta(minutes=5)
            ]
            
            if recent_latencies:
                avg_latency = statistics.mean(recent_latencies)
                
                if avg_latency > self.config["latency_thresholds"]["quote_critical_ms"]:
                    await self._create_alert(
                        "feed_health",
                        f"High average latency during trading hours: {avg_latency:.1f}ms",
                        "critical",
                        {"avg_latency_ms": avg_latency}
                    )
            
        except Exception as e:
            logger.error(f"Error checking trading hours health: {e}")
    
    async def _check_non_trading_hours_health(self):
        """Check feed health outside trading hours."""
        try:
            # Outside trading hours, we expect:
            # 1. Cached data availability
            # 2. System responsiveness
            
            # Check if cache is responsive
            cache_healthy = await self.cache.health_check()
            
            if not cache_healthy:
                await self._create_alert(
                    "feed_health",
                    "Cache health check failed during non-trading hours",
                    "warning",
                    {"cache_healthy": cache_healthy}
                )
            
        except Exception as e:
            logger.error(f"Error checking non-trading hours health: {e}")
    
    async def _create_alert(self, alert_type: str, message: str, severity: str, details: Dict[str, Any]):
        """Create and queue an alert."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "message": message,
            "severity": severity,
            "details": details
        }
        
        self.alerts.append(alert)
        logger.warning(f"Data quality alert [{severity}]: {message}")
        
        # Update statistics
        if severity in ["warning", "critical"]:
            if alert_type == "latency":
                self.stats["latency_issues"] += 1
            else:
                self.stats["quality_issues"] += 1
    
    async def _alert_processor(self):
        """Process and potentially escalate alerts."""
        while self.is_monitoring:
            try:
                await asyncio.sleep(60)  # Process every minute
                
                # Count recent critical alerts
                recent_critical = [
                    alert for alert in list(self.alerts)[-20:]
                    if alert["severity"] == "critical"
                    and datetime.fromisoformat(alert["timestamp"]) > datetime.now() - timedelta(minutes=5)
                ]
                
                if len(recent_critical) >= 3:
                    logger.critical(f"Multiple critical data quality issues detected: {len(recent_critical)} in last 5 minutes")
                
            except Exception as e:
                logger.error(f"Error in alert processor: {e}")
                await asyncio.sleep(120)
    
    async def get_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        try:
            # Calculate statistics
            recent_metrics = [
                m for m in list(self.metrics_history)[-100:]
                if m.timestamp > datetime.now() - timedelta(hours=1)
            ]
            
            recent_latencies = [
                m for m in list(self.latency_history)[-100:]
                if m.timestamp > datetime.now() - timedelta(hours=1)
            ]
            
            recent_alerts = [
                a for a in list(self.alerts)[-50:]
                if datetime.fromisoformat(a["timestamp"]) > datetime.now() - timedelta(hours=1)
            ]
            
            # Calculate averages
            avg_latency = statistics.mean([m.latency_ms for m in recent_latencies]) if recent_latencies else 0
            
            quality_score = self._calculate_quality_score(recent_metrics, recent_latencies, recent_alerts)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "quality_score": quality_score,
                "statistics": {
                    **self.stats,
                    "avg_latency_ms": round(avg_latency, 2),
                    "recent_metrics_count": len(recent_metrics),
                    "recent_alerts_count": len(recent_alerts)
                },
                "recent_alerts": [dict(a) for a in list(recent_alerts)[-10:]],
                "latency_summary": {
                    "avg_ms": round(avg_latency, 2),
                    "samples": len(recent_latencies)
                },
                "cache_stats": await self.cache.get_cache_stats(),
                "monitoring_status": "active" if self.is_monitoring else "inactive"
            }
            
        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
            return {"error": str(e)}
    
    def _calculate_quality_score(self, metrics: List, latencies: List, alerts: List) -> float:
        """Calculate overall data quality score (0-100)."""
        try:
            score = 100.0
            
            # Deduct for high latency
            if latencies:
                avg_latency = statistics.mean([m.latency_ms for m in latencies])
                if avg_latency > 500:
                    score -= 30
                elif avg_latency > 200:
                    score -= 15
            
            # Deduct for alerts
            critical_alerts = [a for a in alerts if a["severity"] == "critical"]
            warning_alerts = [a for a in alerts if a["severity"] == "warning"]
            
            score -= len(critical_alerts) * 10
            score -= len(warning_alerts) * 5
            
            # Deduct for poor metrics
            poor_metrics = [m for m in metrics if m.status in ["warning", "critical"]]
            score -= len(poor_metrics) * 2
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 50.0  # Default score on error


# Global monitor instance
data_quality_monitor = DataQualityMonitor()
