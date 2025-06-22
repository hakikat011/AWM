"""
Health check and connectivity validation for Zerodha API integration.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .client import ZerodhaClient, ZerodhaAPIError
from .auth import ZerodhaAuthService
from .config import config_manager
from .utils import validate_trading_hours

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    
    service: str
    status: str  # "healthy", "degraded", "unhealthy"
    response_time_ms: float
    timestamp: datetime
    details: Dict[str, Any]
    error: Optional[str] = None


class ZerodhaHealthChecker:
    """Comprehensive health checker for Zerodha integration."""
    
    def __init__(self):
        self.auth_service = ZerodhaAuthService()
        self.config = config_manager.get_zerodha_config()
        self.last_check_time = None
        self.check_history: List[HealthCheckResult] = []
        
    async def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """
        Run comprehensive health check covering all Zerodha integration aspects.
        
        Returns:
            Comprehensive health status report
        """
        logger.info("Starting comprehensive Zerodha health check")
        start_time = time.time()
        
        checks = [
            self._check_authentication(),
            self._check_api_connectivity(),
            self._check_market_data_access(),
            self._check_order_management_access(),
            self._check_account_access(),
            self._check_rate_limiting(),
            self._check_trading_hours(),
            self._check_system_resources()
        ]
        
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        # Process results
        health_results = []
        overall_status = "healthy"
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                health_results.append(HealthCheckResult(
                    service=f"check_{i}",
                    status="unhealthy",
                    response_time_ms=0,
                    timestamp=datetime.now(),
                    details={},
                    error=str(result)
                ))
                overall_status = "unhealthy"
            else:
                health_results.append(result)
                if result.status == "unhealthy":
                    overall_status = "unhealthy"
                elif result.status == "degraded" and overall_status != "unhealthy":
                    overall_status = "degraded"
        
        total_time = (time.time() - start_time) * 1000
        self.last_check_time = datetime.now()
        
        # Store in history (keep last 100 checks)
        self.check_history.extend(health_results)
        if len(self.check_history) > 100:
            self.check_history = self.check_history[-100:]
        
        return {
            "overall_status": overall_status,
            "total_check_time_ms": total_time,
            "timestamp": self.last_check_time.isoformat(),
            "checks": [
                {
                    "service": result.service,
                    "status": result.status,
                    "response_time_ms": result.response_time_ms,
                    "details": result.details,
                    "error": result.error
                }
                for result in health_results
            ],
            "summary": self._generate_summary(health_results)
        }
    
    async def _check_authentication(self) -> HealthCheckResult:
        """Check authentication status and token validity."""
        start_time = time.time()
        
        try:
            auth_status = await self.auth_service.get_auth_status()
            response_time = (time.time() - start_time) * 1000
            
            if auth_status.get("is_authenticated"):
                status = "healthy"
                details = {
                    "authenticated": True,
                    "user_id": auth_status.get("user_id"),
                    "time_remaining_hours": auth_status.get("time_remaining_hours", 0),
                    "expires_at": auth_status.get("expires_at")
                }
                
                # Check if token is expiring soon (less than 2 hours)
                if auth_status.get("time_remaining_hours", 0) < 2:
                    status = "degraded"
                    details["warning"] = "Access token expires soon"
                
            else:
                status = "unhealthy"
                details = {
                    "authenticated": False,
                    "message": auth_status.get("message", "Not authenticated")
                }
            
            return HealthCheckResult(
                service="authentication",
                status=status,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details=details
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service="authentication",
                status="unhealthy",
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={},
                error=str(e)
            )
    
    async def _check_api_connectivity(self) -> HealthCheckResult:
        """Check basic API connectivity."""
        start_time = time.time()
        
        try:
            client = await self.auth_service.get_authenticated_client()
            profile = await client.get_profile()
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                service="api_connectivity",
                status="healthy",
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={
                    "user_id": profile.get("user_id"),
                    "user_name": profile.get("user_name"),
                    "broker": profile.get("broker")
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service="api_connectivity",
                status="unhealthy",
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={},
                error=str(e)
            )
    
    async def _check_market_data_access(self) -> HealthCheckResult:
        """Check market data access."""
        start_time = time.time()
        
        try:
            client = await self.auth_service.get_authenticated_client()
            
            # Test with a common stock
            test_symbol = "NSE:RELIANCE"
            quote = await client.get_quote([test_symbol])
            response_time = (time.time() - start_time) * 1000
            
            if test_symbol in quote:
                return HealthCheckResult(
                    service="market_data",
                    status="healthy",
                    response_time_ms=response_time,
                    timestamp=datetime.now(),
                    details={
                        "test_symbol": test_symbol,
                        "last_price": quote[test_symbol].get("last_price"),
                        "timestamp": quote[test_symbol].get("timestamp")
                    }
                )
            else:
                return HealthCheckResult(
                    service="market_data",
                    status="degraded",
                    response_time_ms=response_time,
                    timestamp=datetime.now(),
                    details={"message": "No data for test symbol"},
                    error="No market data returned"
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service="market_data",
                status="unhealthy",
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={},
                error=str(e)
            )
    
    async def _check_order_management_access(self) -> HealthCheckResult:
        """Check order management API access."""
        start_time = time.time()
        
        try:
            client = await self.auth_service.get_authenticated_client()
            orders = await client.get_orders()
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                service="order_management",
                status="healthy",
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={
                    "orders_count": len(orders),
                    "can_access_orders": True
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service="order_management",
                status="unhealthy",
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={},
                error=str(e)
            )
    
    async def _check_account_access(self) -> HealthCheckResult:
        """Check account and portfolio access."""
        start_time = time.time()
        
        try:
            client = await self.auth_service.get_authenticated_client()
            
            # Get margins and positions
            margins = await client.get_margins()
            positions = await client.get_positions()
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                service="account_access",
                status="healthy",
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={
                    "available_cash": margins.get("equity", {}).get("available", {}).get("cash", 0),
                    "positions_count": len(positions.get("net", [])),
                    "can_access_account": True
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service="account_access",
                status="unhealthy",
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={},
                error=str(e)
            )
    
    async def _check_rate_limiting(self) -> HealthCheckResult:
        """Check rate limiting behavior."""
        start_time = time.time()
        
        try:
            client = await self.auth_service.get_authenticated_client()
            
            # Make multiple rapid requests to test rate limiting
            tasks = []
            for _ in range(3):  # 3 requests (at the limit)
                tasks.append(client.get_profile())
            
            await asyncio.gather(*tasks)
            response_time = (time.time() - start_time) * 1000
            
            # If we get here without errors, rate limiting is working
            status = "healthy" if response_time < 5000 else "degraded"  # 5 second threshold
            
            return HealthCheckResult(
                service="rate_limiting",
                status=status,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={
                    "requests_completed": 3,
                    "rate_limiting_working": True
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service="rate_limiting",
                status="degraded",
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={},
                error=str(e)
            )
    
    async def _check_trading_hours(self) -> HealthCheckResult:
        """Check trading hours and market status."""
        start_time = time.time()
        
        try:
            trading_status = validate_trading_hours()
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                service="trading_hours",
                status="healthy",
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details=trading_status
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service="trading_hours",
                status="unhealthy",
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={},
                error=str(e)
            )
    
    async def _check_system_resources(self) -> HealthCheckResult:
        """Check system resources and performance."""
        start_time = time.time()
        
        try:
            # Basic system checks
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine status based on resource usage
            status = "healthy"
            if cpu_percent > 80 or memory.percent > 85 or disk.percent > 90:
                status = "degraded"
            if cpu_percent > 95 or memory.percent > 95 or disk.percent > 95:
                status = "unhealthy"
            
            return HealthCheckResult(
                service="system_resources",
                status=status,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                    "available_memory_gb": memory.available / (1024**3)
                }
            )
            
        except ImportError:
            # psutil not available
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service="system_resources",
                status="degraded",
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={"message": "System monitoring not available"}
            )
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service="system_resources",
                status="unhealthy",
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details={},
                error=str(e)
            )
    
    def _generate_summary(self, results: List[HealthCheckResult]) -> Dict[str, Any]:
        """Generate summary statistics from health check results."""
        total_checks = len(results)
        healthy_count = sum(1 for r in results if r.status == "healthy")
        degraded_count = sum(1 for r in results if r.status == "degraded")
        unhealthy_count = sum(1 for r in results if r.status == "unhealthy")
        
        avg_response_time = sum(r.response_time_ms for r in results) / total_checks if total_checks > 0 else 0
        
        return {
            "total_checks": total_checks,
            "healthy_count": healthy_count,
            "degraded_count": degraded_count,
            "unhealthy_count": unhealthy_count,
            "health_percentage": (healthy_count / total_checks * 100) if total_checks > 0 else 0,
            "average_response_time_ms": avg_response_time
        }
    
    async def get_health_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get health check history for the specified number of hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            {
                "service": result.service,
                "status": result.status,
                "response_time_ms": result.response_time_ms,
                "timestamp": result.timestamp.isoformat(),
                "details": result.details,
                "error": result.error
            }
            for result in self.check_history
            if result.timestamp >= cutoff_time
        ]
    
    async def close(self):
        """Clean up resources."""
        await self.auth_service.close()


# Global health checker instance
health_checker = ZerodhaHealthChecker()
