"""
Risk Management Engine for AWM system.
Real-time risk monitoring and automatic risk controls.
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Set
from decimal import Decimal
from enum import Enum
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from shared.mcp_client.server import MCPServer, validate_required_fields, sanitize_input
from shared.database.connection import init_database, close_database, db_manager
from shared.models.trading import RiskLevel

logger = logging.getLogger(__name__)


class RiskViolationType(Enum):
    POSITION_SIZE_EXCEEDED = "POSITION_SIZE_EXCEEDED"
    DAILY_LOSS_EXCEEDED = "DAILY_LOSS_EXCEEDED"
    PORTFOLIO_RISK_EXCEEDED = "PORTFOLIO_RISK_EXCEEDED"
    CONCENTRATION_RISK = "CONCENTRATION_RISK"
    CORRELATION_RISK = "CORRELATION_RISK"
    LEVERAGE_EXCEEDED = "LEVERAGE_EXCEEDED"
    DRAWDOWN_EXCEEDED = "DRAWDOWN_EXCEEDED"


class RiskAction(Enum):
    ALLOW = "ALLOW"
    WARN = "WARN"
    BLOCK = "BLOCK"
    LIQUIDATE = "LIQUIDATE"


class RiskManagementEngine(MCPServer):
    """Real-time risk management engine with automatic controls."""
    
    def __init__(self):
        host = os.getenv("RISK_MANAGEMENT_ENGINE_HOST", "0.0.0.0")
        port = int(os.getenv("RISK_MANAGEMENT_ENGINE_PORT", "8010"))
        super().__init__("risk_management_engine", host, port)
        
        # Risk parameters from environment
        self.max_position_size = float(os.getenv("MAX_POSITION_SIZE", "100000"))
        self.max_daily_loss = float(os.getenv("MAX_DAILY_LOSS", "10000"))
        self.max_portfolio_risk = float(os.getenv("MAX_PORTFOLIO_RISK", "0.02"))
        self.max_leverage = float(os.getenv("MAX_LEVERAGE", "3.0"))
        self.max_drawdown = float(os.getenv("MAX_DRAWDOWN", "0.15"))
        
        # Concentration limits
        self.max_single_position = 0.20  # 20% of portfolio
        self.max_sector_concentration = 0.40  # 40% in single sector
        self.max_correlation_exposure = 0.60  # 60% in correlated positions
        
        # Monitoring state
        self.active_portfolios: Set[str] = set()
        self.risk_violations: Dict[str, List[Dict]] = {}
        self.emergency_mode = False
        
        # Register handlers
        self.register_handlers()
    
    def register_handlers(self):
        """Register all MCP handlers."""
        
        @self.handler("evaluate_trade_risk")
        async def evaluate_trade_risk(content: Dict[str, Any]) -> Dict[str, Any]:
            """Evaluate risk for a proposed trade."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["trade_proposal"])
            
            trade_proposal = content["trade_proposal"]
            
            try:
                risk_evaluation = await self._evaluate_trade_risk(trade_proposal)
                
                # Log risk evaluation
                await self._log_risk_evaluation(trade_proposal, risk_evaluation)
                
                return risk_evaluation
                
            except Exception as e:
                logger.error(f"Error evaluating trade risk: {e}")
                return {
                    "action": RiskAction.BLOCK.value,
                    "reason": f"Risk evaluation failed: {str(e)}",
                    "violations": []
                }
        
        @self.handler("monitor_portfolio_risk")
        async def monitor_portfolio_risk(content: Dict[str, Any]) -> Dict[str, Any]:
            """Monitor real-time portfolio risk."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["portfolio_id"])
            
            portfolio_id = content["portfolio_id"]
            
            try:
                risk_status = await self._monitor_portfolio_risk(portfolio_id)
                return risk_status
                
            except Exception as e:
                logger.error(f"Error monitoring portfolio risk: {e}")
                return {
                    "status": "ERROR",
                    "error": str(e)
                }
        
        @self.handler("get_risk_limits")
        async def get_risk_limits(content: Dict[str, Any]) -> Dict[str, Any]:
            """Get current risk limits and parameters."""
            return {
                "max_position_size": self.max_position_size,
                "max_daily_loss": self.max_daily_loss,
                "max_portfolio_risk": self.max_portfolio_risk,
                "max_leverage": self.max_leverage,
                "max_drawdown": self.max_drawdown,
                "max_single_position": self.max_single_position,
                "max_sector_concentration": self.max_sector_concentration,
                "max_correlation_exposure": self.max_correlation_exposure,
                "emergency_mode": self.emergency_mode
            }
        
        @self.handler("update_risk_limits")
        async def update_risk_limits(content: Dict[str, Any]) -> Dict[str, Any]:
            """Update risk limits (admin function)."""
            content = await sanitize_input(content)
            
            updated_params = []
            
            if "max_position_size" in content:
                self.max_position_size = float(content["max_position_size"])
                updated_params.append("max_position_size")
            
            if "max_daily_loss" in content:
                self.max_daily_loss = float(content["max_daily_loss"])
                updated_params.append("max_daily_loss")
            
            if "max_portfolio_risk" in content:
                self.max_portfolio_risk = float(content["max_portfolio_risk"])
                updated_params.append("max_portfolio_risk")
            
            logger.info(f"Updated risk parameters: {updated_params}")
            
            return {
                "status": "SUCCESS",
                "updated_parameters": updated_params,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        @self.handler("emergency_stop")
        async def emergency_stop(content: Dict[str, Any]) -> Dict[str, Any]:
            """Emergency stop - block all new trades."""
            content = await sanitize_input(content)
            reason = content.get("reason", "Manual emergency stop")
            
            self.emergency_mode = True
            
            logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
            
            # Send alerts
            await self._send_emergency_alert(reason)
            
            return {
                "status": "EMERGENCY_STOP_ACTIVATED",
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        @self.handler("reset_emergency")
        async def reset_emergency(content: Dict[str, Any]) -> Dict[str, Any]:
            """Reset emergency mode (admin function)."""
            content = await sanitize_input(content)
            authorized_by = content.get("authorized_by", "system")
            
            self.emergency_mode = False
            
            logger.warning(f"Emergency mode reset by: {authorized_by}")
            
            return {
                "status": "EMERGENCY_MODE_RESET",
                "authorized_by": authorized_by,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        @self.handler("get_risk_violations")
        async def get_risk_violations(content: Dict[str, Any]) -> Dict[str, Any]:
            """Get current risk violations."""
            portfolio_id = content.get("portfolio_id")
            
            if portfolio_id:
                violations = self.risk_violations.get(portfolio_id, [])
            else:
                violations = {}
                for pid, viols in self.risk_violations.items():
                    violations[pid] = viols
            
            return {
                "violations": violations,
                "emergency_mode": self.emergency_mode,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _evaluate_trade_risk(self, trade_proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate risk for a proposed trade."""
        
        # Check emergency mode first
        if self.emergency_mode:
            return {
                "action": RiskAction.BLOCK.value,
                "reason": "Emergency mode active - all trades blocked",
                "violations": [{"type": "EMERGENCY_MODE", "severity": "CRITICAL"}]
            }
        
        portfolio_id = trade_proposal["portfolio_id"]
        symbol = trade_proposal["symbol"]
        side = trade_proposal["side"]
        quantity = trade_proposal["quantity"]
        entry_price = trade_proposal.get("entry_price", 0)
        
        violations = []
        max_severity = "LOW"
        
        # Get current portfolio data
        portfolio_data = await self._get_portfolio_data(portfolio_id)
        
        # 1. Check position size limits
        position_value = quantity * entry_price
        if position_value > self.max_position_size:
            violations.append({
                "type": RiskViolationType.POSITION_SIZE_EXCEEDED.value,
                "severity": "HIGH",
                "current_value": position_value,
                "limit": self.max_position_size,
                "message": f"Position value {position_value} exceeds limit {self.max_position_size}"
            })
            max_severity = "HIGH"
        
        # 2. Check portfolio concentration
        portfolio_value = float(portfolio_data.get("current_value", 0))
        if portfolio_value > 0:
            position_weight = position_value / portfolio_value
            if position_weight > self.max_single_position:
                violations.append({
                    "type": RiskViolationType.CONCENTRATION_RISK.value,
                    "severity": "MEDIUM",
                    "position_weight": position_weight,
                    "limit": self.max_single_position,
                    "message": f"Position weight {position_weight:.2%} exceeds limit {self.max_single_position:.2%}"
                })
                if max_severity == "LOW":
                    max_severity = "MEDIUM"
        
        # 3. Check daily loss limits
        daily_pnl = await self._calculate_daily_pnl(portfolio_id)
        if daily_pnl < -self.max_daily_loss:
            violations.append({
                "type": RiskViolationType.DAILY_LOSS_EXCEEDED.value,
                "severity": "HIGH",
                "daily_pnl": daily_pnl,
                "limit": -self.max_daily_loss,
                "message": f"Daily loss {daily_pnl} exceeds limit {self.max_daily_loss}"
            })
            max_severity = "HIGH"
        
        # 4. Check portfolio risk (VaR)
        portfolio_risk = await self._calculate_portfolio_var(portfolio_id)
        if portfolio_risk > self.max_portfolio_risk:
            violations.append({
                "type": RiskViolationType.PORTFOLIO_RISK_EXCEEDED.value,
                "severity": "HIGH",
                "portfolio_risk": portfolio_risk,
                "limit": self.max_portfolio_risk,
                "message": f"Portfolio VaR {portfolio_risk:.2%} exceeds limit {self.max_portfolio_risk:.2%}"
            })
            max_severity = "HIGH"
        
        # 5. Check leverage
        leverage = await self._calculate_leverage(portfolio_id, trade_proposal)
        if leverage > self.max_leverage:
            violations.append({
                "type": RiskViolationType.LEVERAGE_EXCEEDED.value,
                "severity": "MEDIUM",
                "leverage": leverage,
                "limit": self.max_leverage,
                "message": f"Leverage {leverage:.2f} exceeds limit {self.max_leverage:.2f}"
            })
            if max_severity == "LOW":
                max_severity = "MEDIUM"
        
        # Determine action based on violations
        if max_severity == "HIGH":
            action = RiskAction.BLOCK
        elif max_severity == "MEDIUM":
            action = RiskAction.WARN
        else:
            action = RiskAction.ALLOW
        
        # Store violations for monitoring
        if violations:
            if portfolio_id not in self.risk_violations:
                self.risk_violations[portfolio_id] = []
            
            self.risk_violations[portfolio_id].extend(violations)
            
            # Send alerts for high severity violations
            if max_severity == "HIGH":
                await self._send_risk_alert(portfolio_id, violations)
        
        return {
            "action": action.value,
            "severity": max_severity,
            "violations": violations,
            "portfolio_id": portfolio_id,
            "trade_proposal": trade_proposal,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def _get_portfolio_data(self, portfolio_id: str) -> Dict[str, Any]:
        """Get portfolio data from database."""
        try:
            query = "SELECT * FROM portfolios WHERE id = $1"
            portfolio = await db_manager.execute_query(query, portfolio_id, fetch="one")
            return portfolio or {}
        except Exception as e:
            logger.error(f"Error getting portfolio data: {e}")
            return {}
    
    async def _calculate_daily_pnl(self, portfolio_id: str) -> float:
        """Calculate daily P&L for portfolio."""
        try:
            today = datetime.now(timezone.utc).date()
            
            query = """
                SELECT SUM(
                    CASE 
                        WHEN side = 'BUY' THEN -value 
                        ELSE value 
                    END
                ) as daily_pnl
                FROM trades 
                WHERE portfolio_id = $1 
                AND DATE(executed_at) = $2
            """
            
            result = await db_manager.execute_query(query, portfolio_id, today, fetch="one")
            return float(result.get("daily_pnl", 0)) if result else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating daily P&L: {e}")
            return 0.0
    
    async def _calculate_portfolio_var(self, portfolio_id: str) -> float:
        """Calculate portfolio Value at Risk."""
        try:
            # Get latest risk metrics
            query = """
                SELECT var_1d 
                FROM risk_metrics 
                WHERE portfolio_id = $1 
                ORDER BY time DESC 
                LIMIT 1
            """
            
            result = await db_manager.execute_query(query, portfolio_id, fetch="one")
            return abs(float(result.get("var_1d", 0))) if result else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            return 0.0
    
    async def _calculate_leverage(self, portfolio_id: str, trade_proposal: Dict[str, Any]) -> float:
        """Calculate portfolio leverage including proposed trade."""
        try:
            # Get current portfolio value
            portfolio_data = await self._get_portfolio_data(portfolio_id)
            portfolio_value = float(portfolio_data.get("current_value", 0))
            
            if portfolio_value == 0:
                return 0.0
            
            # Get current positions value
            query = """
                SELECT SUM(ABS(market_value)) as total_exposure
                FROM portfolio_positions 
                WHERE portfolio_id = $1
            """
            
            result = await db_manager.execute_query(query, portfolio_id, fetch="one")
            current_exposure = float(result.get("total_exposure", 0)) if result else 0.0
            
            # Add proposed trade exposure
            trade_value = trade_proposal["quantity"] * trade_proposal.get("entry_price", 0)
            total_exposure = current_exposure + trade_value
            
            return total_exposure / portfolio_value
            
        except Exception as e:
            logger.error(f"Error calculating leverage: {e}")
            return 0.0
    
    async def _monitor_portfolio_risk(self, portfolio_id: str) -> Dict[str, Any]:
        """Monitor real-time portfolio risk."""
        try:
            # Add to active monitoring
            self.active_portfolios.add(portfolio_id)
            
            # Get current risk metrics
            risk_metrics = await self._get_current_risk_metrics(portfolio_id)
            
            # Check for violations
            violations = []
            
            # Check VaR
            var_1d = risk_metrics.get("var_1d", 0)
            if abs(var_1d) > self.max_portfolio_risk:
                violations.append({
                    "type": "PORTFOLIO_RISK_EXCEEDED",
                    "current": abs(var_1d),
                    "limit": self.max_portfolio_risk
                })
            
            # Check drawdown
            max_drawdown = risk_metrics.get("max_drawdown", 0)
            if abs(max_drawdown) > self.max_drawdown:
                violations.append({
                    "type": "DRAWDOWN_EXCEEDED",
                    "current": abs(max_drawdown),
                    "limit": self.max_drawdown
                })
            
            # Determine risk level
            if violations:
                risk_level = "HIGH"
            elif abs(var_1d) > self.max_portfolio_risk * 0.8:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            return {
                "portfolio_id": portfolio_id,
                "risk_level": risk_level,
                "risk_metrics": risk_metrics,
                "violations": violations,
                "monitoring_active": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error monitoring portfolio risk: {e}")
            return {
                "portfolio_id": portfolio_id,
                "status": "ERROR",
                "error": str(e)
            }
    
    async def _get_current_risk_metrics(self, portfolio_id: str) -> Dict[str, Any]:
        """Get current risk metrics for portfolio."""
        try:
            query = """
                SELECT * FROM risk_metrics 
                WHERE portfolio_id = $1 
                ORDER BY time DESC 
                LIMIT 1
            """
            
            result = await db_manager.execute_query(query, portfolio_id, fetch="one")
            
            if result:
                return {
                    "var_1d": float(result.get("var_1d", 0)),
                    "var_5d": float(result.get("var_5d", 0)),
                    "expected_shortfall": float(result.get("expected_shortfall", 0)),
                    "max_drawdown": float(result.get("max_drawdown", 0)),
                    "sharpe_ratio": float(result.get("sharpe_ratio", 0)),
                    "volatility": float(result.get("volatility", 0)),
                    "last_updated": result.get("time")
                }
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return {}
    
    async def _send_risk_alert(self, portfolio_id: str, violations: List[Dict]) -> None:
        """Send risk alert for violations."""
        try:
            alert_data = {
                "alert_type": "RISK_VIOLATION",
                "severity": "HIGH",
                "portfolio_id": portfolio_id,
                "violations": violations,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Store alert in database
            query = """
                INSERT INTO system_alerts 
                (alert_type, severity, title, message, data)
                VALUES ($1, $2, $3, $4, $5)
            """
            
            await db_manager.execute_query(
                query,
                "RISK_VIOLATION",
                "ERROR",
                f"Risk violations detected for portfolio {portfolio_id}",
                json.dumps(violations),
                json.dumps(alert_data)
            )
            
            logger.warning(f"Risk alert sent for portfolio {portfolio_id}: {len(violations)} violations")
            
        except Exception as e:
            logger.error(f"Error sending risk alert: {e}")
    
    async def _send_emergency_alert(self, reason: str) -> None:
        """Send emergency alert."""
        try:
            alert_data = {
                "alert_type": "EMERGENCY_STOP",
                "severity": "CRITICAL",
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Store alert in database
            query = """
                INSERT INTO system_alerts 
                (alert_type, severity, title, message, data)
                VALUES ($1, $2, $3, $4, $5)
            """
            
            await db_manager.execute_query(
                query,
                "EMERGENCY_STOP",
                "CRITICAL",
                "Emergency stop activated",
                reason,
                json.dumps(alert_data)
            )
            
            logger.critical(f"Emergency alert sent: {reason}")
            
        except Exception as e:
            logger.error(f"Error sending emergency alert: {e}")
    
    async def _log_risk_evaluation(self, trade_proposal: Dict[str, Any], evaluation: Dict[str, Any]) -> None:
        """Log risk evaluation to database."""
        try:
            query = """
                INSERT INTO risk_evaluations 
                (portfolio_id, trade_proposal, evaluation_result, timestamp)
                VALUES ($1, $2, $3, $4)
            """
            
            await db_manager.execute_query(
                query,
                trade_proposal.get("portfolio_id"),
                json.dumps(trade_proposal),
                json.dumps(evaluation),
                datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Error logging risk evaluation: {e}")


async def main():
    """Main function to run the Risk Management Engine."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize database
    await init_database()
    
    try:
        # Create and start server
        engine = RiskManagementEngine()
        logger.info("Starting Risk Management Engine...")
        await engine.start()
    finally:
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
