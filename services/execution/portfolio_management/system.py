"""
Portfolio Management System for AWM system.
Real-time portfolio tracking, P&L calculation, and performance analytics.
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from decimal import Decimal
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from shared.mcp_client.server import MCPServer, validate_required_fields, sanitize_input
from shared.mcp_client.base import MCPClient
from shared.database.connection import init_database, close_database, db_manager

logger = logging.getLogger(__name__)


class PortfolioManagementSystem(MCPServer):
    """Real-time portfolio tracking and management system."""
    
    def __init__(self):
        host = os.getenv("PORTFOLIO_MANAGEMENT_SYSTEM_HOST", "0.0.0.0")
        port = int(os.getenv("PORTFOLIO_MANAGEMENT_SYSTEM_PORT", "8012"))
        super().__init__("portfolio_management_system", host, port)
        
        # Configuration
        self.market_data_url = "http://market-data-server:8001"
        
        # Portfolio cache for real-time updates
        self.portfolio_cache: Dict[str, Dict[str, Any]] = {}
        self.position_cache: Dict[str, Dict[str, Any]] = {}
        
        # MCP client for communication
        self.mcp_client = MCPClient("portfolio_management_system")
        
        # Register handlers
        self.register_handlers()
        
        # Start background tasks
        asyncio.create_task(self._portfolio_update_loop())
        asyncio.create_task(self._performance_calculation_loop())
    
    def register_handlers(self):
        """Register all MCP handlers."""
        
        @self.handler("get_portfolio")
        async def get_portfolio(content: Dict[str, Any]) -> Dict[str, Any]:
            """Get portfolio details."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["portfolio_id"])
            
            portfolio_id = content["portfolio_id"]
            
            try:
                portfolio = await self._get_portfolio_details(portfolio_id)
                return portfolio
                
            except Exception as e:
                logger.error(f"Error getting portfolio {portfolio_id}: {e}")
                return {
                    "status": "ERROR",
                    "error": str(e)
                }
        
        @self.handler("get_positions")
        async def get_positions(content: Dict[str, Any]) -> Dict[str, Any]:
            """Get portfolio positions."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["portfolio_id"])
            
            portfolio_id = content["portfolio_id"]
            
            try:
                positions = await self._get_portfolio_positions(portfolio_id)
                return {
                    "portfolio_id": portfolio_id,
                    "positions": positions,
                    "count": len(positions),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting positions for {portfolio_id}: {e}")
                return {
                    "status": "ERROR",
                    "error": str(e)
                }
        
        @self.handler("get_performance")
        async def get_performance(content: Dict[str, Any]) -> Dict[str, Any]:
            """Get portfolio performance metrics."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["portfolio_id"])
            
            portfolio_id = content["portfolio_id"]
            days = content.get("days", 30)
            
            try:
                performance = await self._get_portfolio_performance(portfolio_id, days)
                return performance
                
            except Exception as e:
                logger.error(f"Error getting performance for {portfolio_id}: {e}")
                return {
                    "status": "ERROR",
                    "error": str(e)
                }
        
        @self.handler("update_positions")
        async def update_positions(content: Dict[str, Any]) -> Dict[str, Any]:
            """Update portfolio positions (called after trades)."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["portfolio_id", "trade"])
            
            portfolio_id = content["portfolio_id"]
            trade = content["trade"]
            
            try:
                update_result = await self._update_positions_from_trade(portfolio_id, trade)
                return update_result
                
            except Exception as e:
                logger.error(f"Error updating positions: {e}")
                return {
                    "status": "ERROR",
                    "error": str(e)
                }
        
        @self.handler("calculate_pnl")
        async def calculate_pnl(content: Dict[str, Any]) -> Dict[str, Any]:
            """Calculate real-time P&L."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["portfolio_id"])
            
            portfolio_id = content["portfolio_id"]
            
            try:
                pnl_data = await self._calculate_realtime_pnl(portfolio_id)
                return pnl_data
                
            except Exception as e:
                logger.error(f"Error calculating P&L for {portfolio_id}: {e}")
                return {
                    "status": "ERROR",
                    "error": str(e)
                }
        
        @self.handler("rebalance_portfolio")
        async def rebalance_portfolio(content: Dict[str, Any]) -> Dict[str, Any]:
            """Generate portfolio rebalancing recommendations."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["portfolio_id"])
            
            portfolio_id = content["portfolio_id"]
            target_allocation = content.get("target_allocation", {})
            
            try:
                rebalance_plan = await self._generate_rebalance_plan(portfolio_id, target_allocation)
                return rebalance_plan
                
            except Exception as e:
                logger.error(f"Error generating rebalance plan: {e}")
                return {
                    "status": "ERROR",
                    "error": str(e)
                }
        
        @self.handler("get_analytics")
        async def get_analytics(content: Dict[str, Any]) -> Dict[str, Any]:
            """Get portfolio analytics and attribution."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["portfolio_id"])
            
            portfolio_id = content["portfolio_id"]
            
            try:
                analytics = await self._get_portfolio_analytics(portfolio_id)
                return analytics
                
            except Exception as e:
                logger.error(f"Error getting analytics for {portfolio_id}: {e}")
                return {
                    "status": "ERROR",
                    "error": str(e)
                }
    
    async def _get_portfolio_details(self, portfolio_id: str) -> Dict[str, Any]:
        """Get comprehensive portfolio details."""
        
        # Check cache first
        if portfolio_id in self.portfolio_cache:
            cached_portfolio = self.portfolio_cache[portfolio_id]
            # Return cached if recent (within 1 minute)
            if (datetime.now(timezone.utc) - cached_portfolio["last_updated"]).seconds < 60:
                return cached_portfolio["data"]
        
        try:
            # Get portfolio from database
            query = "SELECT * FROM portfolios WHERE id = $1"
            portfolio = await db_manager.execute_query(query, portfolio_id, fetch="one")
            
            if not portfolio:
                return {"error": "Portfolio not found"}
            
            # Get current positions
            positions = await self._get_portfolio_positions(portfolio_id)
            
            # Calculate current values
            total_market_value = sum(pos["market_value"] for pos in positions)
            available_cash = float(portfolio["available_cash"])
            current_value = total_market_value + available_cash
            
            # Calculate P&L
            initial_capital = float(portfolio["initial_capital"])
            total_pnl = current_value - initial_capital
            total_return = (total_pnl / initial_capital) if initial_capital > 0 else 0
            
            portfolio_data = {
                "portfolio_id": portfolio_id,
                "name": portfolio["name"],
                "description": portfolio["description"],
                "initial_capital": initial_capital,
                "current_value": current_value,
                "available_cash": available_cash,
                "invested_value": total_market_value,
                "total_pnl": total_pnl,
                "total_return": total_return,
                "number_of_positions": len(positions),
                "is_active": portfolio["is_active"],
                "created_at": portfolio["created_at"].isoformat(),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
            # Update cache
            self.portfolio_cache[portfolio_id] = {
                "data": portfolio_data,
                "last_updated": datetime.now(timezone.utc)
            }
            
            return portfolio_data
            
        except Exception as e:
            logger.error(f"Error getting portfolio details: {e}")
            raise
    
    async def _get_portfolio_positions(self, portfolio_id: str) -> List[Dict[str, Any]]:
        """Get current portfolio positions with real-time values."""
        
        try:
            query = """
                SELECT pp.*, i.symbol, i.name as instrument_name
                FROM portfolio_positions pp
                JOIN instruments i ON pp.instrument_id = i.id
                WHERE pp.portfolio_id = $1 AND pp.quantity != 0
                ORDER BY pp.market_value DESC
            """
            
            positions = await db_manager.execute_query(query, portfolio_id, fetch="all")
            
            # Update with current market prices
            updated_positions = []
            for position in positions:
                try:
                    # Get current market price
                    current_price = await self._get_current_price(position["symbol"])
                    
                    quantity = position["quantity"]
                    average_price = float(position["average_price"])
                    market_value = quantity * current_price
                    unrealized_pnl = market_value - (quantity * average_price)
                    unrealized_return = (unrealized_pnl / (quantity * average_price)) if average_price > 0 else 0
                    
                    position_data = {
                        "symbol": position["symbol"],
                        "instrument_name": position["instrument_name"],
                        "quantity": quantity,
                        "average_price": average_price,
                        "current_price": current_price,
                        "market_value": market_value,
                        "unrealized_pnl": unrealized_pnl,
                        "unrealized_return": unrealized_return,
                        "realized_pnl": float(position["realized_pnl"]),
                        "last_updated": datetime.now(timezone.utc).isoformat()
                    }
                    
                    updated_positions.append(position_data)
                    
                    # Update position in database
                    await self._update_position_values(position["id"], current_price, market_value, unrealized_pnl)
                    
                except Exception as e:
                    logger.error(f"Error updating position {position['symbol']}: {e}")
                    # Use cached values if price update fails
                    updated_positions.append({
                        "symbol": position["symbol"],
                        "instrument_name": position["instrument_name"],
                        "quantity": position["quantity"],
                        "average_price": float(position["average_price"]),
                        "current_price": float(position["current_price"]),
                        "market_value": float(position["market_value"]),
                        "unrealized_pnl": float(position["unrealized_pnl"]),
                        "unrealized_return": float(position["unrealized_pnl"]) / (position["quantity"] * float(position["average_price"])) if float(position["average_price"]) > 0 else 0,
                        "realized_pnl": float(position["realized_pnl"]),
                        "last_updated": position["last_updated"].isoformat()
                    })
            
            return updated_positions
            
        except Exception as e:
            logger.error(f"Error getting portfolio positions: {e}")
            return []
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol."""
        
        try:
            async with self.mcp_client as client:
                response = await client.send_request(
                    self.market_data_url,
                    "get_current_quote",
                    {"symbol": symbol}
                )
                
                return float(response.content.get("price", 0))
                
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return 0.0
    
    async def _update_position_values(self, position_id: str, current_price: float, market_value: float, unrealized_pnl: float) -> None:
        """Update position values in database."""
        
        try:
            query = """
                UPDATE portfolio_positions 
                SET current_price = $1, market_value = $2, unrealized_pnl = $3, last_updated = $4
                WHERE id = $5
            """
            
            await db_manager.execute_query(
                query,
                Decimal(str(current_price)),
                Decimal(str(market_value)),
                Decimal(str(unrealized_pnl)),
                datetime.now(timezone.utc),
                position_id
            )
            
        except Exception as e:
            logger.error(f"Error updating position values: {e}")
    
    async def _get_portfolio_performance(self, portfolio_id: str, days: int) -> Dict[str, Any]:
        """Get portfolio performance metrics over specified period."""
        
        try:
            # Get portfolio value history
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            # For now, calculate simple performance metrics
            # In a real implementation, you would have historical portfolio values
            
            current_portfolio = await self._get_portfolio_details(portfolio_id)
            
            # Get trades in the period for performance calculation
            query = """
                SELECT t.*, i.symbol
                FROM trades t
                JOIN instruments i ON t.instrument_id = i.id
                WHERE t.portfolio_id = $1 
                AND t.executed_at >= $2 
                AND t.executed_at <= $3
                ORDER BY t.executed_at
            """
            
            trades = await db_manager.execute_query(query, portfolio_id, start_date, end_date, fetch="all")
            
            # Calculate performance metrics
            total_trade_value = sum(float(trade["value"]) for trade in trades)
            total_pnl = current_portfolio.get("total_pnl", 0)
            total_return = current_portfolio.get("total_return", 0)
            
            # Calculate daily returns (simplified)
            daily_returns = []
            if trades:
                # This is a simplified calculation
                # Real implementation would track daily portfolio values
                for trade in trades:
                    if trade["side"] == "SELL":
                        # Calculate return for this trade
                        trade_return = (float(trade["price"]) - float(trade["value"]) / trade["quantity"]) / (float(trade["value"]) / trade["quantity"])
                        daily_returns.append(trade_return)
            
            # Calculate risk metrics
            if daily_returns:
                volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized
                sharpe_ratio = (np.mean(daily_returns) * 252) / volatility if volatility > 0 else 0
                max_drawdown = min(daily_returns) if daily_returns else 0
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
            
            return {
                "portfolio_id": portfolio_id,
                "period_days": days,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "total_return": total_return,
                "total_pnl": total_pnl,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "number_of_trades": len(trades),
                "total_trade_value": total_trade_value,
                "current_value": current_portfolio.get("current_value", 0),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio performance: {e}")
            return {
                "status": "ERROR",
                "error": str(e)
            }
    
    async def _update_positions_from_trade(self, portfolio_id: str, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Update portfolio positions after a trade execution."""
        
        try:
            symbol = trade["symbol"]
            side = trade["side"]
            quantity = trade["quantity"]
            price = float(trade["price"])
            
            # Get instrument ID
            instrument_query = "SELECT id FROM instruments WHERE symbol = $1"
            instrument = await db_manager.execute_query(instrument_query, symbol, fetch="one")
            
            if not instrument:
                return {"status": "ERROR", "error": f"Instrument {symbol} not found"}
            
            instrument_id = instrument["id"]
            
            # Get current position
            position_query = """
                SELECT * FROM portfolio_positions 
                WHERE portfolio_id = $1 AND instrument_id = $2
            """
            position = await db_manager.execute_query(position_query, portfolio_id, instrument_id, fetch="one")
            
            if position:
                # Update existing position
                current_quantity = position["quantity"]
                current_avg_price = float(position["average_price"])
                current_realized_pnl = float(position["realized_pnl"])
                
                if side == "BUY":
                    # Add to position
                    new_quantity = current_quantity + quantity
                    new_avg_price = ((current_quantity * current_avg_price) + (quantity * price)) / new_quantity
                    new_realized_pnl = current_realized_pnl
                else:  # SELL
                    # Reduce position
                    new_quantity = current_quantity - quantity
                    new_avg_price = current_avg_price  # Keep same average price
                    # Calculate realized P&L
                    realized_pnl_from_trade = quantity * (price - current_avg_price)
                    new_realized_pnl = current_realized_pnl + realized_pnl_from_trade
                
                # Update position
                update_query = """
                    UPDATE portfolio_positions 
                    SET quantity = $1, average_price = $2, realized_pnl = $3, last_updated = $4
                    WHERE portfolio_id = $5 AND instrument_id = $6
                """
                
                await db_manager.execute_query(
                    update_query,
                    new_quantity,
                    Decimal(str(new_avg_price)),
                    Decimal(str(new_realized_pnl)),
                    datetime.now(timezone.utc),
                    portfolio_id,
                    instrument_id
                )
                
            else:
                # Create new position (only for BUY orders)
                if side == "BUY":
                    insert_query = """
                        INSERT INTO portfolio_positions 
                        (portfolio_id, instrument_id, quantity, average_price, current_price, 
                         market_value, unrealized_pnl, realized_pnl)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """
                    
                    market_value = quantity * price
                    
                    await db_manager.execute_query(
                        insert_query,
                        portfolio_id,
                        instrument_id,
                        quantity,
                        Decimal(str(price)),
                        Decimal(str(price)),
                        Decimal(str(market_value)),
                        Decimal("0"),
                        Decimal("0")
                    )
            
            # Update portfolio cash
            trade_value = quantity * price
            if side == "BUY":
                cash_change = -trade_value
            else:
                cash_change = trade_value
            
            portfolio_update_query = """
                UPDATE portfolios 
                SET available_cash = available_cash + $1, updated_at = $2
                WHERE id = $3
            """
            
            await db_manager.execute_query(
                portfolio_update_query,
                Decimal(str(cash_change)),
                datetime.now(timezone.utc),
                portfolio_id
            )
            
            return {
                "status": "SUCCESS",
                "portfolio_id": portfolio_id,
                "symbol": symbol,
                "trade_processed": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating positions from trade: {e}")
            return {
                "status": "ERROR",
                "error": str(e)
            }
    
    async def _calculate_realtime_pnl(self, portfolio_id: str) -> Dict[str, Any]:
        """Calculate real-time P&L for portfolio."""
        
        try:
            positions = await self._get_portfolio_positions(portfolio_id)
            
            total_unrealized_pnl = sum(pos["unrealized_pnl"] for pos in positions)
            total_realized_pnl = sum(pos["realized_pnl"] for pos in positions)
            total_pnl = total_unrealized_pnl + total_realized_pnl
            
            # Get portfolio details for initial capital
            portfolio = await self._get_portfolio_details(portfolio_id)
            initial_capital = portfolio.get("initial_capital", 0)
            
            total_return = (total_pnl / initial_capital) if initial_capital > 0 else 0
            
            return {
                "portfolio_id": portfolio_id,
                "total_pnl": total_pnl,
                "unrealized_pnl": total_unrealized_pnl,
                "realized_pnl": total_realized_pnl,
                "total_return": total_return,
                "current_value": portfolio.get("current_value", 0),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating real-time P&L: {e}")
            return {
                "status": "ERROR",
                "error": str(e)
            }
    
    async def _generate_rebalance_plan(self, portfolio_id: str, target_allocation: Dict[str, float]) -> Dict[str, Any]:
        """Generate portfolio rebalancing plan."""
        
        try:
            positions = await self._get_portfolio_positions(portfolio_id)
            portfolio = await self._get_portfolio_details(portfolio_id)
            
            current_value = portfolio.get("current_value", 0)
            
            if current_value == 0:
                return {"status": "ERROR", "error": "Portfolio has no value"}
            
            # Calculate current allocation
            current_allocation = {}
            for position in positions:
                symbol = position["symbol"]
                weight = position["market_value"] / current_value
                current_allocation[symbol] = weight
            
            # Generate rebalancing trades
            rebalance_trades = []
            
            for symbol, target_weight in target_allocation.items():
                current_weight = current_allocation.get(symbol, 0)
                weight_diff = target_weight - current_weight
                
                if abs(weight_diff) > 0.01:  # Only rebalance if difference > 1%
                    target_value = target_weight * current_value
                    current_value_symbol = current_weight * current_value
                    trade_value = target_value - current_value_symbol
                    
                    # Get current price
                    current_price = await self._get_current_price(symbol)
                    
                    if current_price > 0:
                        quantity = int(trade_value / current_price)
                        
                        if quantity != 0:
                            rebalance_trades.append({
                                "symbol": symbol,
                                "side": "BUY" if quantity > 0 else "SELL",
                                "quantity": abs(quantity),
                                "current_weight": current_weight,
                                "target_weight": target_weight,
                                "trade_value": trade_value
                            })
            
            return {
                "portfolio_id": portfolio_id,
                "current_allocation": current_allocation,
                "target_allocation": target_allocation,
                "rebalance_trades": rebalance_trades,
                "total_trades": len(rebalance_trades),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating rebalance plan: {e}")
            return {
                "status": "ERROR",
                "error": str(e)
            }
    
    async def _get_portfolio_analytics(self, portfolio_id: str) -> Dict[str, Any]:
        """Get comprehensive portfolio analytics."""
        
        try:
            positions = await self._get_portfolio_positions(portfolio_id)
            portfolio = await self._get_portfolio_details(portfolio_id)
            performance = await self._get_portfolio_performance(portfolio_id, 30)
            
            # Sector allocation
            sector_allocation = {}
            # This would require sector information in the instruments table
            
            # Top performers
            top_performers = sorted(positions, key=lambda x: x["unrealized_return"], reverse=True)[:5]
            worst_performers = sorted(positions, key=lambda x: x["unrealized_return"])[:5]
            
            # Concentration metrics
            total_value = portfolio.get("current_value", 0)
            position_weights = [pos["market_value"] / total_value for pos in positions if total_value > 0]
            
            if position_weights:
                max_position_weight = max(position_weights)
                herfindahl_index = sum(w**2 for w in position_weights)
                top_5_concentration = sum(sorted(position_weights, reverse=True)[:5])
            else:
                max_position_weight = 0
                herfindahl_index = 0
                top_5_concentration = 0
            
            return {
                "portfolio_id": portfolio_id,
                "summary": {
                    "total_value": portfolio.get("current_value", 0),
                    "total_pnl": portfolio.get("total_pnl", 0),
                    "total_return": portfolio.get("total_return", 0),
                    "number_of_positions": len(positions)
                },
                "performance": performance,
                "concentration": {
                    "max_position_weight": max_position_weight,
                    "herfindahl_index": herfindahl_index,
                    "top_5_concentration": top_5_concentration
                },
                "top_performers": top_performers,
                "worst_performers": worst_performers,
                "sector_allocation": sector_allocation,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio analytics: {e}")
            return {
                "status": "ERROR",
                "error": str(e)
            }
    
    async def _portfolio_update_loop(self):
        """Background loop to update portfolio values."""
        
        while True:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                # Get all active portfolios
                query = "SELECT id FROM portfolios WHERE is_active = true"
                portfolios = await db_manager.execute_query(query, fetch="all")
                
                for portfolio in portfolios:
                    portfolio_id = portfolio["id"]
                    try:
                        # Update positions with current prices
                        await self._get_portfolio_positions(portfolio_id)
                        
                        # Update portfolio cache
                        await self._get_portfolio_details(portfolio_id)
                        
                    except Exception as e:
                        logger.error(f"Error updating portfolio {portfolio_id}: {e}")
                
            except Exception as e:
                logger.error(f"Error in portfolio update loop: {e}")
                await asyncio.sleep(300)  # Wait longer on error
    
    async def _performance_calculation_loop(self):
        """Background loop to calculate and store performance metrics."""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Calculate every hour
                
                # Get all active portfolios
                query = "SELECT id FROM portfolios WHERE is_active = true"
                portfolios = await db_manager.execute_query(query, fetch="all")
                
                for portfolio in portfolios:
                    portfolio_id = portfolio["id"]
                    try:
                        # Calculate and store performance metrics
                        performance = await self._get_portfolio_performance(portfolio_id, 1)
                        
                        # Store in risk_metrics table for historical tracking
                        # This would be expanded to include more comprehensive metrics
                        
                    except Exception as e:
                        logger.error(f"Error calculating performance for {portfolio_id}: {e}")
                
            except Exception as e:
                logger.error(f"Error in performance calculation loop: {e}")
                await asyncio.sleep(3600)  # Wait longer on error


async def main():
    """Main function to run the Portfolio Management System."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize database
    await init_database()
    
    try:
        # Create and start server
        pms = PortfolioManagementSystem()
        logger.info("Starting Portfolio Management System...")
        await pms.start()
    finally:
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
