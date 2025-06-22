"""
Order Management System (OMS) for AWM system.
Handles order lifecycle management, execution tracking, and broker integration.
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from decimal import Decimal
from enum import Enum
import uuid

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from shared.mcp_client.server import MCPServer, validate_required_fields, sanitize_input
from shared.mcp_client.base import MCPClient
from shared.database.connection import init_database, close_database, db_manager
from shared.models.trading import OrderType, OrderSide, OrderStatus

logger = logging.getLogger(__name__)


class OrderState(Enum):
    PENDING_RISK_CHECK = "PENDING_RISK_CHECK"
    RISK_APPROVED = "RISK_APPROVED"
    RISK_REJECTED = "RISK_REJECTED"
    PENDING_EXECUTION = "PENDING_EXECUTION"
    SUBMITTED_TO_BROKER = "SUBMITTED_TO_BROKER"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    ERROR = "ERROR"


class OrderManagementSystem(MCPServer):
    """Comprehensive order lifecycle management system."""
    
    def __init__(self):
        host = os.getenv("OMS_HOST", "0.0.0.0")
        port = int(os.getenv("OMS_PORT", "8011"))
        super().__init__("order_management_system", host, port)
        
        # Configuration
        self.paper_trading = os.getenv("PAPER_TRADING_MODE", "true").lower() == "true"
        self.risk_management_url = os.getenv("RISK_MANAGEMENT_ENGINE_URL", "http://risk-management-engine:8010")
        self.trade_execution_url = os.getenv("TRADE_EXECUTION_SERVER_URL", "http://trade-execution-server:8005")
        
        # Order tracking
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        self.order_history: List[Dict[str, Any]] = []
        
        # MCP client for communication
        self.mcp_client = MCPClient("oms")
        
        # Register handlers
        self.register_handlers()
        
        # Start background tasks
        asyncio.create_task(self._order_monitoring_loop())
        asyncio.create_task(self._reconciliation_loop())
    
    def register_handlers(self):
        """Register all MCP handlers."""
        
        @self.handler("place_order")
        async def place_order(content: Dict[str, Any]) -> Dict[str, Any]:
            """Place a new order."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["symbol", "side", "quantity", "portfolio_id"])
            
            try:
                order_result = await self._place_order(content)
                return order_result
                
            except Exception as e:
                logger.error(f"Error placing order: {e}")
                return {
                    "status": "ERROR",
                    "error": str(e),
                    "order_request": content
                }
        
        @self.handler("cancel_order")
        async def cancel_order(content: Dict[str, Any]) -> Dict[str, Any]:
            """Cancel an existing order."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["order_id"])
            
            order_id = content["order_id"]
            
            try:
                cancel_result = await self._cancel_order(order_id)
                return cancel_result
                
            except Exception as e:
                logger.error(f"Error cancelling order {order_id}: {e}")
                return {
                    "status": "ERROR",
                    "error": str(e),
                    "order_id": order_id
                }
        
        @self.handler("get_order_status")
        async def get_order_status(content: Dict[str, Any]) -> Dict[str, Any]:
            """Get status of an order."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["order_id"])
            
            order_id = content["order_id"]
            
            try:
                order_status = await self._get_order_status(order_id)
                return order_status
                
            except Exception as e:
                logger.error(f"Error getting order status for {order_id}: {e}")
                return {
                    "status": "ERROR",
                    "error": str(e),
                    "order_id": order_id
                }
        
        @self.handler("get_active_orders")
        async def get_active_orders(content: Dict[str, Any]) -> Dict[str, Any]:
            """Get all active orders."""
            portfolio_id = content.get("portfolio_id")
            
            try:
                if portfolio_id:
                    orders = await self._get_orders_by_portfolio(portfolio_id)
                else:
                    orders = await self._get_all_active_orders()
                
                return {
                    "orders": orders,
                    "count": len(orders),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting active orders: {e}")
                return {
                    "status": "ERROR",
                    "error": str(e)
                }
        
        @self.handler("get_order_history")
        async def get_order_history(content: Dict[str, Any]) -> Dict[str, Any]:
            """Get order history."""
            portfolio_id = content.get("portfolio_id")
            limit = content.get("limit", 100)
            
            try:
                history = await self._get_order_history(portfolio_id, limit)
                
                return {
                    "orders": history,
                    "count": len(history),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting order history: {e}")
                return {
                    "status": "ERROR",
                    "error": str(e)
                }
        
        @self.handler("reconcile_orders")
        async def reconcile_orders(content: Dict[str, Any]) -> Dict[str, Any]:
            """Reconcile orders with broker."""
            try:
                reconciliation_result = await self._reconcile_with_broker()
                return reconciliation_result
                
            except Exception as e:
                logger.error(f"Error during reconciliation: {e}")
                return {
                    "status": "ERROR",
                    "error": str(e)
                }
    
    async def _place_order(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """Place a new order through the complete workflow."""
        
        # Generate order ID
        order_id = str(uuid.uuid4())
        
        # Create order object
        order = {
            "id": order_id,
            "portfolio_id": order_request["portfolio_id"],
            "symbol": order_request["symbol"],
            "side": order_request["side"],
            "quantity": order_request["quantity"],
            "order_type": order_request.get("order_type", "MARKET"),
            "price": order_request.get("price"),
            "trigger_price": order_request.get("trigger_price"),
            "state": OrderState.PENDING_RISK_CHECK.value,
            "status": OrderStatus.PENDING.value,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "filled_quantity": 0,
            "average_price": 0,
            "broker_order_id": None,
            "error_message": None
        }
        
        # Store order in tracking
        self.active_orders[order_id] = order
        
        try:
            # Step 1: Risk check
            risk_result = await self._perform_risk_check(order)
            
            if risk_result["action"] != "ALLOW":
                order["state"] = OrderState.RISK_REJECTED.value
                order["status"] = OrderStatus.REJECTED.value
                order["error_message"] = risk_result.get("reason", "Risk check failed")
                
                await self._update_order_in_db(order)
                
                return {
                    "order_id": order_id,
                    "status": "REJECTED",
                    "reason": order["error_message"],
                    "risk_evaluation": risk_result
                }
            
            # Step 2: Risk approved
            order["state"] = OrderState.RISK_APPROVED.value
            order["updated_at"] = datetime.now(timezone.utc)
            
            # Step 3: Execute order
            execution_result = await self._execute_order(order)
            
            if execution_result["status"] == "ERROR":
                order["state"] = OrderState.ERROR.value
                order["status"] = OrderStatus.REJECTED.value
                order["error_message"] = execution_result.get("error", "Execution failed")
            else:
                order["state"] = OrderState.SUBMITTED_TO_BROKER.value
                order["status"] = OrderStatus.OPEN.value
                order["broker_order_id"] = execution_result.get("broker_order_id")
            
            order["updated_at"] = datetime.now(timezone.utc)
            
            # Store in database
            await self._update_order_in_db(order)
            
            return {
                "order_id": order_id,
                "status": order["status"],
                "state": order["state"],
                "broker_order_id": order.get("broker_order_id"),
                "execution_result": execution_result
            }
            
        except Exception as e:
            order["state"] = OrderState.ERROR.value
            order["status"] = OrderStatus.REJECTED.value
            order["error_message"] = str(e)
            order["updated_at"] = datetime.now(timezone.utc)
            
            await self._update_order_in_db(order)
            
            logger.error(f"Error in order workflow: {e}")
            return {
                "order_id": order_id,
                "status": "ERROR",
                "error": str(e)
            }
    
    async def _perform_risk_check(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Perform risk check through Risk Management Engine."""
        
        trade_proposal = {
            "portfolio_id": order["portfolio_id"],
            "symbol": order["symbol"],
            "side": order["side"],
            "quantity": order["quantity"],
            "entry_price": order.get("price", 0),
            "order_type": order["order_type"]
        }
        
        try:
            async with self.mcp_client as client:
                response = await client.send_request(
                    self.risk_management_url,
                    "evaluate_trade_risk",
                    {"trade_proposal": trade_proposal}
                )
                
                return response.content
                
        except Exception as e:
            logger.error(f"Risk check failed: {e}")
            return {
                "action": "BLOCK",
                "reason": f"Risk check service unavailable: {str(e)}"
            }
    
    async def _execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Execute order through Trade Execution Server or simulate."""
        
        if self.paper_trading:
            return await self._simulate_order_execution(order)
        else:
            return await self._execute_real_order(order)
    
    async def _simulate_order_execution(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate order execution for paper trading."""
        
        # Simulate immediate fill for market orders
        if order["order_type"] == "MARKET":
            # Get current market price
            try:
                async with self.mcp_client as client:
                    quote_response = await client.send_request(
                        "http://market-data-server:8001",
                        "get_current_quote",
                        {"symbol": order["symbol"]}
                    )
                    
                    market_price = quote_response.content.get("price", order.get("price", 0))
                    
                    # Simulate slippage
                    slippage = 0.001  # 0.1%
                    if order["side"] == "BUY":
                        execution_price = market_price * (1 + slippage)
                    else:
                        execution_price = market_price * (1 - slippage)
                    
                    # Update order
                    order["filled_quantity"] = order["quantity"]
                    order["average_price"] = execution_price
                    order["state"] = OrderState.FILLED.value
                    order["status"] = OrderStatus.COMPLETE.value
                    
                    # Create trade record
                    await self._create_trade_record(order, execution_price)
                    
                    return {
                        "status": "FILLED",
                        "broker_order_id": f"PAPER_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        "execution_price": execution_price,
                        "filled_quantity": order["quantity"]
                    }
                    
            except Exception as e:
                logger.error(f"Error simulating order execution: {e}")
                return {
                    "status": "ERROR",
                    "error": str(e)
                }
        
        else:
            # For limit orders, just mark as submitted
            return {
                "status": "SUBMITTED",
                "broker_order_id": f"PAPER_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
    
    async def _execute_real_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Execute real order through broker API."""
        
        try:
            async with self.mcp_client as client:
                response = await client.send_request(
                    self.trade_execution_url,
                    "place_order",
                    {
                        "symbol": order["symbol"],
                        "side": order["side"],
                        "quantity": order["quantity"],
                        "order_type": order["order_type"],
                        "price": order.get("price"),
                        "portfolio_id": order["portfolio_id"]
                    }
                )
                
                return response.content
                
        except Exception as e:
            logger.error(f"Real order execution failed: {e}")
            return {
                "status": "ERROR",
                "error": str(e)
            }
    
    async def _create_trade_record(self, order: Dict[str, Any], execution_price: float) -> None:
        """Create trade record in database."""
        
        try:
            # Get instrument ID
            instrument_query = "SELECT id FROM instruments WHERE symbol = $1"
            instrument = await db_manager.execute_query(instrument_query, order["symbol"], fetch="one")
            
            if not instrument:
                logger.error(f"Instrument not found: {order['symbol']}")
                return
            
            # Calculate trade value
            trade_value = order["quantity"] * execution_price
            
            # Insert trade record
            trade_query = """
                INSERT INTO trades 
                (order_id, portfolio_id, instrument_id, side, quantity, price, value, executed_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """
            
            await db_manager.execute_query(
                trade_query,
                order["id"],
                order["portfolio_id"],
                instrument["id"],
                order["side"],
                order["quantity"],
                Decimal(str(execution_price)),
                Decimal(str(trade_value)),
                datetime.now(timezone.utc)
            )
            
            logger.info(f"Trade record created for order {order['id']}")
            
        except Exception as e:
            logger.error(f"Error creating trade record: {e}")
    
    async def _cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order."""
        
        if order_id not in self.active_orders:
            return {
                "status": "ERROR",
                "error": "Order not found"
            }
        
        order = self.active_orders[order_id]
        
        # Check if order can be cancelled
        if order["state"] in [OrderState.FILLED.value, OrderState.CANCELLED.value, OrderState.REJECTED.value]:
            return {
                "status": "ERROR",
                "error": f"Cannot cancel order in state: {order['state']}"
            }
        
        try:
            # Cancel with broker if real trading
            if not self.paper_trading and order.get("broker_order_id"):
                # Cancel with broker
                async with self.mcp_client as client:
                    response = await client.send_request(
                        self.trade_execution_url,
                        "cancel_order",
                        {"broker_order_id": order["broker_order_id"]}
                    )
            
            # Update order state
            order["state"] = OrderState.CANCELLED.value
            order["status"] = OrderStatus.CANCELLED.value
            order["updated_at"] = datetime.now(timezone.utc)
            
            await self._update_order_in_db(order)
            
            return {
                "status": "CANCELLED",
                "order_id": order_id,
                "timestamp": order["updated_at"].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return {
                "status": "ERROR",
                "error": str(e)
            }
    
    async def _get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get current status of an order."""
        
        # Check active orders first
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            return {
                "order_id": order_id,
                "status": order["status"],
                "state": order["state"],
                "filled_quantity": order["filled_quantity"],
                "average_price": order["average_price"],
                "created_at": order["created_at"].isoformat(),
                "updated_at": order["updated_at"].isoformat()
            }
        
        # Check database
        try:
            query = "SELECT * FROM orders WHERE id = $1"
            order = await db_manager.execute_query(query, order_id, fetch="one")
            
            if order:
                return {
                    "order_id": order_id,
                    "status": order["status"],
                    "filled_quantity": order["filled_quantity"],
                    "average_price": float(order["average_price"]),
                    "created_at": order["created_at"].isoformat(),
                    "updated_at": order["updated_at"].isoformat()
                }
            else:
                return {
                    "status": "ERROR",
                    "error": "Order not found"
                }
                
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return {
                "status": "ERROR",
                "error": str(e)
            }
    
    async def _get_all_active_orders(self) -> List[Dict[str, Any]]:
        """Get all active orders."""
        
        try:
            query = """
                SELECT o.*, i.symbol 
                FROM orders o
                JOIN instruments i ON o.instrument_id = i.id
                WHERE o.status IN ('PENDING', 'OPEN')
                ORDER BY o.created_at DESC
            """
            
            orders = await db_manager.execute_query(query, fetch="all")
            
            return [
                {
                    "order_id": order["id"],
                    "symbol": order["symbol"],
                    "side": order["order_side"],
                    "quantity": order["quantity"],
                    "order_type": order["order_type"],
                    "status": order["status"],
                    "created_at": order["created_at"].isoformat()
                }
                for order in orders
            ]
            
        except Exception as e:
            logger.error(f"Error getting active orders: {e}")
            return []
    
    async def _get_orders_by_portfolio(self, portfolio_id: str) -> List[Dict[str, Any]]:
        """Get orders for a specific portfolio."""
        
        try:
            query = """
                SELECT o.*, i.symbol 
                FROM orders o
                JOIN instruments i ON o.instrument_id = i.id
                WHERE o.portfolio_id = $1 AND o.status IN ('PENDING', 'OPEN')
                ORDER BY o.created_at DESC
            """
            
            orders = await db_manager.execute_query(query, portfolio_id, fetch="all")
            
            return [
                {
                    "order_id": order["id"],
                    "symbol": order["symbol"],
                    "side": order["order_side"],
                    "quantity": order["quantity"],
                    "order_type": order["order_type"],
                    "status": order["status"],
                    "created_at": order["created_at"].isoformat()
                }
                for order in orders
            ]
            
        except Exception as e:
            logger.error(f"Error getting orders by portfolio: {e}")
            return []
    
    async def _get_order_history(self, portfolio_id: Optional[str], limit: int) -> List[Dict[str, Any]]:
        """Get order history."""
        
        try:
            if portfolio_id:
                query = """
                    SELECT o.*, i.symbol 
                    FROM orders o
                    JOIN instruments i ON o.instrument_id = i.id
                    WHERE o.portfolio_id = $1
                    ORDER BY o.created_at DESC
                    LIMIT $2
                """
                orders = await db_manager.execute_query(query, portfolio_id, limit, fetch="all")
            else:
                query = """
                    SELECT o.*, i.symbol 
                    FROM orders o
                    JOIN instruments i ON o.instrument_id = i.id
                    ORDER BY o.created_at DESC
                    LIMIT $1
                """
                orders = await db_manager.execute_query(query, limit, fetch="all")
            
            return [
                {
                    "order_id": order["id"],
                    "symbol": order["symbol"],
                    "side": order["order_side"],
                    "quantity": order["quantity"],
                    "order_type": order["order_type"],
                    "status": order["status"],
                    "filled_quantity": order["filled_quantity"],
                    "average_price": float(order["average_price"]),
                    "created_at": order["created_at"].isoformat(),
                    "updated_at": order["updated_at"].isoformat()
                }
                for order in orders
            ]
            
        except Exception as e:
            logger.error(f"Error getting order history: {e}")
            return []
    
    async def _update_order_in_db(self, order: Dict[str, Any]) -> None:
        """Update order in database."""
        
        try:
            # Get instrument ID
            instrument_query = "SELECT id FROM instruments WHERE symbol = $1"
            instrument = await db_manager.execute_query(instrument_query, order["symbol"], fetch="one")
            
            if not instrument:
                logger.error(f"Instrument not found: {order['symbol']}")
                return
            
            # Upsert order
            query = """
                INSERT INTO orders 
                (id, portfolio_id, instrument_id, order_type, order_side, quantity, 
                 price, trigger_price, status, filled_quantity, average_price, 
                 order_id, placed_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    filled_quantity = EXCLUDED.filled_quantity,
                    average_price = EXCLUDED.average_price,
                    updated_at = EXCLUDED.updated_at
            """
            
            await db_manager.execute_query(
                query,
                order["id"],
                order["portfolio_id"],
                instrument["id"],
                order["order_type"],
                order["side"],
                order["quantity"],
                Decimal(str(order.get("price", 0))),
                Decimal(str(order.get("trigger_price", 0))),
                order["status"],
                order["filled_quantity"],
                Decimal(str(order["average_price"])),
                order.get("broker_order_id"),
                order["created_at"],
                order["updated_at"]
            )
            
        except Exception as e:
            logger.error(f"Error updating order in database: {e}")
    
    async def _order_monitoring_loop(self):
        """Background loop to monitor order status."""
        
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Monitor active orders
                for order_id, order in list(self.active_orders.items()):
                    if order["state"] in [OrderState.SUBMITTED_TO_BROKER.value, OrderState.PARTIALLY_FILLED.value]:
                        await self._check_order_status_with_broker(order)
                
            except Exception as e:
                logger.error(f"Error in order monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _check_order_status_with_broker(self, order: Dict[str, Any]) -> None:
        """Check order status with broker."""
        
        if self.paper_trading:
            # In paper trading, simulate fills for limit orders occasionally
            if order["order_type"] == "LIMIT" and order["state"] == OrderState.SUBMITTED_TO_BROKER.value:
                # 10% chance of fill per check
                import random
                if random.random() < 0.1:
                    order["filled_quantity"] = order["quantity"]
                    order["average_price"] = order.get("price", 0)
                    order["state"] = OrderState.FILLED.value
                    order["status"] = OrderStatus.COMPLETE.value
                    order["updated_at"] = datetime.now(timezone.utc)
                    
                    await self._create_trade_record(order, order["average_price"])
                    await self._update_order_in_db(order)
            return
        
        # Real broker status check would go here
        # This would integrate with the actual broker API
    
    async def _reconciliation_loop(self):
        """Background loop for order reconciliation."""
        
        while True:
            try:
                await asyncio.sleep(300)  # Reconcile every 5 minutes
                await self._reconcile_with_broker()
                
            except Exception as e:
                logger.error(f"Error in reconciliation loop: {e}")
                await asyncio.sleep(600)  # Wait longer on error
    
    async def _reconcile_with_broker(self) -> Dict[str, Any]:
        """Reconcile orders with broker."""
        
        if self.paper_trading:
            return {
                "status": "SUCCESS",
                "message": "Paper trading mode - no reconciliation needed",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        # Real reconciliation logic would go here
        # This would compare our order states with broker states
        
        return {
            "status": "SUCCESS",
            "reconciled_orders": 0,
            "discrepancies": 0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


async def main():
    """Main function to run the Order Management System."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize database
    await init_database()
    
    try:
        # Create and start server
        oms = OrderManagementSystem()
        logger.info("Starting Order Management System...")
        await oms.start()
    finally:
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
