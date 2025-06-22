"""
Trade Execution MCP Server for AWM system.
Handles order placement, modification, and cancellation through Zerodha Kite Connect API.
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from decimal import Decimal
from enum import Enum

# Add the project root to Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from shared.mcp_client.server import MCPServer, validate_required_fields, sanitize_input
from shared.database.connection import init_database, close_database, db_manager
from shared.zerodha import ZerodhaAuthService, ZerodhaClient, format_indian_symbol, get_lot_size, get_tick_size

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    MODIFIED = "MODIFIED"


class TradeExecutionServer(MCPServer):
    """Trade Execution MCP Server implementation with Zerodha integration."""
    
    def __init__(self):
        host = os.getenv("TRADE_EXECUTION_SERVER_HOST", "0.0.0.0")
        port = int(os.getenv("TRADE_EXECUTION_SERVER_PORT", "8005"))
        super().__init__("trade_execution_server", host, port)
        
        # Zerodha integration
        self.zerodha_auth = ZerodhaAuthService()
        self.zerodha_client = None
        
        # Configuration
        self.paper_trading = os.getenv("PAPER_TRADING_MODE", "true").lower() == "true"
        
        # Order tracking
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        self.order_history: List[Dict[str, Any]] = []
        
        # Execution statistics
        self.execution_stats = {
            "total_orders": 0,
            "successful_orders": 0,
            "failed_orders": 0,
            "cancelled_orders": 0,
            "avg_execution_time_ms": 0,
            "last_execution": None
        }
        
        # Register handlers
        self.register_handlers()
        
        # Start background tasks
        asyncio.create_task(self._initialize_zerodha_client())
        asyncio.create_task(self._order_status_monitor())
    
    async def _initialize_zerodha_client(self):
        """Initialize Zerodha client connection."""
        try:
            if not self.paper_trading and await self.zerodha_auth.is_authenticated():
                self.zerodha_client = await self.zerodha_auth.get_authenticated_client()
                logger.info("Zerodha trade execution client initialized successfully")
            else:
                logger.info("Running in paper trading mode or Zerodha not authenticated")
        except Exception as e:
            logger.error(f"Failed to initialize Zerodha client: {e}")
    
    async def _order_status_monitor(self):
        """Background task to monitor order status."""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                if not self.paper_trading and self.zerodha_client:
                    await self._sync_order_status()
                
            except Exception as e:
                logger.error(f"Error in order status monitor: {e}")
                await asyncio.sleep(30)
    
    async def _sync_order_status(self):
        """Sync order status with Zerodha."""
        try:
            # Get all orders from Zerodha
            zerodha_orders = await self.zerodha_client.get_orders()
            
            # Update local order tracking
            for zerodha_order in zerodha_orders:
                broker_order_id = zerodha_order.get("order_id")
                if broker_order_id:
                    await self._update_order_from_zerodha(zerodha_order)
                    
        except Exception as e:
            logger.error(f"Error syncing order status: {e}")
    
    async def _update_order_from_zerodha(self, zerodha_order: Dict[str, Any]):
        """Update local order from Zerodha order data."""
        try:
            broker_order_id = zerodha_order.get("order_id")
            
            # Find local order by broker order ID
            local_order = None
            for order_id, order in self.active_orders.items():
                if order.get("broker_order_id") == broker_order_id:
                    local_order = order
                    break
            
            if local_order:
                # Update order status
                zerodha_status = zerodha_order.get("status", "").upper()
                local_order["status"] = self._map_zerodha_status(zerodha_status)
                local_order["filled_quantity"] = zerodha_order.get("filled_quantity", 0)
                local_order["average_price"] = zerodha_order.get("average_price", 0)
                local_order["updated_at"] = datetime.now(timezone.utc)
                
                # Update in database
                await self._update_order_in_db(local_order)
                
        except Exception as e:
            logger.error(f"Error updating order from Zerodha: {e}")
    
    def _map_zerodha_status(self, zerodha_status: str) -> str:
        """Map Zerodha order status to local status."""
        status_mapping = {
            "OPEN": OrderStatus.OPEN.value,
            "COMPLETE": OrderStatus.COMPLETE.value,
            "CANCELLED": OrderStatus.CANCELLED.value,
            "REJECTED": OrderStatus.REJECTED.value,
            "MODIFIED": OrderStatus.MODIFIED.value
        }
        return status_mapping.get(zerodha_status, OrderStatus.PENDING.value)
    
    def register_handlers(self):
        """Register all MCP handlers."""
        
        @self.handler("place_order")
        async def place_order(content: Dict[str, Any]) -> Dict[str, Any]:
            """Place a new order."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["symbol", "side", "quantity", "order_type"])
            
            try:
                order_result = await self._place_order(content)
                self.execution_stats["total_orders"] += 1
                
                if order_result.get("status") == "SUCCESS":
                    self.execution_stats["successful_orders"] += 1
                else:
                    self.execution_stats["failed_orders"] += 1
                
                self.execution_stats["last_execution"] = datetime.now(timezone.utc).isoformat()
                
                return order_result
                
            except Exception as e:
                logger.error(f"Error placing order: {e}")
                self.execution_stats["failed_orders"] += 1
                return {
                    "status": "ERROR",
                    "error": str(e),
                    "order_request": content
                }
        
        @self.handler("modify_order")
        async def modify_order(content: Dict[str, Any]) -> Dict[str, Any]:
            """Modify an existing order."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["order_id"])
            
            try:
                return await self._modify_order(content)
            except Exception as e:
                logger.error(f"Error modifying order: {e}")
                return {"status": "ERROR", "error": str(e)}
        
        @self.handler("cancel_order")
        async def cancel_order(content: Dict[str, Any]) -> Dict[str, Any]:
            """Cancel an order."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["order_id"])
            
            try:
                result = await self._cancel_order(content)
                if result.get("status") == "SUCCESS":
                    self.execution_stats["cancelled_orders"] += 1
                return result
            except Exception as e:
                logger.error(f"Error cancelling order: {e}")
                return {"status": "ERROR", "error": str(e)}
        
        @self.handler("get_order_status")
        async def get_order_status(content: Dict[str, Any]) -> Dict[str, Any]:
            """Get order status."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["order_id"])
            
            try:
                return await self._get_order_status(content)
            except Exception as e:
                logger.error(f"Error getting order status: {e}")
                return {"status": "ERROR", "error": str(e)}
        
        @self.handler("get_order_history")
        async def get_order_history(content: Dict[str, Any]) -> Dict[str, Any]:
            """Get order history."""
            content = await sanitize_input(content)
            
            try:
                return await self._get_order_history(content)
            except Exception as e:
                logger.error(f"Error getting order history: {e}")
                return {"status": "ERROR", "error": str(e)}
        
        @self.handler("get_execution_stats")
        async def get_execution_stats(content: Dict[str, Any]) -> Dict[str, Any]:
            """Get execution statistics."""
            try:
                return {
                    "status": "SUCCESS",
                    "stats": self.execution_stats,
                    "paper_trading": self.paper_trading,
                    "zerodha_connected": self.zerodha_client is not None,
                    "active_orders_count": len(self.active_orders)
                }
            except Exception as e:
                logger.error(f"Error getting execution stats: {e}")
                return {"status": "ERROR", "error": str(e)}
    
    async def _place_order(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """Place a new order."""
        start_time = datetime.now()
        
        # Generate order ID
        order_id = str(uuid.uuid4())
        
        # Validate and format order
        formatted_order = await self._format_order_request(order_request, order_id)
        if "error" in formatted_order:
            return formatted_order
        
        # Store order locally
        self.active_orders[order_id] = formatted_order
        
        try:
            if self.paper_trading:
                # Simulate order execution
                result = await self._simulate_order_execution(formatted_order)
            else:
                # Execute real order through Zerodha
                result = await self._execute_real_order(formatted_order)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_avg_execution_time(execution_time)
            
            # Update order with result
            formatted_order.update(result)
            formatted_order["execution_time_ms"] = execution_time
            
            # Store in database
            await self._store_order_in_db(formatted_order)
            
            return {
                "status": "SUCCESS",
                "order_id": order_id,
                "broker_order_id": result.get("broker_order_id"),
                "execution_time_ms": execution_time,
                "order_details": formatted_order
            }
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            formatted_order["status"] = OrderStatus.REJECTED.value
            formatted_order["error_message"] = str(e)
            
            return {
                "status": "ERROR",
                "order_id": order_id,
                "error": str(e)
            }
    
    async def _format_order_request(self, order_request: Dict[str, Any], order_id: str) -> Dict[str, Any]:
        """Format and validate order request."""
        try:
            # Format symbol
            symbol = format_indian_symbol(order_request["symbol"])
            
            # Validate order parameters
            side = order_request["side"].upper()
            if side not in ["BUY", "SELL"]:
                return {"error": "Invalid order side. Must be BUY or SELL"}
            
            quantity = int(order_request["quantity"])
            if quantity <= 0:
                return {"error": "Quantity must be positive"}
            
            order_type = order_request["order_type"].upper()
            if order_type not in ["MARKET", "LIMIT", "SL", "SL-M"]:
                return {"error": "Invalid order type"}
            
            # Get product type (default to CNC for delivery)
            product = order_request.get("product", "CNC").upper()
            if product not in ["CNC", "MIS", "NRML"]:
                return {"error": "Invalid product type"}
            
            # Validate price for limit orders
            price = None
            if order_type in ["LIMIT", "SL"]:
                if "price" not in order_request:
                    return {"error": f"{order_type} order requires price"}
                price = float(order_request["price"])
                if price <= 0:
                    return {"error": "Price must be positive"}
            
            # Validate trigger price for stop loss orders
            trigger_price = None
            if order_type in ["SL", "SL-M"]:
                if "trigger_price" not in order_request:
                    return {"error": f"{order_type} order requires trigger_price"}
                trigger_price = float(order_request["trigger_price"])
                if trigger_price <= 0:
                    return {"error": "Trigger price must be positive"}
            
            return {
                "id": order_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "order_type": order_type,
                "product": product,
                "price": price,
                "trigger_price": trigger_price,
                "validity": order_request.get("validity", "DAY"),
                "portfolio_id": order_request.get("portfolio_id"),
                "status": OrderStatus.PENDING.value,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "filled_quantity": 0,
                "average_price": 0,
                "broker_order_id": None,
                "error_message": None
            }
            
        except Exception as e:
            return {"error": f"Order formatting failed: {str(e)}"}
    
    def _update_avg_execution_time(self, execution_time_ms: float):
        """Update average execution time."""
        current_avg = self.execution_stats["avg_execution_time_ms"]
        total_orders = self.execution_stats["total_orders"]

        if total_orders == 0:
            self.execution_stats["avg_execution_time_ms"] = execution_time_ms
        else:
            # Calculate running average
            new_avg = ((current_avg * (total_orders - 1)) + execution_time_ms) / total_orders
            self.execution_stats["avg_execution_time_ms"] = new_avg

    async def _simulate_order_execution(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate order execution for paper trading."""
        try:
            # Simulate execution delay
            await asyncio.sleep(0.1)

            # Generate fake broker order ID
            broker_order_id = f"PAPER_{order['id'][:8]}"

            # Simulate immediate fill for market orders
            if order["order_type"] == "MARKET":
                status = OrderStatus.COMPLETE.value
                filled_quantity = order["quantity"]
                # Simulate average price (would get from market data in real implementation)
                average_price = order.get("price", 100.0)  # Default price for simulation
            else:
                # Limit orders start as open
                status = OrderStatus.OPEN.value
                filled_quantity = 0
                average_price = 0

            return {
                "broker_order_id": broker_order_id,
                "status": status,
                "filled_quantity": filled_quantity,
                "average_price": average_price,
                "message": "Paper trading simulation"
            }

        except Exception as e:
            logger.error(f"Error simulating order execution: {e}")
            return {
                "status": OrderStatus.REJECTED.value,
                "error_message": str(e)
            }

    async def _execute_real_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Execute real order through Zerodha."""
        try:
            if not self.zerodha_client:
                raise Exception("Zerodha client not available")

            # Map order parameters to Zerodha format
            zerodha_params = {
                "variety": "regular",
                "exchange": "NSE",  # Extract from symbol
                "tradingsymbol": order["symbol"].split(":")[-1],  # Remove exchange prefix
                "transaction_type": order["side"],
                "quantity": order["quantity"],
                "product": order["product"],
                "order_type": order["order_type"],
                "validity": order["validity"]
            }

            # Add price parameters
            if order["price"]:
                zerodha_params["price"] = order["price"]
            if order["trigger_price"]:
                zerodha_params["trigger_price"] = order["trigger_price"]

            # Place order with Zerodha
            broker_order_id = await self.zerodha_client.place_order(**zerodha_params)

            return {
                "broker_order_id": broker_order_id,
                "status": OrderStatus.SUBMITTED.value,
                "filled_quantity": 0,
                "average_price": 0,
                "message": "Order submitted to Zerodha"
            }

        except Exception as e:
            logger.error(f"Error executing real order: {e}")
            return {
                "status": OrderStatus.REJECTED.value,
                "error_message": str(e)
            }

    async def _modify_order(self, modify_request: Dict[str, Any]) -> Dict[str, Any]:
        """Modify an existing order."""
        try:
            order_id = modify_request["order_id"]

            if order_id not in self.active_orders:
                return {"status": "ERROR", "error": "Order not found"}

            order = self.active_orders[order_id]

            if self.paper_trading:
                # Simulate modification
                if "quantity" in modify_request:
                    order["quantity"] = int(modify_request["quantity"])
                if "price" in modify_request:
                    order["price"] = float(modify_request["price"])

                order["status"] = OrderStatus.MODIFIED.value
                order["updated_at"] = datetime.now(timezone.utc)

                await self._update_order_in_db(order)

                return {
                    "status": "SUCCESS",
                    "order_id": order_id,
                    "message": "Order modified (paper trading)"
                }
            else:
                # Real modification through Zerodha
                if not order.get("broker_order_id"):
                    return {"status": "ERROR", "error": "No broker order ID found"}

                modify_params = {
                    "variety": "regular",
                    "order_id": order["broker_order_id"]
                }

                if "quantity" in modify_request:
                    modify_params["quantity"] = int(modify_request["quantity"])
                if "price" in modify_request:
                    modify_params["price"] = float(modify_request["price"])
                if "order_type" in modify_request:
                    modify_params["order_type"] = modify_request["order_type"]

                result = await self.zerodha_client.modify_order(**modify_params)

                # Update local order
                order.update(modify_request)
                order["status"] = OrderStatus.MODIFIED.value
                order["updated_at"] = datetime.now(timezone.utc)

                await self._update_order_in_db(order)

                return {
                    "status": "SUCCESS",
                    "order_id": order_id,
                    "broker_result": result
                }

        except Exception as e:
            logger.error(f"Error modifying order: {e}")
            return {"status": "ERROR", "error": str(e)}

    async def _cancel_order(self, cancel_request: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel an order."""
        try:
            order_id = cancel_request["order_id"]

            if order_id not in self.active_orders:
                return {"status": "ERROR", "error": "Order not found"}

            order = self.active_orders[order_id]

            if self.paper_trading:
                # Simulate cancellation
                order["status"] = OrderStatus.CANCELLED.value
                order["updated_at"] = datetime.now(timezone.utc)

                await self._update_order_in_db(order)

                return {
                    "status": "SUCCESS",
                    "order_id": order_id,
                    "message": "Order cancelled (paper trading)"
                }
            else:
                # Real cancellation through Zerodha
                if not order.get("broker_order_id"):
                    return {"status": "ERROR", "error": "No broker order ID found"}

                result = await self.zerodha_client.cancel_order(
                    variety="regular",
                    order_id=order["broker_order_id"]
                )

                # Update local order
                order["status"] = OrderStatus.CANCELLED.value
                order["updated_at"] = datetime.now(timezone.utc)

                await self._update_order_in_db(order)

                return {
                    "status": "SUCCESS",
                    "order_id": order_id,
                    "broker_result": result
                }

        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return {"status": "ERROR", "error": str(e)}

    async def _get_order_status(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get order status."""
        try:
            order_id = request["order_id"]

            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                return {
                    "status": "SUCCESS",
                    "order": order
                }
            else:
                # Check database for historical orders
                query = "SELECT * FROM orders WHERE id = $1"
                result = await db_manager.execute_query(query, order_id, fetch="one")

                if result:
                    return {
                        "status": "SUCCESS",
                        "order": dict(result)
                    }
                else:
                    return {"status": "ERROR", "error": "Order not found"}

        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return {"status": "ERROR", "error": str(e)}

    async def _get_order_history(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get order history."""
        try:
            limit = request.get("limit", 100)
            portfolio_id = request.get("portfolio_id")

            query = "SELECT * FROM orders"
            params = []

            if portfolio_id:
                query += " WHERE portfolio_id = $1"
                params.append(portfolio_id)

            query += " ORDER BY created_at DESC LIMIT $" + str(len(params) + 1)
            params.append(limit)

            results = await db_manager.execute_query(query, *params, fetch="all")

            orders = [dict(row) for row in results]

            return {
                "status": "SUCCESS",
                "orders": orders,
                "count": len(orders)
            }

        except Exception as e:
            logger.error(f"Error getting order history: {e}")
            return {"status": "ERROR", "error": str(e)}

    async def _store_order_in_db(self, order: Dict[str, Any]):
        """Store order in database using existing schema."""
        try:
            # Get instrument ID from symbol
            instrument_id = await self._get_instrument_id_by_symbol(order["symbol"])
            if not instrument_id:
                logger.error(f"Instrument not found for symbol: {order['symbol']}")
                return

            query = """
                INSERT INTO orders
                (id, portfolio_id, instrument_id, order_type, order_side, quantity,
                 price, trigger_price, filled_quantity, average_price, status,
                 status_message, placed_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """

            await db_manager.execute_query(
                query,
                order["id"],
                order.get("portfolio_id"),
                instrument_id,
                order["order_type"],
                order["side"],
                order["quantity"],
                order.get("price"),
                order.get("trigger_price"),
                order["filled_quantity"],
                order["average_price"],
                order["status"],
                order.get("error_message"),
                order["created_at"],
                order["updated_at"]
            )

        except Exception as e:
            logger.error(f"Error storing order in database: {e}")

    async def _get_instrument_id_by_symbol(self, symbol: str) -> Optional[str]:
        """Get instrument ID by symbol."""
        try:
            query = "SELECT id FROM instruments WHERE symbol = $1 AND is_active = true"
            result = await db_manager.execute_query(query, symbol, fetch="one")
            return result["id"] if result else None
        except Exception as e:
            logger.error(f"Error getting instrument ID for {symbol}: {e}")
            return None

    async def _update_order_in_db(self, order: Dict[str, Any]):
        """Update order in database."""
        try:
            query = """
                UPDATE orders SET
                    status = $2,
                    updated_at = $3,
                    filled_quantity = $4,
                    average_price = $5,
                    status_message = $6
                WHERE id = $1
            """

            await db_manager.execute_query(
                query,
                order["id"],
                order["status"],
                order["updated_at"],
                order["filled_quantity"],
                order["average_price"],
                order.get("error_message")
            )

        except Exception as e:
            logger.error(f"Error updating order in database: {e}")


async def main():
    """Main function to run the trade execution server."""
    await init_database()

    server = TradeExecutionServer()

    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down trade execution server...")
    finally:
        await server.stop()
        await close_database()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
