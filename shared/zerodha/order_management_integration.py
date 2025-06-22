"""
Order Management Integration for Zerodha Trade Execution.
Provides seamless integration between AWM OMS and Zerodha trade execution.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from decimal import Decimal
from enum import Enum

from .client import ZerodhaClient
from .auth import ZerodhaAuthService
from .utils import format_indian_symbol, validate_trading_hours

logger = logging.getLogger(__name__)


class OrderLifecycleStatus(Enum):
    """Order lifecycle status enumeration."""
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderManagementIntegration:
    """
    Integration service between AWM Order Management System and Zerodha execution.
    Handles order lifecycle, validation, routing, and status synchronization.
    """
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.zerodha_auth = ZerodhaAuthService()
        self.zerodha_client = None
        
        # Order routing configuration
        self.routing_config = {
            "default_exchange": "NSE",
            "default_product": "CNC",
            "default_validity": "DAY",
            "enable_smart_routing": True,
            "enable_pre_trade_validation": True,
            "enable_post_trade_reconciliation": True
        }
        
        # Order validation rules
        self.validation_rules = {
            "min_order_value": Decimal("1.0"),
            "max_order_value": Decimal("1000000.0"),
            "max_quantity_per_order": 10000,
            "allowed_order_types": ["MARKET", "LIMIT", "SL", "SL-M"],
            "allowed_products": ["CNC", "MIS", "NRML"],
            "require_portfolio_validation": True
        }
        
        # Order tracking
        self.order_cache = {}
        self.execution_queue = asyncio.Queue()
        
        # Statistics
        self.stats = {
            "orders_processed": 0,
            "orders_executed": 0,
            "orders_rejected": 0,
            "validation_failures": 0,
            "execution_failures": 0,
            "avg_processing_time_ms": 0
        }
        
        # Background tasks
        self.background_tasks = []
        self.is_running = False
    
    async def start(self):
        """Start the order management integration service."""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Starting Order Management Integration service")
        
        # Initialize Zerodha client
        await self._initialize_zerodha_client()
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._order_processor()),
            asyncio.create_task(self._status_synchronizer()),
            asyncio.create_task(self._reconciliation_service())
        ]
    
    async def stop(self):
        """Stop the order management integration service."""
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        
        logger.info("Order Management Integration service stopped")
    
    async def _initialize_zerodha_client(self):
        """Initialize Zerodha client."""
        try:
            if await self.zerodha_auth.is_authenticated():
                self.zerodha_client = await self.zerodha_auth.get_authenticated_client()
                logger.info("Zerodha client initialized for order management")
            else:
                logger.warning("Zerodha not authenticated - order execution will be limited")
        except Exception as e:
            logger.error(f"Failed to initialize Zerodha client: {e}")
    
    async def submit_order(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit an order through the integrated order management system.
        
        Args:
            order_request: Order details
            
        Returns:
            Order submission result
        """
        start_time = datetime.now()
        order_id = str(uuid.uuid4())
        
        try:
            self.stats["orders_processed"] += 1
            
            # Step 1: Pre-trade validation
            validation_result = await self._validate_order(order_request)
            if not validation_result["valid"]:
                self.stats["validation_failures"] += 1
                return {
                    "status": "REJECTED",
                    "order_id": order_id,
                    "reason": "Validation failed",
                    "details": validation_result["errors"]
                }
            
            # Step 2: Create order record
            order = await self._create_order_record(order_id, order_request, validation_result)
            
            # Step 3: Queue for execution
            await self.execution_queue.put(order)
            
            # Step 4: Store in cache for tracking
            self.order_cache[order_id] = order
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_avg_processing_time(processing_time)
            
            return {
                "status": "ACCEPTED",
                "order_id": order_id,
                "processing_time_ms": processing_time,
                "estimated_execution_time": "< 1 second"
            }
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            self.stats["validation_failures"] += 1
            return {
                "status": "ERROR",
                "order_id": order_id,
                "error": str(e)
            }
    
    async def _validate_order(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate order request against business rules."""
        errors = []
        
        try:
            # Basic field validation
            required_fields = ["symbol", "side", "quantity", "order_type"]
            for field in required_fields:
                if field not in order_request:
                    errors.append(f"Missing required field: {field}")
            
            if errors:
                return {"valid": False, "errors": errors}
            
            # Symbol validation
            symbol = order_request["symbol"]
            if not await self._validate_symbol(symbol):
                errors.append(f"Invalid or inactive symbol: {symbol}")
            
            # Quantity validation
            quantity = int(order_request["quantity"])
            if quantity <= 0:
                errors.append("Quantity must be positive")
            elif quantity > self.validation_rules["max_quantity_per_order"]:
                errors.append(f"Quantity exceeds maximum: {self.validation_rules['max_quantity_per_order']}")
            
            # Order type validation
            order_type = order_request["order_type"].upper()
            if order_type not in self.validation_rules["allowed_order_types"]:
                errors.append(f"Invalid order type: {order_type}")
            
            # Product validation
            product = order_request.get("product", "CNC").upper()
            if product not in self.validation_rules["allowed_products"]:
                errors.append(f"Invalid product type: {product}")
            
            # Price validation for limit orders
            if order_type in ["LIMIT", "SL"] and "price" not in order_request:
                errors.append(f"{order_type} order requires price")
            
            # Order value validation
            if "price" in order_request:
                order_value = Decimal(str(order_request["price"])) * Decimal(str(quantity))
                if order_value < self.validation_rules["min_order_value"]:
                    errors.append(f"Order value below minimum: {self.validation_rules['min_order_value']}")
                elif order_value > self.validation_rules["max_order_value"]:
                    errors.append(f"Order value exceeds maximum: {self.validation_rules['max_order_value']}")
            
            # Trading hours validation
            trading_status = validate_trading_hours()
            if not trading_status["is_trading_hours"] and order_type == "MARKET":
                errors.append("Market orders not allowed outside trading hours")
            
            # Portfolio validation
            if self.validation_rules["require_portfolio_validation"]:
                portfolio_validation = await self._validate_portfolio_constraints(order_request)
                if not portfolio_validation["valid"]:
                    errors.extend(portfolio_validation["errors"])
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "validated_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"]
            }
    
    async def _validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is active and tradeable."""
        try:
            if not self.db_manager:
                return True  # Skip validation if no DB
            
            formatted_symbol = format_indian_symbol(symbol)
            query = "SELECT id FROM instruments WHERE symbol = $1 AND is_active = true"
            result = await self.db_manager.execute_query(query, formatted_symbol, fetch="one")
            return result is not None
            
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False
    
    async def _validate_portfolio_constraints(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate order against portfolio constraints."""
        errors = []
        
        try:
            portfolio_id = order_request.get("portfolio_id")
            if not portfolio_id:
                return {"valid": True, "errors": []}  # Skip if no portfolio specified
            
            # Check portfolio exists and is active
            if self.db_manager:
                query = "SELECT * FROM portfolios WHERE id = $1 AND is_active = true"
                portfolio = await self.db_manager.execute_query(query, portfolio_id, fetch="one")
                
                if not portfolio:
                    errors.append("Portfolio not found or inactive")
                    return {"valid": False, "errors": errors}
                
                # Check available cash for buy orders
                if order_request["side"].upper() == "BUY":
                    required_cash = self._calculate_required_cash(order_request)
                    available_cash = Decimal(str(portfolio.get("cash_balance", 0)))
                    
                    if required_cash > available_cash:
                        errors.append(f"Insufficient cash: required {required_cash}, available {available_cash}")
                
                # Check position limits
                # Add more portfolio-specific validations here
            
            return {"valid": len(errors) == 0, "errors": errors}
            
        except Exception as e:
            logger.error(f"Error validating portfolio constraints: {e}")
            return {"valid": False, "errors": [f"Portfolio validation error: {str(e)}"]}
    
    def _calculate_required_cash(self, order_request: Dict[str, Any]) -> Decimal:
        """Calculate required cash for order."""
        try:
            quantity = Decimal(str(order_request["quantity"]))
            price = Decimal(str(order_request.get("price", 0)))
            
            if price <= 0:
                # For market orders, estimate with buffer
                price = Decimal("100")  # Default estimate
            
            # Add buffer for charges and slippage (5%)
            required_cash = quantity * price * Decimal("1.05")
            return required_cash
            
        except Exception as e:
            logger.error(f"Error calculating required cash: {e}")
            return Decimal("0")
    
    async def _create_order_record(self, order_id: str, order_request: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create order record for tracking."""
        return {
            "id": order_id,
            "symbol": format_indian_symbol(order_request["symbol"]),
            "side": order_request["side"].upper(),
            "quantity": int(order_request["quantity"]),
            "order_type": order_request["order_type"].upper(),
            "product": order_request.get("product", "CNC").upper(),
            "price": float(order_request["price"]) if "price" in order_request else None,
            "trigger_price": float(order_request["trigger_price"]) if "trigger_price" in order_request else None,
            "validity": order_request.get("validity", "DAY").upper(),
            "portfolio_id": order_request.get("portfolio_id"),
            "status": OrderLifecycleStatus.VALIDATED.value,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "validation_result": validation_result,
            "execution_attempts": 0,
            "broker_order_id": None,
            "filled_quantity": 0,
            "average_price": 0,
            "error_message": None
        }
    
    def _update_avg_processing_time(self, processing_time_ms: float):
        """Update average processing time."""
        current_avg = self.stats["avg_processing_time_ms"]
        total_orders = self.stats["orders_processed"]

        if total_orders == 1:
            self.stats["avg_processing_time_ms"] = processing_time_ms
        else:
            new_avg = ((current_avg * (total_orders - 1)) + processing_time_ms) / total_orders
            self.stats["avg_processing_time_ms"] = new_avg

    async def _order_processor(self):
        """Background task to process orders from the execution queue."""
        while self.is_running:
            try:
                # Get order from queue with timeout
                order = await asyncio.wait_for(self.execution_queue.get(), timeout=1.0)

                # Process the order
                await self._execute_order(order)

            except asyncio.TimeoutError:
                # No orders in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error in order processor: {e}")
                await asyncio.sleep(1)

    async def _execute_order(self, order: Dict[str, Any]):
        """Execute an order through Zerodha."""
        try:
            order["execution_attempts"] += 1
            order["status"] = OrderLifecycleStatus.SUBMITTED.value
            order["updated_at"] = datetime.now(timezone.utc)

            if not self.zerodha_client:
                raise Exception("Zerodha client not available")

            # Prepare Zerodha order parameters
            zerodha_params = {
                "variety": "regular",
                "exchange": self._extract_exchange(order["symbol"]),
                "tradingsymbol": self._extract_tradingsymbol(order["symbol"]),
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

            # Execute order
            broker_order_id = await self.zerodha_client.place_order(**zerodha_params)

            # Update order with broker details
            order["broker_order_id"] = broker_order_id
            order["status"] = OrderLifecycleStatus.ACKNOWLEDGED.value
            order["updated_at"] = datetime.now(timezone.utc)

            # Store in database
            await self._store_order_in_db(order)

            self.stats["orders_executed"] += 1
            logger.info(f"Order {order['id']} executed successfully with broker ID {broker_order_id}")

        except Exception as e:
            logger.error(f"Error executing order {order['id']}: {e}")

            order["status"] = OrderLifecycleStatus.REJECTED.value
            order["error_message"] = str(e)
            order["updated_at"] = datetime.now(timezone.utc)

            # Store failed order
            await self._store_order_in_db(order)

            self.stats["execution_failures"] += 1

    def _extract_exchange(self, symbol: str) -> str:
        """Extract exchange from symbol."""
        if ":" in symbol:
            return symbol.split(":")[0]
        return self.routing_config["default_exchange"]

    def _extract_tradingsymbol(self, symbol: str) -> str:
        """Extract trading symbol from formatted symbol."""
        if ":" in symbol:
            return symbol.split(":")[1]
        return symbol

    async def _status_synchronizer(self):
        """Background task to synchronize order status with Zerodha."""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                if self.zerodha_client:
                    await self._sync_order_statuses()

            except Exception as e:
                logger.error(f"Error in status synchronizer: {e}")
                await asyncio.sleep(30)

    async def _sync_order_statuses(self):
        """Sync order statuses with Zerodha."""
        try:
            # Get all orders from Zerodha
            zerodha_orders = await self.zerodha_client.get_orders()

            # Update local orders
            for zerodha_order in zerodha_orders:
                await self._update_order_from_zerodha(zerodha_order)

        except Exception as e:
            logger.error(f"Error syncing order statuses: {e}")

    async def _update_order_from_zerodha(self, zerodha_order: Dict[str, Any]):
        """Update local order from Zerodha order data."""
        try:
            broker_order_id = zerodha_order.get("order_id")
            if not broker_order_id:
                return

            # Find local order
            local_order = None
            for order in self.order_cache.values():
                if order.get("broker_order_id") == broker_order_id:
                    local_order = order
                    break

            if not local_order:
                return

            # Update status
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
        """Map Zerodha status to local status."""
        status_mapping = {
            "OPEN": OrderLifecycleStatus.ACKNOWLEDGED.value,
            "COMPLETE": OrderLifecycleStatus.FILLED.value,
            "CANCELLED": OrderLifecycleStatus.CANCELLED.value,
            "REJECTED": OrderLifecycleStatus.REJECTED.value
        }
        return status_mapping.get(zerodha_status, OrderLifecycleStatus.ACKNOWLEDGED.value)

    async def _reconciliation_service(self):
        """Background task for trade reconciliation."""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                if self.zerodha_client:
                    await self._reconcile_trades()

            except Exception as e:
                logger.error(f"Error in reconciliation service: {e}")
                await asyncio.sleep(600)

    async def _reconcile_trades(self):
        """Reconcile trades between local records and Zerodha."""
        try:
            # Get trades from Zerodha
            zerodha_trades = await self.zerodha_client.get_trades()

            # Process each trade
            for trade in zerodha_trades:
                await self._process_trade_reconciliation(trade)

        except Exception as e:
            logger.error(f"Error reconciling trades: {e}")

    async def _process_trade_reconciliation(self, zerodha_trade: Dict[str, Any]):
        """Process individual trade reconciliation."""
        try:
            # Implementation would check if trade exists locally
            # and create/update trade records as needed
            logger.debug(f"Processing trade reconciliation for trade: {zerodha_trade.get('trade_id')}")

        except Exception as e:
            logger.error(f"Error processing trade reconciliation: {e}")

    async def _store_order_in_db(self, order: Dict[str, Any]):
        """Store order in database."""
        try:
            if not self.db_manager:
                return

            # Get instrument ID
            instrument_id = await self._get_instrument_id_by_symbol(order["symbol"])
            if not instrument_id:
                logger.error(f"Instrument not found for symbol: {order['symbol']}")
                return

            # Check if order exists
            existing_order = await self.db_manager.execute_query(
                "SELECT id FROM orders WHERE id = $1",
                order["id"],
                fetch="one"
            )

            if existing_order:
                # Update existing order
                await self._update_order_in_db(order)
            else:
                # Insert new order
                query = """
                    INSERT INTO orders
                    (id, portfolio_id, instrument_id, order_type, order_side, quantity,
                     price, trigger_price, filled_quantity, average_price, status,
                     status_message, placed_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """

                await self.db_manager.execute_query(
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

    async def _update_order_in_db(self, order: Dict[str, Any]):
        """Update order in database."""
        try:
            if not self.db_manager:
                return

            query = """
                UPDATE orders SET
                    status = $2,
                    updated_at = $3,
                    filled_quantity = $4,
                    average_price = $5,
                    status_message = $6
                WHERE id = $1
            """

            await self.db_manager.execute_query(
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

    async def _get_instrument_id_by_symbol(self, symbol: str) -> Optional[str]:
        """Get instrument ID by symbol."""
        try:
            if not self.db_manager:
                return None

            query = "SELECT id FROM instruments WHERE symbol = $1 AND is_active = true"
            result = await self.db_manager.execute_query(query, symbol, fetch="one")
            return result["id"] if result else None

        except Exception as e:
            logger.error(f"Error getting instrument ID for {symbol}: {e}")
            return None

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status."""
        try:
            if order_id in self.order_cache:
                return {
                    "status": "SUCCESS",
                    "order": self.order_cache[order_id]
                }

            # Check database
            if self.db_manager:
                query = "SELECT * FROM orders WHERE id = $1"
                result = await self.db_manager.execute_query(query, order_id, fetch="one")

                if result:
                    return {
                        "status": "SUCCESS",
                        "order": dict(result)
                    }

            return {"status": "ERROR", "error": "Order not found"}

        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return {"status": "ERROR", "error": str(e)}

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order."""
        try:
            if order_id not in self.order_cache:
                return {"status": "ERROR", "error": "Order not found"}

            order = self.order_cache[order_id]

            if not order.get("broker_order_id"):
                return {"status": "ERROR", "error": "Order not yet submitted to broker"}

            # Cancel with Zerodha
            if self.zerodha_client:
                result = await self.zerodha_client.cancel_order(
                    variety="regular",
                    order_id=order["broker_order_id"]
                )

                # Update local order
                order["status"] = OrderLifecycleStatus.CANCELLED.value
                order["updated_at"] = datetime.now(timezone.utc)

                await self._update_order_in_db(order)

                return {
                    "status": "SUCCESS",
                    "order_id": order_id,
                    "broker_result": result
                }
            else:
                return {"status": "ERROR", "error": "Zerodha client not available"}

        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return {"status": "ERROR", "error": str(e)}

    def get_statistics(self) -> Dict[str, Any]:
        """Get order management statistics."""
        return {
            "stats": self.stats,
            "active_orders": len(self.order_cache),
            "queue_size": self.execution_queue.qsize(),
            "is_running": self.is_running,
            "zerodha_connected": self.zerodha_client is not None
        }


# Global order management integration instance
order_management_integration = OrderManagementIntegration()
