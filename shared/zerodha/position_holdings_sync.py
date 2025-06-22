"""
Position and Holdings Sync Service for AWM System.
Syncs positions and holdings from Zerodha to local portfolio management system.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from decimal import Decimal

from .client import ZerodhaClient
from .auth import ZerodhaAuthService
from .utils import format_indian_symbol, calculate_indian_taxes

logger = logging.getLogger(__name__)


class PositionHoldingsSync:
    """
    Service to sync positions and holdings from Zerodha to local portfolio system.
    Handles real-time position updates, holdings reconciliation, and P&L calculations.
    """
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.zerodha_auth = ZerodhaAuthService()
        self.zerodha_client = None
        
        # Sync configuration
        self.sync_config = {
            "positions_sync_interval": 30,  # seconds
            "holdings_sync_interval": 300,  # 5 minutes
            "enable_real_time_sync": True,
            "enable_reconciliation": True,
            "reconciliation_interval": 3600  # 1 hour
        }
        
        # Sync state
        self.last_positions_sync = None
        self.last_holdings_sync = None
        self.last_reconciliation = None
        
        # Statistics
        self.sync_stats = {
            "positions_synced": 0,
            "holdings_synced": 0,
            "sync_errors": 0,
            "reconciliation_runs": 0,
            "discrepancies_found": 0,
            "last_sync_duration_ms": 0
        }
        
        # Background tasks
        self.background_tasks = []
        self.is_running = False
    
    async def start(self):
        """Start the position and holdings sync service."""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Starting Position and Holdings Sync service")
        
        # Initialize Zerodha client
        await self._initialize_zerodha_client()
        
        # Start background sync tasks
        if self.sync_config["enable_real_time_sync"]:
            self.background_tasks = [
                asyncio.create_task(self._positions_sync_loop()),
                asyncio.create_task(self._holdings_sync_loop()),
                asyncio.create_task(self._reconciliation_loop())
            ]
    
    async def stop(self):
        """Stop the sync service."""
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        
        logger.info("Position and Holdings Sync service stopped")
    
    async def _initialize_zerodha_client(self):
        """Initialize Zerodha client."""
        try:
            if await self.zerodha_auth.is_authenticated():
                self.zerodha_client = await self.zerodha_auth.get_authenticated_client()
                logger.info("Zerodha client initialized for position sync")
            else:
                logger.warning("Zerodha not authenticated - sync will be limited")
        except Exception as e:
            logger.error(f"Failed to initialize Zerodha client: {e}")
    
    async def sync_positions(self, portfolio_id: str = None) -> Dict[str, Any]:
        """
        Sync positions from Zerodha to local portfolio.
        
        Args:
            portfolio_id: Target portfolio ID (optional)
            
        Returns:
            Sync result
        """
        start_time = datetime.now()
        
        try:
            if not self.zerodha_client:
                return {"status": "ERROR", "error": "Zerodha client not available"}
            
            # Get positions from Zerodha
            positions_data = await self.zerodha_client.get_positions()
            
            # Process net positions
            net_positions = positions_data.get("net", [])
            day_positions = positions_data.get("day", [])
            
            # Sync net positions
            net_result = await self._sync_net_positions(net_positions, portfolio_id)
            
            # Sync day positions
            day_result = await self._sync_day_positions(day_positions, portfolio_id)
            
            # Update sync timestamp
            self.last_positions_sync = datetime.now(timezone.utc)
            self.sync_stats["positions_synced"] += len(net_positions) + len(day_positions)
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.sync_stats["last_sync_duration_ms"] = duration_ms
            
            return {
                "status": "SUCCESS",
                "net_positions": net_result,
                "day_positions": day_result,
                "sync_duration_ms": duration_ms,
                "synced_at": self.last_positions_sync.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error syncing positions: {e}")
            self.sync_stats["sync_errors"] += 1
            return {"status": "ERROR", "error": str(e)}
    
    async def sync_holdings(self, portfolio_id: str = None) -> Dict[str, Any]:
        """
        Sync holdings from Zerodha to local portfolio.
        
        Args:
            portfolio_id: Target portfolio ID (optional)
            
        Returns:
            Sync result
        """
        start_time = datetime.now()
        
        try:
            if not self.zerodha_client:
                return {"status": "ERROR", "error": "Zerodha client not available"}
            
            # Get holdings from Zerodha
            holdings = await self.zerodha_client.get_holdings()
            
            # Process holdings
            result = await self._sync_holdings_data(holdings, portfolio_id)
            
            # Update sync timestamp
            self.last_holdings_sync = datetime.now(timezone.utc)
            self.sync_stats["holdings_synced"] += len(holdings)
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "status": "SUCCESS",
                "holdings": result,
                "count": len(holdings),
                "sync_duration_ms": duration_ms,
                "synced_at": self.last_holdings_sync.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error syncing holdings: {e}")
            self.sync_stats["sync_errors"] += 1
            return {"status": "ERROR", "error": str(e)}
    
    async def _sync_net_positions(self, positions: List[Dict[str, Any]], portfolio_id: str = None) -> Dict[str, Any]:
        """Sync net positions to database."""
        try:
            synced_count = 0
            errors = []
            
            for position in positions:
                try:
                    await self._store_position(position, "NET", portfolio_id)
                    synced_count += 1
                except Exception as e:
                    errors.append(f"Error syncing position {position.get('tradingsymbol', 'unknown')}: {str(e)}")
            
            return {
                "synced_count": synced_count,
                "total_count": len(positions),
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"Error syncing net positions: {e}")
            return {"synced_count": 0, "total_count": len(positions), "errors": [str(e)]}
    
    async def _sync_day_positions(self, positions: List[Dict[str, Any]], portfolio_id: str = None) -> Dict[str, Any]:
        """Sync day positions to database."""
        try:
            synced_count = 0
            errors = []
            
            for position in positions:
                try:
                    await self._store_position(position, "DAY", portfolio_id)
                    synced_count += 1
                except Exception as e:
                    errors.append(f"Error syncing day position {position.get('tradingsymbol', 'unknown')}: {str(e)}")
            
            return {
                "synced_count": synced_count,
                "total_count": len(positions),
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"Error syncing day positions: {e}")
            return {"synced_count": 0, "total_count": len(positions), "errors": [str(e)]}
    
    async def _sync_holdings_data(self, holdings: List[Dict[str, Any]], portfolio_id: str = None) -> Dict[str, Any]:
        """Sync holdings data to database."""
        try:
            synced_count = 0
            errors = []
            
            for holding in holdings:
                try:
                    await self._store_holding(holding, portfolio_id)
                    synced_count += 1
                except Exception as e:
                    errors.append(f"Error syncing holding {holding.get('tradingsymbol', 'unknown')}: {str(e)}")
            
            return {
                "synced_count": synced_count,
                "total_count": len(holdings),
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"Error syncing holdings: {e}")
            return {"synced_count": 0, "total_count": len(holdings), "errors": [str(e)]}
    
    async def _store_position(self, position: Dict[str, Any], position_type: str, portfolio_id: str = None):
        """Store position in database."""
        try:
            if not self.db_manager:
                return
            
            # Get or create portfolio
            if not portfolio_id:
                portfolio_id = await self._get_default_portfolio_id()
            
            # Get instrument ID
            symbol = format_indian_symbol(position.get("tradingsymbol", ""), position.get("exchange", "NSE"))
            instrument_id = await self._get_instrument_id_by_symbol(symbol)
            
            if not instrument_id:
                logger.warning(f"Instrument not found for symbol: {symbol}")
                return
            
            # Calculate P&L and other metrics
            quantity = int(position.get("quantity", 0))
            if quantity == 0:
                return  # Skip zero quantity positions
            
            buy_quantity = int(position.get("buy_quantity", 0))
            sell_quantity = int(position.get("sell_quantity", 0))
            buy_price = Decimal(str(position.get("buy_price", 0)))
            sell_price = Decimal(str(position.get("sell_price", 0)))
            last_price = Decimal(str(position.get("last_price", 0)))
            
            # Calculate average price
            if quantity > 0:  # Long position
                average_price = buy_price
            else:  # Short position
                average_price = sell_price
            
            # Calculate unrealized P&L
            unrealized_pnl = quantity * (last_price - average_price)
            
            # Calculate realized P&L
            realized_pnl = Decimal(str(position.get("pnl", 0)))
            
            # Store position
            query = """
                INSERT INTO portfolio_positions 
                (portfolio_id, instrument_id, quantity, average_price, current_price, 
                 unrealized_pnl, realized_pnl, position_type, buy_quantity, sell_quantity,
                 buy_price, sell_price, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (portfolio_id, instrument_id, position_type) DO UPDATE SET
                    quantity = EXCLUDED.quantity,
                    average_price = EXCLUDED.average_price,
                    current_price = EXCLUDED.current_price,
                    unrealized_pnl = EXCLUDED.unrealized_pnl,
                    realized_pnl = EXCLUDED.realized_pnl,
                    buy_quantity = EXCLUDED.buy_quantity,
                    sell_quantity = EXCLUDED.sell_quantity,
                    buy_price = EXCLUDED.buy_price,
                    sell_price = EXCLUDED.sell_price,
                    updated_at = EXCLUDED.updated_at
            """
            
            await self.db_manager.execute_query(
                query,
                portfolio_id,
                instrument_id,
                quantity,
                average_price,
                last_price,
                unrealized_pnl,
                realized_pnl,
                position_type,
                buy_quantity,
                sell_quantity,
                buy_price,
                sell_price,
                datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Error storing position: {e}")
            raise
    
    async def _store_holding(self, holding: Dict[str, Any], portfolio_id: str = None):
        """Store holding in database."""
        try:
            if not self.db_manager:
                return
            
            # Get or create portfolio
            if not portfolio_id:
                portfolio_id = await self._get_default_portfolio_id()
            
            # Get instrument ID
            symbol = format_indian_symbol(holding.get("tradingsymbol", ""), holding.get("exchange", "NSE"))
            instrument_id = await self._get_instrument_id_by_symbol(symbol)
            
            if not instrument_id:
                logger.warning(f"Instrument not found for symbol: {symbol}")
                return
            
            # Extract holding data
            quantity = int(holding.get("quantity", 0))
            if quantity <= 0:
                return  # Skip zero or negative holdings
            
            average_price = Decimal(str(holding.get("average_price", 0)))
            last_price = Decimal(str(holding.get("last_price", 0)))
            
            # Calculate P&L
            unrealized_pnl = quantity * (last_price - average_price)
            
            # Store holding
            query = """
                INSERT INTO portfolio_holdings 
                (portfolio_id, instrument_id, quantity, average_price, current_price, 
                 unrealized_pnl, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (portfolio_id, instrument_id) DO UPDATE SET
                    quantity = EXCLUDED.quantity,
                    average_price = EXCLUDED.average_price,
                    current_price = EXCLUDED.current_price,
                    unrealized_pnl = EXCLUDED.unrealized_pnl,
                    updated_at = EXCLUDED.updated_at
            """
            
            await self.db_manager.execute_query(
                query,
                portfolio_id,
                instrument_id,
                quantity,
                average_price,
                last_price,
                unrealized_pnl,
                datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Error storing holding: {e}")
            raise

    async def _get_default_portfolio_id(self) -> str:
        """Get or create default portfolio ID."""
        try:
            if not self.db_manager:
                return "default"

            # Try to get existing default portfolio
            query = """
                SELECT id FROM portfolios
                WHERE name = 'Zerodha Default' AND is_active = true
                LIMIT 1
            """
            result = await self.db_manager.execute_query(query, fetch="one")

            if result:
                return result["id"]

            # Create default portfolio
            import uuid
            portfolio_id = str(uuid.uuid4())

            create_query = """
                INSERT INTO portfolios (id, name, description, user_id, is_active, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
            """

            await self.db_manager.execute_query(
                create_query,
                portfolio_id,
                "Zerodha Default",
                "Default portfolio for Zerodha positions and holdings",
                "system",
                True,
                datetime.now(timezone.utc)
            )

            return portfolio_id

        except Exception as e:
            logger.error(f"Error getting default portfolio: {e}")
            return "default"

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

    async def _positions_sync_loop(self):
        """Background loop for positions sync."""
        while self.is_running:
            try:
                await asyncio.sleep(self.sync_config["positions_sync_interval"])

                if self.zerodha_client:
                    await self.sync_positions()

            except Exception as e:
                logger.error(f"Error in positions sync loop: {e}")
                await asyncio.sleep(60)

    async def _holdings_sync_loop(self):
        """Background loop for holdings sync."""
        while self.is_running:
            try:
                await asyncio.sleep(self.sync_config["holdings_sync_interval"])

                if self.zerodha_client:
                    await self.sync_holdings()

            except Exception as e:
                logger.error(f"Error in holdings sync loop: {e}")
                await asyncio.sleep(300)

    async def _reconciliation_loop(self):
        """Background loop for reconciliation."""
        while self.is_running:
            try:
                await asyncio.sleep(self.sync_config["reconciliation_interval"])

                if self.sync_config["enable_reconciliation"]:
                    await self.run_reconciliation()

            except Exception as e:
                logger.error(f"Error in reconciliation loop: {e}")
                await asyncio.sleep(1800)  # 30 minutes on error

    async def run_reconciliation(self) -> Dict[str, Any]:
        """Run reconciliation between local and Zerodha data."""
        try:
            logger.info("Starting position and holdings reconciliation")
            start_time = datetime.now()

            # Reconcile positions
            positions_result = await self._reconcile_positions()

            # Reconcile holdings
            holdings_result = await self._reconcile_holdings()

            # Update reconciliation timestamp
            self.last_reconciliation = datetime.now(timezone.utc)
            self.sync_stats["reconciliation_runs"] += 1

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            return {
                "status": "SUCCESS",
                "positions_reconciliation": positions_result,
                "holdings_reconciliation": holdings_result,
                "duration_ms": duration_ms,
                "reconciled_at": self.last_reconciliation.isoformat()
            }

        except Exception as e:
            logger.error(f"Error in reconciliation: {e}")
            return {"status": "ERROR", "error": str(e)}

    async def _reconcile_positions(self) -> Dict[str, Any]:
        """Reconcile positions between local and Zerodha."""
        try:
            discrepancies = []

            if not self.zerodha_client or not self.db_manager:
                return {"discrepancies": [], "status": "SKIPPED"}

            # Get positions from Zerodha
            zerodha_positions = await self.zerodha_client.get_positions()
            net_positions = zerodha_positions.get("net", [])

            # Get local positions
            local_positions = await self._get_local_positions()

            # Create lookup dictionaries
            zerodha_lookup = {}
            for pos in net_positions:
                symbol = format_indian_symbol(pos.get("tradingsymbol", ""), pos.get("exchange", "NSE"))
                zerodha_lookup[symbol] = pos

            local_lookup = {}
            for pos in local_positions:
                local_lookup[pos["symbol"]] = pos

            # Find discrepancies
            all_symbols = set(zerodha_lookup.keys()) | set(local_lookup.keys())

            for symbol in all_symbols:
                zerodha_pos = zerodha_lookup.get(symbol)
                local_pos = local_lookup.get(symbol)

                if not zerodha_pos and local_pos:
                    # Position exists locally but not in Zerodha
                    discrepancies.append({
                        "symbol": symbol,
                        "type": "MISSING_IN_ZERODHA",
                        "local_quantity": local_pos["quantity"],
                        "zerodha_quantity": 0
                    })
                elif zerodha_pos and not local_pos:
                    # Position exists in Zerodha but not locally
                    discrepancies.append({
                        "symbol": symbol,
                        "type": "MISSING_LOCALLY",
                        "local_quantity": 0,
                        "zerodha_quantity": zerodha_pos.get("quantity", 0)
                    })
                elif zerodha_pos and local_pos:
                    # Both exist, check for quantity differences
                    zerodha_qty = int(zerodha_pos.get("quantity", 0))
                    local_qty = int(local_pos["quantity"])

                    if zerodha_qty != local_qty:
                        discrepancies.append({
                            "symbol": symbol,
                            "type": "QUANTITY_MISMATCH",
                            "local_quantity": local_qty,
                            "zerodha_quantity": zerodha_qty,
                            "difference": zerodha_qty - local_qty
                        })

            self.sync_stats["discrepancies_found"] += len(discrepancies)

            return {
                "discrepancies": discrepancies,
                "total_discrepancies": len(discrepancies),
                "status": "COMPLETED"
            }

        except Exception as e:
            logger.error(f"Error reconciling positions: {e}")
            return {"discrepancies": [], "status": "ERROR", "error": str(e)}

    async def _reconcile_holdings(self) -> Dict[str, Any]:
        """Reconcile holdings between local and Zerodha."""
        try:
            discrepancies = []

            if not self.zerodha_client or not self.db_manager:
                return {"discrepancies": [], "status": "SKIPPED"}

            # Get holdings from Zerodha
            zerodha_holdings = await self.zerodha_client.get_holdings()

            # Get local holdings
            local_holdings = await self._get_local_holdings()

            # Create lookup dictionaries
            zerodha_lookup = {}
            for holding in zerodha_holdings:
                symbol = format_indian_symbol(holding.get("tradingsymbol", ""), holding.get("exchange", "NSE"))
                zerodha_lookup[symbol] = holding

            local_lookup = {}
            for holding in local_holdings:
                local_lookup[holding["symbol"]] = holding

            # Find discrepancies
            all_symbols = set(zerodha_lookup.keys()) | set(local_lookup.keys())

            for symbol in all_symbols:
                zerodha_holding = zerodha_lookup.get(symbol)
                local_holding = local_lookup.get(symbol)

                if not zerodha_holding and local_holding:
                    discrepancies.append({
                        "symbol": symbol,
                        "type": "MISSING_IN_ZERODHA",
                        "local_quantity": local_holding["quantity"],
                        "zerodha_quantity": 0
                    })
                elif zerodha_holding and not local_holding:
                    discrepancies.append({
                        "symbol": symbol,
                        "type": "MISSING_LOCALLY",
                        "local_quantity": 0,
                        "zerodha_quantity": zerodha_holding.get("quantity", 0)
                    })
                elif zerodha_holding and local_holding:
                    zerodha_qty = int(zerodha_holding.get("quantity", 0))
                    local_qty = int(local_holding["quantity"])

                    if zerodha_qty != local_qty:
                        discrepancies.append({
                            "symbol": symbol,
                            "type": "QUANTITY_MISMATCH",
                            "local_quantity": local_qty,
                            "zerodha_quantity": zerodha_qty,
                            "difference": zerodha_qty - local_qty
                        })

            return {
                "discrepancies": discrepancies,
                "total_discrepancies": len(discrepancies),
                "status": "COMPLETED"
            }

        except Exception as e:
            logger.error(f"Error reconciling holdings: {e}")
            return {"discrepancies": [], "status": "ERROR", "error": str(e)}

    async def _get_local_positions(self) -> List[Dict[str, Any]]:
        """Get local positions from database."""
        try:
            if not self.db_manager:
                return []

            query = """
                SELECT pp.*, i.symbol
                FROM portfolio_positions pp
                JOIN instruments i ON pp.instrument_id = i.id
                WHERE pp.quantity != 0
            """

            results = await self.db_manager.execute_query(query, fetch="all")
            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Error getting local positions: {e}")
            return []

    async def _get_local_holdings(self) -> List[Dict[str, Any]]:
        """Get local holdings from database."""
        try:
            if not self.db_manager:
                return []

            query = """
                SELECT ph.*, i.symbol
                FROM portfolio_holdings ph
                JOIN instruments i ON ph.instrument_id = i.id
                WHERE ph.quantity > 0
            """

            results = await self.db_manager.execute_query(query, fetch="all")
            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Error getting local holdings: {e}")
            return []

    def get_sync_status(self) -> Dict[str, Any]:
        """Get sync service status."""
        return {
            "is_running": self.is_running,
            "zerodha_connected": self.zerodha_client is not None,
            "last_positions_sync": self.last_positions_sync.isoformat() if self.last_positions_sync else None,
            "last_holdings_sync": self.last_holdings_sync.isoformat() if self.last_holdings_sync else None,
            "last_reconciliation": self.last_reconciliation.isoformat() if self.last_reconciliation else None,
            "sync_config": self.sync_config,
            "sync_stats": self.sync_stats
        }

    async def force_full_sync(self) -> Dict[str, Any]:
        """Force a complete sync of positions and holdings."""
        try:
            logger.info("Starting forced full sync")

            # Sync positions
            positions_result = await self.sync_positions()

            # Sync holdings
            holdings_result = await self.sync_holdings()

            # Run reconciliation
            reconciliation_result = await self.run_reconciliation()

            return {
                "status": "SUCCESS",
                "positions": positions_result,
                "holdings": holdings_result,
                "reconciliation": reconciliation_result
            }

        except Exception as e:
            logger.error(f"Error in forced full sync: {e}")
            return {"status": "ERROR", "error": str(e)}


# Global instance
position_holdings_sync = PositionHoldingsSync()
