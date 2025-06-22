"""
Indian Instruments Database Sync Service.
Syncs instrument data from Zerodha to local database with proper formatting.
"""

import asyncio
import logging
import csv
import io
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from decimal import Decimal

from .client import ZerodhaClient, ZerodhaAPIError
from .auth import ZerodhaAuthService
from .utils import format_indian_symbol, get_lot_size, get_tick_size

logger = logging.getLogger(__name__)


class InstrumentsSyncService:
    """Service to sync Indian market instruments from Zerodha to local database."""
    
    def __init__(self, db_manager=None):
        self.auth_service = ZerodhaAuthService()
        self.db_manager = db_manager
        self.last_sync_time = None
        
        # Supported exchanges
        self.supported_exchanges = ["NSE", "BSE", "NFO", "BFO", "CDS", "MCX"]
        
        # Instrument type mapping
        self.instrument_type_mapping = {
            "EQ": "EQUITY",
            "CE": "CALL_OPTION",
            "PE": "PUT_OPTION",
            "FUT": "FUTURE",
            "COMMODITY": "COMMODITY",
            "CURRENCY": "CURRENCY"
        }
    
    async def sync_all_instruments(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Sync all instruments from Zerodha to local database.
        
        Args:
            force_refresh: Force refresh even if recently synced
            
        Returns:
            Sync result summary
        """
        logger.info("Starting instruments sync from Zerodha")
        start_time = datetime.now()
        
        # Check if sync is needed
        if not force_refresh and await self._is_recent_sync():
            return {
                "status": "skipped",
                "reason": "Recent sync found",
                "last_sync": self.last_sync_time
            }
        
        try:
            # Get authenticated client
            client = await self.auth_service.get_authenticated_client()
            
            sync_results = {}
            total_instruments = 0
            
            # Sync instruments for each exchange
            for exchange in self.supported_exchanges:
                try:
                    result = await self._sync_exchange_instruments(client, exchange)
                    sync_results[exchange] = result
                    total_instruments += result.get("count", 0)
                    
                except Exception as e:
                    logger.error(f"Failed to sync {exchange} instruments: {e}")
                    sync_results[exchange] = {"error": str(e)}
            
            # Update sync timestamp
            await self._update_sync_timestamp()
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": "success",
                "total_instruments": total_instruments,
                "exchanges": sync_results,
                "duration_seconds": duration,
                "synced_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Instruments sync failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "duration_seconds": (datetime.now() - start_time).total_seconds()
            }
    
    async def _sync_exchange_instruments(self, client: ZerodhaClient, exchange: str) -> Dict[str, Any]:
        """Sync instruments for a specific exchange."""
        logger.info(f"Syncing instruments for {exchange}")
        
        try:
            # Get instruments from Zerodha
            instruments = await client.get_instruments(exchange)
            
            if not instruments:
                return {"count": 0, "message": "No instruments found"}
            
            # Process and store instruments
            processed_count = 0
            error_count = 0
            
            for instrument in instruments:
                try:
                    await self._process_instrument(instrument, exchange)
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Error processing instrument {instrument.get('tradingsymbol', 'unknown')}: {e}")
                    error_count += 1
            
            logger.info(f"Synced {processed_count} instruments for {exchange} (errors: {error_count})")
            
            return {
                "count": processed_count,
                "errors": error_count,
                "total_received": len(instruments)
            }
            
        except Exception as e:
            logger.error(f"Failed to sync {exchange} instruments: {e}")
            raise
    
    async def _process_instrument(self, instrument: Dict[str, Any], exchange: str):
        """Process and store a single instrument."""
        try:
            # Extract instrument data
            symbol = instrument.get("tradingsymbol", "")
            name = instrument.get("name", symbol)
            instrument_token = instrument.get("instrument_token")
            
            # Determine instrument type
            segment = instrument.get("segment", "")
            instrument_type = self._determine_instrument_type(instrument, segment)
            
            # Get lot size and tick size
            lot_size = instrument.get("lot_size", 1)
            tick_size = instrument.get("tick_size", 0.01)
            
            # Format symbol for consistency
            formatted_symbol = format_indian_symbol(symbol, exchange)
            
            # Prepare instrument data for database
            instrument_data = {
                "symbol": formatted_symbol,
                "name": name,
                "instrument_type": instrument_type,
                "exchange": exchange,
                "segment": segment,
                "lot_size": lot_size,
                "tick_size": Decimal(str(tick_size)),
                "instrument_token": instrument_token,
                "expiry": instrument.get("expiry"),
                "strike": instrument.get("strike"),
                "is_active": True,
                "updated_at": datetime.now()
            }
            
            # Store in database
            await self._store_instrument(instrument_data)
            
        except Exception as e:
            logger.error(f"Error processing instrument: {e}")
            raise
    
    def _determine_instrument_type(self, instrument: Dict[str, Any], segment: str) -> str:
        """Determine instrument type from Zerodha data."""
        
        # Check instrument type field
        inst_type = instrument.get("instrument_type", "")
        if inst_type in self.instrument_type_mapping:
            return self.instrument_type_mapping[inst_type]
        
        # Determine from segment
        if segment in ["NSE", "BSE"]:
            return "EQUITY"
        elif segment in ["NFO", "BFO"]:
            if "CE" in instrument.get("tradingsymbol", ""):
                return "CALL_OPTION"
            elif "PE" in instrument.get("tradingsymbol", ""):
                return "PUT_OPTION"
            elif "FUT" in instrument.get("tradingsymbol", ""):
                return "FUTURE"
        elif segment == "CDS":
            return "CURRENCY"
        elif segment == "MCX":
            return "COMMODITY"
        
        # Default
        return "EQUITY"
    
    async def _store_instrument(self, instrument_data: Dict[str, Any]):
        """Store instrument in database."""
        if not self.db_manager:
            logger.warning("No database manager available")
            return
        
        try:
            # Upsert instrument
            query = """
                INSERT INTO instruments 
                (symbol, name, instrument_type, exchange, segment, lot_size, tick_size, 
                 instrument_token, expiry, strike, is_active, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (symbol) DO UPDATE SET
                    name = EXCLUDED.name,
                    instrument_type = EXCLUDED.instrument_type,
                    exchange = EXCLUDED.exchange,
                    segment = EXCLUDED.segment,
                    lot_size = EXCLUDED.lot_size,
                    tick_size = EXCLUDED.tick_size,
                    instrument_token = EXCLUDED.instrument_token,
                    expiry = EXCLUDED.expiry,
                    strike = EXCLUDED.strike,
                    is_active = EXCLUDED.is_active,
                    updated_at = EXCLUDED.updated_at
            """
            
            await self.db_manager.execute_query(
                query,
                instrument_data["symbol"],
                instrument_data["name"],
                instrument_data["instrument_type"],
                instrument_data["exchange"],
                instrument_data["segment"],
                instrument_data["lot_size"],
                instrument_data["tick_size"],
                instrument_data.get("instrument_token"),
                instrument_data.get("expiry"),
                instrument_data.get("strike"),
                instrument_data["is_active"],
                instrument_data["updated_at"]
            )
            
        except Exception as e:
            logger.error(f"Error storing instrument {instrument_data['symbol']}: {e}")
            raise
    
    async def _is_recent_sync(self) -> bool:
        """Check if instruments were synced recently."""
        if not self.db_manager:
            return False
        
        try:
            # Check for recent sync (within last 24 hours)
            query = """
                SELECT MAX(updated_at) as last_update 
                FROM instruments 
                WHERE updated_at > $1
            """
            
            cutoff_time = datetime.now() - timedelta(hours=24)
            result = await self.db_manager.execute_query(query, cutoff_time, fetch="one")
            
            if result and result["last_update"]:
                self.last_sync_time = result["last_update"]
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking recent sync: {e}")
            return False
    
    async def _update_sync_timestamp(self):
        """Update sync timestamp in system config."""
        if not self.db_manager:
            return
        
        try:
            query = """
                INSERT INTO system_config (key, value, description, updated_at)
                VALUES ('instruments_last_sync', $1, 'Last instruments sync timestamp', $2)
                ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = EXCLUDED.updated_at
            """
            
            timestamp = datetime.now().isoformat()
            await self.db_manager.execute_query(query, timestamp, datetime.now())
            
        except Exception as e:
            logger.error(f"Error updating sync timestamp: {e}")
    
    async def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync status."""
        if not self.db_manager:
            return {"status": "no_database"}
        
        try:
            # Get last sync time
            query = "SELECT value FROM system_config WHERE key = 'instruments_last_sync'"
            result = await self.db_manager.execute_query(query, fetch="one")
            
            last_sync = result["value"] if result else None
            
            # Get instrument counts by exchange
            count_query = """
                SELECT exchange, COUNT(*) as count 
                FROM instruments 
                WHERE is_active = true 
                GROUP BY exchange
            """
            
            counts = await self.db_manager.execute_query(count_query, fetch="all")
            
            exchange_counts = {row["exchange"]: row["count"] for row in counts}
            total_instruments = sum(exchange_counts.values())
            
            return {
                "last_sync": last_sync,
                "total_instruments": total_instruments,
                "exchange_counts": exchange_counts,
                "is_recent": await self._is_recent_sync()
            }
            
        except Exception as e:
            logger.error(f"Error getting sync status: {e}")
            return {"status": "error", "error": str(e)}
    
    async def cleanup_inactive_instruments(self, days_old: int = 30) -> int:
        """Clean up instruments that haven't been updated recently."""
        if not self.db_manager:
            return 0
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            query = """
                UPDATE instruments 
                SET is_active = false 
                WHERE updated_at < $1 AND is_active = true
            """
            
            result = await self.db_manager.execute_query(query, cutoff_date)
            
            # Get count of deactivated instruments
            count_query = """
                SELECT COUNT(*) as count 
                FROM instruments 
                WHERE updated_at < $1 AND is_active = false
            """
            
            count_result = await self.db_manager.execute_query(count_query, cutoff_date, fetch="one")
            deactivated_count = count_result["count"] if count_result else 0
            
            logger.info(f"Deactivated {deactivated_count} old instruments")
            return deactivated_count
            
        except Exception as e:
            logger.error(f"Error cleaning up instruments: {e}")
            return 0
