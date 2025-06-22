"""
Trade Reconciliation Service for AWM System.
Implements automated trade reconciliation between local records and Zerodha.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta, date
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum

from .client import ZerodhaClient
from .auth import ZerodhaAuthService
from .utils import format_indian_symbol, calculate_indian_taxes

logger = logging.getLogger(__name__)


class ReconciliationStatus(Enum):
    """Reconciliation status enumeration."""
    MATCHED = "MATCHED"
    MISSING_LOCAL = "MISSING_LOCAL"
    MISSING_ZERODHA = "MISSING_ZERODHA"
    QUANTITY_MISMATCH = "QUANTITY_MISMATCH"
    PRICE_MISMATCH = "PRICE_MISMATCH"
    TIME_MISMATCH = "TIME_MISMATCH"
    MULTIPLE_ISSUES = "MULTIPLE_ISSUES"


@dataclass
class TradeDiscrepancy:
    """Trade discrepancy structure."""
    trade_id: str
    symbol: str
    status: ReconciliationStatus
    local_trade: Optional[Dict[str, Any]]
    zerodha_trade: Optional[Dict[str, Any]]
    issues: List[str]
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    auto_fixable: bool
    detected_at: datetime


class TradeReconciliationService:
    """
    Automated trade reconciliation service between local records and Zerodha.
    Handles trade matching, discrepancy detection, and automated resolution.
    """
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.zerodha_auth = ZerodhaAuthService()
        self.zerodha_client = None
        
        # Reconciliation configuration
        self.config = {
            "reconciliation_interval": 300,  # 5 minutes
            "daily_reconciliation_time": "16:30",  # 4:30 PM IST
            "lookback_days": 7,  # Days to look back for trades
            "tolerance_price_pct": 0.01,  # 1% price tolerance
            "tolerance_time_minutes": 5,  # 5 minutes time tolerance
            "auto_fix_enabled": True,
            "alert_threshold": 5  # Alert if more than 5 discrepancies
        }
        
        # Reconciliation state
        self.last_reconciliation = None
        self.daily_reconciliation_done = False
        self.discrepancies: List[TradeDiscrepancy] = []
        
        # Statistics
        self.stats = {
            "total_reconciliations": 0,
            "trades_reconciled": 0,
            "discrepancies_found": 0,
            "discrepancies_resolved": 0,
            "auto_fixes_applied": 0,
            "manual_reviews_required": 0,
            "last_reconciliation_duration_ms": 0
        }
        
        # Background tasks
        self.background_tasks = []
        self.is_running = False
    
    async def start(self):
        """Start the trade reconciliation service."""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Starting Trade Reconciliation service")
        
        # Initialize Zerodha client
        await self._initialize_zerodha_client()
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._reconciliation_loop()),
            asyncio.create_task(self._daily_reconciliation_scheduler())
        ]
    
    async def stop(self):
        """Stop the reconciliation service."""
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        
        logger.info("Trade Reconciliation service stopped")
    
    async def _initialize_zerodha_client(self):
        """Initialize Zerodha client."""
        try:
            if await self.zerodha_auth.is_authenticated():
                self.zerodha_client = await self.zerodha_auth.get_authenticated_client()
                logger.info("Zerodha client initialized for trade reconciliation")
            else:
                logger.warning("Zerodha not authenticated - reconciliation will be limited")
        except Exception as e:
            logger.error(f"Failed to initialize Zerodha client: {e}")
    
    async def run_reconciliation(self, target_date: date = None, force: bool = False) -> Dict[str, Any]:
        """
        Run trade reconciliation for a specific date.
        
        Args:
            target_date: Date to reconcile (defaults to today)
            force: Force reconciliation even if already done
            
        Returns:
            Reconciliation result
        """
        start_time = datetime.now()
        
        try:
            if not target_date:
                target_date = date.today()
            
            logger.info(f"Starting trade reconciliation for {target_date}")
            
            # Get trades from both sources
            local_trades = await self._get_local_trades(target_date)
            zerodha_trades = await self._get_zerodha_trades(target_date)
            
            # Perform reconciliation
            reconciliation_result = await self._reconcile_trades(local_trades, zerodha_trades, target_date)
            
            # Process discrepancies
            if reconciliation_result["discrepancies"]:
                await self._process_discrepancies(reconciliation_result["discrepancies"])
            
            # Update statistics
            self.stats["total_reconciliations"] += 1
            self.stats["trades_reconciled"] += reconciliation_result["matched_trades"]
            self.stats["discrepancies_found"] += len(reconciliation_result["discrepancies"])
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.stats["last_reconciliation_duration_ms"] = duration_ms
            
            # Update reconciliation timestamp
            self.last_reconciliation = datetime.now(timezone.utc)
            
            return {
                "status": "SUCCESS",
                "date": target_date.isoformat(),
                "local_trades_count": len(local_trades),
                "zerodha_trades_count": len(zerodha_trades),
                "matched_trades": reconciliation_result["matched_trades"],
                "discrepancies": reconciliation_result["discrepancies"],
                "discrepancy_count": len(reconciliation_result["discrepancies"]),
                "duration_ms": duration_ms,
                "reconciled_at": self.last_reconciliation.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in trade reconciliation: {e}")
            return {
                "status": "ERROR",
                "error": str(e),
                "date": target_date.isoformat() if target_date else None
            }
    
    async def _get_local_trades(self, target_date: date) -> List[Dict[str, Any]]:
        """Get local trades for a specific date."""
        try:
            if not self.db_manager:
                return []
            
            start_datetime = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=timezone.utc)
            end_datetime = start_datetime + timedelta(days=1)
            
            query = """
                SELECT t.*, i.symbol, o.broker_order_id
                FROM trades t
                JOIN instruments i ON t.instrument_id = i.id
                LEFT JOIN orders o ON t.order_id = o.id
                WHERE t.executed_at >= $1 AND t.executed_at < $2
                ORDER BY t.executed_at
            """
            
            results = await self.db_manager.execute_query(query, start_datetime, end_datetime, fetch="all")
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error getting local trades for {target_date}: {e}")
            return []
    
    async def _get_zerodha_trades(self, target_date: date) -> List[Dict[str, Any]]:
        """Get Zerodha trades for a specific date."""
        try:
            if not self.zerodha_client:
                return []
            
            # Get trades from Zerodha
            trades = await self.zerodha_client.get_trades()
            
            # Filter trades for the target date
            filtered_trades = []
            for trade in trades:
                trade_date_str = trade.get("order_timestamp", "")
                if trade_date_str:
                    try:
                        trade_datetime = datetime.fromisoformat(trade_date_str.replace("Z", "+00:00"))
                        if trade_datetime.date() == target_date:
                            filtered_trades.append(trade)
                    except Exception as e:
                        logger.warning(f"Error parsing trade timestamp {trade_date_str}: {e}")
            
            return filtered_trades
            
        except Exception as e:
            logger.error(f"Error getting Zerodha trades for {target_date}: {e}")
            return []
    
    async def _reconcile_trades(self, local_trades: List[Dict[str, Any]], zerodha_trades: List[Dict[str, Any]], target_date: date) -> Dict[str, Any]:
        """Reconcile trades between local and Zerodha."""
        try:
            matched_trades = 0
            discrepancies = []
            
            # Create lookup dictionaries
            local_lookup = self._create_trade_lookup(local_trades, "local")
            zerodha_lookup = self._create_trade_lookup(zerodha_trades, "zerodha")
            
            # Find matches and discrepancies
            all_trade_keys = set(local_lookup.keys()) | set(zerodha_lookup.keys())
            
            for trade_key in all_trade_keys:
                local_trade = local_lookup.get(trade_key)
                zerodha_trade = zerodha_lookup.get(trade_key)
                
                if local_trade and zerodha_trade:
                    # Both exist, check for discrepancies
                    match_result = await self._match_trades(local_trade, zerodha_trade)
                    if match_result["matched"]:
                        matched_trades += 1
                    else:
                        discrepancies.append(self._create_discrepancy(
                            trade_key, local_trade, zerodha_trade, match_result["issues"]
                        ))
                elif local_trade and not zerodha_trade:
                    # Missing in Zerodha
                    discrepancies.append(self._create_discrepancy(
                        trade_key, local_trade, None, ["Trade missing in Zerodha"]
                    ))
                elif zerodha_trade and not local_trade:
                    # Missing locally
                    discrepancies.append(self._create_discrepancy(
                        trade_key, None, zerodha_trade, ["Trade missing locally"]
                    ))
            
            return {
                "matched_trades": matched_trades,
                "discrepancies": discrepancies
            }
            
        except Exception as e:
            logger.error(f"Error reconciling trades: {e}")
            return {"matched_trades": 0, "discrepancies": []}
    
    def _create_trade_lookup(self, trades: List[Dict[str, Any]], source: str) -> Dict[str, Dict[str, Any]]:
        """Create trade lookup dictionary."""
        lookup = {}
        
        for trade in trades:
            if source == "local":
                # Use symbol + side + quantity as key for local trades
                symbol = trade.get("symbol", "")
                side = trade.get("side", "")
                quantity = trade.get("quantity", 0)
                executed_at = trade.get("executed_at", "")
            else:  # zerodha
                # Use tradingsymbol + transaction_type + quantity for Zerodha trades
                symbol = format_indian_symbol(trade.get("tradingsymbol", ""), trade.get("exchange", "NSE"))
                side = trade.get("transaction_type", "")
                quantity = trade.get("quantity", 0)
                executed_at = trade.get("order_timestamp", "")
            
            # Create composite key
            key = f"{symbol}_{side}_{quantity}_{executed_at[:10]}"  # Include date only
            
            if key in lookup:
                # Handle multiple trades with same key
                if not isinstance(lookup[key], list):
                    lookup[key] = [lookup[key]]
                lookup[key].append(trade)
            else:
                lookup[key] = trade
        
        return lookup
    
    async def _match_trades(self, local_trade: Dict[str, Any], zerodha_trade: Dict[str, Any]) -> Dict[str, Any]:
        """Match individual trades and identify discrepancies."""
        issues = []
        matched = True
        
        try:
            # Check quantity
            local_qty = int(local_trade.get("quantity", 0))
            zerodha_qty = int(zerodha_trade.get("quantity", 0))
            
            if local_qty != zerodha_qty:
                issues.append(f"Quantity mismatch: local={local_qty}, zerodha={zerodha_qty}")
                matched = False
            
            # Check price (with tolerance)
            local_price = Decimal(str(local_trade.get("price", 0)))
            zerodha_price = Decimal(str(zerodha_trade.get("average_price", 0)))
            
            if local_price > 0 and zerodha_price > 0:
                price_diff_pct = abs((local_price - zerodha_price) / local_price) * 100
                if price_diff_pct > self.config["tolerance_price_pct"]:
                    issues.append(f"Price mismatch: local={local_price}, zerodha={zerodha_price} (diff: {price_diff_pct:.2f}%)")
                    matched = False
            
            # Check side/transaction type
            local_side = local_trade.get("side", "").upper()
            zerodha_side = zerodha_trade.get("transaction_type", "").upper()
            
            if local_side != zerodha_side:
                issues.append(f"Side mismatch: local={local_side}, zerodha={zerodha_side}")
                matched = False
            
            # Check timing (with tolerance)
            local_time_str = local_trade.get("executed_at", "")
            zerodha_time_str = zerodha_trade.get("order_timestamp", "")
            
            if local_time_str and zerodha_time_str:
                try:
                    local_time = datetime.fromisoformat(local_time_str.replace("Z", "+00:00"))
                    zerodha_time = datetime.fromisoformat(zerodha_time_str.replace("Z", "+00:00"))
                    
                    time_diff_minutes = abs((local_time - zerodha_time).total_seconds()) / 60
                    if time_diff_minutes > self.config["tolerance_time_minutes"]:
                        issues.append(f"Time mismatch: diff={time_diff_minutes:.1f} minutes")
                        matched = False
                        
                except Exception as e:
                    issues.append(f"Time parsing error: {str(e)}")
                    matched = False
            
            return {"matched": matched, "issues": issues}
            
        except Exception as e:
            logger.error(f"Error matching trades: {e}")
            return {"matched": False, "issues": [f"Matching error: {str(e)}"]}
    
    def _create_discrepancy(self, trade_id: str, local_trade: Optional[Dict[str, Any]], zerodha_trade: Optional[Dict[str, Any]], issues: List[str]) -> TradeDiscrepancy:
        """Create trade discrepancy object."""
        
        # Determine status
        if not local_trade:
            status = ReconciliationStatus.MISSING_LOCAL
            severity = "HIGH"
            auto_fixable = True
        elif not zerodha_trade:
            status = ReconciliationStatus.MISSING_ZERODHA
            severity = "CRITICAL"
            auto_fixable = False
        elif len(issues) == 1:
            if "quantity" in issues[0].lower():
                status = ReconciliationStatus.QUANTITY_MISMATCH
                severity = "HIGH"
                auto_fixable = False
            elif "price" in issues[0].lower():
                status = ReconciliationStatus.PRICE_MISMATCH
                severity = "MEDIUM"
                auto_fixable = True
            elif "time" in issues[0].lower():
                status = ReconciliationStatus.TIME_MISMATCH
                severity = "LOW"
                auto_fixable = True
            else:
                status = ReconciliationStatus.MULTIPLE_ISSUES
                severity = "HIGH"
                auto_fixable = False
        else:
            status = ReconciliationStatus.MULTIPLE_ISSUES
            severity = "HIGH"
            auto_fixable = False
        
        # Determine symbol
        symbol = ""
        if local_trade:
            symbol = local_trade.get("symbol", "")
        elif zerodha_trade:
            symbol = format_indian_symbol(zerodha_trade.get("tradingsymbol", ""), zerodha_trade.get("exchange", "NSE"))
        
        return TradeDiscrepancy(
            trade_id=trade_id,
            symbol=symbol,
            status=status,
            local_trade=local_trade,
            zerodha_trade=zerodha_trade,
            issues=issues,
            severity=severity,
            auto_fixable=auto_fixable,
            detected_at=datetime.now(timezone.utc)
        )

    async def _process_discrepancies(self, discrepancies: List[TradeDiscrepancy]):
        """Process and attempt to resolve discrepancies."""
        try:
            for discrepancy in discrepancies:
                # Store discrepancy
                self.discrepancies.append(discrepancy)

                # Attempt auto-fix if enabled and possible
                if self.config["auto_fix_enabled"] and discrepancy.auto_fixable:
                    await self._attempt_auto_fix(discrepancy)
                else:
                    self.stats["manual_reviews_required"] += 1
                    logger.warning(f"Manual review required for discrepancy: {discrepancy.trade_id}")

            # Check if alert threshold is exceeded
            if len(discrepancies) > self.config["alert_threshold"]:
                await self._send_reconciliation_alert(discrepancies)

        except Exception as e:
            logger.error(f"Error processing discrepancies: {e}")

    async def _attempt_auto_fix(self, discrepancy: TradeDiscrepancy):
        """Attempt to automatically fix a discrepancy."""
        try:
            fixed = False

            if discrepancy.status == ReconciliationStatus.MISSING_LOCAL:
                # Create missing local trade from Zerodha data
                fixed = await self._create_missing_local_trade(discrepancy)

            elif discrepancy.status == ReconciliationStatus.PRICE_MISMATCH:
                # Update local trade price if within tolerance
                fixed = await self._fix_price_mismatch(discrepancy)

            elif discrepancy.status == ReconciliationStatus.TIME_MISMATCH:
                # Update local trade timestamp
                fixed = await self._fix_time_mismatch(discrepancy)

            if fixed:
                self.stats["auto_fixes_applied"] += 1
                self.stats["discrepancies_resolved"] += 1
                logger.info(f"Auto-fixed discrepancy: {discrepancy.trade_id}")
            else:
                self.stats["manual_reviews_required"] += 1
                logger.warning(f"Auto-fix failed for discrepancy: {discrepancy.trade_id}")

        except Exception as e:
            logger.error(f"Error in auto-fix for {discrepancy.trade_id}: {e}")
            self.stats["manual_reviews_required"] += 1

    async def _create_missing_local_trade(self, discrepancy: TradeDiscrepancy) -> bool:
        """Create missing local trade from Zerodha data."""
        try:
            if not discrepancy.zerodha_trade or not self.db_manager:
                return False

            zerodha_trade = discrepancy.zerodha_trade

            # Get instrument ID
            symbol = format_indian_symbol(zerodha_trade.get("tradingsymbol", ""), zerodha_trade.get("exchange", "NSE"))
            instrument_id = await self._get_instrument_id_by_symbol(symbol)

            if not instrument_id:
                logger.warning(f"Instrument not found for symbol: {symbol}")
                return False

            # Create trade record
            import uuid
            trade_id = str(uuid.uuid4())

            query = """
                INSERT INTO trades
                (id, portfolio_id, instrument_id, order_id, side, quantity, price,
                 executed_at, settlement_date, commission, taxes, net_amount)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """

            # Get default portfolio
            portfolio_id = await self._get_default_portfolio_id()

            # Parse trade data
            quantity = int(zerodha_trade.get("quantity", 0))
            price = Decimal(str(zerodha_trade.get("average_price", 0)))
            executed_at = datetime.fromisoformat(zerodha_trade.get("order_timestamp", "").replace("Z", "+00:00"))

            # Calculate settlement date (T+2 for equity)
            settlement_date = executed_at.date() + timedelta(days=2)

            # Calculate charges
            trade_value = quantity * price
            taxes = calculate_indian_taxes(trade_value, zerodha_trade.get("transaction_type", "BUY"))
            commission = Decimal("0")  # Zerodha is zero brokerage
            net_amount = trade_value + taxes.get("total_charges", Decimal("0"))

            await self.db_manager.execute_query(
                query,
                trade_id,
                portfolio_id,
                instrument_id,
                None,  # No local order ID
                zerodha_trade.get("transaction_type", "BUY"),
                quantity,
                price,
                executed_at,
                settlement_date,
                commission,
                taxes.get("total_charges", Decimal("0")),
                net_amount
            )

            logger.info(f"Created missing local trade: {trade_id}")
            return True

        except Exception as e:
            logger.error(f"Error creating missing local trade: {e}")
            return False

    async def _fix_price_mismatch(self, discrepancy: TradeDiscrepancy) -> bool:
        """Fix price mismatch by updating local trade."""
        try:
            if not discrepancy.local_trade or not discrepancy.zerodha_trade or not self.db_manager:
                return False

            local_trade = discrepancy.local_trade
            zerodha_trade = discrepancy.zerodha_trade

            # Update local trade price
            new_price = Decimal(str(zerodha_trade.get("average_price", 0)))

            query = """
                UPDATE trades SET
                    price = $2,
                    updated_at = $3
                WHERE id = $1
            """

            await self.db_manager.execute_query(
                query,
                local_trade["id"],
                new_price,
                datetime.now(timezone.utc)
            )

            logger.info(f"Fixed price mismatch for trade: {local_trade['id']}")
            return True

        except Exception as e:
            logger.error(f"Error fixing price mismatch: {e}")
            return False

    async def _fix_time_mismatch(self, discrepancy: TradeDiscrepancy) -> bool:
        """Fix time mismatch by updating local trade timestamp."""
        try:
            if not discrepancy.local_trade or not discrepancy.zerodha_trade or not self.db_manager:
                return False

            local_trade = discrepancy.local_trade
            zerodha_trade = discrepancy.zerodha_trade

            # Update local trade timestamp
            new_timestamp = datetime.fromisoformat(zerodha_trade.get("order_timestamp", "").replace("Z", "+00:00"))

            query = """
                UPDATE trades SET
                    executed_at = $2,
                    updated_at = $3
                WHERE id = $1
            """

            await self.db_manager.execute_query(
                query,
                local_trade["id"],
                new_timestamp,
                datetime.now(timezone.utc)
            )

            logger.info(f"Fixed time mismatch for trade: {local_trade['id']}")
            return True

        except Exception as e:
            logger.error(f"Error fixing time mismatch: {e}")
            return False

    async def _send_reconciliation_alert(self, discrepancies: List[TradeDiscrepancy]):
        """Send alert for high number of discrepancies."""
        try:
            alert_message = f"Trade reconciliation alert: {len(discrepancies)} discrepancies found"

            # Group by severity
            critical = [d for d in discrepancies if d.severity == "CRITICAL"]
            high = [d for d in discrepancies if d.severity == "HIGH"]
            medium = [d for d in discrepancies if d.severity == "MEDIUM"]

            alert_details = {
                "total_discrepancies": len(discrepancies),
                "critical": len(critical),
                "high": len(high),
                "medium": len(medium),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            logger.critical(f"{alert_message}: {alert_details}")

            # Here you would integrate with your alerting system
            # (email, Slack, SMS, etc.)

        except Exception as e:
            logger.error(f"Error sending reconciliation alert: {e}")

    async def _reconciliation_loop(self):
        """Background reconciliation loop."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config["reconciliation_interval"])

                if self.zerodha_client:
                    # Run reconciliation for today
                    await self.run_reconciliation()

            except Exception as e:
                logger.error(f"Error in reconciliation loop: {e}")
                await asyncio.sleep(600)  # 10 minutes on error

    async def _daily_reconciliation_scheduler(self):
        """Daily reconciliation scheduler."""
        while self.is_running:
            try:
                now = datetime.now()
                target_time = datetime.strptime(self.config["daily_reconciliation_time"], "%H:%M").time()
                target_datetime = datetime.combine(now.date(), target_time)

                # If target time has passed today, schedule for tomorrow
                if now.time() > target_time:
                    target_datetime += timedelta(days=1)
                    self.daily_reconciliation_done = False

                # Wait until target time
                wait_seconds = (target_datetime - now).total_seconds()
                if wait_seconds > 0:
                    await asyncio.sleep(min(wait_seconds, 3600))  # Check every hour max

                # Run daily reconciliation
                if not self.daily_reconciliation_done and now.time() >= target_time:
                    await self._run_daily_reconciliation()
                    self.daily_reconciliation_done = True

            except Exception as e:
                logger.error(f"Error in daily reconciliation scheduler: {e}")
                await asyncio.sleep(3600)  # 1 hour on error

    async def _run_daily_reconciliation(self):
        """Run comprehensive daily reconciliation."""
        try:
            logger.info("Starting daily trade reconciliation")

            # Reconcile last few days
            for days_back in range(self.config["lookback_days"]):
                target_date = date.today() - timedelta(days=days_back)
                await self.run_reconciliation(target_date)

            # Generate daily report
            await self._generate_daily_report()

        except Exception as e:
            logger.error(f"Error in daily reconciliation: {e}")

    async def _generate_daily_report(self):
        """Generate daily reconciliation report."""
        try:
            # Get recent discrepancies
            recent_discrepancies = [
                d for d in self.discrepancies
                if d.detected_at.date() >= date.today() - timedelta(days=1)
            ]

            report = {
                "date": date.today().isoformat(),
                "total_discrepancies": len(recent_discrepancies),
                "by_severity": {
                    "critical": len([d for d in recent_discrepancies if d.severity == "CRITICAL"]),
                    "high": len([d for d in recent_discrepancies if d.severity == "HIGH"]),
                    "medium": len([d for d in recent_discrepancies if d.severity == "MEDIUM"]),
                    "low": len([d for d in recent_discrepancies if d.severity == "LOW"])
                },
                "auto_fixes_applied": self.stats["auto_fixes_applied"],
                "manual_reviews_required": self.stats["manual_reviews_required"],
                "stats": self.stats
            }

            logger.info(f"Daily reconciliation report: {report}")

        except Exception as e:
            logger.error(f"Error generating daily report: {e}")

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

    async def _get_default_portfolio_id(self) -> str:
        """Get default portfolio ID."""
        try:
            if not self.db_manager:
                return "default"

            query = """
                SELECT id FROM portfolios
                WHERE name = 'Zerodha Default' AND is_active = true
                LIMIT 1
            """
            result = await self.db_manager.execute_query(query, fetch="one")

            if result:
                return result["id"]

            return "default"

        except Exception as e:
            logger.error(f"Error getting default portfolio: {e}")
            return "default"

    def get_reconciliation_status(self) -> Dict[str, Any]:
        """Get reconciliation service status."""
        return {
            "is_running": self.is_running,
            "zerodha_connected": self.zerodha_client is not None,
            "last_reconciliation": self.last_reconciliation.isoformat() if self.last_reconciliation else None,
            "daily_reconciliation_done": self.daily_reconciliation_done,
            "active_discrepancies": len(self.discrepancies),
            "config": self.config,
            "stats": self.stats
        }

    def get_discrepancies(self, severity: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get current discrepancies."""
        discrepancies = self.discrepancies

        if severity:
            discrepancies = [d for d in discrepancies if d.severity == severity.upper()]

        # Convert to dict format
        result = []
        for d in discrepancies[-limit:]:
            result.append({
                "trade_id": d.trade_id,
                "symbol": d.symbol,
                "status": d.status.value,
                "severity": d.severity,
                "issues": d.issues,
                "auto_fixable": d.auto_fixable,
                "detected_at": d.detected_at.isoformat(),
                "local_trade": d.local_trade,
                "zerodha_trade": d.zerodha_trade
            })

        return result


# Global instance
trade_reconciliation_service = TradeReconciliationService()
