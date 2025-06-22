"""
Autonomous Trading Agent for AWM system.
Continuously monitors markets, analyzes data, generates signals, makes decisions, and executes trades.
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timedelta, timezone
import time

# Add the project root to Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from shared.agents.base import BaseAgent
from shared.mcp_client.base import MCPClient

logger = logging.getLogger(__name__)


class AutonomousTradingAgent(BaseAgent):
    """Autonomous Trading Agent implementation."""
    
    def __init__(self):
        super().__init__("autonomous_trading_agent")
        
        # Agent configuration
        self.config = {
            "trading_enabled": os.getenv("AUTONOMOUS_TRADING_ENABLED", "false").lower() == "true",
            "paper_trading_only": os.getenv("PAPER_TRADING_ONLY", "true").lower() == "true",
            "scan_interval_seconds": int(os.getenv("SCAN_INTERVAL_SECONDS", "300")),  # 5 minutes
            "decision_config": os.getenv("DECISION_CONFIG", "moderate"),
            "max_daily_trades": int(os.getenv("MAX_DAILY_TRADES", "10")),
            "portfolio_id": os.getenv("DEFAULT_PORTFOLIO_ID", "default-portfolio-id"),
            "watchlist": os.getenv("TRADING_WATCHLIST", "RELIANCE,TCS,HDFCBANK,INFY,HINDUNILVR").split(",")
        }
        
        # Server URLs
        self.server_urls = {
            "signal_generation": os.getenv("SIGNAL_GENERATION_SERVER_URL", "http://signal-generation-server:8004"),
            "decision_engine": os.getenv("DECISION_ENGINE_SERVER_URL", "http://decision-engine-server:8005"),
            "risk_management": os.getenv("RISK_MANAGEMENT_ENGINE_URL", "http://risk-management-engine:8010"),
            "oms": os.getenv("OMS_URL", "http://order-management-system:8011"),
            "portfolio_management": os.getenv("PORTFOLIO_MANAGEMENT_SYSTEM_URL", "http://portfolio-management-system:8012"),
            "llm_market_intelligence": os.getenv("LLM_MARKET_INTELLIGENCE_SERVER_URL", "http://llm-market-intelligence-server:8007")
        }
        
        # Trading state
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.active_orders = {}
        self.position_monitors = {}
        
        # Performance tracking
        self.performance_metrics = {
            "total_signals_generated": 0,
            "total_decisions_made": 0,
            "total_trades_executed": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "total_pnl": 0.0
        }
        
        # Register task handlers
        self.register_task_handlers()
    
    def register_task_handlers(self):
        """Register task handlers for the agent."""
        
        @self.task_handler("start_autonomous_trading")
        async def start_autonomous_trading(parameters: Dict[str, Any]) -> Dict[str, Any]:
            """Start autonomous trading operations."""
            if not self.config["trading_enabled"]:
                return {
                    "status": "DISABLED",
                    "message": "Autonomous trading is disabled in configuration"
                }
            
            self.logger.info("Starting autonomous trading operations...")
            
            # Start the main trading loop
            asyncio.create_task(self._autonomous_trading_loop())
            
            return {
                "status": "STARTED",
                "message": "Autonomous trading operations started",
                "config": self.config
            }
        
        @self.task_handler("stop_autonomous_trading")
        async def stop_autonomous_trading(parameters: Dict[str, Any]) -> Dict[str, Any]:
            """Stop autonomous trading operations."""
            self.logger.info("Stopping autonomous trading operations...")
            
            # Cancel all pending orders
            await self._cancel_all_pending_orders()
            
            return {
                "status": "STOPPED",
                "message": "Autonomous trading operations stopped",
                "performance_metrics": self.performance_metrics
            }
        
        @self.task_handler("get_trading_status")
        async def get_trading_status(parameters: Dict[str, Any]) -> Dict[str, Any]:
            """Get current trading status and performance."""
            return {
                "status": "ACTIVE" if self.config["trading_enabled"] else "INACTIVE",
                "config": self.config,
                "performance_metrics": self.performance_metrics,
                "daily_trade_count": self.daily_trade_count,
                "active_orders": len(self.active_orders),
                "monitored_positions": len(self.position_monitors)
            }
        
        @self.task_handler("update_trading_config")
        async def update_trading_config(parameters: Dict[str, Any]) -> Dict[str, Any]:
            """Update trading configuration."""
            config_updates = parameters.get("config_updates", {})
            
            # Update configuration
            self.config.update(config_updates)
            
            self.logger.info(f"Updated trading configuration: {config_updates}")
            
            return {
                "status": "UPDATED",
                "updated_config": self.config
            }
        
        @self.task_handler("force_market_scan")
        async def force_market_scan(parameters: Dict[str, Any]) -> Dict[str, Any]:
            """Force an immediate market scan and decision cycle."""
            symbols = parameters.get("symbols", self.config["watchlist"])
            
            self.logger.info(f"Forcing market scan for symbols: {symbols}")
            
            scan_results = await self._perform_market_scan(symbols)
            
            return {
                "status": "COMPLETED",
                "scan_results": scan_results
            }

    async def _autonomous_trading_loop(self):
        """Main autonomous trading loop."""
        self.logger.info("Starting autonomous trading loop...")
        
        while self.config["trading_enabled"]:
            try:
                # Check if we need to reset daily trade count
                await self._check_daily_reset()
                
                # Check if we've reached daily trade limit
                if self.daily_trade_count >= self.config["max_daily_trades"]:
                    self.logger.info(f"Daily trade limit reached ({self.daily_trade_count}), skipping scan")
                    await asyncio.sleep(self.config["scan_interval_seconds"])
                    continue
                
                # Perform market scan and decision making
                scan_results = await self._perform_market_scan(self.config["watchlist"])
                
                # Process decisions and execute trades
                await self._process_trading_decisions(scan_results)
                
                # Monitor existing positions
                await self._monitor_positions()
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Wait for next scan interval
                await asyncio.sleep(self.config["scan_interval_seconds"])
                
            except Exception as e:
                self.logger.error(f"Error in autonomous trading loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _perform_market_scan(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Perform market scan and generate trading decisions with LLM intelligence."""
        scan_results = []

        # Get overall market intelligence before scanning individual symbols
        market_intelligence = await self._get_market_intelligence_overview()

        for symbol in symbols:
            try:
                # Generate signals for the symbol
                signals = await self._generate_signals(symbol)

                # Get symbol-specific LLM insights for significant decisions
                symbol_insights = await self._get_symbol_insights(symbol, signals)

                # Make trading decision with LLM context
                decision = await self._make_trading_decision(symbol, signals)

                # Enhance decision with LLM rationale
                enhanced_decision = await self._enhance_decision_with_llm(symbol, decision, symbol_insights, market_intelligence)

                scan_results.append({
                    "symbol": symbol,
                    "signals": signals,
                    "decision": enhanced_decision,
                    "symbol_insights": symbol_insights,
                    "market_intelligence": market_intelligence,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

                self.performance_metrics["total_signals_generated"] += 1
                self.performance_metrics["total_decisions_made"] += 1

            except Exception as e:
                self.logger.error(f"Error scanning {symbol}: {e}")
                scan_results.append({
                    "symbol": symbol,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

        return scan_results
    
    async def _generate_signals(self, symbol: str) -> Dict[str, Any]:
        """Generate trading signals for a symbol."""
        try:
            async with self.mcp_client as client:
                response = await client.send_request(
                    self.server_urls["signal_generation"],
                    "generate_signals",
                    {"symbol": symbol}
                )
                return response
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")
            return {}
    
    async def _make_trading_decision(self, symbol: str, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Make trading decision based on signals."""
        try:
            async with self.mcp_client as client:
                response = await client.send_request(
                    self.server_urls["decision_engine"],
                    "make_trading_decision",
                    {
                        "symbol": symbol,
                        "portfolio_id": self.config["portfolio_id"],
                        "config": self.config["decision_config"],
                        "override_params": {
                            "paper_trading": self.config["paper_trading_only"]
                        }
                    }
                )
                return response.get("decision", {})
        except Exception as e:
            self.logger.error(f"Error making trading decision for {symbol}: {e}")
            return {}

    async def _process_trading_decisions(self, scan_results: List[Dict[str, Any]]):
        """Process trading decisions and execute trades."""
        for result in scan_results:
            if "error" in result:
                continue

            symbol = result["symbol"]
            decision = result.get("decision", {})
            action = decision.get("action")

            if action in ["BUY", "SELL"]:
                # Execute the trade
                trade_result = await self._execute_trade(decision)

                if trade_result.get("status") == "SUCCESS":
                    self.daily_trade_count += 1
                    self.performance_metrics["total_trades_executed"] += 1
                    self.performance_metrics["successful_trades"] += 1

                    # Add to active orders for monitoring
                    order_id = trade_result.get("order_id")
                    if order_id:
                        self.active_orders[order_id] = {
                            "symbol": symbol,
                            "decision": decision,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }

                    self.logger.info(f"Successfully executed {action} order for {symbol}: {order_id}")
                else:
                    self.performance_metrics["failed_trades"] += 1
                    self.logger.error(f"Failed to execute {action} order for {symbol}: {trade_result.get('error')}")

    async def _execute_trade(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trading decision."""
        try:
            # Prepare order request
            order_request = {
                "portfolio_id": self.config["portfolio_id"],
                "symbol": decision["symbol"],
                "side": decision["action"],
                "quantity": decision["quantity"],
                "order_type": decision.get("order_type", "MARKET"),
                "paper_trading": decision.get("paper_trading", True)
            }

            # Add price for limit orders
            if order_request["order_type"] == "LIMIT":
                order_request["price"] = decision.get("estimated_price")

            # Execute order through OMS
            async with self.mcp_client as client:
                response = await client.send_request(
                    self.server_urls["oms"],
                    "place_order",
                    order_request
                )
                return response
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return {"status": "ERROR", "error": str(e)}

    async def _monitor_positions(self):
        """Monitor existing positions for stop loss and take profit."""
        try:
            # Get current portfolio positions
            async with self.mcp_client as client:
                portfolio_response = await client.send_request(
                    self.server_urls["portfolio_management"],
                    "get_portfolio_state",
                    {"portfolio_id": self.config["portfolio_id"]}
                )

            positions = portfolio_response.get("positions", [])

            for position in positions:
                symbol = position.get("symbol")
                quantity = position.get("quantity", 0)

                if quantity != 0:  # Only monitor non-zero positions
                    await self._check_position_exits(position)

        except Exception as e:
            self.logger.error(f"Error monitoring positions: {e}")

    async def _check_position_exits(self, position: Dict[str, Any]):
        """Check if position should be exited based on stop loss or take profit."""
        symbol = position.get("symbol")
        quantity = position.get("quantity", 0)
        avg_price = position.get("average_price", 0)

        if not symbol or quantity == 0:
            return

        try:
            # Get current market price
            async with self.mcp_client as client:
                market_data = await client.send_request(
                    self.server_urls["market_data"] if "market_data" in self.server_urls else self.server_urls["oms"],
                    "get_quote",
                    {"symbol": symbol}
                )

            current_price = market_data.get("price", market_data.get("close", 0))
            if not current_price:
                return

            # Calculate P&L percentage
            if quantity > 0:  # Long position
                pnl_pct = (current_price - avg_price) / avg_price
            else:  # Short position
                pnl_pct = (avg_price - current_price) / avg_price

            # Check for exit conditions
            should_exit = False
            exit_reason = ""

            # Stop loss check
            stop_loss_pct = 0.05  # 5% stop loss (should come from config)
            if pnl_pct <= -stop_loss_pct:
                should_exit = True
                exit_reason = f"Stop loss triggered: {pnl_pct:.2%}"

            # Take profit check
            take_profit_pct = 0.10  # 10% take profit (should come from config)
            if pnl_pct >= take_profit_pct:
                should_exit = True
                exit_reason = f"Take profit triggered: {pnl_pct:.2%}"

            if should_exit:
                # Execute exit order
                exit_decision = {
                    "symbol": symbol,
                    "action": "SELL" if quantity > 0 else "BUY",
                    "quantity": abs(quantity),
                    "order_type": "MARKET",
                    "paper_trading": self.config["paper_trading_only"]
                }

                trade_result = await self._execute_trade(exit_decision)

                if trade_result.get("status") == "SUCCESS":
                    self.logger.info(f"Executed exit order for {symbol}: {exit_reason}")
                    self.daily_trade_count += 1
                    self.performance_metrics["total_trades_executed"] += 1
                else:
                    self.logger.error(f"Failed to execute exit order for {symbol}: {trade_result.get('error')}")

        except Exception as e:
            self.logger.error(f"Error checking position exits for {symbol}: {e}")

    async def _check_daily_reset(self):
        """Check if we need to reset daily counters."""
        today = datetime.now().date()

        if self.last_trade_date != today:
            self.daily_trade_count = 0
            self.last_trade_date = today
            self.logger.info(f"Reset daily trade count for {today}")

    async def _cancel_all_pending_orders(self):
        """Cancel all pending orders."""
        try:
            for order_id in list(self.active_orders.keys()):
                async with self.mcp_client as client:
                    await client.send_request(
                        self.server_urls["oms"],
                        "cancel_order",
                        {"order_id": order_id}
                    )

                del self.active_orders[order_id]
                self.logger.info(f"Cancelled order: {order_id}")

        except Exception as e:
            self.logger.error(f"Error cancelling orders: {e}")

    async def _update_performance_metrics(self):
        """Update performance metrics."""
        try:
            # Get portfolio performance
            async with self.mcp_client as client:
                portfolio_response = await client.send_request(
                    self.server_urls["portfolio_management"],
                    "get_portfolio_analytics",
                    {"portfolio_id": self.config["portfolio_id"]}
                )

            summary = portfolio_response.get("summary", {})
            self.performance_metrics["total_pnl"] = summary.get("total_pnl", 0.0)

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    async def _get_market_intelligence_overview(self) -> Dict[str, Any]:
        """Get overall market intelligence for trading session."""
        try:
            # Get current portfolio state for context
            async with self.mcp_client as client:
                portfolio_response = await client.send_request(
                    self.server_urls["portfolio_management"],
                    "get_portfolio_state",
                    {"portfolio_id": self.config["portfolio_id"]}
                )

                # Prepare market context
                market_context = {
                    "trading_session": "autonomous",
                    "portfolio_value": portfolio_response.get("total_value", 0),
                    "cash_available": portfolio_response.get("cash_available", 0),
                    "positions_count": len(portfolio_response.get("positions", [])),
                    "daily_trades_executed": self.daily_trade_count,
                    "max_daily_trades": self.config["max_daily_trades"],
                    "watchlist": self.config["watchlist"]
                }

                # Get market insights
                insights_response = await client.send_request(
                    self.server_urls["llm_market_intelligence"],
                    "generate_market_insights",
                    {
                        "market_context": market_context,
                        "focus_areas": ["autonomous_trading", "risk_management", "market_timing"]
                    }
                )

                return insights_response

        except Exception as e:
            self.logger.error(f"Error getting market intelligence overview: {e}")
            return {"actionable_insights": [], "market_outlook": "neutral", "confidence": 0.5}

    async def _get_symbol_insights(self, symbol: str, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Get LLM insights for specific symbol before significant position changes."""
        try:
            # Check if this would be a significant decision
            combined_signals = signals.get("signals", {}).get("combined_signals", {})
            signal_action = combined_signals.get("signal", "HOLD")
            signal_confidence = combined_signals.get("confidence", 0.0)

            # Only get detailed insights for high-confidence signals
            if signal_action in ["BUY", "SELL"] and signal_confidence > 0.6:

                # Get recent news for sentiment analysis
                async with self.mcp_client as client:
                    # Get market context for this symbol
                    current_conditions = {
                        "symbol": symbol,
                        "signal_action": signal_action,
                        "signal_confidence": signal_confidence,
                        "quantitative_signals": signals.get("signals", {}).get("quantitative_signals", []),
                        "technical_analysis": signals.get("signals", {}).get("technical_analysis", {}),
                        "news_sentiment": signals.get("signals", {}).get("news_sentiment", {}),
                        "llm_sentiment": signals.get("signals", {}).get("llm_sentiment", {}),
                        "market_regime": signals.get("signals", {}).get("market_regime", {})
                    }

                    # Get detailed market context explanation
                    explanation_response = await client.send_request(
                        self.server_urls["llm_market_intelligence"],
                        "explain_market_context",
                        {
                            "current_conditions": current_conditions,
                            "detail_level": "detailed"
                        }
                    )

                    return explanation_response
            else:
                # For low-confidence or HOLD signals, return basic context
                return {
                    "explanation": f"Low confidence {signal_action} signal for {symbol}",
                    "key_points": [f"Signal confidence: {signal_confidence:.2f}"],
                    "trader_focus": "Monitor for stronger signals",
                    "risk_warning": "Avoid trading on weak signals"
                }

        except Exception as e:
            self.logger.error(f"Error getting symbol insights for {symbol}: {e}")
            return {"explanation": "Symbol insights unavailable", "key_points": []}

    async def _enhance_decision_with_llm(self, symbol: str, decision: Dict[str, Any],
                                       symbol_insights: Dict[str, Any],
                                       market_intelligence: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance trading decision with LLM-generated rationale and context."""
        try:
            action = decision.get("action", "NO_ACTION")

            # Add LLM context to decision
            enhanced_decision = decision.copy()
            enhanced_decision["llm_context"] = {
                "symbol_insights": symbol_insights,
                "market_intelligence": market_intelligence,
                "decision_rationale": self._generate_decision_rationale(symbol, decision, symbol_insights, market_intelligence),
                "risk_assessment": self._assess_decision_risk(decision, symbol_insights, market_intelligence),
                "confidence_adjustment": self._adjust_confidence_with_llm(decision, symbol_insights, market_intelligence)
            }

            # Log significant decisions with LLM context
            if action in ["BUY", "SELL"]:
                self.logger.info(f"Enhanced {action} decision for {symbol}: {enhanced_decision['llm_context']['decision_rationale']}")

            return enhanced_decision

        except Exception as e:
            self.logger.error(f"Error enhancing decision with LLM for {symbol}: {e}")
            return decision

    def _generate_decision_rationale(self, symbol: str, decision: Dict[str, Any],
                                   symbol_insights: Dict[str, Any],
                                   market_intelligence: Dict[str, Any]) -> str:
        """Generate human-readable rationale for the trading decision."""
        action = decision.get("action", "NO_ACTION")
        confidence = decision.get("confidence", 0.0)

        # Extract key insights
        market_outlook = market_intelligence.get("market_outlook", "neutral")
        key_insights = market_intelligence.get("actionable_insights", [])
        symbol_explanation = symbol_insights.get("explanation", "")

        if action == "BUY":
            rationale = f"Initiating BUY position in {symbol} (confidence: {confidence:.2f}). "
            rationale += f"Market outlook: {market_outlook}. "
            if key_insights:
                rationale += f"Key market insight: {key_insights[0] if key_insights else 'N/A'}. "
            if symbol_explanation:
                rationale += f"Symbol context: {symbol_explanation[:100]}..."
        elif action == "SELL":
            rationale = f"Executing SELL for {symbol} (confidence: {confidence:.2f}). "
            rationale += f"Market outlook: {market_outlook}. "
            if key_insights:
                rationale += f"Key market insight: {key_insights[0] if key_insights else 'N/A'}. "
            if symbol_explanation:
                rationale += f"Symbol context: {symbol_explanation[:100]}..."
        else:
            rationale = f"Holding position in {symbol}. Market outlook: {market_outlook}. "
            rationale += "Waiting for stronger signals or better market conditions."

        return rationale

    def _assess_decision_risk(self, decision: Dict[str, Any],
                            symbol_insights: Dict[str, Any],
                            market_intelligence: Dict[str, Any]) -> str:
        """Assess risk level of the trading decision based on LLM insights."""
        action = decision.get("action", "NO_ACTION")

        # Extract risk factors
        market_risk_factors = market_intelligence.get("risk_factors", [])
        symbol_risk_warning = symbol_insights.get("risk_warning", "")

        if action in ["BUY", "SELL"]:
            if len(market_risk_factors) > 2:
                return "HIGH - Multiple market risk factors identified"
            elif symbol_risk_warning and "risk" in symbol_risk_warning.lower():
                return "MEDIUM - Symbol-specific risks noted"
            else:
                return "LOW - Normal market conditions"
        else:
            return "MINIMAL - No position change"

    def _adjust_confidence_with_llm(self, decision: Dict[str, Any],
                                  symbol_insights: Dict[str, Any],
                                  market_intelligence: Dict[str, Any]) -> float:
        """Adjust decision confidence based on LLM market intelligence."""
        original_confidence = decision.get("confidence", 0.0)

        # Extract LLM confidence indicators
        market_confidence = market_intelligence.get("confidence", 0.5)
        market_outlook = market_intelligence.get("market_outlook", "neutral")

        # Adjust confidence based on market conditions
        if market_outlook == "bullish" and market_confidence > 0.7:
            adjusted_confidence = min(1.0, original_confidence * 1.1)
        elif market_outlook == "bearish" and market_confidence > 0.7:
            adjusted_confidence = original_confidence * 0.9
        else:
            adjusted_confidence = original_confidence

        return adjusted_confidence

if __name__ == "__main__":
    async def main():
        """Main function to run the Autonomous Trading Agent."""
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        try:
            # Create and start agent
            agent = AutonomousTradingAgent()
            logger.info("Starting Autonomous Trading Agent...")
            await agent.start()
        except Exception as e:
            logger.error(f"Failed to start agent: {e}")
            raise
    
    asyncio.run(main())
