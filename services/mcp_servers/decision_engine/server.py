"""
Decision Engine MCP Server for AWM system.
Makes autonomous trading decisions by combining signals with risk management and portfolio constraints.
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from decimal import Decimal

# Add the project root to Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from shared.mcp_client.server import MCPServer, validate_required_fields, sanitize_input
from shared.mcp_client.base import MCPClient

logger = logging.getLogger(__name__)


class DecisionEngineServer(MCPServer):
    """Decision Engine MCP Server implementation."""
    
    def __init__(self):
        host = os.getenv("DECISION_ENGINE_SERVER_HOST", "0.0.0.0")
        port = int(os.getenv("DECISION_ENGINE_SERVER_PORT", "8005"))
        super().__init__("decision_engine_server", host, port)
        
        # MCP client for calling other servers
        self.mcp_client = MCPClient("decision_engine")
        
        # Server URLs
        self.server_urls = {
            "signal_generation": os.getenv("SIGNAL_GENERATION_SERVER_URL", "http://signal-generation-server:8004"),
            "risk_management": os.getenv("RISK_MANAGEMENT_ENGINE_URL", "http://risk-management-engine:8010"),
            "portfolio_management": os.getenv("PORTFOLIO_MANAGEMENT_SYSTEM_URL", "http://portfolio-management-system:8012"),
            "market_data": os.getenv("MARKET_DATA_SERVER_URL", "http://market-data-server:8001"),
            "llm_market_intelligence": os.getenv("LLM_MARKET_INTELLIGENCE_SERVER_URL", "http://llm-market-intelligence-server:8007")
        }
        
        # Decision engine configurations
        self.decision_configs = {
            "conservative": {
                "min_signal_confidence": 0.8,
                "max_position_size_pct": 0.05,  # 5% of portfolio
                "max_portfolio_risk": 0.02,     # 2% VaR
                "require_consensus": True,
                "stop_loss_pct": 0.03,          # 3% stop loss
                "take_profit_pct": 0.06,        # 6% take profit
                "paper_trading": True
            },
            "moderate": {
                "min_signal_confidence": 0.7,
                "max_position_size_pct": 0.08,  # 8% of portfolio
                "max_portfolio_risk": 0.03,     # 3% VaR
                "require_consensus": True,
                "stop_loss_pct": 0.04,          # 4% stop loss
                "take_profit_pct": 0.08,        # 8% take profit
                "paper_trading": True
            },
            "aggressive": {
                "min_signal_confidence": 0.6,
                "max_position_size_pct": 0.12,  # 12% of portfolio
                "max_portfolio_risk": 0.05,     # 5% VaR
                "require_consensus": False,
                "stop_loss_pct": 0.05,          # 5% stop loss
                "take_profit_pct": 0.10,        # 10% take profit
                "paper_trading": True
            }
        }
        
        # Register handlers
        self.register_handlers()
    
    def register_handlers(self):
        """Register all MCP handlers."""
        
        @self.handler("make_trading_decision")
        async def make_trading_decision(content: Dict[str, Any]) -> Dict[str, Any]:
            """Make a comprehensive trading decision for a symbol."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["symbol", "portfolio_id"])
            
            symbol = content["symbol"]
            portfolio_id = content["portfolio_id"]
            config_name = content.get("config", "moderate")
            override_params = content.get("override_params", {})
            
            try:
                decision = await self._make_comprehensive_decision(symbol, portfolio_id, config_name, override_params)
                return {
                    "symbol": symbol,
                    "portfolio_id": portfolio_id,
                    "decision": decision,
                    "decision_timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error making trading decision for {symbol}: {str(e)}")
                return {"error": f"Failed to make trading decision: {str(e)}"}
        
        @self.handler("evaluate_trade_proposal")
        async def evaluate_trade_proposal(content: Dict[str, Any]) -> Dict[str, Any]:
            """Evaluate a specific trade proposal against risk constraints."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["trade_proposal"])
            
            trade_proposal = content["trade_proposal"]
            config_name = content.get("config", "moderate")
            
            try:
                evaluation = await self._evaluate_trade_proposal(trade_proposal, config_name)
                return {
                    "trade_proposal": trade_proposal,
                    "evaluation": evaluation,
                    "evaluation_timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error evaluating trade proposal: {str(e)}")
                return {"error": f"Failed to evaluate trade proposal: {str(e)}"}
        
        @self.handler("generate_portfolio_decisions")
        async def generate_portfolio_decisions(content: Dict[str, Any]) -> Dict[str, Any]:
            """Generate trading decisions for entire portfolio."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["portfolio_id"])
            
            portfolio_id = content["portfolio_id"]
            watchlist = content.get("watchlist", [])
            config_name = content.get("config", "moderate")
            
            try:
                portfolio_decisions = await self._generate_portfolio_decisions(portfolio_id, watchlist, config_name)
                return {
                    "portfolio_id": portfolio_id,
                    "portfolio_decisions": portfolio_decisions,
                    "generation_timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error generating portfolio decisions: {str(e)}")
                return {"error": f"Failed to generate portfolio decisions: {str(e)}"}
        
        @self.handler("update_decision_config")
        async def update_decision_config(content: Dict[str, Any]) -> Dict[str, Any]:
            """Update decision engine configuration."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["config_name", "config_updates"])
            
            config_name = content["config_name"]
            config_updates = content["config_updates"]
            
            try:
                updated_config = await self._update_decision_config(config_name, config_updates)
                return {
                    "config_name": config_name,
                    "updated_config": updated_config,
                    "update_timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error updating decision config: {str(e)}")
                return {"error": f"Failed to update decision config: {str(e)}"}

    async def _make_comprehensive_decision(self, symbol: str, portfolio_id: str, config_name: str, override_params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a comprehensive trading decision combining signals and risk management."""
        config = self.decision_configs.get(config_name, self.decision_configs["moderate"]).copy()
        config.update(override_params)
        
        # Step 1: Get trading signals
        signals = await self._get_trading_signals(symbol)
        if not signals:
            return {
                "action": "NO_ACTION",
                "reason": "No trading signals available",
                "confidence": 0.0
            }
        
        # Step 2: Get current portfolio state
        portfolio_state = await self._get_portfolio_state(portfolio_id)
        
        # Step 3: Assess current risk
        risk_assessment = await self._assess_portfolio_risk(portfolio_id)
        
        # Step 4: Get current market data
        market_data = await self._get_current_market_data(symbol)

        # Step 5: Get LLM market intelligence
        market_insights = await self._get_market_insights(symbol, portfolio_state, risk_assessment)

        # Step 6: Make decision based on all factors
        decision = await self._synthesize_decision(
            symbol, signals, portfolio_state, risk_assessment, market_data, market_insights, config
        )

        return decision
    
    async def _get_trading_signals(self, symbol: str) -> Dict[str, Any]:
        """Get trading signals from signal generation server."""
        try:
            async with self.mcp_client as client:
                response = await client.send_request(
                    self.server_urls["signal_generation"],
                    "generate_signals",
                    {"symbol": symbol}
                )
                return response
        except Exception as e:
            logger.error(f"Error getting trading signals for {symbol}: {e}")
            return {}
    
    async def _get_portfolio_state(self, portfolio_id: str) -> Dict[str, Any]:
        """Get current portfolio state."""
        try:
            async with self.mcp_client as client:
                response = await client.send_request(
                    self.server_urls["portfolio_management"],
                    "get_portfolio_state",
                    {"portfolio_id": portfolio_id}
                )
                return response
        except Exception as e:
            logger.error(f"Error getting portfolio state for {portfolio_id}: {e}")
            return {}
    
    async def _assess_portfolio_risk(self, portfolio_id: str) -> Dict[str, Any]:
        """Assess current portfolio risk."""
        try:
            async with self.mcp_client as client:
                response = await client.send_request(
                    self.server_urls["risk_management"],
                    "monitor_portfolio_risk",
                    {"portfolio_id": portfolio_id}
                )
                return response
        except Exception as e:
            logger.error(f"Error assessing portfolio risk for {portfolio_id}: {e}")
            return {}
    
    async def _get_current_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data for the symbol."""
        try:
            async with self.mcp_client as client:
                response = await client.send_request(
                    self.server_urls["market_data"],
                    "get_quote",
                    {"symbol": symbol}
                )
                return response
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return {}

    async def _get_market_insights(self, symbol: str, portfolio_state: Dict[str, Any], risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Get LLM-powered market insights for enhanced decision making."""
        try:
            # Prepare market context for LLM analysis
            market_context = {
                "symbol": symbol,
                "portfolio_metrics": {
                    "total_value": portfolio_state.get("total_value", 0),
                    "cash_available": portfolio_state.get("cash_available", 0),
                    "positions_count": len(portfolio_state.get("positions", [])),
                    "current_risk_level": risk_assessment.get("risk_level", "UNKNOWN")
                },
                "risk_metrics": risk_assessment.get("risk_metrics", {}),
                "market_conditions": {
                    "timestamp": datetime.now().isoformat(),
                    "trading_session": "regular"  # Could be enhanced with actual session detection
                }
            }

            async with self.mcp_client as client:
                # Get market insights
                insights_response = await client.send_request(
                    self.server_urls["llm_market_intelligence"],
                    "generate_market_insights",
                    {
                        "market_context": market_context,
                        "focus_areas": ["trading", "risk_management", "position_sizing"]
                    }
                )

                # Get market context explanation
                explanation_response = await client.send_request(
                    self.server_urls["llm_market_intelligence"],
                    "explain_market_context",
                    {
                        "current_conditions": market_context,
                        "detail_level": "medium"
                    }
                )

                return {
                    "insights": insights_response,
                    "explanation": explanation_response,
                    "context_used": market_context
                }

        except Exception as e:
            logger.error(f"Error getting market insights for {symbol}: {e}")
            return {
                "insights": {"actionable_insights": [], "confidence": 0.5, "market_outlook": "neutral"},
                "explanation": {"explanation": "Market insights unavailable", "key_points": []},
                "context_used": {}
            }

    async def _synthesize_decision(self, symbol: str, signals: Dict[str, Any], portfolio_state: Dict[str, Any],
                                 risk_assessment: Dict[str, Any], market_data: Dict[str, Any],
                                 market_insights: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize all information into a trading decision."""

        # Extract signal information
        combined_signals = signals.get("signals", {}).get("combined_signals", {})
        signal_type = combined_signals.get("signal", "HOLD")
        signal_confidence = combined_signals.get("confidence", 0.0)

        # Check minimum confidence threshold
        if signal_confidence < config["min_signal_confidence"]:
            return {
                "action": "NO_ACTION",
                "reason": f"Signal confidence {signal_confidence:.2f} below threshold {config['min_signal_confidence']}",
                "confidence": signal_confidence,
                "signal_analysis": combined_signals
            }

        # Check risk constraints
        current_risk = risk_assessment.get("risk_metrics", {}).get("var_1d", 0.0)
        if abs(current_risk) > config["max_portfolio_risk"]:
            return {
                "action": "NO_ACTION",
                "reason": f"Portfolio risk {abs(current_risk):.3f} exceeds limit {config['max_portfolio_risk']}",
                "confidence": signal_confidence,
                "risk_analysis": risk_assessment
            }

        # Get current position in this symbol
        positions = portfolio_state.get("positions", [])
        current_position = next((p for p in positions if p.get("symbol") == symbol), None)
        current_quantity = current_position.get("quantity", 0) if current_position else 0

        # Get portfolio value for position sizing
        portfolio_value = portfolio_state.get("total_value", 100000)  # Default to 100k
        current_price = market_data.get("price", market_data.get("close", 0))

        if not current_price:
            return {
                "action": "NO_ACTION",
                "reason": "Unable to get current market price",
                "confidence": 0.0
            }

        # Apply LLM market intelligence adjustments
        llm_adjustments = self._apply_llm_adjustments(signal_confidence, config, market_insights)
        adjusted_confidence = llm_adjustments["adjusted_confidence"]
        position_size_multiplier = llm_adjustments["position_size_multiplier"]
        risk_adjustment = llm_adjustments["risk_adjustment"]

        # Calculate position size with LLM adjustments
        base_position_value = portfolio_value * config["max_position_size_pct"]
        adjusted_position_value = base_position_value * position_size_multiplier
        max_shares = int(adjusted_position_value / current_price)

        # Make decision based on signal type
        if signal_type == "BUY":
            if current_quantity >= max_shares:
                return {
                    "action": "NO_ACTION",
                    "reason": f"Already at maximum position size ({current_quantity} shares)",
                    "confidence": signal_confidence
                }

            # Calculate buy quantity
            buy_quantity = min(max_shares - current_quantity, max_shares // 2)  # Buy in increments

            if buy_quantity <= 0:
                return {
                    "action": "NO_ACTION",
                    "reason": "Calculated buy quantity is zero or negative",
                    "confidence": signal_confidence
                }

            # Calculate stop loss and take profit with risk adjustment
            stop_loss_pct = config["stop_loss_pct"] * (1 + risk_adjustment)
            take_profit_pct = config["take_profit_pct"] * (1 - risk_adjustment * 0.5)

            stop_loss_price = current_price * (1 - stop_loss_pct)
            take_profit_price = current_price * (1 + take_profit_pct)

            return {
                "action": "BUY",
                "symbol": symbol,
                "quantity": buy_quantity,
                "order_type": "MARKET",
                "estimated_price": current_price,
                "estimated_value": buy_quantity * current_price,
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
                "confidence": adjusted_confidence,
                "reason": combined_signals.get("reason", "Buy signal generated"),
                "paper_trading": config.get("paper_trading", True),
                "risk_metrics": {
                    "position_size_pct": (buy_quantity * current_price) / portfolio_value,
                    "portfolio_impact": buy_quantity * current_price / portfolio_value,
                    "llm_adjustments": llm_adjustments
                },
                "llm_insights": {
                    "market_outlook": market_insights.get("insights", {}).get("market_outlook", "neutral"),
                    "key_insights": market_insights.get("insights", {}).get("actionable_insights", [])[:3],
                    "risk_factors": market_insights.get("insights", {}).get("risk_factors", [])[:3]
                }
            }

        elif signal_type == "SELL":
            if current_quantity <= 0:
                return {
                    "action": "NO_ACTION",
                    "reason": "No position to sell",
                    "confidence": signal_confidence
                }

            # Calculate sell quantity (sell partial or full position)
            sell_quantity = min(current_quantity, max(1, current_quantity // 2))  # Sell in increments

            return {
                "action": "SELL",
                "symbol": symbol,
                "quantity": sell_quantity,
                "order_type": "MARKET",
                "estimated_price": current_price,
                "estimated_value": sell_quantity * current_price,
                "confidence": adjusted_confidence,
                "reason": combined_signals.get("reason", "Sell signal generated"),
                "paper_trading": config.get("paper_trading", True),
                "risk_metrics": {
                    "position_reduction_pct": sell_quantity / current_quantity if current_quantity > 0 else 0,
                    "portfolio_impact": sell_quantity * current_price / portfolio_value,
                    "llm_adjustments": llm_adjustments
                },
                "llm_insights": {
                    "market_outlook": market_insights.get("insights", {}).get("market_outlook", "neutral"),
                    "key_insights": market_insights.get("insights", {}).get("actionable_insights", [])[:3],
                    "risk_factors": market_insights.get("insights", {}).get("risk_factors", [])[:3]
                }
            }

        else:  # HOLD or other
            return {
                "action": "HOLD",
                "reason": combined_signals.get("reason", "Hold signal or no clear direction"),
                "confidence": signal_confidence,
                "current_position": current_quantity,
                "signal_analysis": combined_signals
            }

    async def _evaluate_trade_proposal(self, trade_proposal: Dict[str, Any], config_name: str) -> Dict[str, Any]:
        """Evaluate a specific trade proposal against risk constraints."""
        config = self.decision_configs.get(config_name, self.decision_configs["moderate"])

        symbol = trade_proposal.get("symbol")
        side = trade_proposal.get("side")  # BUY or SELL
        quantity = trade_proposal.get("quantity", 0)
        price = trade_proposal.get("price", 0)
        portfolio_id = trade_proposal.get("portfolio_id")

        if not all([symbol, side, quantity, price, portfolio_id]):
            return {
                "approved": False,
                "reason": "Missing required trade proposal fields",
                "risk_score": 1.0
            }

        # Get portfolio state for evaluation
        portfolio_state = await self._get_portfolio_state(portfolio_id)
        portfolio_value = portfolio_state.get("total_value", 100000)

        # Calculate trade value and position size
        trade_value = quantity * price
        position_size_pct = trade_value / portfolio_value

        # Check position size limits
        if position_size_pct > config["max_position_size_pct"]:
            return {
                "approved": False,
                "reason": f"Position size {position_size_pct:.2%} exceeds limit {config['max_position_size_pct']:.2%}",
                "risk_score": position_size_pct / config["max_position_size_pct"]
            }

        # Check portfolio risk
        risk_assessment = await self._assess_portfolio_risk(portfolio_id)
        current_risk = risk_assessment.get("risk_metrics", {}).get("var_1d", 0.0)

        if abs(current_risk) > config["max_portfolio_risk"]:
            return {
                "approved": False,
                "reason": f"Portfolio risk {abs(current_risk):.3f} exceeds limit {config['max_portfolio_risk']}",
                "risk_score": abs(current_risk) / config["max_portfolio_risk"]
            }

        # Calculate risk score (0 = low risk, 1 = high risk)
        risk_score = max(
            position_size_pct / config["max_position_size_pct"],
            abs(current_risk) / config["max_portfolio_risk"]
        )

        return {
            "approved": True,
            "reason": "Trade proposal meets all risk constraints",
            "risk_score": risk_score,
            "position_size_pct": position_size_pct,
            "estimated_portfolio_impact": trade_value / portfolio_value,
            "paper_trading_recommended": config.get("paper_trading", True)
        }

    async def _generate_portfolio_decisions(self, portfolio_id: str, watchlist: List[str], config_name: str) -> List[Dict[str, Any]]:
        """Generate trading decisions for entire portfolio."""
        config = self.decision_configs.get(config_name, self.decision_configs["moderate"])
        portfolio_decisions = []

        # Get current portfolio state
        portfolio_state = await self._get_portfolio_state(portfolio_id)

        # If no watchlist provided, use current positions
        if not watchlist:
            positions = portfolio_state.get("positions", [])
            watchlist = [p.get("symbol") for p in positions if p.get("symbol")]

        # Generate decisions for each symbol
        for symbol in watchlist:
            try:
                decision = await self._make_comprehensive_decision(symbol, portfolio_id, config_name, {})
                portfolio_decisions.append({
                    "symbol": symbol,
                    "decision": decision,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error generating decision for {symbol}: {e}")
                portfolio_decisions.append({
                    "symbol": symbol,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

        # Rank decisions by confidence and potential impact
        portfolio_decisions.sort(key=lambda x: x.get("decision", {}).get("confidence", 0), reverse=True)

        # Add portfolio-level analysis
        total_actions = len([d for d in portfolio_decisions if d.get("decision", {}).get("action") not in ["NO_ACTION", "HOLD"]])

        return {
            "decisions": portfolio_decisions,
            "summary": {
                "total_symbols_analyzed": len(watchlist),
                "actionable_decisions": total_actions,
                "config_used": config_name,
                "portfolio_id": portfolio_id
            }
        }

    async def _update_decision_config(self, config_name: str, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update decision engine configuration."""
        if config_name not in self.decision_configs:
            # Create new config based on moderate defaults
            self.decision_configs[config_name] = self.decision_configs["moderate"].copy()

        # Update configuration
        self.decision_configs[config_name].update(config_updates)

        # Validate configuration
        config = self.decision_configs[config_name]

        # Ensure required fields exist
        required_fields = [
            "min_signal_confidence", "max_position_size_pct", "max_portfolio_risk",
            "stop_loss_pct", "take_profit_pct", "paper_trading"
        ]

        for field in required_fields:
            if field not in config:
                raise ValueError(f"Required configuration field '{field}' is missing")

        # Validate ranges
        if not (0 <= config["min_signal_confidence"] <= 1):
            raise ValueError("min_signal_confidence must be between 0 and 1")

        if not (0 < config["max_position_size_pct"] <= 1):
            raise ValueError("max_position_size_pct must be between 0 and 1")

        if not (0 < config["max_portfolio_risk"] <= 1):
            raise ValueError("max_portfolio_risk must be between 0 and 1")

        if not (0 < config["stop_loss_pct"] <= 1):
            raise ValueError("stop_loss_pct must be between 0 and 1")

        if not (0 < config["take_profit_pct"] <= 1):
            raise ValueError("take_profit_pct must be between 0 and 1")

        logger.info(f"Updated decision config '{config_name}': {config}")

        return config

    def _apply_llm_adjustments(self, signal_confidence: float, config: Dict[str, Any], market_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Apply LLM market intelligence adjustments to trading decisions."""
        insights = market_insights.get("insights", {})

        # Extract LLM analysis
        market_outlook = insights.get("market_outlook", "neutral")
        llm_confidence = insights.get("confidence", 0.5)
        risk_factors = insights.get("risk_factors", [])
        opportunities = insights.get("opportunities", [])

        # Base adjustments
        adjusted_confidence = signal_confidence
        position_size_multiplier = 1.0
        risk_adjustment = 0.0

        # Market outlook adjustments
        if market_outlook == "bullish" and llm_confidence > 0.7:
            adjusted_confidence *= 1.1  # Increase confidence in bullish market
            position_size_multiplier *= 1.2  # Increase position size
            risk_adjustment -= 0.1  # Reduce risk adjustment (tighter stops)
        elif market_outlook == "bearish" and llm_confidence > 0.7:
            adjusted_confidence *= 0.9  # Decrease confidence in bearish market
            position_size_multiplier *= 0.8  # Decrease position size
            risk_adjustment += 0.2  # Increase risk adjustment (wider stops)
        elif market_outlook == "neutral":
            # No significant adjustments for neutral outlook
            pass

        # Risk factor adjustments
        high_risk_factors = len([rf for rf in risk_factors if "high" in str(rf).lower() or "risk" in str(rf).lower()])
        if high_risk_factors > 2:
            adjusted_confidence *= 0.85
            position_size_multiplier *= 0.7
            risk_adjustment += 0.15

        # Opportunity adjustments
        strong_opportunities = len([op for op in opportunities if "strong" in str(op).lower() or "opportunity" in str(op).lower()])
        if strong_opportunities > 1 and llm_confidence > 0.6:
            adjusted_confidence *= 1.05
            position_size_multiplier *= 1.1

        # Ensure bounds
        adjusted_confidence = max(0.1, min(1.0, adjusted_confidence))
        position_size_multiplier = max(0.3, min(1.5, position_size_multiplier))
        risk_adjustment = max(-0.2, min(0.3, risk_adjustment))

        return {
            "adjusted_confidence": adjusted_confidence,
            "position_size_multiplier": position_size_multiplier,
            "risk_adjustment": risk_adjustment,
            "market_outlook": market_outlook,
            "llm_confidence": llm_confidence,
            "risk_factors_count": high_risk_factors,
            "opportunities_count": strong_opportunities
        }

if __name__ == "__main__":
    async def main():
        """Main function to run the Decision Engine MCP Server."""
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        try:
            # Create and start server
            server = DecisionEngineServer()
            logger.info("Starting Decision Engine MCP Server...")
            await server.start()
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise
    
    asyncio.run(main())
