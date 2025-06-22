"""
Trade Execution Agent for AWM system.
Responsible for intelligent order routing and execution logic.
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from decimal import Decimal
from enum import Enum
import openai

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from shared.agents.base_agent import BaseAgent
from shared.models.trading import OrderType, OrderSide, OrderStatus

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    TWAP = "TWAP"  # Time Weighted Average Price
    VWAP = "VWAP"  # Volume Weighted Average Price
    ICEBERG = "ICEBERG"
    SMART = "SMART"


class TradeExecutionAgent(BaseAgent):
    """Agent responsible for intelligent trade execution."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("trade_execution_agent", config)
        
        # OpenAI configuration
        self.openai_client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        
        # Execution parameters
        self.max_order_size = float(os.getenv("MAX_POSITION_SIZE", "100000"))
        self.min_order_value = float(os.getenv("MIN_ORDER_VALUE", "1000"))
        self.default_timeout = self.config.get("order_timeout", 300)  # 5 minutes
        
        # Market impact parameters
        self.impact_threshold = self.config.get("impact_threshold", 0.001)  # 0.1%
        self.slice_size = self.config.get("slice_size", 0.1)  # 10% of order
        
        # Paper trading mode
        self.paper_trading = os.getenv("PAPER_TRADING_MODE", "true").lower() == "true"
    
    async def initialize(self):
        """Initialize the Trade Execution Agent."""
        self.logger.info("Initializing Trade Execution Agent...")
        
        # Test connections to required MCP servers
        try:
            # Test trade execution server
            await self.call_mcp_server("trade_execution", "health", {})
            self.logger.info("✓ Trade Execution Server connection verified")
            
            # Test portfolio management server
            await self.call_mcp_server("portfolio_management", "health", {})
            self.logger.info("✓ Portfolio Management Server connection verified")
            
            # Test market data server
            await self.call_mcp_server("market_data", "health", {})
            self.logger.info("✓ Market Data Server connection verified")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to required servers: {e}")
            raise
        
        if self.paper_trading:
            self.logger.warning("⚠️  PAPER TRADING MODE ENABLED - No real trades will be executed")
        
        self.logger.info("Trade Execution Agent initialized successfully")
    
    async def cleanup(self):
        """Cleanup the Trade Execution Agent."""
        self.logger.info("Cleaning up Trade Execution Agent...")
        # Cancel any pending orders
        await self._cancel_pending_orders()
    
    async def process_task(self, task_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process a trade execution task."""
        
        if task_type == "execute_trade":
            return await self._execute_trade(parameters)
        elif task_type == "optimize_execution":
            return await self._optimize_execution(parameters)
        elif task_type == "monitor_orders":
            return await self._monitor_orders(parameters)
        elif task_type == "cancel_order":
            return await self._cancel_order(parameters)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _execute_trade(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade with intelligent routing."""
        trade_proposal = parameters["trade_proposal"]
        execution_strategy = parameters.get("execution_strategy", "SMART")
        
        self.logger.info(f"Executing trade: {trade_proposal.get('symbol')} {trade_proposal.get('side')} {trade_proposal.get('quantity')}")
        
        try:
            # Validate trade proposal
            validation_result = await self._validate_trade_proposal(trade_proposal)
            if not validation_result["valid"]:
                return {
                    "status": "REJECTED",
                    "reason": validation_result["reason"],
                    "trade_proposal": trade_proposal
                }
            
            # Get market conditions
            market_conditions = await self._analyze_market_conditions(trade_proposal["symbol"])
            
            # Determine optimal execution strategy
            optimal_strategy = await self._determine_execution_strategy(
                trade_proposal, market_conditions, execution_strategy
            )
            
            # Execute based on strategy
            if optimal_strategy["strategy"] == ExecutionStrategy.MARKET:
                result = await self._execute_market_order(trade_proposal, optimal_strategy)
            elif optimal_strategy["strategy"] == ExecutionStrategy.LIMIT:
                result = await self._execute_limit_order(trade_proposal, optimal_strategy)
            elif optimal_strategy["strategy"] == ExecutionStrategy.TWAP:
                result = await self._execute_twap_order(trade_proposal, optimal_strategy)
            elif optimal_strategy["strategy"] == ExecutionStrategy.VWAP:
                result = await self._execute_vwap_order(trade_proposal, optimal_strategy)
            elif optimal_strategy["strategy"] == ExecutionStrategy.ICEBERG:
                result = await self._execute_iceberg_order(trade_proposal, optimal_strategy)
            else:  # SMART
                result = await self._execute_smart_order(trade_proposal, optimal_strategy)
            
            # Log execution result
            await self._log_execution_result(trade_proposal, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return {
                "status": "ERROR",
                "error": str(e),
                "trade_proposal": trade_proposal
            }
    
    async def _validate_trade_proposal(self, trade_proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trade proposal before execution."""
        
        # Check required fields
        required_fields = ["symbol", "side", "quantity", "portfolio_id"]
        for field in required_fields:
            if field not in trade_proposal:
                return {"valid": False, "reason": f"Missing required field: {field}"}
        
        # Validate quantity
        quantity = trade_proposal["quantity"]
        if quantity <= 0:
            return {"valid": False, "reason": "Quantity must be positive"}
        
        # Validate order value
        entry_price = trade_proposal.get("entry_price", 0)
        order_value = quantity * entry_price
        
        if order_value < self.min_order_value:
            return {"valid": False, "reason": f"Order value {order_value} below minimum {self.min_order_value}"}
        
        if order_value > self.max_order_size:
            return {"valid": False, "reason": f"Order value {order_value} exceeds maximum {self.max_order_size}"}
        
        # Check portfolio balance
        portfolio_check = await self._check_portfolio_balance(trade_proposal)
        if not portfolio_check["sufficient"]:
            return {"valid": False, "reason": portfolio_check["reason"]}
        
        return {"valid": True}
    
    async def _check_portfolio_balance(self, trade_proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Check if portfolio has sufficient balance for the trade."""
        try:
            portfolio_response = await self.call_mcp_server(
                "portfolio_management",
                "get_portfolio",
                {"portfolio_id": trade_proposal["portfolio_id"]}
            )
            
            portfolio = portfolio_response.get("portfolio", {})
            available_cash = float(portfolio.get("available_cash", 0))
            
            if trade_proposal["side"] == "BUY":
                required_cash = trade_proposal["quantity"] * trade_proposal.get("entry_price", 0)
                if available_cash < required_cash:
                    return {
                        "sufficient": False,
                        "reason": f"Insufficient cash: {available_cash} < {required_cash}"
                    }
            
            return {"sufficient": True}
            
        except Exception as e:
            return {"sufficient": False, "reason": f"Error checking balance: {e}"}
    
    async def _analyze_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Analyze current market conditions for the symbol."""
        try:
            # Get current quote
            quote_response = await self.call_mcp_server(
                "market_data",
                "get_current_quote",
                {"symbol": symbol}
            )
            
            # Get recent volume data
            history_response = await self.call_mcp_server(
                "market_data",
                "get_price_history",
                {"symbol": symbol, "limit": 20}
            )
            
            recent_data = history_response.get("data", [])
            
            # Calculate average volume
            avg_volume = sum(data["volume"] for data in recent_data) / len(recent_data) if recent_data else 0
            current_volume = quote_response.get("volume", 0)
            
            # Assess liquidity
            liquidity_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if liquidity_ratio > 1.5:
                liquidity = "HIGH"
            elif liquidity_ratio > 0.8:
                liquidity = "NORMAL"
            else:
                liquidity = "LOW"
            
            # Calculate volatility
            if len(recent_data) > 1:
                returns = []
                for i in range(1, len(recent_data)):
                    prev_close = float(recent_data[i-1]["close"])
                    curr_close = float(recent_data[i]["close"])
                    returns.append((curr_close - prev_close) / prev_close)
                
                volatility = np.std(returns) if returns else 0
            else:
                volatility = 0
            
            return {
                "current_price": quote_response.get("price"),
                "volume": current_volume,
                "avg_volume": avg_volume,
                "liquidity": liquidity,
                "volatility": volatility,
                "spread_estimate": 0.001  # Simplified spread estimate
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {e}")
            return {
                "liquidity": "UNKNOWN",
                "volatility": 0,
                "spread_estimate": 0.002
            }
    
    async def _determine_execution_strategy(
        self,
        trade_proposal: Dict[str, Any],
        market_conditions: Dict[str, Any],
        preferred_strategy: str
    ) -> Dict[str, Any]:
        """Determine optimal execution strategy using AI."""
        
        context = {
            "trade_proposal": trade_proposal,
            "market_conditions": market_conditions,
            "preferred_strategy": preferred_strategy
        }
        
        prompt = f"""
        As a trade execution expert, determine the optimal execution strategy for this trade:
        
        Context: {json.dumps(context, indent=2)}
        
        Available strategies:
        - MARKET: Immediate execution at market price
        - LIMIT: Execute at specified price or better
        - TWAP: Time-weighted average price over time
        - VWAP: Volume-weighted average price
        - ICEBERG: Hide order size, execute in small chunks
        - SMART: Adaptive strategy based on conditions
        
        Provide recommendation in JSON format:
        {{
            "strategy": "MARKET|LIMIT|TWAP|VWAP|ICEBERG|SMART",
            "reasoning": "explanation of choice",
            "parameters": {{
                "price": number,
                "time_horizon": minutes,
                "slice_size": percentage,
                "urgency": "low|medium|high"
            }},
            "expected_impact": percentage,
            "confidence": 0.0-1.0
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert algorithmic trading execution specialist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            
            ai_response = response.choices[0].message.content
            
            # Extract JSON
            try:
                start_idx = ai_response.find('{')
                end_idx = ai_response.rfind('}') + 1
                json_str = ai_response[start_idx:end_idx]
                
                strategy_recommendation = json.loads(json_str)
                strategy_recommendation["strategy"] = ExecutionStrategy(strategy_recommendation["strategy"])
                
                return strategy_recommendation
                
            except (json.JSONDecodeError, ValueError):
                # Fallback to simple strategy
                return self._fallback_strategy(trade_proposal, market_conditions)
                
        except Exception as e:
            self.logger.error(f"Error determining execution strategy: {e}")
            return self._fallback_strategy(trade_proposal, market_conditions)
    
    def _fallback_strategy(self, trade_proposal: Dict[str, Any], market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback execution strategy when AI fails."""
        order_value = trade_proposal["quantity"] * trade_proposal.get("entry_price", 0)
        
        # Simple rules-based strategy
        if order_value < 10000 and market_conditions.get("liquidity") == "HIGH":
            strategy = ExecutionStrategy.MARKET
        elif market_conditions.get("volatility", 0) > 0.02:
            strategy = ExecutionStrategy.LIMIT
        else:
            strategy = ExecutionStrategy.SMART
        
        return {
            "strategy": strategy,
            "reasoning": "Fallback strategy based on simple rules",
            "parameters": {
                "urgency": "medium"
            },
            "confidence": 0.6
        }
    
    async def _execute_market_order(self, trade_proposal: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute market order."""
        if self.paper_trading:
            return await self._simulate_market_order(trade_proposal)
        
        try:
            response = await self.call_mcp_server(
                "trade_execution",
                "place_order",
                {
                    "symbol": trade_proposal["symbol"],
                    "side": trade_proposal["side"],
                    "quantity": trade_proposal["quantity"],
                    "order_type": "MARKET",
                    "portfolio_id": trade_proposal["portfolio_id"]
                }
            )
            
            return {
                "status": "SUBMITTED",
                "order_id": response.get("order_id"),
                "strategy": "MARKET",
                "execution_details": response
            }
            
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}
    
    async def _execute_limit_order(self, trade_proposal: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute limit order."""
        if self.paper_trading:
            return await self._simulate_limit_order(trade_proposal, strategy)
        
        try:
            limit_price = strategy["parameters"].get("price", trade_proposal.get("entry_price"))
            
            response = await self.call_mcp_server(
                "trade_execution",
                "place_order",
                {
                    "symbol": trade_proposal["symbol"],
                    "side": trade_proposal["side"],
                    "quantity": trade_proposal["quantity"],
                    "order_type": "LIMIT",
                    "price": limit_price,
                    "portfolio_id": trade_proposal["portfolio_id"]
                }
            )
            
            return {
                "status": "SUBMITTED",
                "order_id": response.get("order_id"),
                "strategy": "LIMIT",
                "limit_price": limit_price,
                "execution_details": response
            }
            
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}
    
    async def _execute_smart_order(self, trade_proposal: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute smart order with adaptive logic."""
        # For now, default to limit order
        return await self._execute_limit_order(trade_proposal, strategy)
    
    async def _execute_twap_order(self, trade_proposal: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute TWAP (Time Weighted Average Price) order."""
        # Implementation for TWAP execution
        return {"status": "NOT_IMPLEMENTED", "strategy": "TWAP"}
    
    async def _execute_vwap_order(self, trade_proposal: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute VWAP (Volume Weighted Average Price) order."""
        # Implementation for VWAP execution
        return {"status": "NOT_IMPLEMENTED", "strategy": "VWAP"}
    
    async def _execute_iceberg_order(self, trade_proposal: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute iceberg order."""
        # Implementation for iceberg execution
        return {"status": "NOT_IMPLEMENTED", "strategy": "ICEBERG"}
    
    async def _simulate_market_order(self, trade_proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate market order execution for paper trading."""
        # Get current market price
        try:
            quote_response = await self.call_mcp_server(
                "market_data",
                "get_current_quote",
                {"symbol": trade_proposal["symbol"]}
            )
            
            execution_price = quote_response.get("price", trade_proposal.get("entry_price", 0))
            
            # Simulate slippage
            slippage = 0.001  # 0.1%
            if trade_proposal["side"] == "BUY":
                execution_price *= (1 + slippage)
            else:
                execution_price *= (1 - slippage)
            
            return {
                "status": "FILLED",
                "order_id": f"PAPER_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "strategy": "MARKET",
                "execution_price": execution_price,
                "quantity": trade_proposal["quantity"],
                "paper_trading": True
            }
            
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}
    
    async def _simulate_limit_order(self, trade_proposal: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate limit order execution for paper trading."""
        limit_price = strategy["parameters"].get("price", trade_proposal.get("entry_price"))
        
        return {
            "status": "SUBMITTED",
            "order_id": f"PAPER_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "strategy": "LIMIT",
            "limit_price": limit_price,
            "quantity": trade_proposal["quantity"],
            "paper_trading": True
        }
    
    async def _log_execution_result(self, trade_proposal: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Log execution result to database."""
        try:
            # Store execution log
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trade_proposal": trade_proposal,
                "execution_result": result,
                "agent": self.agent_name
            }
            
            # In a real implementation, you would store this in a dedicated execution log table
            self.logger.info(f"Execution logged: {json.dumps(log_entry, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Failed to log execution result: {e}")
    
    async def _optimize_execution(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize execution for a large order."""
        # Implementation for execution optimization
        pass
    
    async def _monitor_orders(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor active orders."""
        # Implementation for order monitoring
        pass
    
    async def _cancel_order(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel a specific order."""
        # Implementation for order cancellation
        pass
    
    async def _cancel_pending_orders(self) -> None:
        """Cancel all pending orders during shutdown."""
        # Implementation for bulk order cancellation
        pass


async def main():
    """Main function to run the Trade Execution Agent."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start agent
    agent = TradeExecutionAgent()
    
    try:
        await agent.start()
        
        # Keep the agent running
        while agent.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
