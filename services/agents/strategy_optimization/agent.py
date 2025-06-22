"""
Strategy Optimization Agent for AWM system.
Backtests strategies and optimizes parameters using historical data.
"""

import asyncio
import logging
import json
import os
import itertools
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
import numpy as np
import openai

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from shared.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class StrategyOptimizationAgent(BaseAgent):
    """Agent responsible for strategy backtesting and parameter optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("strategy_optimization_agent", config)
        
        # OpenAI configuration
        self.openai_client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        
        # Optimization parameters
        self.max_iterations = self.config.get("max_iterations", 100)
        self.min_trades = self.config.get("min_trades", 10)
        self.optimization_metric = self.config.get("optimization_metric", "sharpe_ratio")
        
        # Backtesting parameters
        self.initial_capital = self.config.get("initial_capital", 100000)
        self.commission = self.config.get("commission", 0.001)  # 0.1%
        self.slippage = self.config.get("slippage", 0.0005)  # 0.05%
    
    async def initialize(self):
        """Initialize the Strategy Optimization Agent."""
        self.logger.info("Initializing Strategy Optimization Agent...")
        
        # Test connections to required MCP servers
        try:
            # Test market data server
            await self.call_mcp_server("market_data", "health", {})
            self.logger.info("✓ Market Data Server connection verified")
            
            # Test technical analysis server
            await self.call_mcp_server("technical_analysis", "health", {})
            self.logger.info("✓ Technical Analysis Server connection verified")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to required servers: {e}")
            raise
        
        self.logger.info("Strategy Optimization Agent initialized successfully")
    
    async def cleanup(self):
        """Cleanup the Strategy Optimization Agent."""
        self.logger.info("Cleaning up Strategy Optimization Agent...")
        # No specific cleanup needed
    
    async def process_task(self, task_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process a strategy optimization task."""
        
        if task_type == "backtest_strategy":
            return await self._backtest_strategy(parameters)
        elif task_type == "optimize_parameters":
            return await self._optimize_parameters(parameters)
        elif task_type == "compare_strategies":
            return await self._compare_strategies(parameters)
        elif task_type == "generate_strategy":
            return await self._generate_strategy(parameters)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _backtest_strategy(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Backtest a trading strategy."""
        strategy_name = parameters["strategy"]
        symbol = parameters["symbol"]
        strategy_params = parameters.get("params", {})
        start_date = parameters.get("start_date")
        end_date = parameters.get("end_date")
        
        self.logger.info(f"Backtesting {strategy_name} for {symbol}")
        
        try:
            # Get historical data
            market_data = await self._get_historical_data(symbol, start_date, end_date)
            
            if not market_data:
                raise ValueError("No market data available for backtesting")
            
            # Run backtest based on strategy type
            if strategy_name == "sma_crossover":
                results = await self._backtest_sma_crossover(market_data, strategy_params)
            elif strategy_name == "rsi_mean_reversion":
                results = await self._backtest_rsi_mean_reversion(market_data, strategy_params)
            elif strategy_name == "bollinger_bands":
                results = await self._backtest_bollinger_bands(market_data, strategy_params)
            elif strategy_name == "macd_momentum":
                results = await self._backtest_macd_momentum(market_data, strategy_params)
            else:
                # Use technical analysis server for generic backtesting
                results = await self._backtest_generic_strategy(strategy_name, market_data, strategy_params)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(results)
            
            # Generate AI analysis of results
            ai_analysis = await self._analyze_backtest_results(
                strategy_name, symbol, strategy_params, performance_metrics
            )
            
            return {
                "strategy": strategy_name,
                "symbol": symbol,
                "parameters": strategy_params,
                "backtest_period": {
                    "start": start_date,
                    "end": end_date,
                    "days": len(market_data)
                },
                "performance_metrics": performance_metrics,
                "trade_details": results.get("trades", []),
                "ai_analysis": ai_analysis,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error backtesting {strategy_name}: {e}")
            raise
    
    async def _get_historical_data(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get historical market data for backtesting."""
        response = await self.call_mcp_server(
            "market_data",
            "get_price_history",
            {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "limit": 10000
            }
        )
        
        return response.get("data", [])
    
    async def _backtest_sma_crossover(self, market_data: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        """Backtest SMA crossover strategy."""
        fast_period = params.get("fast_period", 10)
        slow_period = params.get("slow_period", 20)
        
        # Calculate SMAs using technical analysis server
        fast_sma_response = await self.call_mcp_server(
            "technical_analysis",
            "calculate_indicator",
            {
                "indicator": "SMA",
                "data": market_data,
                "params": {"period": fast_period}
            }
        )
        
        slow_sma_response = await self.call_mcp_server(
            "technical_analysis",
            "calculate_indicator",
            {
                "indicator": "SMA",
                "data": market_data,
                "params": {"period": slow_period}
            }
        )
        
        fast_sma = {item["timestamp"]: item["value"] for item in fast_sma_response.get("data", [])}
        slow_sma = {item["timestamp"]: item["value"] for item in slow_sma_response.get("data", [])}
        
        # Generate signals and simulate trades
        trades = []
        position = 0
        entry_price = 0
        
        for i, data_point in enumerate(market_data):
            timestamp = data_point["timestamp"]
            price = float(data_point["close"])
            
            if timestamp not in fast_sma or timestamp not in slow_sma:
                continue
            
            fast_value = fast_sma[timestamp]
            slow_value = slow_sma[timestamp]
            
            # Generate signals
            if position == 0 and fast_value > slow_value:
                # Buy signal
                position = 1
                entry_price = price * (1 + self.slippage)  # Account for slippage
                trades.append({
                    "timestamp": timestamp,
                    "action": "BUY",
                    "price": entry_price,
                    "signal": "SMA_CROSSOVER_UP"
                })
            elif position == 1 and fast_value < slow_value:
                # Sell signal
                exit_price = price * (1 - self.slippage)
                pnl = (exit_price - entry_price) / entry_price - self.commission
                
                trades.append({
                    "timestamp": timestamp,
                    "action": "SELL",
                    "price": exit_price,
                    "pnl": pnl,
                    "signal": "SMA_CROSSOVER_DOWN"
                })
                position = 0
        
        return {"trades": trades, "final_position": position}
    
    async def _backtest_rsi_mean_reversion(self, market_data: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        """Backtest RSI mean reversion strategy."""
        rsi_period = params.get("rsi_period", 14)
        oversold_threshold = params.get("oversold", 30)
        overbought_threshold = params.get("overbought", 70)
        
        # Calculate RSI
        rsi_response = await self.call_mcp_server(
            "technical_analysis",
            "calculate_indicator",
            {
                "indicator": "RSI",
                "data": market_data,
                "params": {"period": rsi_period}
            }
        )
        
        rsi_data = {item["timestamp"]: item["value"] for item in rsi_response.get("data", [])}
        
        # Generate signals and simulate trades
        trades = []
        position = 0
        entry_price = 0
        
        for data_point in market_data:
            timestamp = data_point["timestamp"]
            price = float(data_point["close"])
            
            if timestamp not in rsi_data:
                continue
            
            rsi_value = rsi_data[timestamp]
            
            # Generate signals
            if position == 0 and rsi_value < oversold_threshold:
                # Buy signal (oversold)
                position = 1
                entry_price = price * (1 + self.slippage)
                trades.append({
                    "timestamp": timestamp,
                    "action": "BUY",
                    "price": entry_price,
                    "signal": "RSI_OVERSOLD",
                    "rsi": rsi_value
                })
            elif position == 1 and rsi_value > overbought_threshold:
                # Sell signal (overbought)
                exit_price = price * (1 - self.slippage)
                pnl = (exit_price - entry_price) / entry_price - self.commission
                
                trades.append({
                    "timestamp": timestamp,
                    "action": "SELL",
                    "price": exit_price,
                    "pnl": pnl,
                    "signal": "RSI_OVERBOUGHT",
                    "rsi": rsi_value
                })
                position = 0
        
        return {"trades": trades, "final_position": position}
    
    async def _backtest_bollinger_bands(self, market_data: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        """Backtest Bollinger Bands strategy."""
        period = params.get("period", 20)
        std_dev = params.get("std_dev", 2)
        
        # Calculate Bollinger Bands
        bb_response = await self.call_mcp_server(
            "technical_analysis",
            "calculate_indicator",
            {
                "indicator": "BOLLINGER_BANDS",
                "data": market_data,
                "params": {"period": period, "std_dev": std_dev}
            }
        )
        
        bb_data = {item["timestamp"]: item for item in bb_response.get("data", [])}
        
        # Generate signals and simulate trades
        trades = []
        position = 0
        entry_price = 0
        
        for data_point in market_data:
            timestamp = data_point["timestamp"]
            price = float(data_point["close"])
            
            if timestamp not in bb_data:
                continue
            
            bb = bb_data[timestamp]
            lower_band = bb["lower"]
            upper_band = bb["upper"]
            
            # Generate signals
            if position == 0 and price < lower_band:
                # Buy signal (price below lower band)
                position = 1
                entry_price = price * (1 + self.slippage)
                trades.append({
                    "timestamp": timestamp,
                    "action": "BUY",
                    "price": entry_price,
                    "signal": "BB_LOWER_BREACH"
                })
            elif position == 1 and price > upper_band:
                # Sell signal (price above upper band)
                exit_price = price * (1 - self.slippage)
                pnl = (exit_price - entry_price) / entry_price - self.commission
                
                trades.append({
                    "timestamp": timestamp,
                    "action": "SELL",
                    "price": exit_price,
                    "pnl": pnl,
                    "signal": "BB_UPPER_BREACH"
                })
                position = 0
        
        return {"trades": trades, "final_position": position}
    
    async def _backtest_macd_momentum(self, market_data: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        """Backtest MACD momentum strategy."""
        fast = params.get("fast", 12)
        slow = params.get("slow", 26)
        signal = params.get("signal", 9)
        
        # Calculate MACD
        macd_response = await self.call_mcp_server(
            "technical_analysis",
            "calculate_indicator",
            {
                "indicator": "MACD",
                "data": market_data,
                "params": {"fast": fast, "slow": slow, "signal": signal}
            }
        )
        
        macd_data = {item["timestamp"]: item for item in macd_response.get("data", [])}
        
        # Generate signals and simulate trades
        trades = []
        position = 0
        entry_price = 0
        prev_macd = None
        prev_signal = None
        
        for data_point in market_data:
            timestamp = data_point["timestamp"]
            price = float(data_point["close"])
            
            if timestamp not in macd_data:
                continue
            
            macd_item = macd_data[timestamp]
            macd_value = macd_item.get("macd")
            signal_value = macd_item.get("signal")
            
            if prev_macd is None or prev_signal is None:
                prev_macd = macd_value
                prev_signal = signal_value
                continue
            
            # Generate signals based on MACD crossover
            if position == 0 and prev_macd <= prev_signal and macd_value > signal_value:
                # Buy signal (MACD crosses above signal)
                position = 1
                entry_price = price * (1 + self.slippage)
                trades.append({
                    "timestamp": timestamp,
                    "action": "BUY",
                    "price": entry_price,
                    "signal": "MACD_BULLISH_CROSSOVER"
                })
            elif position == 1 and prev_macd >= prev_signal and macd_value < signal_value:
                # Sell signal (MACD crosses below signal)
                exit_price = price * (1 - self.slippage)
                pnl = (exit_price - entry_price) / entry_price - self.commission
                
                trades.append({
                    "timestamp": timestamp,
                    "action": "SELL",
                    "price": exit_price,
                    "pnl": pnl,
                    "signal": "MACD_BEARISH_CROSSOVER"
                })
                position = 0
            
            prev_macd = macd_value
            prev_signal = signal_value
        
        return {"trades": trades, "final_position": position}
    
    async def _backtest_generic_strategy(self, strategy_name: str, market_data: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        """Backtest using technical analysis server."""
        response = await self.call_mcp_server(
            "technical_analysis",
            "run_backtest",
            {
                "strategy": strategy_name,
                "data": market_data,
                "params": params
            }
        )
        
        return response
    
    def _calculate_performance_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        trades = backtest_results.get("trades", [])
        
        if not trades:
            return {"error": "No trades executed"}
        
        # Extract PnL data
        pnls = [trade.get("pnl", 0) for trade in trades if "pnl" in trade]
        
        if not pnls:
            return {"error": "No PnL data available"}
        
        # Calculate basic metrics
        total_trades = len(pnls)
        winning_trades = len([pnl for pnl in pnls if pnl > 0])
        losing_trades = len([pnl for pnl in pnls if pnl < 0])
        
        total_return = sum(pnls)
        avg_return = np.mean(pnls)
        
        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum([pnl for pnl in pnls if pnl > 0])
        gross_loss = abs(sum([pnl for pnl in pnls if pnl < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe ratio (simplified)
        if len(pnls) > 1:
            volatility = np.std(pnls)
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        cumulative_returns = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": float(win_rate),
            "total_return": float(total_return),
            "average_return": float(avg_return),
            "profit_factor": float(profit_factor),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "gross_profit": float(gross_profit),
            "gross_loss": float(gross_loss)
        }
    
    async def _analyze_backtest_results(
        self,
        strategy_name: str,
        symbol: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate AI analysis of backtest results."""
        
        prompt = f"""
        Analyze the following backtest results for the {strategy_name} strategy on {symbol}:
        
        Strategy Parameters: {json.dumps(params, indent=2)}
        Performance Metrics: {json.dumps(metrics, indent=2)}
        
        Provide analysis in JSON format:
        {{
            "overall_assessment": "excellent" | "good" | "fair" | "poor",
            "strengths": ["list", "of", "strengths"],
            "weaknesses": ["list", "of", "weaknesses"],
            "optimization_suggestions": ["specific", "suggestions"],
            "risk_assessment": "low" | "medium" | "high",
            "recommended_action": "deploy" | "optimize" | "reject"
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert quantitative analyst specializing in trading strategy evaluation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            ai_response = response.choices[0].message.content
            
            # Extract JSON
            try:
                start_idx = ai_response.find('{')
                end_idx = ai_response.rfind('}') + 1
                json_str = ai_response[start_idx:end_idx]
                
                return json.loads(json_str)
                
            except (json.JSONDecodeError, ValueError):
                return {"error": "Failed to parse AI analysis"}
                
        except Exception as e:
            self.logger.error(f"Error generating AI analysis: {e}")
            return {"error": str(e)}
    
    async def _optimize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize strategy parameters using grid search."""
        # Implementation for parameter optimization
        pass
    
    async def _compare_strategies(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Compare multiple strategies."""
        # Implementation for strategy comparison
        pass
    
    async def _generate_strategy(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate new strategy using AI."""
        # Implementation for AI-generated strategies
        pass


async def main():
    """Main function to run the Strategy Optimization Agent."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start agent
    agent = StrategyOptimizationAgent()
    
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
