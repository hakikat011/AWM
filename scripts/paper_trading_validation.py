"""
Paper trading validation script for A/B testing LLM-enhanced vs quantitative-only trading.
"""

import asyncio
import json
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
import statistics
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.mcp_client.base import MCPClient

logger = logging.getLogger(__name__)


class PaperTradingValidator:
    """Validates LLM-enhanced trading system through paper trading."""
    
    def __init__(self):
        self.server_urls = {
            "signal_generation": "http://localhost:8004",
            "decision_engine": "http://localhost:8005",
            "llm_market_intelligence": "http://localhost:8007",
            "market_data": "http://localhost:8001"
        }
        
        self.test_symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR"]
        self.initial_capital = 1000000  # 10 Lakh INR
        
        # Track performance for both systems
        self.llm_enhanced_portfolio = {
            "cash": self.initial_capital,
            "positions": {},
            "trades": [],
            "daily_pnl": []
        }
        
        self.quantitative_only_portfolio = {
            "cash": self.initial_capital,
            "positions": {},
            "trades": [],
            "daily_pnl": []
        }
    
    async def run_validation(self, days: int = 30) -> Dict[str, Any]:
        """Run paper trading validation for specified number of days."""
        logger.info(f"Starting {days}-day paper trading validation")
        
        results = {
            "validation_period": days,
            "start_date": datetime.now(timezone.utc).isoformat(),
            "llm_enhanced_results": {},
            "quantitative_only_results": {},
            "comparison_metrics": {}
        }
        
        try:
            # Run parallel paper trading simulations
            for day in range(days):
                logger.info(f"Processing day {day + 1}/{days}")
                
                # Get market data for all symbols
                market_data = await self._get_market_data_for_day(day)
                
                # Run LLM-enhanced trading
                await self._run_llm_enhanced_trading(market_data, day)
                
                # Run quantitative-only trading
                await self._run_quantitative_only_trading(market_data, day)
                
                # Calculate daily P&L
                self._calculate_daily_pnl(day)
                
                # Add small delay to avoid overwhelming servers
                await asyncio.sleep(0.1)
            
            # Calculate final results
            results["llm_enhanced_results"] = self._calculate_portfolio_metrics(self.llm_enhanced_portfolio)
            results["quantitative_only_results"] = self._calculate_portfolio_metrics(self.quantitative_only_portfolio)
            results["comparison_metrics"] = self._compare_strategies(results)
            
            # Save results
            await self._save_results(results)
            
            logger.info("Paper trading validation completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            raise
    
    async def _get_market_data_for_day(self, day: int) -> Dict[str, Any]:
        """Get market data for all symbols for a specific day."""
        market_data = {}
        
        async with MCPClient("paper_trading_validator") as client:
            for symbol in self.test_symbols:
                try:
                    response = await client.send_request(
                        self.server_urls["market_data"],
                        "get_historical_data",
                        {
                            "symbol": symbol,
                            "days": 1,
                            "offset_days": day
                        }
                    )
                    market_data[symbol] = response.get("data", [])
                except Exception as e:
                    logger.error(f"Error getting market data for {symbol}: {e}")
                    market_data[symbol] = []
        
        return market_data
    
    async def _run_llm_enhanced_trading(self, market_data: Dict[str, Any], day: int):
        """Run LLM-enhanced trading simulation."""
        async with MCPClient("llm_enhanced_trader") as client:
            for symbol in self.test_symbols:
                try:
                    # Generate LLM-enhanced signals
                    signals_response = await client.send_request(
                        self.server_urls["signal_generation"],
                        "generate_signals",
                        {"symbol": symbol, "config": "moderate"}
                    )
                    
                    # Make LLM-enhanced decision
                    decision_response = await client.send_request(
                        self.server_urls["decision_engine"],
                        "make_trading_decision",
                        {
                            "symbol": symbol,
                            "portfolio_id": "llm_enhanced_test",
                            "config": "moderate"
                        }
                    )
                    
                    # Execute paper trade
                    decision = decision_response.get("decision", {})
                    await self._execute_paper_trade(
                        symbol, decision, self.llm_enhanced_portfolio, 
                        market_data.get(symbol, []), "LLM_ENHANCED", day
                    )
                    
                except Exception as e:
                    logger.error(f"Error in LLM-enhanced trading for {symbol}: {e}")
    
    async def _run_quantitative_only_trading(self, market_data: Dict[str, Any], day: int):
        """Run quantitative-only trading simulation."""
        async with MCPClient("quantitative_trader") as client:
            for symbol in self.test_symbols:
                try:
                    # Generate quantitative-only signals (disable LLM components)
                    signals_response = await client.send_request(
                        self.server_urls["signal_generation"],
                        "generate_signals",
                        {
                            "symbol": symbol, 
                            "config": "quantitative_only",
                            "disable_llm": True
                        }
                    )
                    
                    # Make quantitative-only decision
                    decision_response = await client.send_request(
                        self.server_urls["decision_engine"],
                        "make_trading_decision",
                        {
                            "symbol": symbol,
                            "portfolio_id": "quantitative_only_test",
                            "config": "conservative",
                            "override_params": {"disable_llm_insights": True}
                        }
                    )
                    
                    # Execute paper trade
                    decision = decision_response.get("decision", {})
                    await self._execute_paper_trade(
                        symbol, decision, self.quantitative_only_portfolio,
                        market_data.get(symbol, []), "QUANTITATIVE_ONLY", day
                    )
                    
                except Exception as e:
                    logger.error(f"Error in quantitative-only trading for {symbol}: {e}")
    
    async def _execute_paper_trade(self, symbol: str, decision: Dict[str, Any], 
                                 portfolio: Dict[str, Any], market_data: List[Dict[str, Any]], 
                                 strategy_type: str, day: int):
        """Execute a paper trade based on decision."""
        if not market_data:
            return
        
        action = decision.get("action", "NO_ACTION")
        quantity = decision.get("quantity", 0)
        current_price = market_data[-1].get("close", 0) if market_data else 0
        
        if action == "BUY" and quantity > 0 and current_price > 0:
            trade_value = quantity * current_price
            
            if portfolio["cash"] >= trade_value:
                # Execute buy
                portfolio["cash"] -= trade_value
                portfolio["positions"][symbol] = portfolio["positions"].get(symbol, 0) + quantity
                
                trade_record = {
                    "day": day,
                    "symbol": symbol,
                    "action": "BUY",
                    "quantity": quantity,
                    "price": current_price,
                    "value": trade_value,
                    "strategy": strategy_type,
                    "decision_confidence": decision.get("confidence", 0),
                    "llm_insights": decision.get("llm_insights", {}),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                portfolio["trades"].append(trade_record)
                
                logger.debug(f"{strategy_type}: BUY {quantity} {symbol} @ {current_price}")
        
        elif action == "SELL" and quantity > 0 and current_price > 0:
            current_position = portfolio["positions"].get(symbol, 0)
            sell_quantity = min(quantity, current_position)
            
            if sell_quantity > 0:
                # Execute sell
                trade_value = sell_quantity * current_price
                portfolio["cash"] += trade_value
                portfolio["positions"][symbol] = current_position - sell_quantity
                
                if portfolio["positions"][symbol] == 0:
                    del portfolio["positions"][symbol]
                
                trade_record = {
                    "day": day,
                    "symbol": symbol,
                    "action": "SELL",
                    "quantity": sell_quantity,
                    "price": current_price,
                    "value": trade_value,
                    "strategy": strategy_type,
                    "decision_confidence": decision.get("confidence", 0),
                    "llm_insights": decision.get("llm_insights", {}),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                portfolio["trades"].append(trade_record)
                
                logger.debug(f"{strategy_type}: SELL {sell_quantity} {symbol} @ {current_price}")
    
    def _calculate_daily_pnl(self, day: int):
        """Calculate daily P&L for both portfolios."""
        # This would calculate mark-to-market P&L based on current positions
        # For simplicity, we'll track the portfolio value changes
        
        llm_portfolio_value = self._calculate_portfolio_value(self.llm_enhanced_portfolio)
        quant_portfolio_value = self._calculate_portfolio_value(self.quantitative_only_portfolio)
        
        self.llm_enhanced_portfolio["daily_pnl"].append({
            "day": day,
            "portfolio_value": llm_portfolio_value,
            "cash": self.llm_enhanced_portfolio["cash"],
            "positions_value": llm_portfolio_value - self.llm_enhanced_portfolio["cash"]
        })
        
        self.quantitative_only_portfolio["daily_pnl"].append({
            "day": day,
            "portfolio_value": quant_portfolio_value,
            "cash": self.quantitative_only_portfolio["cash"],
            "positions_value": quant_portfolio_value - self.quantitative_only_portfolio["cash"]
        })
    
    def _calculate_portfolio_value(self, portfolio: Dict[str, Any]) -> float:
        """Calculate current portfolio value (simplified)."""
        # In real implementation, this would use current market prices
        # For simulation, we'll use a simplified calculation
        return portfolio["cash"] + sum(
            quantity * 100  # Simplified: assume each share worth 100
            for quantity in portfolio["positions"].values()
        )
    
    def _calculate_portfolio_metrics(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive portfolio performance metrics."""
        trades = portfolio["trades"]
        daily_pnl = portfolio["daily_pnl"]
        
        if not daily_pnl:
            return {"error": "No P&L data available"}
        
        # Calculate returns
        initial_value = self.initial_capital
        final_value = daily_pnl[-1]["portfolio_value"]
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(daily_pnl)):
            prev_value = daily_pnl[i-1]["portfolio_value"]
            curr_value = daily_pnl[i]["portfolio_value"]
            daily_return = (curr_value - prev_value) / prev_value
            daily_returns.append(daily_return)
        
        # Calculate metrics
        metrics = {
            "total_return": total_return,
            "final_portfolio_value": final_value,
            "total_trades": len(trades),
            "winning_trades": len([t for t in trades if t.get("pnl", 0) > 0]),
            "losing_trades": len([t for t in trades if t.get("pnl", 0) < 0]),
            "average_daily_return": statistics.mean(daily_returns) if daily_returns else 0,
            "volatility": statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0,
            "sharpe_ratio": 0,  # Would calculate with risk-free rate
            "max_drawdown": self._calculate_max_drawdown(daily_pnl),
            "win_rate": 0
        }
        
        if metrics["total_trades"] > 0:
            metrics["win_rate"] = metrics["winning_trades"] / metrics["total_trades"]
        
        if metrics["volatility"] > 0:
            metrics["sharpe_ratio"] = metrics["average_daily_return"] / metrics["volatility"]
        
        return metrics
    
    def _calculate_max_drawdown(self, daily_pnl: List[Dict[str, Any]]) -> float:
        """Calculate maximum drawdown."""
        if len(daily_pnl) < 2:
            return 0.0
        
        peak = daily_pnl[0]["portfolio_value"]
        max_drawdown = 0.0
        
        for pnl in daily_pnl[1:]:
            current_value = pnl["portfolio_value"]
            if current_value > peak:
                peak = current_value
            else:
                drawdown = (peak - current_value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _compare_strategies(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare LLM-enhanced vs quantitative-only strategies."""
        llm_metrics = results["llm_enhanced_results"]
        quant_metrics = results["quantitative_only_results"]
        
        if "error" in llm_metrics or "error" in quant_metrics:
            return {"error": "Cannot compare due to missing data"}
        
        comparison = {
            "return_improvement": llm_metrics["total_return"] - quant_metrics["total_return"],
            "sharpe_improvement": llm_metrics["sharpe_ratio"] - quant_metrics["sharpe_ratio"],
            "volatility_change": llm_metrics["volatility"] - quant_metrics["volatility"],
            "drawdown_improvement": quant_metrics["max_drawdown"] - llm_metrics["max_drawdown"],
            "win_rate_improvement": llm_metrics["win_rate"] - quant_metrics["win_rate"],
            "trade_count_difference": llm_metrics["total_trades"] - quant_metrics["total_trades"],
            "statistical_significance": "TBD"  # Would require proper statistical testing
        }
        
        # Determine overall performance
        improvements = 0
        if comparison["return_improvement"] > 0:
            improvements += 1
        if comparison["sharpe_improvement"] > 0:
            improvements += 1
        if comparison["drawdown_improvement"] > 0:
            improvements += 1
        if comparison["win_rate_improvement"] > 0:
            improvements += 1
        
        comparison["overall_assessment"] = "BETTER" if improvements >= 3 else "MIXED" if improvements >= 2 else "WORSE"
        
        return comparison
    
    async def _save_results(self, results: Dict[str, Any]):
        """Save validation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"paper_trading_results_{timestamp}.json"
        
        os.makedirs("test_results", exist_ok=True)
        filepath = os.path.join("test_results", filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")


async def main():
    """Main function to run paper trading validation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    validator = PaperTradingValidator()
    
    try:
        print("Starting Paper Trading Validation...")
        print("This will compare LLM-enhanced vs quantitative-only trading strategies")
        
        # Run validation for 30 days
        results = await validator.run_validation(days=30)
        
        # Print summary
        print("\n" + "="*60)
        print("PAPER TRADING VALIDATION RESULTS")
        print("="*60)
        
        llm_results = results["llm_enhanced_results"]
        quant_results = results["quantitative_only_results"]
        comparison = results["comparison_metrics"]
        
        print(f"\nLLM-Enhanced Strategy:")
        print(f"  Total Return: {llm_results.get('total_return', 0):.2%}")
        print(f"  Sharpe Ratio: {llm_results.get('sharpe_ratio', 0):.3f}")
        print(f"  Max Drawdown: {llm_results.get('max_drawdown', 0):.2%}")
        print(f"  Win Rate: {llm_results.get('win_rate', 0):.2%}")
        print(f"  Total Trades: {llm_results.get('total_trades', 0)}")
        
        print(f"\nQuantitative-Only Strategy:")
        print(f"  Total Return: {quant_results.get('total_return', 0):.2%}")
        print(f"  Sharpe Ratio: {quant_results.get('sharpe_ratio', 0):.3f}")
        print(f"  Max Drawdown: {quant_results.get('max_drawdown', 0):.2%}")
        print(f"  Win Rate: {quant_results.get('win_rate', 0):.2%}")
        print(f"  Total Trades: {quant_results.get('total_trades', 0)}")
        
        print(f"\nComparison:")
        print(f"  Return Improvement: {comparison.get('return_improvement', 0):.2%}")
        print(f"  Sharpe Improvement: {comparison.get('sharpe_improvement', 0):.3f}")
        print(f"  Drawdown Improvement: {comparison.get('drawdown_improvement', 0):.2%}")
        print(f"  Overall Assessment: {comparison.get('overall_assessment', 'UNKNOWN')}")
        
        print("\nValidation completed successfully!")
        
    except Exception as e:
        print(f"Error during validation: {e}")
        logger.error(f"Validation failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
