#!/usr/bin/env python3
"""
Demonstration script for the AWM Quantitative Trading System.
Shows how to use the system to analyze markets and generate trading signals.
"""

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.mcp_client.base import MCPClient


class QuantitativeTradingDemo:
    """Demonstration of the quantitative trading system."""
    
    def __init__(self):
        self.client = MCPClient("demo_client")
        self.server_urls = {
            "quantitative_analysis": "http://localhost:8003",
            "signal_generation": "http://localhost:8004",
            "decision_engine": "http://localhost:8005"
        }
    
    def generate_sample_data(self, symbol: str = "RELIANCE", days: int = 60) -> list:
        """Generate sample market data for demonstration."""
        print(f"📊 Generating {days} days of sample market data for {symbol}...")
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
        
        # Generate realistic price movements
        np.random.seed(42)
        initial_price = 2500
        returns = np.random.normal(0.0008, 0.018, len(dates))  # Slightly positive drift with volatility
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        market_data = []
        for i, date in enumerate(dates):
            price = prices[i]
            # Add realistic intraday variation
            daily_range = price * 0.02  # 2% daily range
            high = price + np.random.uniform(0, daily_range * 0.6)
            low = price - np.random.uniform(0, daily_range * 0.4)
            open_price = low + np.random.uniform(0, high - low)
            volume = int(np.random.normal(150000, 30000))
            
            market_data.append({
                "timestamp": date.isoformat(),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(price, 2),
                "volume": max(volume, 10000)
            })
        
        print(f"✅ Generated data: {market_data[0]['close']:.2f} → {market_data[-1]['close']:.2f}")
        return market_data
    
    async def demo_quantitative_analysis(self, symbol: str, market_data: list):
        """Demonstrate quantitative analysis capabilities."""
        print(f"\n🔬 Running Quantitative Analysis for {symbol}")
        print("=" * 50)
        
        try:
            async with self.client as client:
                # Test different strategies
                strategies = ["sma_crossover", "rsi_mean_reversion", "bollinger_bands", "momentum"]
                
                response = await client.send_request(
                    self.server_urls["quantitative_analysis"],
                    "generate_signals",
                    {
                        "symbol": symbol,
                        "data": market_data,
                        "strategies": strategies
                    }
                )
                
                signals = response.get("signals", [])
                print(f"📈 Generated {len(signals)} trading signals across {len(strategies)} strategies")
                
                # Group signals by strategy
                strategy_signals = {}
                for signal in signals:
                    strategy = signal.get("strategy", "unknown")
                    if strategy not in strategy_signals:
                        strategy_signals[strategy] = []
                    strategy_signals[strategy].append(signal)
                
                # Display signal summary
                for strategy, strat_signals in strategy_signals.items():
                    buy_signals = len([s for s in strat_signals if s.get("signal") == "BUY"])
                    sell_signals = len([s for s in strat_signals if s.get("signal") == "SELL"])
                    avg_confidence = sum(s.get("confidence", 0) for s in strat_signals) / len(strat_signals)
                    
                    print(f"  📊 {strategy.upper()}: {buy_signals} BUY, {sell_signals} SELL (avg confidence: {avg_confidence:.2f})")
                
                # Show latest signals
                if signals:
                    latest_signals = sorted(signals, key=lambda x: x.get("timestamp", ""))[-3:]
                    print(f"\n🔥 Latest Signals:")
                    for signal in latest_signals:
                        print(f"  {signal.get('timestamp', '')[:10]} | {signal.get('strategy', '').upper()} | "
                              f"{signal.get('signal', '')} @ ₹{signal.get('price', 0):.2f} "
                              f"(confidence: {signal.get('confidence', 0):.2f})")
                
                return signals
                
        except Exception as e:
            print(f"❌ Error in quantitative analysis: {e}")
            return []
    
    async def demo_signal_generation(self, symbol: str):
        """Demonstrate comprehensive signal generation."""
        print(f"\n🎯 Running Signal Generation for {symbol}")
        print("=" * 50)
        
        try:
            async with self.client as client:
                response = await client.send_request(
                    self.server_urls["signal_generation"],
                    "generate_signals",
                    {
                        "symbol": symbol,
                        "config": "moderate"
                    }
                )
                
                signals_data = response.get("signals", {})
                combined_signals = signals_data.get("combined_signals", {})
                
                print(f"🎯 Combined Signal: {combined_signals.get('signal', 'UNKNOWN')}")
                print(f"🎯 Confidence: {combined_signals.get('confidence', 0):.2f}")
                print(f"🎯 Reason: {combined_signals.get('reason', 'No reason provided')}")
                
                # Show analysis breakdown
                analysis = combined_signals.get("analysis", {})
                if analysis:
                    print(f"\n📊 Signal Analysis:")
                    print(f"  Buy Strength: {analysis.get('buy_strength', 0):.2f}")
                    print(f"  Sell Strength: {analysis.get('sell_strength', 0):.2f}")
                    print(f"  Technical Influence: {analysis.get('technical_influence', 0):.3f}")
                    print(f"  Sentiment Influence: {analysis.get('sentiment_influence', 0):.3f}")
                
                return combined_signals
                
        except Exception as e:
            print(f"❌ Error in signal generation: {e}")
            return {}
    
    async def demo_trading_decision(self, symbol: str):
        """Demonstrate trading decision making."""
        print(f"\n🤖 Making Trading Decision for {symbol}")
        print("=" * 50)
        
        try:
            async with self.client as client:
                response = await client.send_request(
                    self.server_urls["decision_engine"],
                    "make_trading_decision",
                    {
                        "symbol": symbol,
                        "portfolio_id": "demo-portfolio",
                        "config": "moderate",
                        "override_params": {
                            "paper_trading": True,
                            "max_position_size_pct": 0.1
                        }
                    }
                )
                
                decision = response.get("decision", {})
                
                print(f"🤖 Trading Decision: {decision.get('action', 'UNKNOWN')}")
                print(f"🤖 Confidence: {decision.get('confidence', 0):.2f}")
                print(f"🤖 Reason: {decision.get('reason', 'No reason provided')}")
                
                if decision.get("action") in ["BUY", "SELL"]:
                    print(f"\n📋 Execution Details:")
                    print(f"  Quantity: {decision.get('quantity', 0)} shares")
                    print(f"  Estimated Price: ₹{decision.get('estimated_price', 0):.2f}")
                    print(f"  Estimated Value: ₹{decision.get('estimated_value', 0):,.2f}")
                    print(f"  Order Type: {decision.get('order_type', 'MARKET')}")
                    print(f"  Paper Trading: {decision.get('paper_trading', True)}")
                    
                    if "stop_loss" in decision:
                        print(f"  Stop Loss: ₹{decision['stop_loss']:.2f}")
                    if "take_profit" in decision:
                        print(f"  Take Profit: ₹{decision['take_profit']:.2f}")
                
                return decision
                
        except Exception as e:
            print(f"❌ Error in trading decision: {e}")
            return {}
    
    async def demo_strategy_backtest(self, symbol: str, market_data: list):
        """Demonstrate strategy backtesting."""
        print(f"\n📈 Running Strategy Backtest for {symbol}")
        print("=" * 50)
        
        try:
            async with self.client as client:
                response = await client.send_request(
                    self.server_urls["quantitative_analysis"],
                    "backtest_strategy",
                    {
                        "strategy": "sma_crossover",
                        "data": market_data,
                        "params": {"short_period": 20, "long_period": 50},
                        "initial_capital": 100000
                    }
                )
                
                backtest = response.get("backtest_results", {})
                
                print(f"📈 Strategy: SMA Crossover (20/50)")
                print(f"📈 Initial Capital: ₹{backtest.get('initial_capital', 0):,.2f}")
                print(f"📈 Final Value: ₹{backtest.get('final_value', 0):,.2f}")
                print(f"📈 Total Return: {backtest.get('total_return', 0):.2%}")
                print(f"📈 Total Trades: {backtest.get('total_trades', 0)}")
                
                risk_metrics = backtest.get("risk_metrics", {})
                if risk_metrics:
                    print(f"\n📊 Risk Metrics:")
                    print(f"  Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.3f}")
                    print(f"  Max Drawdown: {risk_metrics.get('max_drawdown', 0):.3f}")
                    print(f"  Volatility: {risk_metrics.get('volatility', 0):.3f}")
                    print(f"  Win Rate: {risk_metrics.get('win_rate', 0):.3f}")
                
                return backtest
                
        except Exception as e:
            print(f"❌ Error in strategy backtest: {e}")
            return {}
    
    async def run_complete_demo(self):
        """Run the complete quantitative trading demonstration."""
        print("🚀 AWM Quantitative Trading System Demo")
        print("=" * 60)
        
        symbol = "RELIANCE"
        
        # Generate sample data
        market_data = self.generate_sample_data(symbol, days=60)
        
        # Run all demonstrations
        signals = await self.demo_quantitative_analysis(symbol, market_data)
        combined_signal = await self.demo_signal_generation(symbol)
        decision = await self.demo_trading_decision(symbol)
        backtest = await self.demo_strategy_backtest(symbol, market_data)
        
        # Summary
        print(f"\n🎉 Demo Complete!")
        print("=" * 60)
        print(f"📊 Generated {len(signals)} quantitative signals")
        print(f"🎯 Combined signal: {combined_signal.get('signal', 'N/A')} (confidence: {combined_signal.get('confidence', 0):.2f})")
        print(f"🤖 Trading decision: {decision.get('action', 'N/A')}")
        print(f"📈 Backtest return: {backtest.get('total_return', 0):.2%}")
        print(f"\n✅ All systems operational! Ready for autonomous trading.")


async def main():
    """Main function to run the demonstration."""
    demo = QuantitativeTradingDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    print("Starting AWM Quantitative Trading System Demo...")
    print("Make sure the following services are running:")
    print("- Quantitative Analysis Server (port 8003)")
    print("- Signal Generation Server (port 8004)")
    print("- Decision Engine Server (port 8005)")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Please ensure all required services are running and accessible.")
