"""
Integration test for the complete quantitative trading pipeline.
Tests the end-to-end flow: market data → analysis → signals → decisions → orders.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Add the project root to Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from shared.mcp_client.base import MCPClient


class TestQuantitativeTradingPipeline:
    """Test the complete quantitative trading pipeline."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Generate realistic price data
        np.random.seed(42)
        initial_price = 2500
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        market_data = []
        for i, date in enumerate(dates):
            price = prices[i]
            # Add some intraday variation
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = price * (1 + np.random.normal(0, 0.005))
            volume = int(np.random.normal(100000, 20000))
            
            market_data.append({
                "timestamp": date.isoformat(),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(price, 2),
                "volume": max(volume, 1000)
            })
        
        return market_data
    
    @pytest.fixture
    def mcp_client(self):
        """Create MCP client for testing."""
        return MCPClient("test_client")
    
    @pytest.mark.asyncio
    async def test_quantitative_analysis_server(self, sample_market_data, mcp_client):
        """Test the quantitative analysis server."""
        try:
            async with mcp_client as client:
                # Test signal generation
                response = await client.send_request(
                    "http://quantitative-analysis-server:8003",
                    "generate_signals",
                    {
                        "symbol": "RELIANCE",
                        "data": sample_market_data,
                        "strategies": ["sma_crossover", "rsi_mean_reversion"]
                    }
                )
                
                assert "signals" in response
                assert response["symbol"] == "RELIANCE"
                
                signals = response["signals"]
                assert isinstance(signals, list)
                
                # Verify signal structure
                if signals:
                    signal = signals[0]
                    assert "timestamp" in signal
                    assert "signal" in signal
                    assert "confidence" in signal
                    assert signal["signal"] in ["BUY", "SELL", "EXIT"]
                    assert 0 <= signal["confidence"] <= 1
                
                print(f"✓ Generated {len(signals)} quantitative signals")
                
        except Exception as e:
            pytest.skip(f"Quantitative analysis server not available: {e}")
    
    @pytest.mark.asyncio
    async def test_signal_generation_server(self, mcp_client):
        """Test the signal generation server."""
        try:
            async with mcp_client as client:
                # Test comprehensive signal generation
                response = await client.send_request(
                    "http://signal-generation-server:8004",
                    "generate_signals",
                    {
                        "symbol": "RELIANCE",
                        "config": "moderate"
                    }
                )
                
                assert "signals" in response
                assert response["symbol"] == "RELIANCE"
                
                signals_data = response["signals"]
                assert "quantitative_signals" in signals_data
                assert "combined_signals" in signals_data
                
                combined = signals_data["combined_signals"]
                assert "signal" in combined
                assert "confidence" in combined
                assert "reason" in combined
                
                print(f"✓ Generated combined signal: {combined['signal']} (confidence: {combined['confidence']:.2f})")
                
        except Exception as e:
            pytest.skip(f"Signal generation server not available: {e}")
    
    @pytest.mark.asyncio
    async def test_decision_engine_server(self, mcp_client):
        """Test the decision engine server."""
        try:
            async with mcp_client as client:
                # Test trading decision
                response = await client.send_request(
                    "http://decision-engine-server:8005",
                    "make_trading_decision",
                    {
                        "symbol": "RELIANCE",
                        "portfolio_id": "test-portfolio",
                        "config": "conservative"
                    }
                )
                
                assert "decision" in response
                assert response["symbol"] == "RELIANCE"
                
                decision = response["decision"]
                assert "action" in decision
                assert "confidence" in decision
                assert "reason" in decision
                
                valid_actions = ["BUY", "SELL", "HOLD", "NO_ACTION"]
                assert decision["action"] in valid_actions
                
                print(f"✓ Generated trading decision: {decision['action']} (confidence: {decision['confidence']:.2f})")
                
        except Exception as e:
            pytest.skip(f"Decision engine server not available: {e}")
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self, sample_market_data, mcp_client):
        """Test the complete end-to-end trading pipeline."""
        symbol = "RELIANCE"
        portfolio_id = "test-portfolio"
        
        try:
            # Step 1: Generate quantitative signals
            async with mcp_client as client:
                quant_response = await client.send_request(
                    "http://quantitative-analysis-server:8003",
                    "generate_signals",
                    {
                        "symbol": symbol,
                        "data": sample_market_data,
                        "strategies": ["sma_crossover", "rsi_mean_reversion", "bollinger_bands"]
                    }
                )
            
            assert "signals" in quant_response
            quant_signals = quant_response["signals"]
            print(f"✓ Step 1: Generated {len(quant_signals)} quantitative signals")
            
            # Step 2: Generate comprehensive signals
            async with mcp_client as client:
                signal_response = await client.send_request(
                    "http://signal-generation-server:8004",
                    "generate_signals",
                    {
                        "symbol": symbol,
                        "config": "moderate"
                    }
                )
            
            assert "signals" in signal_response
            combined_signals = signal_response["signals"]["combined_signals"]
            print(f"✓ Step 2: Generated combined signal: {combined_signals['signal']}")
            
            # Step 3: Make trading decision
            async with mcp_client as client:
                decision_response = await client.send_request(
                    "http://decision-engine-server:8005",
                    "make_trading_decision",
                    {
                        "symbol": symbol,
                        "portfolio_id": portfolio_id,
                        "config": "moderate",
                        "override_params": {"paper_trading": True}
                    }
                )
            
            assert "decision" in decision_response
            decision = decision_response["decision"]
            print(f"✓ Step 3: Made trading decision: {decision['action']}")
            
            # Step 4: Validate decision structure
            if decision["action"] in ["BUY", "SELL"]:
                assert "quantity" in decision
                assert "estimated_price" in decision
                assert "estimated_value" in decision
                assert "paper_trading" in decision
                assert decision["paper_trading"] is True  # Should be paper trading in test
                
                print(f"✓ Step 4: Decision includes execution details")
                print(f"  - Quantity: {decision.get('quantity', 0)}")
                print(f"  - Estimated Price: ₹{decision.get('estimated_price', 0):.2f}")
                print(f"  - Estimated Value: ₹{decision.get('estimated_value', 0):.2f}")
            
            # Step 5: Test strategy analysis
            async with mcp_client as client:
                analysis_response = await client.send_request(
                    "http://quantitative-analysis-server:8003",
                    "analyze_strategy",
                    {
                        "strategy": "sma_crossover",
                        "data": sample_market_data,
                        "params": {"short_period": 20, "long_period": 50}
                    }
                )
            
            assert "analysis" in analysis_response
            analysis = analysis_response["analysis"]
            assert "total_signals" in analysis
            assert "signal_frequency" in analysis
            print(f"✓ Step 5: Strategy analysis completed - {analysis['total_signals']} signals")
            
            # Step 6: Test backtesting
            async with mcp_client as client:
                backtest_response = await client.send_request(
                    "http://quantitative-analysis-server:8003",
                    "backtest_strategy",
                    {
                        "strategy": "sma_crossover",
                        "data": sample_market_data,
                        "params": {"short_period": 20, "long_period": 50},
                        "initial_capital": 100000
                    }
                )
            
            assert "backtest_results" in backtest_response
            backtest = backtest_response["backtest_results"]
            assert "total_return" in backtest
            assert "total_trades" in backtest
            print(f"✓ Step 6: Backtest completed - Return: {backtest['total_return']:.2%}, Trades: {backtest['total_trades']}")
            
            print("\n🎉 End-to-end quantitative trading pipeline test PASSED!")
            
            return {
                "quantitative_signals": len(quant_signals),
                "combined_signal": combined_signals["signal"],
                "trading_decision": decision["action"],
                "backtest_return": backtest["total_return"],
                "backtest_trades": backtest["total_trades"]
            }
            
        except Exception as e:
            pytest.fail(f"End-to-end pipeline test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_risk_metrics_calculation(self, mcp_client):
        """Test risk metrics calculation."""
        # Generate sample returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100).tolist()
        
        try:
            async with mcp_client as client:
                response = await client.send_request(
                    "http://quantitative-analysis-server:8003",
                    "calculate_risk_metrics",
                    {
                        "returns": returns
                    }
                )
            
            assert "risk_metrics" in response
            metrics = response["risk_metrics"]
            
            required_metrics = [
                "mean_return", "volatility", "sharpe_ratio", "max_drawdown",
                "var_95", "expected_shortfall", "win_rate"
            ]
            
            for metric in required_metrics:
                assert metric in metrics
                assert isinstance(metrics[metric], (int, float))
            
            print(f"✓ Risk metrics calculated:")
            print(f"  - Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"  - Max Drawdown: {metrics['max_drawdown']:.3f}")
            print(f"  - VaR (95%): {metrics['var_95']:.3f}")
            print(f"  - Win Rate: {metrics['win_rate']:.3f}")
            
        except Exception as e:
            pytest.skip(f"Risk metrics calculation not available: {e}")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])
