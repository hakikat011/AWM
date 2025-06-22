"""
Unit tests for quantitative trading strategies.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from services.mcp_servers.quantitative_analysis.server import QuantitativeAnalysisServer


class TestQuantitativeStrategies:
    """Test quantitative trading strategies."""
    
    @pytest.fixture
    def server(self):
        """Create quantitative analysis server instance."""
        return QuantitativeAnalysisServer()
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample market data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Generate trending price data for testing crossover
        np.random.seed(42)
        trend = np.linspace(2400, 2600, len(dates))  # Upward trend
        noise = np.random.normal(0, 20, len(dates))
        prices = trend + noise
        
        data = []
        for i, date in enumerate(dates):
            price = max(prices[i], 100)  # Ensure positive prices
            high = price * 1.02
            low = price * 0.98
            open_price = price * (1 + np.random.normal(0, 0.005))
            
            data.append({
                "timestamp": date.isoformat(),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(price, 2),
                "volume": int(np.random.normal(100000, 20000))
            })
        
        return data
    
    @pytest.fixture
    def sample_dataframe(self, server, sample_data):
        """Convert sample data to DataFrame."""
        return server._prepare_dataframe(sample_data)
    
    def test_prepare_dataframe(self, server, sample_data):
        """Test DataFrame preparation."""
        df = server._prepare_dataframe(sample_data)
        
        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_data)
        
        # Check columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in df.columns
        
        # Check index
        assert isinstance(df.index, pd.DatetimeIndex)
        
        # Check data types
        for col in ['open', 'high', 'low', 'close']:
            assert df[col].dtype in [np.float64, np.int64]
    
    @pytest.mark.asyncio
    async def test_sma_crossover_signals(self, server, sample_dataframe):
        """Test SMA crossover strategy."""
        signals = await server._sma_crossover_signals(sample_dataframe)
        
        # Should generate some signals
        assert isinstance(signals, list)
        
        # Check signal structure
        if signals:
            signal = signals[0]
            assert "timestamp" in signal
            assert "signal" in signal
            assert "price" in signal
            assert "confidence" in signal
            assert "reason" in signal
            assert "indicators" in signal
            
            # Check signal values
            assert signal["signal"] in ["BUY", "SELL"]
            assert 0 <= signal["confidence"] <= 1
            assert signal["price"] > 0
            assert "sma_short" in signal["indicators"]
            assert "sma_long" in signal["indicators"]
    
    @pytest.mark.asyncio
    async def test_rsi_mean_reversion_signals(self, server, sample_dataframe):
        """Test RSI mean reversion strategy."""
        signals = await server._rsi_mean_reversion_signals(sample_dataframe)
        
        assert isinstance(signals, list)
        
        if signals:
            signal = signals[0]
            assert "timestamp" in signal
            assert "signal" in signal
            assert "price" in signal
            assert "confidence" in signal
            assert "indicators" in signal
            
            # Check RSI-specific fields
            assert "rsi" in signal["indicators"]
            rsi_value = signal["indicators"]["rsi"]
            assert 0 <= rsi_value <= 100
    
    @pytest.mark.asyncio
    async def test_bollinger_bands_signals(self, server, sample_dataframe):
        """Test Bollinger Bands strategy."""
        signals = await server._bollinger_bands_signals(sample_dataframe)
        
        assert isinstance(signals, list)
        
        if signals:
            signal = signals[0]
            assert "indicators" in signal
            indicators = signal["indicators"]
            
            # Check Bollinger Bands indicators
            required_bb_fields = ["bb_upper", "bb_lower", "bb_middle", "bb_width"]
            for field in required_bb_fields:
                assert field in indicators
                assert indicators[field] > 0
            
            # Check logical relationships
            assert indicators["bb_upper"] > indicators["bb_middle"]
            assert indicators["bb_middle"] > indicators["bb_lower"]
    
    @pytest.mark.asyncio
    async def test_momentum_signals(self, server, sample_dataframe):
        """Test momentum strategy."""
        signals = await server._momentum_signals(sample_dataframe)
        
        assert isinstance(signals, list)
        
        if signals:
            signal = signals[0]
            assert "indicators" in signal
            indicators = signal["indicators"]
            
            # Check momentum indicators
            assert "momentum" in indicators
            assert "momentum_ma" in indicators
            
            # Momentum can be positive or negative
            assert isinstance(indicators["momentum"], (int, float))
            assert isinstance(indicators["momentum_ma"], (int, float))
    
    @pytest.mark.asyncio
    async def test_generate_trading_signals(self, server, sample_dataframe):
        """Test comprehensive signal generation."""
        strategies = ["sma_crossover", "rsi_mean_reversion", "bollinger_bands", "momentum"]
        
        signals = await server._generate_trading_signals(sample_dataframe, "TEST", strategies)
        
        assert isinstance(signals, list)
        
        # Should have signals from multiple strategies
        strategy_names = set(signal.get("strategy") for signal in signals)
        assert len(strategy_names) > 0
        
        # Check signal structure
        for signal in signals:
            assert "strategy" in signal
            assert "symbol" in signal
            assert signal["symbol"] == "TEST"
            assert signal["strategy"] in strategies
    
    @pytest.mark.asyncio
    async def test_analyze_strategy(self, server, sample_dataframe):
        """Test strategy analysis."""
        analysis = await server._analyze_strategy(sample_dataframe, "sma_crossover", {})
        
        assert isinstance(analysis, dict)
        
        # Check analysis fields
        required_fields = [
            "total_signals", "buy_signals", "sell_signals", "exit_signals",
            "average_confidence", "signal_frequency", "config_used"
        ]
        
        for field in required_fields:
            assert field in analysis
        
        # Check data types and ranges
        assert isinstance(analysis["total_signals"], int)
        assert isinstance(analysis["buy_signals"], int)
        assert isinstance(analysis["sell_signals"], int)
        assert isinstance(analysis["exit_signals"], int)
        assert 0 <= analysis["average_confidence"] <= 1
        assert analysis["signal_frequency"] >= 0
    
    @pytest.mark.asyncio
    async def test_calculate_risk_metrics(self, server):
        """Test risk metrics calculation."""
        # Generate sample returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100).tolist()
        
        risk_metrics = await server._calculate_risk_metrics(returns, [])
        
        assert isinstance(risk_metrics, dict)
        
        # Check required metrics
        required_metrics = [
            "mean_return", "volatility", "sharpe_ratio", "sortino_ratio",
            "calmar_ratio", "max_drawdown", "var_95", "expected_shortfall",
            "win_rate", "total_periods"
        ]
        
        for metric in required_metrics:
            assert metric in risk_metrics
            assert isinstance(risk_metrics[metric], (int, float))
        
        # Check logical constraints
        assert risk_metrics["volatility"] >= 0
        assert risk_metrics["max_drawdown"] <= 0
        assert 0 <= risk_metrics["win_rate"] <= 1
        assert risk_metrics["total_periods"] == 100
    
    @pytest.mark.asyncio
    async def test_run_comprehensive_backtest(self, server, sample_dataframe):
        """Test comprehensive backtesting."""
        backtest = await server._run_comprehensive_backtest(
            sample_dataframe, "sma_crossover", {}, 100000
        )
        
        assert isinstance(backtest, dict)
        
        # Check backtest results structure
        required_fields = [
            "initial_capital", "final_value", "total_return", "total_trades",
            "portfolio_history", "trades", "risk_metrics", "strategy"
        ]
        
        for field in required_fields:
            assert field in backtest
        
        # Check data types
        assert isinstance(backtest["initial_capital"], (int, float))
        assert isinstance(backtest["final_value"], (int, float))
        assert isinstance(backtest["total_return"], (int, float))
        assert isinstance(backtest["total_trades"], int)
        assert isinstance(backtest["portfolio_history"], list)
        assert isinstance(backtest["trades"], list)
        assert isinstance(backtest["risk_metrics"], dict)
        
        # Check logical constraints
        assert backtest["initial_capital"] > 0
        assert backtest["final_value"] > 0
        assert backtest["total_trades"] >= 0
        
        # Check portfolio history structure
        if backtest["portfolio_history"]:
            history_point = backtest["portfolio_history"][0]
            assert "timestamp" in history_point
            assert "portfolio_value" in history_point
            assert "cash" in history_point
            assert "position" in history_point
            assert "price" in history_point
        
        # Check trades structure
        if backtest["trades"]:
            trade = backtest["trades"][0]
            assert "timestamp" in trade
            assert "action" in trade
            assert "price" in trade
            assert "quantity" in trade
            assert "value" in trade
            assert "strategy" in trade
    
    @pytest.mark.asyncio
    async def test_detect_market_regime(self, server, sample_dataframe):
        """Test market regime detection."""
        regime = await server._detect_market_regime(sample_dataframe, 30)
        
        assert isinstance(regime, dict)
        
        # Check regime fields
        required_fields = [
            "regime", "trend_strength", "volatility_ratio", "price_change",
            "high_low_range", "confidence", "lookback_period"
        ]
        
        for field in required_fields:
            assert field in regime
        
        # Check regime types
        valid_regimes = [
            "BULL_TRENDING", "BEAR_TRENDING", "HIGH_VOLATILITY",
            "LOW_VOLATILITY_RANGING", "RANGING"
        ]
        assert regime["regime"] in valid_regimes
        
        # Check data types and ranges
        assert isinstance(regime["trend_strength"], (int, float))
        assert isinstance(regime["volatility_ratio"], (int, float))
        assert isinstance(regime["price_change"], (int, float))
        assert isinstance(regime["high_low_range"], (int, float))
        assert 0 <= regime["confidence"] <= 1
        assert regime["lookback_period"] == 30
    
    def test_strategy_configs(self, server):
        """Test strategy configurations."""
        configs = server.strategy_configs
        
        # Check that all expected strategies are configured
        expected_strategies = ["sma_crossover", "rsi_mean_reversion", "bollinger_bands", "momentum"]
        
        for strategy in expected_strategies:
            assert strategy in configs
            assert isinstance(configs[strategy], dict)
        
        # Check SMA crossover config
        sma_config = configs["sma_crossover"]
        assert "short_period" in sma_config
        assert "long_period" in sma_config
        assert sma_config["short_period"] < sma_config["long_period"]
        
        # Check RSI config
        rsi_config = configs["rsi_mean_reversion"]
        assert "rsi_period" in rsi_config
        assert "oversold_threshold" in rsi_config
        assert "overbought_threshold" in rsi_config
        assert rsi_config["oversold_threshold"] < rsi_config["overbought_threshold"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
