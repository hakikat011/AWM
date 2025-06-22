"""
Unit tests for historical backtesting framework.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.historical_backtesting import HistoricalBacktester
from scripts.run_historical_backtest import ConfigurableHistoricalBacktester, load_config


class TestHistoricalBacktester:
    """Test cases for historical backtesting framework."""
    
    @pytest.fixture
    def backtester(self):
        """Create test backtester instance."""
        return HistoricalBacktester()
    
    @pytest.fixture
    def sample_historical_data(self):
        """Create sample historical data for testing."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        dates = dates[dates.dayofweek < 5]  # Only weekdays
        
        data = []
        base_price = 100
        
        for i, date in enumerate(dates):
            # Generate realistic OHLCV data
            daily_return = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% volatility
            
            open_price = base_price * (1 + daily_return)
            high_price = open_price * (1 + abs(np.random.normal(0, 0.01)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.01)))
            close_price = open_price + np.random.normal(0, 0.005) * open_price
            volume = np.random.randint(100000, 1000000)
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
            
            base_price = close_price
        
        return pd.DataFrame(data).set_index('timestamp')
    
    def test_initialization(self, backtester):
        """Test backtester initialization."""
        assert backtester.initial_capital == 1000000
        assert len(backtester.test_symbols) == 5
        assert backtester.transaction_costs['total_cost_pct'] == 0.00228
        assert backtester.slippage_config['market_order_slippage_bps'] == 5
    
    def test_filter_indian_market_hours(self, backtester, sample_historical_data):
        """Test filtering for Indian market hours."""
        filtered_data = backtester._filter_indian_market_hours(sample_historical_data)
        
        # Should only have weekdays
        assert all(filtered_data.index.dayofweek < 5)
        assert len(filtered_data) <= len(sample_historical_data)
    
    def test_calculate_technical_indicators(self, backtester, sample_historical_data):
        """Test technical indicators calculation."""
        indicators_data = backtester._calculate_technical_indicators(sample_historical_data)
        
        # Check that indicators are calculated
        expected_indicators = ['sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi', 'macd', 'bb_upper', 'bb_lower']
        for indicator in expected_indicators:
            assert indicator in indicators_data.columns
        
        # Check RSI is in valid range (0-100)
        rsi_values = indicators_data['rsi'].dropna()
        assert all(0 <= rsi <= 100 for rsi in rsi_values)
    
    def test_calculate_slippage(self, backtester):
        """Test slippage calculation."""
        symbol_data = {
            "current": {
                "close": 100,
                "volatility_20": 0.15,
                "volume_ratio": 1.0
            }
        }
        
        # Test normal conditions
        slippage = backtester._calculate_slippage(symbol_data, 100, "BUY")
        assert 0 <= slippage <= 0.005  # Should be within max slippage
        
        # Test high volatility
        symbol_data["current"]["volatility_20"] = 0.5
        high_vol_slippage = backtester._calculate_slippage(symbol_data, 100, "BUY")
        assert high_vol_slippage >= slippage  # Should be higher
        
        # Test low liquidity
        symbol_data["current"]["volume_ratio"] = 0.3
        low_liq_slippage = backtester._calculate_slippage(symbol_data, 100, "BUY")
        assert low_liq_slippage >= slippage  # Should be higher
    
    @pytest.mark.asyncio
    async def test_execute_historical_trade_buy(self, backtester):
        """Test historical trade execution for buy orders."""
        portfolio = {
            "cash": 100000,
            "positions": {},
            "trades": []
        }
        
        symbol_data = {
            "current": {
                "close": 100,
                "volatility_20": 0.15,
                "volume_ratio": 1.0
            }
        }
        
        decision = {
            "action": "BUY",
            "quantity": 100,
            "confidence": 0.8
        }
        
        current_date = datetime(2023, 6, 15)
        
        await backtester._execute_historical_trade(
            "TEST", decision, portfolio, symbol_data, "TEST_STRATEGY", 0, current_date
        )
        
        # Check trade was executed
        assert len(portfolio["trades"]) == 1
        assert portfolio["positions"]["TEST"] == 100
        assert portfolio["cash"] < 100000  # Cash should be reduced
        
        trade = portfolio["trades"][0]
        assert trade["action"] == "BUY"
        assert trade["quantity"] == 100
        assert trade["symbol"] == "TEST"
        assert "transaction_cost" in trade
        assert "slippage" in trade
    
    @pytest.mark.asyncio
    async def test_execute_historical_trade_sell(self, backtester):
        """Test historical trade execution for sell orders."""
        portfolio = {
            "cash": 50000,
            "positions": {"TEST": 200},
            "trades": []
        }
        
        symbol_data = {
            "current": {
                "close": 110,
                "volatility_20": 0.15,
                "volume_ratio": 1.0
            }
        }
        
        decision = {
            "action": "SELL",
            "quantity": 100,
            "confidence": 0.7
        }
        
        current_date = datetime(2023, 6, 15)
        
        await backtester._execute_historical_trade(
            "TEST", decision, portfolio, symbol_data, "TEST_STRATEGY", 0, current_date
        )
        
        # Check trade was executed
        assert len(portfolio["trades"]) == 1
        assert portfolio["positions"]["TEST"] == 100  # Reduced by 100
        assert portfolio["cash"] > 50000  # Cash should be increased
        
        trade = portfolio["trades"][0]
        assert trade["action"] == "SELL"
        assert trade["quantity"] == 100
    
    def test_calculate_enhanced_metrics(self, backtester):
        """Test enhanced metrics calculation."""
        # Create sample portfolio with trades and daily P&L
        portfolio = {
            "trades": [
                {
                    "action": "BUY",
                    "net_value": 10000,
                    "transaction_cost": 23,
                    "slippage": 0.0005,
                    "gross_value": 10000
                },
                {
                    "action": "SELL",
                    "net_value": 11000,
                    "transaction_cost": 25,
                    "slippage": 0.0003,
                    "gross_value": 11000
                }
            ],
            "daily_pnl": [
                {"portfolio_value": 1000000, "daily_return": 0.0},
                {"portfolio_value": 1005000, "daily_return": 0.005},
                {"portfolio_value": 1010000, "daily_return": 0.005},
                {"portfolio_value": 1008000, "daily_return": -0.002},
                {"portfolio_value": 1012000, "daily_return": 0.004}
            ]
        }
        
        # Mock the async method
        async def run_test():
            metrics = await backtester._calculate_enhanced_metrics(portfolio)
            
            # Check basic metrics
            assert "total_return" in metrics
            assert "annualized_return" in metrics
            assert "sharpe_ratio" in metrics
            assert "max_drawdown" in metrics
            assert "win_rate" in metrics
            assert "total_trades" in metrics
            
            # Check values are reasonable
            assert metrics["total_trades"] == 2
            assert 0 <= metrics["win_rate"] <= 1
            assert metrics["total_return"] > 0  # Should be positive
        
        asyncio.run(run_test())
    
    def test_calculate_sharpe_ratio(self, backtester):
        """Test Sharpe ratio calculation."""
        returns = [0.01, -0.005, 0.02, 0.015, -0.01, 0.008, 0.012]
        sharpe = backtester._calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        
        # Test with empty returns
        empty_sharpe = backtester._calculate_sharpe_ratio([])
        assert empty_sharpe == 0.0
    
    def test_bootstrap_sharpe_test(self, backtester):
        """Test bootstrap Sharpe ratio test."""
        returns1 = [0.01, 0.02, 0.015, 0.008, 0.012] * 10  # 50 returns
        returns2 = [0.005, 0.01, 0.008, 0.004, 0.006] * 10  # 50 returns
        
        p_value = backtester._bootstrap_sharpe_test(returns1, returns2, n_bootstrap=100)
        
        assert 0 <= p_value <= 1
        assert isinstance(p_value, float)
    
    def test_analyze_drawdowns(self, backtester):
        """Test drawdown analysis."""
        daily_pnl = [
            {"portfolio_value": 1000000},
            {"portfolio_value": 1050000},
            {"portfolio_value": 1020000},  # Start drawdown
            {"portfolio_value": 980000},   # Trough
            {"portfolio_value": 1010000},  # Recovery
            {"portfolio_value": 1080000}   # New peak
        ]
        
        drawdown_analysis = backtester._analyze_drawdowns(daily_pnl)
        
        assert "max_drawdown" in drawdown_analysis
        assert "avg_drawdown" in drawdown_analysis
        assert "drawdown_count" in drawdown_analysis
        assert drawdown_analysis["max_drawdown"] > 0
    
    def test_calculate_tail_risk(self, backtester):
        """Test tail risk calculation."""
        returns = np.random.normal(0.001, 0.02, 1000).tolist()  # 1000 random returns
        
        tail_risk = backtester._calculate_tail_risk(returns)
        
        assert "var_95" in tail_risk
        assert "var_99" in tail_risk
        assert "cvar_95" in tail_risk
        assert "cvar_99" in tail_risk
        assert "worst_day" in tail_risk
        assert "best_day" in tail_risk
        
        # VaR should be negative (loss)
        assert tail_risk["var_95"] < 0
        assert tail_risk["var_99"] < 0
        assert tail_risk["var_99"] <= tail_risk["var_95"]  # 99% VaR should be worse


class TestConfigurableBacktester:
    """Test cases for configurable backtester."""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return {
            'backtest_period': {
                'start_date': '2023-01-01',
                'end_date': '2023-06-30'
            },
            'portfolio': {
                'initial_capital': 500000,
                'max_position_size_pct': 0.15
            },
            'symbols': ['RELIANCE', 'TCS'],
            'transaction_costs': {
                'total_cost_pct': 0.003
            },
            'slippage': {
                'market_order_slippage_bps': 10
            },
            'servers': {
                'market_data': 'http://test:8001'
            }
        }
    
    def test_configuration_loading(self, sample_config):
        """Test configuration loading and application."""
        backtester = ConfigurableHistoricalBacktester(sample_config)
        
        # Check configuration was applied
        assert backtester.initial_capital == 500000
        assert len(backtester.test_symbols) == 2
        assert backtester.test_symbols == ['RELIANCE', 'TCS']
        assert backtester.transaction_costs['total_cost_pct'] == 0.003
        assert backtester.slippage_config['market_order_slippage_bps'] == 10
        assert backtester.server_urls['market_data'] == 'http://test:8001'
    
    def test_date_configuration(self, sample_config):
        """Test date configuration."""
        backtester = ConfigurableHistoricalBacktester(sample_config)
        
        expected_start = datetime(2023, 1, 1)
        expected_end = datetime(2023, 6, 30)
        
        assert backtester.backtest_start_date == expected_start
        assert backtester.backtest_end_date == expected_end


class TestConfigurationFile:
    """Test configuration file loading."""
    
    def test_load_valid_config(self, tmp_path):
        """Test loading valid configuration file."""
        config_content = """
backtest_period:
  start_date: "2023-01-01"
  end_date: "2023-12-31"

portfolio:
  initial_capital: 1000000

symbols:
  - "RELIANCE"
  - "TCS"
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        config = load_config(str(config_file))
        
        assert config['backtest_period']['start_date'] == "2023-01-01"
        assert config['portfolio']['initial_capital'] == 1000000
        assert len(config['symbols']) == 2
    
    def test_load_invalid_config(self, tmp_path):
        """Test loading invalid configuration file."""
        config_file = tmp_path / "invalid_config.yaml"
        config_file.write_text("invalid: yaml: content: [")
        
        with pytest.raises(Exception):
            load_config(str(config_file))


class TestStatisticalMethods:
    """Test statistical analysis methods."""
    
    @pytest.fixture
    def backtester(self):
        return HistoricalBacktester()
    
    @pytest.mark.asyncio
    async def test_statistical_tests_with_data(self, backtester):
        """Test statistical tests with sample data."""
        # Create sample daily P&L data
        llm_returns = np.random.normal(0.002, 0.02, 100).tolist()  # Slightly better returns
        quant_returns = np.random.normal(0.001, 0.02, 100).tolist()
        
        # Mock the portfolio data
        backtester.llm_enhanced_portfolio["daily_pnl"] = [
            {"daily_return": 0.0}  # First day
        ] + [{"daily_return": r} for r in llm_returns]
        
        backtester.quantitative_only_portfolio["daily_pnl"] = [
            {"daily_return": 0.0}  # First day
        ] + [{"daily_return": r} for r in quant_returns]
        
        statistical_tests = await backtester._perform_statistical_tests()
        
        assert "return_differences" in statistical_tests
        assert "mann_whitney_test" in statistical_tests
        assert "sharpe_ratio_comparison" in statistical_tests
        assert "overall_significance" in statistical_tests
        
        # Check p-values are valid
        assert 0 <= statistical_tests["return_differences"]["t_p_value"] <= 1
        assert 0 <= statistical_tests["mann_whitney_test"]["p_value"] <= 1
    
    @pytest.mark.asyncio
    async def test_statistical_tests_insufficient_data(self, backtester):
        """Test statistical tests with insufficient data."""
        # Create insufficient data
        backtester.llm_enhanced_portfolio["daily_pnl"] = [{"daily_return": 0.01}] * 10
        backtester.quantitative_only_portfolio["daily_pnl"] = [{"daily_return": 0.005}] * 10
        
        statistical_tests = await backtester._perform_statistical_tests()
        
        assert "error" in statistical_tests


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
