"""
Unit tests for signal generation system.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from services.mcp_servers.signal_generation.server import SignalGenerationServer


class TestSignalGeneration:
    """Test signal generation system."""
    
    @pytest.fixture
    def server(self):
        """Create signal generation server instance."""
        return SignalGenerationServer()
    
    @pytest.fixture
    def sample_quant_signals(self):
        """Sample quantitative signals."""
        return [
            {
                "timestamp": "2024-01-01T10:00:00",
                "signal": "BUY",
                "price": 2500.0,
                "confidence": 0.8,
                "strategy": "sma_crossover",
                "reason": "Golden cross detected"
            },
            {
                "timestamp": "2024-01-01T11:00:00",
                "signal": "BUY",
                "price": 2505.0,
                "confidence": 0.7,
                "strategy": "rsi_mean_reversion",
                "reason": "RSI oversold"
            },
            {
                "timestamp": "2024-01-01T12:00:00",
                "signal": "SELL",
                "price": 2510.0,
                "confidence": 0.6,
                "strategy": "bollinger_bands",
                "reason": "Price at upper band"
            }
        ]
    
    @pytest.fixture
    def sample_technical_analysis(self):
        """Sample technical analysis data."""
        return {
            "rsi": [{"timestamp": "2024-01-01T12:00:00", "value": 25.5}],
            "macd": [{"timestamp": "2024-01-01T12:00:00", "macd": 5.2, "signal": 3.1}],
            "support_resistance": {
                "support_levels": [2450.0, 2480.0],
                "resistance_levels": [2520.0, 2550.0]
            }
        }
    
    @pytest.fixture
    def sample_news_sentiment(self):
        """Sample news sentiment data."""
        return {
            "sentiment": "positive",
            "confidence": 0.75,
            "score": 0.6
        }
    
    def test_signal_configs(self, server):
        """Test signal generation configurations."""
        configs = server.signal_configs
        
        # Check that all expected configs exist
        expected_configs = ["default", "conservative", "aggressive"]
        for config_name in expected_configs:
            assert config_name in configs
            config = configs[config_name]
            
            # Check required fields
            required_fields = [
                "strategies", "min_confidence", "consensus_threshold", "lookback_days"
            ]
            for field in required_fields:
                assert field in config
            
            # Check data types and ranges
            assert isinstance(config["strategies"], list)
            assert len(config["strategies"]) > 0
            assert 0 <= config["min_confidence"] <= 1
            assert 0 <= config["consensus_threshold"] <= 1
            assert config["lookback_days"] > 0
    
    @pytest.mark.asyncio
    async def test_combine_signals(self, server, sample_quant_signals, sample_technical_analysis, sample_news_sentiment):
        """Test signal combination logic."""
        config = server.signal_configs["default"]
        
        combined = await server._combine_signals(
            sample_quant_signals, sample_technical_analysis, sample_news_sentiment, config
        )
        
        assert isinstance(combined, dict)
        
        # Check required fields
        required_fields = ["signal", "confidence", "reason", "analysis"]
        for field in required_fields:
            assert field in combined
        
        # Check signal values
        assert combined["signal"] in ["BUY", "SELL", "HOLD"]
        assert 0 <= combined["confidence"] <= 1
        assert isinstance(combined["reason"], str)
        assert isinstance(combined["analysis"], dict)
        
        # Check analysis breakdown
        analysis = combined["analysis"]
        analysis_fields = [
            "buy_strength", "sell_strength", "buy_confidence", "sell_confidence",
            "technical_influence", "sentiment_influence", "final_buy_score", "final_sell_score"
        ]
        for field in analysis_fields:
            assert field in analysis
            assert isinstance(analysis[field], (int, float))
    
    @pytest.mark.asyncio
    async def test_combine_signals_no_signals(self, server, sample_technical_analysis, sample_news_sentiment):
        """Test signal combination with no quantitative signals."""
        config = server.signal_configs["default"]
        
        combined = await server._combine_signals(
            [], sample_technical_analysis, sample_news_sentiment, config
        )
        
        assert combined["signal"] == "HOLD"
        assert combined["confidence"] == 0.0
        assert "No quantitative signals available" in combined["reason"]
    
    @pytest.mark.asyncio
    async def test_calculate_signal_consensus(self, server, sample_quant_signals):
        """Test signal consensus calculation."""
        consensus = await server._calculate_signal_consensus(sample_quant_signals, 0.7)
        
        assert isinstance(consensus, dict)
        
        # Check required fields
        required_fields = [
            "consensus", "confidence", "scores", "signal_counts", "consensus_threshold"
        ]
        for field in required_fields:
            assert field in consensus
        
        # Check consensus value
        assert consensus["consensus"] in ["BUY", "SELL", "HOLD"]
        assert 0 <= consensus["confidence"] <= 1
        assert consensus["consensus_threshold"] == 0.7
        
        # Check scores
        scores = consensus["scores"]
        score_fields = ["buy_score", "sell_score", "hold_score"]
        for field in score_fields:
            assert field in scores
            assert isinstance(scores[field], (int, float))
        
        # Check signal counts
        counts = consensus["signal_counts"]
        count_fields = ["buy_signals", "sell_signals", "hold_signals"]
        for field in count_fields:
            assert field in counts
            assert isinstance(counts[field], int)
            assert counts[field] >= 0
    
    @pytest.mark.asyncio
    async def test_rank_signals_confidence_weighted(self, server, sample_quant_signals):
        """Test signal ranking by confidence."""
        ranked = await server._rank_signals(sample_quant_signals, "confidence_weighted")
        
        assert isinstance(ranked, list)
        assert len(ranked) == len(sample_quant_signals)
        
        # Check that signals are ranked by confidence (descending)
        for i in range(len(ranked) - 1):
            assert ranked[i]["confidence"] >= ranked[i + 1]["confidence"]
    
    @pytest.mark.asyncio
    async def test_rank_signals_signal_strength(self, server, sample_quant_signals):
        """Test signal ranking by signal strength."""
        ranked = await server._rank_signals(sample_quant_signals, "signal_strength")
        
        assert isinstance(ranked, list)
        assert len(ranked) == len(sample_quant_signals)
        
        # BUY/SELL signals should be ranked higher than HOLD signals
        buy_sell_signals = [s for s in ranked if s["signal"] in ["BUY", "SELL"]]
        hold_signals = [s for s in ranked if s["signal"] == "HOLD"]
        
        if buy_sell_signals and hold_signals:
            # First BUY/SELL signal should have higher effective score than first HOLD signal
            assert buy_sell_signals[0]["confidence"] >= hold_signals[0]["confidence"] * 0.5
    
    @pytest.mark.asyncio
    async def test_filter_signals_min_confidence(self, server, sample_quant_signals):
        """Test signal filtering by minimum confidence."""
        filters = {"min_confidence": 0.75}
        
        filtered = await server._filter_signals(sample_quant_signals, filters)
        
        assert isinstance(filtered, list)
        
        # All filtered signals should meet minimum confidence
        for signal in filtered:
            assert signal["confidence"] >= 0.75
        
        # Should have fewer signals than original
        assert len(filtered) <= len(sample_quant_signals)
    
    @pytest.mark.asyncio
    async def test_filter_signals_signal_types(self, server, sample_quant_signals):
        """Test signal filtering by signal type."""
        filters = {"signal_types": ["BUY"]}
        
        filtered = await server._filter_signals(sample_quant_signals, filters)
        
        assert isinstance(filtered, list)
        
        # All filtered signals should be BUY signals
        for signal in filtered:
            assert signal["signal"] == "BUY"
    
    @pytest.mark.asyncio
    async def test_filter_signals_strategies(self, server, sample_quant_signals):
        """Test signal filtering by strategy."""
        filters = {"strategies": ["sma_crossover", "rsi_mean_reversion"]}
        
        filtered = await server._filter_signals(sample_quant_signals, filters)
        
        assert isinstance(filtered, list)
        
        # All filtered signals should be from allowed strategies
        for signal in filtered:
            assert signal["strategy"] in ["sma_crossover", "rsi_mean_reversion"]
    
    @pytest.mark.asyncio
    async def test_filter_signals_multiple_criteria(self, server, sample_quant_signals):
        """Test signal filtering with multiple criteria."""
        filters = {
            "min_confidence": 0.7,
            "signal_types": ["BUY", "SELL"],
            "strategies": ["sma_crossover", "rsi_mean_reversion"]
        }
        
        filtered = await server._filter_signals(sample_quant_signals, filters)
        
        assert isinstance(filtered, list)
        
        # All filtered signals should meet all criteria
        for signal in filtered:
            assert signal["confidence"] >= 0.7
            assert signal["signal"] in ["BUY", "SELL"]
            assert signal["strategy"] in ["sma_crossover", "rsi_mean_reversion"]
    
    @pytest.mark.asyncio
    async def test_generate_watchlist_signals(self, server):
        """Test watchlist signal generation."""
        symbols = ["RELIANCE", "TCS"]
        
        # Mock the _generate_comprehensive_signals method
        async def mock_generate_signals(symbol, config_name, custom_strategies):
            return {
                "symbol": symbol,
                "combined_signals": {
                    "signal": "BUY",
                    "confidence": 0.8,
                    "reason": f"Mock signal for {symbol}"
                }
            }
        
        server._generate_comprehensive_signals = mock_generate_signals
        
        watchlist_signals = await server._generate_watchlist_signals(symbols, "default")
        
        assert isinstance(watchlist_signals, list)
        assert len(watchlist_signals) == len(symbols)
        
        for i, signal_data in enumerate(watchlist_signals):
            assert signal_data["symbol"] == symbols[i]
            assert "signal_data" in signal_data
            assert "timestamp" in signal_data
    
    def test_server_urls(self, server):
        """Test server URL configuration."""
        urls = server.server_urls
        
        expected_servers = [
            "quantitative_analysis", "technical_analysis", "market_data", "news"
        ]
        
        for server_name in expected_servers:
            assert server_name in urls
            assert isinstance(urls[server_name], str)
            assert urls[server_name].startswith("http")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
