"""
Unit tests for LLM Market Intelligence MCP Server.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services.mcp_servers.llm_market_intelligence.server import LLMMarketIntelligenceServer
from services.mcp_servers.llm_market_intelligence.llm_engine import LLMEngine
from services.mcp_servers.llm_market_intelligence.config import config


class TestLLMMarketIntelligenceServer:
    """Test cases for LLM Market Intelligence MCP Server."""
    
    @pytest.fixture
    async def server(self):
        """Create test server instance."""
        server = LLMMarketIntelligenceServer()
        server.redis_client = AsyncMock()
        return server
    
    @pytest.fixture
    def mock_llm_engine(self):
        """Mock LLM engine for testing."""
        with patch('services.mcp_servers.llm_market_intelligence.server.llm_engine') as mock:
            mock.analyze_sentiment = AsyncMock()
            mock.detect_market_regime = AsyncMock()
            mock.generate_response = AsyncMock()
            yield mock
    
    @pytest.mark.asyncio
    async def test_analyze_market_sentiment_success(self, server, mock_llm_engine):
        """Test successful market sentiment analysis."""
        # Setup mock response
        mock_llm_engine.analyze_sentiment.return_value = {
            "sentiment": "positive",
            "confidence": 0.8,
            "reasoning": "Strong earnings report",
            "market_impact": "bullish",
            "key_factors": ["earnings", "growth"],
            "inference_time": 0.5
        }
        
        # Test data
        news_data = [
            {"title": "Company reports strong Q3 earnings", "content": "Revenue up 25%"}
        ]
        
        # Mock cache miss
        server._get_cached_result = AsyncMock(return_value=None)
        server._cache_result = AsyncMock()
        
        # Execute test
        content = {
            "news_data": news_data,
            "timeframe": "1d",
            "symbol": "RELIANCE"
        }
        
        # Call the handler directly
        result = await server.register_handlers.__wrapped__(server)
        handler = None
        for endpoint, func in server.handlers.items():
            if endpoint == "analyze_market_sentiment":
                handler = func
                break
        
        assert handler is not None
        response = await handler(content)
        
        # Verify response
        assert "sentiment_score" in response
        assert response["confidence"] == 0.8
        assert response["reasoning"] == "Strong earnings report"
        assert response["market_impact"] == "bullish"
        assert response["symbol"] == "RELIANCE"
        assert "analysis_timestamp" in response
        
        # Verify LLM engine was called
        mock_llm_engine.analyze_sentiment.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_detect_market_regime_success(self, server, mock_llm_engine):
        """Test successful market regime detection."""
        # Setup mock response
        mock_llm_engine.detect_market_regime.return_value = {
            "regime_type": "bull_market",
            "confidence": 0.75,
            "explanation": "Strong upward trend with low volatility",
            "key_indicators": ["price_momentum", "volume_increase"],
            "duration_estimate": 45,
            "risk_level": "low",
            "inference_time": 1.2
        }
        
        # Test data
        market_data = {
            "current_price": 2500,
            "change_1d": 1.5,
            "change_1w": 3.2,
            "volatility": 12.5
        }
        
        # Mock cache miss
        server._get_cached_result = AsyncMock(return_value=None)
        server._cache_result = AsyncMock()
        
        # Execute test
        content = {
            "market_data": market_data,
            "lookback_period": 30
        }
        
        # Get handler
        result = await server.register_handlers.__wrapped__(server)
        handler = server.handlers["detect_market_regime"]
        response = await handler(content)
        
        # Verify response
        assert response["regime_type"] == "bull_market"
        assert response["confidence"] == 0.75
        assert response["explanation"] == "Strong upward trend with low volatility"
        assert response["risk_level"] == "low"
        assert response["lookback_period"] == 30
        
        # Verify LLM engine was called
        mock_llm_engine.detect_market_regime.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_assess_event_impact_success(self, server, mock_llm_engine):
        """Test successful event impact assessment."""
        # Setup mock response
        mock_llm_engine.generate_response.return_value = {
            "response": json.dumps({
                "impact_score": 0.6,
                "duration_estimate": 7,
                "reasoning": "Positive earnings likely to boost sector sentiment",
                "affected_sectors": ["IT", "Technology"],
                "risk_factors": ["Market volatility"],
                "opportunities": ["Sector rotation"],
                "confidence": 0.8
            }),
            "inference_time": 0.8
        }
        
        # Test data
        event_data = {
            "event_type": "earnings_announcement",
            "company": "TCS",
            "details": "Strong Q3 results"
        }
        
        # Execute test
        content = {
            "event_data": event_data,
            "affected_symbols": ["TCS", "INFY"]
        }
        
        # Get handler
        result = await server.register_handlers.__wrapped__(server)
        handler = server.handlers["assess_event_impact"]
        response = await handler(content)
        
        # Verify response
        assert response["impact_score"] == 0.6
        assert response["duration_estimate"] == 7
        assert response["confidence"] == 0.8
        assert response["affected_symbols"] == ["TCS", "INFY"]
        
        # Verify LLM engine was called
        mock_llm_engine.generate_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_market_insights_success(self, server, mock_llm_engine):
        """Test successful market insights generation."""
        # Setup mock response
        mock_llm_engine.generate_response.return_value = {
            "response": json.dumps({
                "actionable_insights": ["Consider defensive positioning", "Monitor IT sector"],
                "risk_factors": ["Global uncertainty", "Inflation concerns"],
                "opportunities": ["Value stocks", "Dividend plays"],
                "market_outlook": "neutral",
                "time_horizon": "medium_term",
                "confidence": 0.7,
                "key_levels": [21000, 21500],
                "sector_recommendations": ["Underweight IT", "Overweight Banking"]
            }),
            "inference_time": 1.5
        }
        
        # Test data
        market_context = {
            "market_conditions": {"trend": "sideways", "volatility": "medium"},
            "economic_indicators": {"inflation": 5.2, "gdp_growth": 6.8}
        }
        
        # Execute test
        content = {
            "market_context": market_context,
            "focus_areas": ["trading", "risk_management"]
        }
        
        # Get handler
        result = await server.register_handlers.__wrapped__(server)
        handler = server.handlers["generate_market_insights"]
        response = await handler(content)
        
        # Verify response
        assert len(response["actionable_insights"]) == 2
        assert response["market_outlook"] == "neutral"
        assert response["confidence"] == 0.7
        assert response["focus_areas"] == ["trading", "risk_management"]
        
        # Verify LLM engine was called
        mock_llm_engine.generate_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_explain_market_context_success(self, server, mock_llm_engine):
        """Test successful market context explanation."""
        # Setup mock response
        mock_llm_engine.generate_response.return_value = {
            "response": json.dumps({
                "explanation": "Current market conditions show consolidation phase...",
                "key_points": ["Sideways trend", "Moderate volatility", "Earnings season"],
                "market_summary": "Markets in consolidation with mixed signals",
                "trader_focus": "Watch for breakout levels",
                "investor_focus": "Quality stocks at reasonable valuations",
                "risk_warning": "Avoid leveraged positions in current volatility"
            }),
            "inference_time": 1.0
        }
        
        # Test data
        current_conditions = {
            "nifty_level": 21500,
            "market_trend": "consolidation",
            "key_events": ["RBI policy", "Q3 earnings"]
        }
        
        # Execute test
        content = {
            "current_conditions": current_conditions,
            "detail_level": "medium"
        }
        
        # Get handler
        result = await server.register_handlers.__wrapped__(server)
        handler = server.handlers["explain_market_context"]
        response = await handler(content)
        
        # Verify response
        assert "explanation" in response
        assert len(response["key_points"]) == 3
        assert response["detail_level"] == "medium"
        
        # Verify LLM engine was called
        mock_llm_engine.generate_response.assert_called_once()
    
    def test_convert_sentiment_to_score(self, server):
        """Test sentiment to score conversion."""
        assert server._convert_sentiment_to_score("positive") == 0.7
        assert server._convert_sentiment_to_score("negative") == -0.7
        assert server._convert_sentiment_to_score("neutral") == 0.0
        assert server._convert_sentiment_to_score("bullish") == 0.8
        assert server._convert_sentiment_to_score("bearish") == -0.8
        assert server._convert_sentiment_to_score("unknown") == 0.0
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self, server):
        """Test caching functionality."""
        # Test cache hit
        cached_data = {"test": "data", "cached": True}
        server.redis_client.get.return_value = json.dumps(cached_data)
        
        result = await server._get_cached_result("test_key")
        assert result == cached_data
        
        # Test cache miss
        server.redis_client.get.return_value = None
        result = await server._get_cached_result("test_key")
        assert result is None
        
        # Test cache storage
        test_data = {"test": "data"}
        await server._cache_result("test_key", test_data, 300)
        server.redis_client.setex.assert_called_with("test_key", 300, json.dumps(test_data))


class TestLLMEngine:
    """Test cases for LLM Engine."""
    
    @pytest.fixture
    def mock_model(self):
        """Mock LLM model for testing."""
        with patch('services.mcp_servers.llm_market_intelligence.llm_engine.AutoModelForCausalLM') as mock:
            yield mock
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer for testing."""
        with patch('services.mcp_servers.llm_market_intelligence.llm_engine.AutoTokenizer') as mock:
            yield mock
    
    @pytest.mark.asyncio
    async def test_llm_engine_initialization(self, mock_model, mock_tokenizer):
        """Test LLM engine initialization."""
        engine = LLMEngine()
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_name', return_value="Tesla V100"):
                await engine.initialize()
        
        assert engine.model_loaded
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()
    
    def test_llm_engine_cleanup(self):
        """Test LLM engine cleanup."""
        engine = LLMEngine()
        engine.model = MagicMock()
        engine.tokenizer = MagicMock()
        engine.model_loaded = True
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.empty_cache') as mock_empty_cache:
                engine.cleanup()
        
        assert not engine.model_loaded
        mock_empty_cache.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
