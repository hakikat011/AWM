"""
Integration tests for LLM-enhanced trading system components.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services.mcp_servers.signal_generation.server import SignalGenerationServer
from services.mcp_servers.decision_engine.server import DecisionEngineServer
from services.agents.autonomous_trading.agent import AutonomousTradingAgent


class TestLLMEnhancedSignalGeneration:
    """Test LLM integration in signal generation."""
    
    @pytest.fixture
    async def signal_server(self):
        """Create test signal generation server."""
        server = SignalGenerationServer()
        server.mcp_client = AsyncMock()
        return server
    
    @pytest.mark.asyncio
    async def test_llm_sentiment_integration(self, signal_server):
        """Test LLM sentiment integration in signal generation."""
        # Mock LLM sentiment response
        llm_sentiment_response = {
            "sentiment_score": 0.6,
            "confidence": 0.8,
            "reasoning": "Positive earnings outlook",
            "market_impact": "bullish",
            "key_factors": ["earnings", "growth"]
        }
        
        # Mock news data response
        news_data_response = [
            {"title": "Strong Q3 results", "content": "Revenue growth of 25%"}
        ]
        
        # Setup mocks
        signal_server.mcp_client.__aenter__ = AsyncMock(return_value=signal_server.mcp_client)
        signal_server.mcp_client.__aexit__ = AsyncMock(return_value=None)
        signal_server.mcp_client.send_request = AsyncMock()
        
        # Configure mock responses
        def mock_send_request(url, endpoint, content):
            if "llm-market-intelligence" in url and endpoint == "analyze_market_sentiment":
                return llm_sentiment_response
            elif "news" in url and endpoint == "get_recent_news":
                return {"articles": news_data_response}
            return {}
        
        signal_server.mcp_client.send_request.side_effect = mock_send_request
        
        # Test LLM sentiment retrieval
        market_data = [{"close": 100, "volume": 1000, "timestamp": "2024-01-01"}]
        result = await signal_server._get_llm_sentiment("RELIANCE", market_data)
        
        assert result["sentiment_score"] == 0.6
        assert result["confidence"] == 0.8
        assert result["market_impact"] == "bullish"
    
    @pytest.mark.asyncio
    async def test_market_regime_integration(self, signal_server):
        """Test market regime integration in signal generation."""
        # Mock market regime response
        regime_response = {
            "regime_type": "bull_market",
            "confidence": 0.75,
            "explanation": "Strong upward momentum",
            "risk_level": "low"
        }
        
        # Setup mocks
        signal_server.mcp_client.__aenter__ = AsyncMock(return_value=signal_server.mcp_client)
        signal_server.mcp_client.__aexit__ = AsyncMock(return_value=None)
        signal_server.mcp_client.send_request = AsyncMock(return_value=regime_response)
        
        # Test market regime detection
        market_data = [
            {"close": 100, "volume": 1000, "timestamp": "2024-01-01"},
            {"close": 105, "volume": 1200, "timestamp": "2024-01-02"}
        ]
        result = await signal_server._get_market_regime(market_data)
        
        assert result["regime_type"] == "bull_market"
        assert result["confidence"] == 0.75
        assert result["risk_level"] == "low"
    
    @pytest.mark.asyncio
    async def test_enhanced_signal_combination(self, signal_server):
        """Test LLM-enhanced signal combination logic."""
        # Mock data
        quant_signals = [
            {"signal": "BUY", "confidence": 0.7, "strategy": "momentum"},
            {"signal": "BUY", "confidence": 0.8, "strategy": "mean_reversion"}
        ]
        
        technical_analysis = {
            "rsi": [{"value": 35, "timestamp": "2024-01-01"}],
            "macd": [{"macd": 0.5, "signal": 0.3, "timestamp": "2024-01-01"}]
        }
        
        news_sentiment = {"sentiment": "positive", "confidence": 0.6}
        
        llm_sentiment = {
            "sentiment_score": 0.7,
            "confidence": 0.85,
            "market_impact": "bullish"
        }
        
        market_regime = {
            "regime_type": "bull_market",
            "confidence": 0.8,
            "risk_level": "low"
        }
        
        config = {
            "consensus_threshold": 0.7,
            "min_confidence": 0.6
        }
        
        # Test signal combination
        result = await signal_server._combine_signals(
            quant_signals, technical_analysis, news_sentiment, 
            llm_sentiment, market_regime, config
        )
        
        assert result["signal"] in ["BUY", "SELL", "HOLD"]
        assert "llm_analysis" in result
        assert result["llm_analysis"]["sentiment_score"] == 0.7
        assert result["llm_analysis"]["regime_type"] == "bull_market"


class TestLLMEnhancedDecisionEngine:
    """Test LLM integration in decision engine."""
    
    @pytest.fixture
    async def decision_server(self):
        """Create test decision engine server."""
        server = DecisionEngineServer()
        server.mcp_client = AsyncMock()
        return server
    
    @pytest.mark.asyncio
    async def test_market_insights_integration(self, decision_server):
        """Test market insights integration in decision making."""
        # Mock market insights response
        insights_response = {
            "actionable_insights": ["Consider defensive positioning"],
            "risk_factors": ["Market volatility"],
            "opportunities": ["Value stocks"],
            "market_outlook": "neutral",
            "confidence": 0.7
        }
        
        explanation_response = {
            "explanation": "Current market shows mixed signals",
            "key_points": ["Consolidation phase", "Earnings season"],
            "trader_focus": "Watch breakout levels"
        }
        
        # Setup mocks
        decision_server.mcp_client.__aenter__ = AsyncMock(return_value=decision_server.mcp_client)
        decision_server.mcp_client.__aexit__ = AsyncMock(return_value=None)
        
        def mock_send_request(url, endpoint, content):
            if endpoint == "generate_market_insights":
                return insights_response
            elif endpoint == "explain_market_context":
                return explanation_response
            return {}
        
        decision_server.mcp_client.send_request.side_effect = mock_send_request
        
        # Test market insights retrieval
        portfolio_state = {"total_value": 100000, "cash_available": 20000}
        risk_assessment = {"risk_level": "MEDIUM", "risk_metrics": {}}
        
        result = await decision_server._get_market_insights("RELIANCE", portfolio_state, risk_assessment)
        
        assert "insights" in result
        assert "explanation" in result
        assert result["insights"]["market_outlook"] == "neutral"
        assert result["explanation"]["trader_focus"] == "Watch breakout levels"
    
    @pytest.mark.asyncio
    async def test_llm_adjustments_application(self, decision_server):
        """Test LLM adjustments in decision making."""
        # Test data
        signal_confidence = 0.7
        config = {"max_position_size_pct": 0.1}
        market_insights = {
            "insights": {
                "market_outlook": "bullish",
                "confidence": 0.8,
                "risk_factors": ["Minor volatility"],
                "opportunities": ["Strong momentum", "Sector rotation"]
            }
        }
        
        # Test LLM adjustments
        adjustments = decision_server._apply_llm_adjustments(signal_confidence, config, market_insights)
        
        assert adjustments["adjusted_confidence"] > signal_confidence  # Should increase for bullish outlook
        assert adjustments["position_size_multiplier"] > 1.0  # Should increase position size
        assert adjustments["market_outlook"] == "bullish"
        assert adjustments["llm_confidence"] == 0.8


class TestLLMEnhancedAutonomousTrading:
    """Test LLM integration in autonomous trading agent."""
    
    @pytest.fixture
    async def trading_agent(self):
        """Create test autonomous trading agent."""
        agent = AutonomousTradingAgent()
        agent.mcp_client = AsyncMock()
        return agent
    
    @pytest.mark.asyncio
    async def test_market_intelligence_overview(self, trading_agent):
        """Test market intelligence overview gathering."""
        # Mock portfolio response
        portfolio_response = {
            "total_value": 100000,
            "cash_available": 20000,
            "positions": [{"symbol": "RELIANCE", "quantity": 100}]
        }
        
        # Mock insights response
        insights_response = {
            "actionable_insights": ["Monitor IT sector closely"],
            "market_outlook": "neutral",
            "confidence": 0.7,
            "risk_factors": ["Global uncertainty"]
        }
        
        # Setup mocks
        trading_agent.mcp_client.__aenter__ = AsyncMock(return_value=trading_agent.mcp_client)
        trading_agent.mcp_client.__aexit__ = AsyncMock(return_value=None)
        
        def mock_send_request(url, endpoint, content):
            if "portfolio-management" in url:
                return portfolio_response
            elif "llm-market-intelligence" in url:
                return insights_response
            return {}
        
        trading_agent.mcp_client.send_request.side_effect = mock_send_request
        
        # Test market intelligence overview
        result = await trading_agent._get_market_intelligence_overview()
        
        assert result["market_outlook"] == "neutral"
        assert result["confidence"] == 0.7
        assert len(result["actionable_insights"]) > 0
    
    @pytest.mark.asyncio
    async def test_symbol_insights_for_high_confidence_signals(self, trading_agent):
        """Test symbol insights for high-confidence trading signals."""
        # Mock high-confidence signals
        signals = {
            "signals": {
                "combined_signals": {
                    "signal": "BUY",
                    "confidence": 0.8
                },
                "quantitative_signals": [{"signal": "BUY", "confidence": 0.7}],
                "technical_analysis": {"rsi": [{"value": 30}]},
                "news_sentiment": {"sentiment": "positive"},
                "llm_sentiment": {"sentiment_score": 0.6},
                "market_regime": {"regime_type": "bull_market"}
            }
        }
        
        # Mock explanation response
        explanation_response = {
            "explanation": "Strong buy signal with bullish market regime",
            "key_points": ["Technical oversold", "Positive sentiment"],
            "trader_focus": "Enter position on pullback",
            "risk_warning": "Monitor for reversal signals"
        }
        
        # Setup mocks
        trading_agent.mcp_client.__aenter__ = AsyncMock(return_value=trading_agent.mcp_client)
        trading_agent.mcp_client.__aexit__ = AsyncMock(return_value=None)
        trading_agent.mcp_client.send_request = AsyncMock(return_value=explanation_response)
        
        # Test symbol insights
        result = await trading_agent._get_symbol_insights("RELIANCE", signals)
        
        assert result["explanation"] == "Strong buy signal with bullish market regime"
        assert len(result["key_points"]) == 2
        assert result["trader_focus"] == "Enter position on pullback"
    
    @pytest.mark.asyncio
    async def test_decision_enhancement_with_llm(self, trading_agent):
        """Test decision enhancement with LLM context."""
        # Mock decision
        decision = {
            "action": "BUY",
            "symbol": "RELIANCE",
            "quantity": 100,
            "confidence": 0.75
        }
        
        # Mock insights
        symbol_insights = {
            "explanation": "Strong technical setup with positive sentiment",
            "key_points": ["Breakout pattern", "Volume confirmation"],
            "risk_warning": "Watch for false breakout"
        }
        
        market_intelligence = {
            "market_outlook": "bullish",
            "actionable_insights": ["Favor momentum stocks"],
            "risk_factors": ["Minor volatility"],
            "confidence": 0.8
        }
        
        # Test decision enhancement
        enhanced_decision = await trading_agent._enhance_decision_with_llm(
            "RELIANCE", decision, symbol_insights, market_intelligence
        )
        
        assert "llm_context" in enhanced_decision
        assert enhanced_decision["llm_context"]["symbol_insights"] == symbol_insights
        assert enhanced_decision["llm_context"]["market_intelligence"] == market_intelligence
        assert "decision_rationale" in enhanced_decision["llm_context"]
        assert "risk_assessment" in enhanced_decision["llm_context"]
    
    def test_decision_rationale_generation(self, trading_agent):
        """Test decision rationale generation."""
        decision = {"action": "BUY", "confidence": 0.8}
        symbol_insights = {"explanation": "Strong technical setup"}
        market_intelligence = {
            "market_outlook": "bullish",
            "actionable_insights": ["Favor growth stocks"]
        }
        
        rationale = trading_agent._generate_decision_rationale(
            "RELIANCE", decision, symbol_insights, market_intelligence
        )
        
        assert "BUY" in rationale
        assert "RELIANCE" in rationale
        assert "bullish" in rationale
        assert "0.80" in rationale
    
    def test_risk_assessment(self, trading_agent):
        """Test risk assessment based on LLM insights."""
        decision = {"action": "BUY"}
        symbol_insights = {"risk_warning": "High volatility risk"}
        market_intelligence = {"risk_factors": ["Global uncertainty", "Market volatility"]}
        
        risk_level = trading_agent._assess_decision_risk(decision, symbol_insights, market_intelligence)
        
        assert "HIGH" in risk_level or "MEDIUM" in risk_level
    
    def test_confidence_adjustment(self, trading_agent):
        """Test confidence adjustment with LLM insights."""
        decision = {"confidence": 0.7}
        symbol_insights = {}
        market_intelligence = {
            "market_outlook": "bullish",
            "confidence": 0.8
        }
        
        adjusted_confidence = trading_agent._adjust_confidence_with_llm(
            decision, symbol_insights, market_intelligence
        )
        
        assert adjusted_confidence >= 0.7  # Should increase for bullish outlook


if __name__ == "__main__":
    pytest.main([__file__])
