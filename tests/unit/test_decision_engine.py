"""
Unit tests for decision engine system.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from services.mcp_servers.decision_engine.server import DecisionEngineServer


class TestDecisionEngine:
    """Test decision engine system."""
    
    @pytest.fixture
    def server(self):
        """Create decision engine server instance."""
        return DecisionEngineServer()
    
    @pytest.fixture
    def sample_signals(self):
        """Sample trading signals."""
        return {
            "signals": {
                "combined_signals": {
                    "signal": "BUY",
                    "confidence": 0.8,
                    "reason": "Strong buy consensus from multiple strategies",
                    "analysis": {
                        "buy_strength": 0.7,
                        "sell_strength": 0.2,
                        "buy_confidence": 0.8,
                        "sell_confidence": 0.6
                    }
                }
            }
        }
    
    @pytest.fixture
    def sample_portfolio_state(self):
        """Sample portfolio state."""
        return {
            "total_value": 1000000,
            "cash": 500000,
            "positions": [
                {
                    "symbol": "RELIANCE",
                    "quantity": 100,
                    "average_price": 2400.0,
                    "current_value": 250000
                }
            ]
        }
    
    @pytest.fixture
    def sample_risk_assessment(self):
        """Sample risk assessment."""
        return {
            "risk_metrics": {
                "var_1d": 0.015,
                "volatility": 0.18,
                "max_drawdown": -0.05
            },
            "risk_level": "MEDIUM",
            "violations": []
        }
    
    @pytest.fixture
    def sample_market_data(self):
        """Sample market data."""
        return {
            "symbol": "RELIANCE",
            "price": 2500.0,
            "close": 2500.0,
            "bid": 2499.5,
            "ask": 2500.5
        }
    
    def test_decision_configs(self, server):
        """Test decision engine configurations."""
        configs = server.decision_configs
        
        # Check that all expected configs exist
        expected_configs = ["conservative", "moderate", "aggressive"]
        for config_name in expected_configs:
            assert config_name in configs
            config = configs[config_name]
            
            # Check required fields
            required_fields = [
                "min_signal_confidence", "max_position_size_pct", "max_portfolio_risk",
                "require_consensus", "stop_loss_pct", "take_profit_pct", "paper_trading"
            ]
            for field in required_fields:
                assert field in config
            
            # Check data types and ranges
            assert 0 <= config["min_signal_confidence"] <= 1
            assert 0 < config["max_position_size_pct"] <= 1
            assert 0 < config["max_portfolio_risk"] <= 1
            assert isinstance(config["require_consensus"], bool)
            assert 0 < config["stop_loss_pct"] <= 1
            assert 0 < config["take_profit_pct"] <= 1
            assert isinstance(config["paper_trading"], bool)
        
        # Check that conservative is more restrictive than aggressive
        conservative = configs["conservative"]
        aggressive = configs["aggressive"]
        
        assert conservative["min_signal_confidence"] >= aggressive["min_signal_confidence"]
        assert conservative["max_position_size_pct"] <= aggressive["max_position_size_pct"]
        assert conservative["max_portfolio_risk"] <= aggressive["max_portfolio_risk"]
    
    @pytest.mark.asyncio
    async def test_synthesize_decision_buy_signal(self, server, sample_signals, sample_portfolio_state, 
                                                sample_risk_assessment, sample_market_data):
        """Test decision synthesis for buy signal."""
        config = server.decision_configs["moderate"]
        
        decision = await server._synthesize_decision(
            "RELIANCE", sample_signals, sample_portfolio_state, 
            sample_risk_assessment, sample_market_data, config
        )
        
        assert isinstance(decision, dict)
        
        # Check required fields
        required_fields = ["action", "confidence", "reason"]
        for field in required_fields:
            assert field in decision
        
        # Should be a buy decision since signal confidence is high
        assert decision["action"] in ["BUY", "NO_ACTION"]  # Could be NO_ACTION if already at max position
        assert 0 <= decision["confidence"] <= 1
        assert isinstance(decision["reason"], str)
        
        # If it's a buy decision, check execution details
        if decision["action"] == "BUY":
            execution_fields = [
                "symbol", "quantity", "order_type", "estimated_price", 
                "estimated_value", "stop_loss", "take_profit", "paper_trading"
            ]
            for field in execution_fields:
                assert field in decision
            
            assert decision["symbol"] == "RELIANCE"
            assert decision["quantity"] > 0
            assert decision["estimated_price"] > 0
            assert decision["estimated_value"] > 0
            assert decision["stop_loss"] < decision["estimated_price"]
            assert decision["take_profit"] > decision["estimated_price"]
            assert decision["paper_trading"] is True
    
    @pytest.mark.asyncio
    async def test_synthesize_decision_low_confidence(self, server, sample_portfolio_state, 
                                                    sample_risk_assessment, sample_market_data):
        """Test decision synthesis with low confidence signal."""
        # Create low confidence signal
        low_confidence_signals = {
            "signals": {
                "combined_signals": {
                    "signal": "BUY",
                    "confidence": 0.5,  # Below moderate threshold of 0.7
                    "reason": "Weak buy signal"
                }
            }
        }
        
        config = server.decision_configs["moderate"]
        
        decision = await server._synthesize_decision(
            "RELIANCE", low_confidence_signals, sample_portfolio_state, 
            sample_risk_assessment, sample_market_data, config
        )
        
        # Should be no action due to low confidence
        assert decision["action"] == "NO_ACTION"
        assert "confidence" in decision["reason"] or "threshold" in decision["reason"]
    
    @pytest.mark.asyncio
    async def test_synthesize_decision_high_risk(self, server, sample_signals, sample_portfolio_state, 
                                               sample_market_data):
        """Test decision synthesis with high portfolio risk."""
        # Create high risk assessment
        high_risk_assessment = {
            "risk_metrics": {
                "var_1d": 0.06,  # Above moderate threshold of 0.03
                "volatility": 0.25
            },
            "risk_level": "HIGH"
        }
        
        config = server.decision_configs["moderate"]
        
        decision = await server._synthesize_decision(
            "RELIANCE", sample_signals, sample_portfolio_state, 
            high_risk_assessment, sample_market_data, config
        )
        
        # Should be no action due to high risk
        assert decision["action"] == "NO_ACTION"
        assert "risk" in decision["reason"].lower()
    
    @pytest.mark.asyncio
    async def test_synthesize_decision_sell_signal(self, server, sample_portfolio_state, 
                                                 sample_risk_assessment, sample_market_data):
        """Test decision synthesis for sell signal."""
        # Create sell signal
        sell_signals = {
            "signals": {
                "combined_signals": {
                    "signal": "SELL",
                    "confidence": 0.8,
                    "reason": "Strong sell signal"
                }
            }
        }
        
        config = server.decision_configs["moderate"]
        
        decision = await server._synthesize_decision(
            "RELIANCE", sell_signals, sample_portfolio_state, 
            sample_risk_assessment, sample_market_data, config
        )
        
        # Should be a sell decision since we have position
        assert decision["action"] == "SELL"
        assert decision["symbol"] == "RELIANCE"
        assert decision["quantity"] > 0
        assert decision["quantity"] <= 100  # Can't sell more than we have
    
    @pytest.mark.asyncio
    async def test_synthesize_decision_sell_no_position(self, server, sample_risk_assessment, sample_market_data):
        """Test decision synthesis for sell signal with no position."""
        # Create sell signal
        sell_signals = {
            "signals": {
                "combined_signals": {
                    "signal": "SELL",
                    "confidence": 0.8,
                    "reason": "Strong sell signal"
                }
            }
        }
        
        # Portfolio with no position in RELIANCE
        no_position_portfolio = {
            "total_value": 1000000,
            "cash": 1000000,
            "positions": []
        }
        
        config = server.decision_configs["moderate"]
        
        decision = await server._synthesize_decision(
            "RELIANCE", sell_signals, no_position_portfolio, 
            sample_risk_assessment, sample_market_data, config
        )
        
        # Should be no action since no position to sell
        assert decision["action"] == "NO_ACTION"
        assert "No position to sell" in decision["reason"]
    
    @pytest.mark.asyncio
    async def test_evaluate_trade_proposal_approved(self, server):
        """Test trade proposal evaluation - approved case."""
        trade_proposal = {
            "symbol": "RELIANCE",
            "side": "BUY",
            "quantity": 10,
            "price": 2500.0,
            "portfolio_id": "test-portfolio"
        }
        
        # Mock portfolio state and risk assessment
        async def mock_get_portfolio_state(portfolio_id):
            return {"total_value": 1000000}
        
        async def mock_assess_portfolio_risk(portfolio_id):
            return {"risk_metrics": {"var_1d": 0.01}}
        
        server._get_portfolio_state = mock_get_portfolio_state
        server._assess_portfolio_risk = mock_assess_portfolio_risk
        
        evaluation = await server._evaluate_trade_proposal(trade_proposal, "moderate")
        
        assert isinstance(evaluation, dict)
        assert evaluation["approved"] is True
        assert "risk_score" in evaluation
        assert 0 <= evaluation["risk_score"] <= 1
        assert "position_size_pct" in evaluation
        assert "estimated_portfolio_impact" in evaluation
    
    @pytest.mark.asyncio
    async def test_evaluate_trade_proposal_rejected_size(self, server):
        """Test trade proposal evaluation - rejected due to position size."""
        trade_proposal = {
            "symbol": "RELIANCE",
            "side": "BUY",
            "quantity": 1000,  # Large quantity
            "price": 2500.0,
            "portfolio_id": "test-portfolio"
        }
        
        # Mock portfolio state
        async def mock_get_portfolio_state(portfolio_id):
            return {"total_value": 1000000}
        
        async def mock_assess_portfolio_risk(portfolio_id):
            return {"risk_metrics": {"var_1d": 0.01}}
        
        server._get_portfolio_state = mock_get_portfolio_state
        server._assess_portfolio_risk = mock_assess_portfolio_risk
        
        evaluation = await server._evaluate_trade_proposal(trade_proposal, "conservative")
        
        assert evaluation["approved"] is False
        assert "Position size" in evaluation["reason"]
        assert "exceeds limit" in evaluation["reason"]
    
    @pytest.mark.asyncio
    async def test_update_decision_config(self, server):
        """Test decision configuration updates."""
        config_updates = {
            "min_signal_confidence": 0.85,
            "max_position_size_pct": 0.06
        }
        
        updated_config = await server._update_decision_config("test_config", config_updates)
        
        assert isinstance(updated_config, dict)
        assert updated_config["min_signal_confidence"] == 0.85
        assert updated_config["max_position_size_pct"] == 0.06
        
        # Should have all required fields
        required_fields = [
            "min_signal_confidence", "max_position_size_pct", "max_portfolio_risk",
            "stop_loss_pct", "take_profit_pct", "paper_trading"
        ]
        for field in required_fields:
            assert field in updated_config
    
    @pytest.mark.asyncio
    async def test_update_decision_config_validation(self, server):
        """Test decision configuration validation."""
        # Test invalid confidence value
        with pytest.raises(ValueError, match="min_signal_confidence must be between 0 and 1"):
            await server._update_decision_config("test_config", {"min_signal_confidence": 1.5})
        
        # Test invalid position size
        with pytest.raises(ValueError, match="max_position_size_pct must be between 0 and 1"):
            await server._update_decision_config("test_config", {"max_position_size_pct": 0})
        
        # Test invalid portfolio risk
        with pytest.raises(ValueError, match="max_portfolio_risk must be between 0 and 1"):
            await server._update_decision_config("test_config", {"max_portfolio_risk": 2.0})
    
    def test_server_urls(self, server):
        """Test server URL configuration."""
        urls = server.server_urls
        
        expected_servers = [
            "signal_generation", "risk_management", "portfolio_management", "market_data"
        ]
        
        for server_name in expected_servers:
            assert server_name in urls
            assert isinstance(urls[server_name], str)
            assert urls[server_name].startswith("http")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
