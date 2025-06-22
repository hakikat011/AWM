"""
Unit tests for AWM system agents.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from shared.agents.base_agent import BaseAgent, AgentTask, AgentStatus
from services.agents.market_analysis.agent import MarketAnalysisAgent
from services.agents.risk_assessment.agent import RiskAssessmentAgent
from services.agents.strategy_optimization.agent import StrategyOptimizationAgent
from services.agents.trade_execution.agent import TradeExecutionAgent


class TestBaseAgent:
    """Test base agent functionality."""
    
    class MockAgent(BaseAgent):
        """Mock agent for testing."""
        
        async def initialize(self):
            pass
        
        async def cleanup(self):
            pass
        
        async def process_task(self, task_type: str, parameters: dict) -> dict:
            if task_type == "test_task":
                return {"result": "success", "parameters": parameters}
            else:
                raise ValueError(f"Unknown task type: {task_type}")
    
    def test_agent_creation(self):
        """Test creating a base agent."""
        agent = self.MockAgent("test_agent")
        
        assert agent.agent_name == "test_agent"
        assert agent.status == AgentStatus.IDLE
        assert agent.metrics.agent_name == "test_agent"
    
    @pytest.mark.asyncio
    async def test_task_creation(self):
        """Test creating and adding tasks."""
        agent = self.MockAgent("test_agent")
        
        task_id = await agent.add_task("test_task", {"param1": "value1"})
        
        assert task_id is not None
        assert agent.task_queue.qsize() == 1
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test agent health check."""
        agent = self.MockAgent("test_agent")
        
        health = await agent.health_check()
        
        assert health["agent_name"] == "test_agent"
        assert health["status"] == "IDLE"
        assert "metrics" in health
        assert "timestamp" in health


class TestMarketAnalysisAgent:
    """Test Market Analysis Agent."""
    
    @pytest.fixture
    def agent(self):
        """Create a Market Analysis Agent for testing."""
        return MarketAnalysisAgent({"lookback_days": 30})
    
    def test_agent_creation(self, agent):
        """Test creating Market Analysis Agent."""
        assert agent.agent_name == "market_analysis_agent"
        assert agent.lookback_days == 30
        assert len(agent.indicators) > 0
    
    @pytest.mark.asyncio
    async def test_market_data_summary(self, agent):
        """Test market data summarization."""
        market_data = [
            {"timestamp": "2024-01-01T00:00:00Z", "close": 100, "volume": 1000},
            {"timestamp": "2024-01-02T00:00:00Z", "close": 105, "volume": 1200},
            {"timestamp": "2024-01-03T00:00:00Z", "close": 95, "volume": 800}
        ]
        
        summary = agent._summarize_market_data(market_data)
        
        assert summary["latest_price"] == 100
        assert summary["high_52w"] == 105
        assert summary["low_52w"] == 95
        assert summary["data_points"] == 3
    
    def test_price_change_calculation(self, agent):
        """Test price change calculation."""
        market_data = [
            {"close": 100},
            {"close": 105},
            {"close": 95},
            {"close": 110}
        ]
        
        change_1d = agent._calculate_price_change(market_data, 1)
        change_2d = agent._calculate_price_change(market_data, 2)
        
        assert abs(change_1d - (-0.047619)) < 0.001  # (100-105)/105
        assert abs(change_2d - 0.052632) < 0.001     # (100-95)/95
    
    def test_volume_trend_calculation(self, agent):
        """Test volume trend calculation."""
        # Increasing volume
        market_data = [
            {"volume": 1000}, {"volume": 1100}, {"volume": 1200}, {"volume": 1300}, {"volume": 1400},
            {"volume": 800}, {"volume": 850}, {"volume": 900}, {"volume": 950}, {"volume": 1000}
        ]
        
        trend = agent._calculate_volume_trend(market_data)
        assert trend == "increasing"
        
        # Stable volume
        market_data = [
            {"volume": 1000}, {"volume": 1050}, {"volume": 950}, {"volume": 1100}, {"volume": 900},
            {"volume": 1000}, {"volume": 1050}, {"volume": 950}, {"volume": 1100}, {"volume": 900}
        ]
        
        trend = agent._calculate_volume_trend(market_data)
        assert trend == "stable"


class TestRiskAssessmentAgent:
    """Test Risk Assessment Agent."""
    
    @pytest.fixture
    def agent(self):
        """Create a Risk Assessment Agent for testing."""
        return RiskAssessmentAgent({"confidence_level": 0.95})
    
    def test_agent_creation(self, agent):
        """Test creating Risk Assessment Agent."""
        assert agent.agent_name == "risk_assessment_agent"
        assert agent.confidence_level == 0.95
        assert agent.max_position_size > 0
    
    def test_correlation_estimation(self, agent):
        """Test correlation estimation between positions."""
        pos1 = {"sector": "Technology", "exchange": "NSE"}
        pos2 = {"sector": "Technology", "exchange": "NSE"}
        pos3 = {"sector": "Banking", "exchange": "NSE"}
        pos4 = {"sector": "Banking", "exchange": "BSE"}
        
        # Same sector should have higher correlation
        corr1 = agent._estimate_correlation(pos1, pos2)
        assert corr1 == 0.7
        
        # Same exchange, different sector
        corr2 = agent._estimate_correlation(pos1, pos3)
        assert corr2 == 0.4
        
        # Different sector and exchange
        corr3 = agent._estimate_correlation(pos3, pos4)
        assert corr3 == 0.2
    
    def test_fallback_strategy(self, agent):
        """Test fallback execution strategy."""
        trade_proposal = {"quantity": 100, "entry_price": 50}
        market_conditions = {"liquidity": "HIGH", "volatility": 0.01}
        
        strategy = agent._fallback_strategy(trade_proposal, market_conditions)
        
        assert "strategy" in strategy
        assert "reasoning" in strategy
        assert strategy["confidence"] == 0.6


class TestStrategyOptimizationAgent:
    """Test Strategy Optimization Agent."""
    
    @pytest.fixture
    def agent(self):
        """Create a Strategy Optimization Agent for testing."""
        return StrategyOptimizationAgent({"max_iterations": 50})
    
    def test_agent_creation(self, agent):
        """Test creating Strategy Optimization Agent."""
        assert agent.agent_name == "strategy_optimization_agent"
        assert agent.max_iterations == 50
        assert agent.initial_capital == 100000
    
    def test_performance_metrics_calculation(self, agent):
        """Test performance metrics calculation."""
        backtest_results = {
            "trades": [
                {"pnl": 0.05},   # 5% gain
                {"pnl": -0.02},  # 2% loss
                {"pnl": 0.03},   # 3% gain
                {"pnl": -0.01},  # 1% loss
                {"pnl": 0.04}    # 4% gain
            ]
        }
        
        metrics = agent._calculate_performance_metrics(backtest_results)
        
        assert metrics["total_trades"] == 5
        assert metrics["winning_trades"] == 3
        assert metrics["losing_trades"] == 2
        assert metrics["win_rate"] == 0.6
        assert abs(metrics["total_return"] - 0.09) < 0.001
        assert metrics["profit_factor"] > 1  # Profitable strategy
    
    def test_empty_trades_handling(self, agent):
        """Test handling of empty trade results."""
        backtest_results = {"trades": []}
        
        metrics = agent._calculate_performance_metrics(backtest_results)
        
        assert "error" in metrics
        assert metrics["error"] == "No trades executed"


class TestTradeExecutionAgent:
    """Test Trade Execution Agent."""
    
    @pytest.fixture
    def agent(self):
        """Create a Trade Execution Agent for testing."""
        return TradeExecutionAgent({"order_timeout": 300})
    
    def test_agent_creation(self, agent):
        """Test creating Trade Execution Agent."""
        assert agent.agent_name == "trade_execution_agent"
        assert agent.default_timeout == 300
        assert agent.paper_trading is True  # Should default to paper trading
    
    @pytest.mark.asyncio
    async def test_trade_validation(self, agent):
        """Test trade proposal validation."""
        # Valid trade proposal
        valid_proposal = {
            "symbol": "RELIANCE",
            "side": "BUY",
            "quantity": 100,
            "entry_price": 2500,
            "portfolio_id": "test-portfolio"
        }
        
        # Mock portfolio balance check
        with patch.object(agent, '_check_portfolio_balance', return_value={"sufficient": True}):
            result = await agent._validate_trade_proposal(valid_proposal)
            assert result["valid"] is True
        
        # Invalid trade proposal (missing fields)
        invalid_proposal = {
            "symbol": "RELIANCE",
            "side": "BUY"
            # Missing quantity and portfolio_id
        }
        
        result = await agent._validate_trade_proposal(invalid_proposal)
        assert result["valid"] is False
        assert "Missing required field" in result["reason"]
        
        # Invalid quantity
        zero_quantity_proposal = {
            "symbol": "RELIANCE",
            "side": "BUY",
            "quantity": 0,
            "entry_price": 2500,
            "portfolio_id": "test-portfolio"
        }
        
        result = await agent._validate_trade_proposal(zero_quantity_proposal)
        assert result["valid"] is False
        assert "Quantity must be positive" in result["reason"]
    
    @pytest.mark.asyncio
    async def test_market_order_simulation(self, agent):
        """Test market order simulation in paper trading mode."""
        trade_proposal = {
            "symbol": "RELIANCE",
            "side": "BUY",
            "quantity": 100,
            "entry_price": 2500
        }
        
        # Mock market data response
        mock_quote = {"price": 2505}
        
        with patch.object(agent, 'call_mcp_server', return_value=mock_quote):
            result = await agent._simulate_market_order(trade_proposal)
            
            assert result["status"] == "FILLED"
            assert result["paper_trading"] is True
            assert result["quantity"] == 100
            assert result["execution_price"] > 2505  # Should include slippage
    
    def test_fallback_strategy_selection(self, agent):
        """Test fallback strategy selection."""
        # Small order with high liquidity -> MARKET
        trade_proposal = {"quantity": 10, "entry_price": 100}
        market_conditions = {"liquidity": "HIGH", "volatility": 0.01}
        
        strategy = agent._fallback_strategy(trade_proposal, market_conditions)
        assert strategy["strategy"].value == "MARKET"
        
        # High volatility -> LIMIT
        market_conditions = {"liquidity": "NORMAL", "volatility": 0.03}
        
        strategy = agent._fallback_strategy(trade_proposal, market_conditions)
        assert strategy["strategy"].value == "LIMIT"


@pytest.mark.asyncio
class TestAgentIntegration:
    """Integration tests for agent interactions."""
    
    async def test_agent_task_processing(self):
        """Test that agents can process tasks correctly."""
        
        class TestAgent(BaseAgent):
            async def initialize(self):
                pass
            
            async def cleanup(self):
                pass
            
            async def process_task(self, task_type: str, parameters: dict) -> dict:
                return {"task_type": task_type, "processed": True}
        
        agent = TestAgent("test_agent")
        
        # Add a task
        task_id = await agent.add_task("test_task", {"param": "value"})
        
        # Process the task manually (without starting the full agent)
        result = await agent.process_task("test_task", {"param": "value"})
        
        assert result["task_type"] == "test_task"
        assert result["processed"] is True


if __name__ == "__main__":
    pytest.main([__file__])
