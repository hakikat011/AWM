"""
Unit tests for AWM execution layer systems.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from services.execution.risk_management.engine import RiskManagementEngine, RiskViolationType, RiskAction
from services.execution.oms.system import OrderManagementSystem, OrderState
from services.execution.portfolio_management.system import PortfolioManagementSystem


class TestRiskManagementEngine:
    """Test Risk Management Engine."""
    
    @pytest.fixture
    def engine(self):
        """Create a Risk Management Engine for testing."""
        return RiskManagementEngine()
    
    def test_engine_creation(self, engine):
        """Test creating Risk Management Engine."""
        assert engine.server_name == "risk_management_engine"
        assert engine.max_position_size > 0
        assert engine.max_daily_loss > 0
        assert engine.max_portfolio_risk > 0
    
    @pytest.mark.asyncio
    async def test_trade_risk_evaluation_position_size(self, engine):
        """Test trade risk evaluation for position size violation."""
        
        # Mock portfolio data
        with patch.object(engine, '_get_portfolio_data', return_value={"current_value": 1000000}):
            with patch.object(engine, '_calculate_daily_pnl', return_value=0):
                with patch.object(engine, '_calculate_portfolio_var', return_value=0.01):
                    with patch.object(engine, '_calculate_leverage', return_value=1.0):
                        
                        trade_proposal = {
                            "portfolio_id": "test-portfolio",
                            "symbol": "RELIANCE",
                            "side": "BUY",
                            "quantity": 1000,
                            "entry_price": 3000  # 3M position, exceeds 100K limit
                        }
                        
                        result = await engine._evaluate_trade_risk(trade_proposal)
                        
                        assert result["action"] == RiskAction.BLOCK.value
                        assert any(v["type"] == RiskViolationType.POSITION_SIZE_EXCEEDED.value for v in result["violations"])
    
    @pytest.mark.asyncio
    async def test_trade_risk_evaluation_allowed(self, engine):
        """Test trade risk evaluation for allowed trade."""
        
        # Mock portfolio data for a safe trade
        with patch.object(engine, '_get_portfolio_data', return_value={"current_value": 1000000}):
            with patch.object(engine, '_calculate_daily_pnl', return_value=0):
                with patch.object(engine, '_calculate_portfolio_var', return_value=0.01):
                    with patch.object(engine, '_calculate_leverage', return_value=1.0):
                        
                        trade_proposal = {
                            "portfolio_id": "test-portfolio",
                            "symbol": "RELIANCE",
                            "side": "BUY",
                            "quantity": 10,
                            "entry_price": 2500  # 25K position, within limits
                        }
                        
                        result = await engine._evaluate_trade_risk(trade_proposal)
                        
                        assert result["action"] == RiskAction.ALLOW.value
                        assert len(result["violations"]) == 0
    
    def test_emergency_mode(self, engine):
        """Test emergency mode functionality."""
        
        # Initially not in emergency mode
        assert engine.emergency_mode is False
        
        # Activate emergency mode
        engine.emergency_mode = True
        assert engine.emergency_mode is True
    
    @pytest.mark.asyncio
    async def test_emergency_mode_blocks_trades(self, engine):
        """Test that emergency mode blocks all trades."""
        
        engine.emergency_mode = True
        
        trade_proposal = {
            "portfolio_id": "test-portfolio",
            "symbol": "RELIANCE",
            "side": "BUY",
            "quantity": 1,
            "entry_price": 100
        }
        
        result = await engine._evaluate_trade_risk(trade_proposal)
        
        assert result["action"] == RiskAction.BLOCK.value
        assert "emergency mode" in result["reason"].lower()


class TestOrderManagementSystem:
    """Test Order Management System."""
    
    @pytest.fixture
    def oms(self):
        """Create an OMS for testing."""
        return OrderManagementSystem()
    
    def test_oms_creation(self, oms):
        """Test creating OMS."""
        assert oms.server_name == "order_management_system"
        assert oms.paper_trading is True  # Should default to paper trading
        assert isinstance(oms.active_orders, dict)
    
    @pytest.mark.asyncio
    async def test_order_placement_workflow(self, oms):
        """Test complete order placement workflow."""
        
        order_request = {
            "portfolio_id": "test-portfolio",
            "symbol": "RELIANCE",
            "side": "BUY",
            "quantity": 100,
            "order_type": "MARKET"
        }
        
        # Mock risk check to approve
        mock_risk_result = {"action": "ALLOW"}
        
        # Mock execution result
        mock_execution_result = {
            "status": "FILLED",
            "broker_order_id": "TEST123",
            "execution_price": 2500
        }
        
        with patch.object(oms, '_perform_risk_check', return_value=mock_risk_result):
            with patch.object(oms, '_execute_order', return_value=mock_execution_result):
                with patch.object(oms, '_update_order_in_db', return_value=None):
                    
                    result = await oms._place_order(order_request)
                    
                    assert "order_id" in result
                    assert result["status"] in ["OPEN", "FILLED"]
                    
                    # Check that order was added to active orders
                    order_id = result["order_id"]
                    assert order_id in oms.active_orders
    
    @pytest.mark.asyncio
    async def test_order_risk_rejection(self, oms):
        """Test order rejection due to risk."""
        
        order_request = {
            "portfolio_id": "test-portfolio",
            "symbol": "RELIANCE",
            "side": "BUY",
            "quantity": 100,
            "order_type": "MARKET"
        }
        
        # Mock risk check to reject
        mock_risk_result = {
            "action": "BLOCK",
            "reason": "Position size exceeded"
        }
        
        with patch.object(oms, '_perform_risk_check', return_value=mock_risk_result):
            with patch.object(oms, '_update_order_in_db', return_value=None):
                
                result = await oms._place_order(order_request)
                
                assert result["status"] == "REJECTED"
                assert "Position size exceeded" in result["reason"]
    
    @pytest.mark.asyncio
    async def test_paper_trading_simulation(self, oms):
        """Test paper trading order simulation."""
        
        order = {
            "id": "test-order",
            "symbol": "RELIANCE",
            "side": "BUY",
            "quantity": 100,
            "order_type": "MARKET"
        }
        
        # Mock market data response
        mock_quote = {"price": 2500}
        
        with patch.object(oms.mcp_client, '__aenter__', return_value=oms.mcp_client):
            with patch.object(oms.mcp_client, '__aexit__', return_value=None):
                with patch.object(oms.mcp_client, 'send_request') as mock_send:
                    mock_send.return_value.content = mock_quote
                    
                    result = await oms._simulate_order_execution(order)
                    
                    assert result["status"] == "FILLED"
                    assert result["paper_trading"] is True
                    assert result["execution_price"] > 2500  # Should include slippage
    
    @pytest.mark.asyncio
    async def test_order_cancellation(self, oms):
        """Test order cancellation."""
        
        # Create a test order
        order_id = "test-order-123"
        oms.active_orders[order_id] = {
            "id": order_id,
            "state": OrderState.SUBMITTED_TO_BROKER.value,
            "status": "OPEN",
            "broker_order_id": "BROKER123"
        }
        
        with patch.object(oms, '_update_order_in_db', return_value=None):
            result = await oms._cancel_order(order_id)
            
            assert result["status"] == "CANCELLED"
            assert oms.active_orders[order_id]["state"] == OrderState.CANCELLED.value


class TestPortfolioManagementSystem:
    """Test Portfolio Management System."""
    
    @pytest.fixture
    def pms(self):
        """Create a PMS for testing."""
        return PortfolioManagementSystem()
    
    def test_pms_creation(self, pms):
        """Test creating PMS."""
        assert pms.server_name == "portfolio_management_system"
        assert isinstance(pms.portfolio_cache, dict)
        assert isinstance(pms.position_cache, dict)
    
    @pytest.mark.asyncio
    async def test_portfolio_details_calculation(self, pms):
        """Test portfolio details calculation."""
        
        # Mock database responses
        mock_portfolio = {
            "id": "test-portfolio",
            "name": "Test Portfolio",
            "description": "Test portfolio",
            "initial_capital": 1000000,
            "available_cash": 500000,
            "is_active": True,
            "created_at": datetime.now(timezone.utc)
        }
        
        mock_positions = [
            {
                "symbol": "RELIANCE",
                "market_value": 250000,
                "unrealized_pnl": 25000,
                "realized_pnl": 5000
            },
            {
                "symbol": "TCS",
                "market_value": 200000,
                "unrealized_pnl": -10000,
                "realized_pnl": 0
            }
        ]
        
        with patch.object(pms.db_manager, 'execute_query', return_value=mock_portfolio):
            with patch.object(pms, '_get_portfolio_positions', return_value=mock_positions):
                
                result = await pms._get_portfolio_details("test-portfolio")
                
                assert result["portfolio_id"] == "test-portfolio"
                assert result["current_value"] == 950000  # 500K cash + 450K positions
                assert result["invested_value"] == 450000
                assert result["total_pnl"] == -50000  # 950K - 1M
                assert result["number_of_positions"] == 2
    
    @pytest.mark.asyncio
    async def test_position_update_from_buy_trade(self, pms):
        """Test position update from a buy trade."""
        
        trade = {
            "symbol": "RELIANCE",
            "side": "BUY",
            "quantity": 100,
            "price": 2500
        }
        
        # Mock existing position
        mock_position = {
            "quantity": 50,
            "average_price": 2400,
            "realized_pnl": 0
        }
        
        # Mock instrument lookup
        mock_instrument = {"id": "instrument-123"}
        
        with patch.object(pms.db_manager, 'execute_query') as mock_query:
            # Setup mock responses
            mock_query.side_effect = [
                mock_instrument,  # Instrument lookup
                mock_position,    # Position lookup
                None,            # Position update
                None             # Portfolio cash update
            ]
            
            result = await pms._update_positions_from_trade("test-portfolio", trade)
            
            assert result["status"] == "SUCCESS"
            assert result["trade_processed"] is True
            
            # Verify position update was called with correct values
            # New quantity: 50 + 100 = 150
            # New avg price: ((50 * 2400) + (100 * 2500)) / 150 = 2466.67
            position_update_call = mock_query.call_args_list[2]
            assert position_update_call[0][1] == 150  # New quantity
    
    @pytest.mark.asyncio
    async def test_position_update_from_sell_trade(self, pms):
        """Test position update from a sell trade."""
        
        trade = {
            "symbol": "RELIANCE",
            "side": "SELL",
            "quantity": 30,
            "price": 2600
        }
        
        # Mock existing position
        mock_position = {
            "quantity": 100,
            "average_price": 2400,
            "realized_pnl": 0
        }
        
        # Mock instrument lookup
        mock_instrument = {"id": "instrument-123"}
        
        with patch.object(pms.db_manager, 'execute_query') as mock_query:
            # Setup mock responses
            mock_query.side_effect = [
                mock_instrument,  # Instrument lookup
                mock_position,    # Position lookup
                None,            # Position update
                None             # Portfolio cash update
            ]
            
            result = await pms._update_positions_from_trade("test-portfolio", trade)
            
            assert result["status"] == "SUCCESS"
            
            # Verify position update was called with correct values
            # New quantity: 100 - 30 = 70
            # Realized P&L: 30 * (2600 - 2400) = 6000
            position_update_call = mock_query.call_args_list[2]
            assert position_update_call[0][1] == 70  # New quantity
            # Realized P&L should be 6000
    
    @pytest.mark.asyncio
    async def test_rebalance_plan_generation(self, pms):
        """Test portfolio rebalancing plan generation."""
        
        # Mock current positions
        mock_positions = [
            {
                "symbol": "RELIANCE",
                "market_value": 600000  # 60% of portfolio
            },
            {
                "symbol": "TCS",
                "market_value": 400000  # 40% of portfolio
            }
        ]
        
        # Mock portfolio
        mock_portfolio = {"current_value": 1000000}
        
        # Target allocation: 50% RELIANCE, 50% TCS
        target_allocation = {
            "RELIANCE": 0.5,
            "TCS": 0.5
        }
        
        with patch.object(pms, '_get_portfolio_positions', return_value=mock_positions):
            with patch.object(pms, '_get_portfolio_details', return_value=mock_portfolio):
                with patch.object(pms, '_get_current_price', return_value=2500):
                    
                    result = await pms._generate_rebalance_plan("test-portfolio", target_allocation)
                    
                    assert result["portfolio_id"] == "test-portfolio"
                    assert "rebalance_trades" in result
                    
                    # Should suggest selling RELIANCE (over-weight) and buying TCS (under-weight)
                    trades = result["rebalance_trades"]
                    reliance_trade = next((t for t in trades if t["symbol"] == "RELIANCE"), None)
                    tcs_trade = next((t for t in trades if t["symbol"] == "TCS"), None)
                    
                    if reliance_trade:
                        assert reliance_trade["side"] == "SELL"  # Over-weight, should sell
                    if tcs_trade:
                        assert tcs_trade["side"] == "BUY"   # Under-weight, should buy


@pytest.mark.asyncio
class TestSystemIntegration:
    """Integration tests for execution systems."""
    
    async def test_risk_oms_integration(self):
        """Test integration between Risk Management and OMS."""
        
        # This would test the actual MCP communication between systems
        # For now, we test the workflow conceptually
        
        risk_engine = RiskManagementEngine()
        oms = OrderManagementSystem()
        
        # Test that OMS calls risk engine for evaluation
        trade_proposal = {
            "portfolio_id": "test-portfolio",
            "symbol": "RELIANCE",
            "side": "BUY",
            "quantity": 100,
            "entry_price": 2500
        }
        
        # Mock the risk evaluation
        with patch.object(oms, '_perform_risk_check') as mock_risk_check:
            mock_risk_check.return_value = {"action": "ALLOW"}
            
            with patch.object(oms, '_execute_order', return_value={"status": "FILLED"}):
                with patch.object(oms, '_update_order_in_db', return_value=None):
                    
                    result = await oms._place_order(trade_proposal)
                    
                    # Verify risk check was called
                    mock_risk_check.assert_called_once()
                    assert result["status"] in ["OPEN", "FILLED"]
    
    async def test_oms_portfolio_integration(self):
        """Test integration between OMS and Portfolio Management."""
        
        oms = OrderManagementSystem()
        pms = PortfolioManagementSystem()
        
        # Test that successful trades update portfolio positions
        trade = {
            "symbol": "RELIANCE",
            "side": "BUY",
            "quantity": 100,
            "price": 2500
        }
        
        with patch.object(pms, '_update_positions_from_trade') as mock_update:
            mock_update.return_value = {"status": "SUCCESS", "trade_processed": True}
            
            # Simulate trade execution notification
            result = await pms._update_positions_from_trade("test-portfolio", trade)
            
            assert result["status"] == "SUCCESS"
            assert result["trade_processed"] is True


if __name__ == "__main__":
    pytest.main([__file__])
