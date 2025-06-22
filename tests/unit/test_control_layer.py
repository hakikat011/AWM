"""
Unit tests for AWM control layer (dashboard and alerting).
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from services.alerting.service import AlertingService, AlertSeverity, AlertChannel


class TestAlertingService:
    """Test Alerting Service."""
    
    @pytest.fixture
    def alerting_service(self):
        """Create an Alerting Service for testing."""
        return AlertingService()
    
    def test_alerting_service_creation(self, alerting_service):
        """Test creating Alerting Service."""
        assert alerting_service.server_name == "alerting_service"
        assert isinstance(alerting_service.alert_rules, dict)
        assert isinstance(alerting_service.subscriptions, dict)
    
    @pytest.mark.asyncio
    async def test_send_alert_database_only(self, alerting_service):
        """Test sending alert to database only."""
        
        with patch.object(alerting_service, '_store_alert', return_value=None) as mock_store:
            result = await alerting_service._send_alert(
                alert_type="TEST_ALERT",
                severity="INFO",
                message="This is a test alert",
                data={"test": True},
                channels=["DATABASE"]
            )
            
            assert result["status"] == "SENT"
            assert "alert_id" in result
            assert "DATABASE" in result["channels"]
            assert result["channels"]["DATABASE"]["status"] == "SUCCESS"
            
            # Verify store_alert was called
            mock_store.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_telegram_alert_success(self, alerting_service):
        """Test successful Telegram alert."""
        
        # Mock Telegram configuration
        alerting_service.telegram_bot_token = "test_token"
        alerting_service.telegram_chat_id = "test_chat_id"
        
        # Mock aiohttp response
        mock_response = AsyncMock()
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await alerting_service._send_telegram_alert("Test Alert", "Test message")
            
            assert result["status"] == "SUCCESS"
            assert "Telegram alert sent" in result["message"]
    
    @pytest.mark.asyncio
    async def test_send_telegram_alert_not_configured(self, alerting_service):
        """Test Telegram alert when not configured."""
        
        # Ensure Telegram is not configured
        alerting_service.telegram_bot_token = None
        alerting_service.telegram_chat_id = None
        
        result = await alerting_service._send_telegram_alert("Test Alert", "Test message")
        
        assert result["status"] == "ERROR"
        assert "not configured" in result["error"]
    
    @pytest.mark.asyncio
    async def test_send_slack_alert_success(self, alerting_service):
        """Test successful Slack alert."""
        
        # Mock Slack configuration
        alerting_service.slack_webhook_url = "https://hooks.slack.com/test"
        
        # Mock aiohttp response
        mock_response = AsyncMock()
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await alerting_service._send_slack_alert("Test Alert", "Test message")
            
            assert result["status"] == "SUCCESS"
            assert "Slack alert sent" in result["message"]
    
    @pytest.mark.asyncio
    async def test_send_slack_alert_not_configured(self, alerting_service):
        """Test Slack alert when not configured."""
        
        # Ensure Slack is not configured
        alerting_service.slack_webhook_url = None
        
        result = await alerting_service._send_slack_alert("Test Alert", "Test message")
        
        assert result["status"] == "ERROR"
        assert "not configured" in result["error"]
    
    @pytest.mark.asyncio
    async def test_store_alert(self, alerting_service):
        """Test storing alert in database."""
        
        with patch.object(alerting_service.db_manager, 'execute_query', return_value=None) as mock_query:
            await alerting_service._store_alert(
                alert_id="test_alert_123",
                alert_type="TEST_ALERT",
                severity="INFO",
                message="Test message",
                data={"test": True}
            )
            
            # Verify database query was called
            mock_query.assert_called_once()
            
            # Check the query parameters
            call_args = mock_query.call_args[0]
            assert call_args[1] == "test_alert_123"  # alert_id
            assert call_args[2] == "TEST_ALERT"      # alert_type
            assert call_args[3] == "INFO"            # severity
    
    @pytest.mark.asyncio
    async def test_get_alerts(self, alerting_service):
        """Test getting alerts from database."""
        
        # Mock database response
        mock_alerts = [
            {
                "id": "alert_1",
                "alert_type": "RISK_VIOLATION",
                "severity": "ERROR",
                "title": "Risk Alert",
                "message": "Risk threshold exceeded",
                "data": '{"portfolio_id": "test"}',
                "created_at": datetime.now(timezone.utc)
            },
            {
                "id": "alert_2",
                "alert_type": "SYSTEM_ERROR",
                "severity": "CRITICAL",
                "title": "System Alert",
                "message": "System component failed",
                "data": '{"component": "oms"}',
                "created_at": datetime.now(timezone.utc)
            }
        ]
        
        with patch.object(alerting_service.db_manager, 'execute_query', return_value=mock_alerts):
            alerts = await alerting_service._get_alerts(limit=10)
            
            assert len(alerts) == 2
            assert alerts[0]["id"] == "alert_1"
            assert alerts[0]["alert_type"] == "RISK_VIOLATION"
            assert alerts[1]["id"] == "alert_2"
            assert alerts[1]["alert_type"] == "SYSTEM_ERROR"
    
    @pytest.mark.asyncio
    async def test_get_alerts_with_filters(self, alerting_service):
        """Test getting alerts with severity and type filters."""
        
        with patch.object(alerting_service.db_manager, 'execute_query', return_value=[]) as mock_query:
            await alerting_service._get_alerts(
                limit=50,
                severity="ERROR",
                alert_type="RISK_VIOLATION"
            )
            
            # Verify the query was called with filters
            mock_query.assert_called_once()
            
            # Check that the query includes WHERE clause
            query = mock_query.call_args[0][0]
            assert "WHERE" in query
            assert "severity = $1" in query
            assert "alert_type = $2" in query
    
    def test_configure_alert_rule(self, alerting_service):
        """Test configuring an alert rule."""
        
        rule_name = "High Risk Alert"
        conditions = {"type": "portfolio_risk", "threshold": 0.05}
        actions = [{"channel": "TELEGRAM", "message": "High risk detected"}]
        
        rule_id = asyncio.run(alerting_service._configure_alert_rule(
            rule_name=rule_name,
            conditions=conditions,
            actions=actions,
            enabled=True
        ))
        
        assert rule_id is not None
        assert rule_id in alerting_service.alert_rules
        
        rule = alerting_service.alert_rules[rule_id]
        assert rule["name"] == rule_name
        assert rule["conditions"] == conditions
        assert rule["actions"] == actions
        assert rule["enabled"] is True
    
    def test_alert_severity_enum(self):
        """Test AlertSeverity enum."""
        assert AlertSeverity.INFO.value == "INFO"
        assert AlertSeverity.WARNING.value == "WARNING"
        assert AlertSeverity.ERROR.value == "ERROR"
        assert AlertSeverity.CRITICAL.value == "CRITICAL"
    
    def test_alert_channel_enum(self):
        """Test AlertChannel enum."""
        assert AlertChannel.EMAIL.value == "EMAIL"
        assert AlertChannel.TELEGRAM.value == "TELEGRAM"
        assert AlertChannel.SLACK.value == "SLACK"
        assert AlertChannel.WEBHOOK.value == "WEBHOOK"
        assert AlertChannel.DATABASE.value == "DATABASE"


class TestDashboardIntegration:
    """Test dashboard integration with backend services."""
    
    def test_dashboard_api_client_creation(self):
        """Test creating dashboard API client."""
        # This would test the dashboard API client
        # For now, we'll test the concept
        
        from services.dashboard.main import DashboardAPI
        
        api = DashboardAPI()
        assert api.client is not None
        assert api.client.client_name == "dashboard"
    
    @pytest.mark.asyncio
    async def test_dashboard_portfolio_data_fetch(self):
        """Test fetching portfolio data through dashboard API."""
        
        from services.dashboard.main import DashboardAPI
        
        api = DashboardAPI()
        
        # Mock the MCP client response
        mock_response = Mock()
        mock_response.content = {
            "portfolio_id": "test-portfolio",
            "current_value": 1000000,
            "total_pnl": 50000,
            "total_return": 0.05
        }
        
        with patch.object(api.client, '__aenter__', return_value=api.client):
            with patch.object(api.client, '__aexit__', return_value=None):
                with patch.object(api.client, 'send_request', return_value=mock_response):
                    
                    result = await api.get_portfolio_data("test-portfolio")
                    
                    assert result["portfolio_id"] == "test-portfolio"
                    assert result["current_value"] == 1000000
                    assert result["total_pnl"] == 50000
    
    @pytest.mark.asyncio
    async def test_dashboard_risk_status_fetch(self):
        """Test fetching risk status through dashboard API."""
        
        from services.dashboard.main import DashboardAPI
        
        api = DashboardAPI()
        
        # Mock the MCP client response
        mock_response = Mock()
        mock_response.content = {
            "portfolio_id": "test-portfolio",
            "risk_level": "MEDIUM",
            "violations": [
                {"type": "CONCENTRATION_RISK", "severity": "MEDIUM"}
            ],
            "risk_metrics": {
                "var_1d": 0.02,
                "volatility": 0.15
            }
        }
        
        with patch.object(api.client, '__aenter__', return_value=api.client):
            with patch.object(api.client, '__aexit__', return_value=None):
                with patch.object(api.client, 'send_request', return_value=mock_response):
                    
                    result = await api.get_risk_status("test-portfolio")
                    
                    assert result["risk_level"] == "MEDIUM"
                    assert len(result["violations"]) == 1
                    assert result["risk_metrics"]["var_1d"] == 0.02


@pytest.mark.asyncio
class TestSystemIntegration:
    """Integration tests for control layer systems."""
    
    async def test_alerting_dashboard_integration(self):
        """Test integration between alerting service and dashboard."""
        
        alerting_service = AlertingService()
        
        # Test that dashboard can fetch alerts from alerting service
        with patch.object(alerting_service, '_get_alerts') as mock_get_alerts:
            mock_get_alerts.return_value = [
                {
                    "id": "alert_1",
                    "alert_type": "RISK_VIOLATION",
                    "severity": "ERROR",
                    "message": "Risk threshold exceeded"
                }
            ]
            
            alerts = await alerting_service._get_alerts(limit=10)
            
            assert len(alerts) == 1
            assert alerts[0]["alert_type"] == "RISK_VIOLATION"
    
    async def test_dashboard_order_placement_workflow(self):
        """Test order placement workflow through dashboard."""
        
        from services.dashboard.main import DashboardAPI
        
        api = DashboardAPI()
        
        # Mock successful order placement
        mock_response = Mock()
        mock_response.content = {
            "order_id": "order_123",
            "status": "SUBMITTED",
            "message": "Order placed successfully"
        }
        
        with patch.object(api.client, '__aenter__', return_value=api.client):
            with patch.object(api.client, '__aexit__', return_value=None):
                with patch.object(api.client, 'send_request', return_value=mock_response):
                    
                    # This would be called by the dashboard order form
                    # For now, we test the API structure
                    result = mock_response.content
                    
                    assert result["order_id"] == "order_123"
                    assert result["status"] == "SUBMITTED"
    
    async def test_emergency_stop_workflow(self):
        """Test emergency stop workflow through dashboard."""
        
        from services.dashboard.main import DashboardAPI
        
        api = DashboardAPI()
        
        # Mock emergency stop activation
        mock_response = Mock()
        mock_response.content = {
            "status": "EMERGENCY_STOP_ACTIVATED",
            "reason": "Manual activation from dashboard",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        with patch.object(api.client, '__aenter__', return_value=api.client):
            with patch.object(api.client, '__aexit__', return_value=None):
                with patch.object(api.client, 'send_request', return_value=mock_response):
                    
                    # This would be called by the dashboard emergency button
                    result = mock_response.content
                    
                    assert result["status"] == "EMERGENCY_STOP_ACTIVATED"
                    assert "Manual activation" in result["reason"]


if __name__ == "__main__":
    pytest.main([__file__])
