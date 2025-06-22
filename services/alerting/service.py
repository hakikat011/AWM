"""
Alerting Service for AWM system.
Handles notifications and alerts for various system events.
"""

import asyncio
import logging
import json
import os
import aiohttp
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from enum import Enum

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from shared.mcp_client.server import MCPServer, validate_required_fields, sanitize_input
from shared.database.connection import init_database, close_database, db_manager

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertChannel(Enum):
    EMAIL = "EMAIL"
    TELEGRAM = "TELEGRAM"
    SLACK = "SLACK"
    WEBHOOK = "WEBHOOK"
    DATABASE = "DATABASE"


class AlertingService(MCPServer):
    """Comprehensive alerting and notification service."""
    
    def __init__(self):
        host = os.getenv("ALERTING_SERVICE_HOST", "0.0.0.0")
        port = int(os.getenv("ALERTING_SERVICE_PORT", "8013"))
        super().__init__("alerting_service", host, port)
        
        # Configuration
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        self.email_smtp_server = os.getenv("EMAIL_SMTP_SERVER")
        self.email_username = os.getenv("EMAIL_USERNAME")
        self.email_password = os.getenv("EMAIL_PASSWORD")
        
        # Alert rules and subscriptions
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.subscriptions: Dict[str, List[str]] = {}  # user_id -> [alert_types]
        
        # Register handlers
        self.register_handlers()
        
        # Start background tasks
        asyncio.create_task(self._alert_processing_loop())
    
    def register_handlers(self):
        """Register all MCP handlers."""
        
        @self.handler("send_alert")
        async def send_alert(content: Dict[str, Any]) -> Dict[str, Any]:
            """Send an alert through configured channels."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["alert_type", "severity", "message"])
            
            try:
                result = await self._send_alert(
                    alert_type=content["alert_type"],
                    severity=content["severity"],
                    message=content["message"],
                    data=content.get("data", {}),
                    channels=content.get("channels", ["DATABASE", "TELEGRAM"])
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Error sending alert: {e}")
                return {
                    "status": "ERROR",
                    "error": str(e)
                }
        
        @self.handler("get_alerts")
        async def get_alerts(content: Dict[str, Any]) -> Dict[str, Any]:
            """Get recent alerts."""
            content = await sanitize_input(content)
            
            limit = content.get("limit", 50)
            severity = content.get("severity")
            alert_type = content.get("alert_type")
            
            try:
                alerts = await self._get_alerts(limit, severity, alert_type)
                
                return {
                    "alerts": alerts,
                    "count": len(alerts),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting alerts: {e}")
                return {
                    "status": "ERROR",
                    "error": str(e)
                }
        
        @self.handler("configure_alert_rule")
        async def configure_alert_rule(content: Dict[str, Any]) -> Dict[str, Any]:
            """Configure an alert rule."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["rule_name", "conditions", "actions"])
            
            try:
                rule_id = await self._configure_alert_rule(
                    rule_name=content["rule_name"],
                    conditions=content["conditions"],
                    actions=content["actions"],
                    enabled=content.get("enabled", True)
                )
                
                return {
                    "status": "SUCCESS",
                    "rule_id": rule_id,
                    "message": f"Alert rule '{content['rule_name']}' configured"
                }
                
            except Exception as e:
                logger.error(f"Error configuring alert rule: {e}")
                return {
                    "status": "ERROR",
                    "error": str(e)
                }
        
        @self.handler("subscribe_to_alerts")
        async def subscribe_to_alerts(content: Dict[str, Any]) -> Dict[str, Any]:
            """Subscribe user to specific alert types."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["user_id", "alert_types"])
            
            user_id = content["user_id"]
            alert_types = content["alert_types"]
            
            self.subscriptions[user_id] = alert_types
            
            return {
                "status": "SUCCESS",
                "message": f"User {user_id} subscribed to {len(alert_types)} alert types"
            }
        
        @self.handler("test_alert_channels")
        async def test_alert_channels(content: Dict[str, Any]) -> Dict[str, Any]:
            """Test alert channels configuration."""
            content = await sanitize_input(content)
            
            channels = content.get("channels", ["TELEGRAM", "SLACK"])
            
            test_results = {}
            
            for channel in channels:
                try:
                    if channel == "TELEGRAM":
                        result = await self._send_telegram_alert("🧪 Test Alert", "This is a test alert from AWM system.")
                    elif channel == "SLACK":
                        result = await self._send_slack_alert("🧪 Test Alert", "This is a test alert from AWM system.")
                    else:
                        result = {"status": "SKIPPED", "message": f"Channel {channel} not implemented for testing"}
                    
                    test_results[channel] = result
                    
                except Exception as e:
                    test_results[channel] = {"status": "ERROR", "error": str(e)}
            
            return {
                "test_results": test_results,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _send_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        data: Dict[str, Any] = None,
        channels: List[str] = None
    ) -> Dict[str, Any]:
        """Send alert through specified channels."""
        
        if data is None:
            data = {}
        
        if channels is None:
            channels = ["DATABASE"]
        
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Store alert in database
        await self._store_alert(alert_id, alert_type, severity, message, data)
        
        # Send through configured channels
        results = {}
        
        for channel in channels:
            try:
                if channel == "TELEGRAM" and self.telegram_bot_token:
                    result = await self._send_telegram_alert(f"🚨 {alert_type}", message)
                    results[channel] = result
                
                elif channel == "SLACK" and self.slack_webhook_url:
                    result = await self._send_slack_alert(f"🚨 {alert_type}", message)
                    results[channel] = result
                
                elif channel == "EMAIL" and self.email_smtp_server:
                    result = await self._send_email_alert(alert_type, message, severity)
                    results[channel] = result
                
                elif channel == "DATABASE":
                    results[channel] = {"status": "SUCCESS", "message": "Stored in database"}
                
                else:
                    results[channel] = {"status": "SKIPPED", "message": f"Channel {channel} not configured"}
                    
            except Exception as e:
                logger.error(f"Error sending alert via {channel}: {e}")
                results[channel] = {"status": "ERROR", "error": str(e)}
        
        return {
            "alert_id": alert_id,
            "status": "SENT",
            "channels": results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def _send_telegram_alert(self, title: str, message: str) -> Dict[str, Any]:
        """Send alert via Telegram."""
        
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return {"status": "ERROR", "error": "Telegram not configured"}
        
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        
        text = f"*{title}*\n\n{message}\n\n_AWM System Alert_"
        
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        return {"status": "SUCCESS", "message": "Telegram alert sent"}
                    else:
                        error_text = await response.text()
                        return {"status": "ERROR", "error": f"Telegram API error: {error_text}"}
        
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}
    
    async def _send_slack_alert(self, title: str, message: str) -> Dict[str, Any]:
        """Send alert via Slack."""
        
        if not self.slack_webhook_url:
            return {"status": "ERROR", "error": "Slack not configured"}
        
        payload = {
            "text": f"{title}\n{message}",
            "username": "AWM System",
            "icon_emoji": ":warning:"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.slack_webhook_url, json=payload) as response:
                    if response.status == 200:
                        return {"status": "SUCCESS", "message": "Slack alert sent"}
                    else:
                        error_text = await response.text()
                        return {"status": "ERROR", "error": f"Slack webhook error: {error_text}"}
        
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}
    
    async def _send_email_alert(self, title: str, message: str, severity: str) -> Dict[str, Any]:
        """Send alert via email."""
        
        # Email implementation would go here
        # For now, return a placeholder
        return {"status": "NOT_IMPLEMENTED", "message": "Email alerts not implemented"}
    
    async def _store_alert(
        self,
        alert_id: str,
        alert_type: str,
        severity: str,
        message: str,
        data: Dict[str, Any]
    ) -> None:
        """Store alert in database."""
        
        try:
            query = """
                INSERT INTO system_alerts 
                (id, alert_type, severity, title, message, data, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """
            
            await db_manager.execute_query(
                query,
                alert_id,
                alert_type,
                severity,
                alert_type,  # Use alert_type as title for now
                message,
                json.dumps(data),
                datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
    
    async def _get_alerts(
        self,
        limit: int,
        severity: Optional[str] = None,
        alert_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get alerts from database."""
        
        try:
            conditions = []
            params = []
            param_count = 0
            
            if severity:
                param_count += 1
                conditions.append(f"severity = ${param_count}")
                params.append(severity)
            
            if alert_type:
                param_count += 1
                conditions.append(f"alert_type = ${param_count}")
                params.append(alert_type)
            
            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)
            
            param_count += 1
            query = f"""
                SELECT * FROM system_alerts 
                {where_clause}
                ORDER BY created_at DESC 
                LIMIT ${param_count}
            """
            params.append(limit)
            
            alerts = await db_manager.execute_query(query, *params, fetch="all")
            
            return [
                {
                    "id": alert["id"],
                    "alert_type": alert["alert_type"],
                    "severity": alert["severity"],
                    "title": alert["title"],
                    "message": alert["message"],
                    "data": json.loads(alert["data"]) if alert["data"] else {},
                    "created_at": alert["created_at"].isoformat()
                }
                for alert in alerts
            ]
            
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    async def _configure_alert_rule(
        self,
        rule_name: str,
        conditions: Dict[str, Any],
        actions: List[Dict[str, Any]],
        enabled: bool = True
    ) -> str:
        """Configure an alert rule."""
        
        rule_id = f"rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        rule = {
            "id": rule_id,
            "name": rule_name,
            "conditions": conditions,
            "actions": actions,
            "enabled": enabled,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        self.alert_rules[rule_id] = rule
        
        logger.info(f"Configured alert rule: {rule_name}")
        
        return rule_id
    
    async def _alert_processing_loop(self):
        """Background loop to process alerts and check rules."""
        
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Process alert rules
                await self._process_alert_rules()
                
                # Clean up old alerts (optional)
                await self._cleanup_old_alerts()
                
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                await asyncio.sleep(300)  # Wait longer on error
    
    async def _process_alert_rules(self):
        """Process configured alert rules."""
        
        for rule_id, rule in self.alert_rules.items():
            if not rule.get("enabled", True):
                continue
            
            try:
                # Check rule conditions
                # This would be expanded to check actual system metrics
                # For now, it's a placeholder
                
                conditions = rule.get("conditions", {})
                
                # Example: Check portfolio risk threshold
                if conditions.get("type") == "portfolio_risk":
                    # This would check actual portfolio risk
                    # await self._check_portfolio_risk_rule(rule)
                    pass
                
                # Example: Check system health
                elif conditions.get("type") == "system_health":
                    # This would check system component health
                    # await self._check_system_health_rule(rule)
                    pass
                
            except Exception as e:
                logger.error(f"Error processing alert rule {rule_id}: {e}")
    
    async def _cleanup_old_alerts(self):
        """Clean up old alerts from database."""
        
        try:
            # Delete alerts older than 30 days
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
            
            query = "DELETE FROM system_alerts WHERE created_at < $1"
            await db_manager.execute_query(query, cutoff_date)
            
        except Exception as e:
            logger.error(f"Error cleaning up old alerts: {e}")


async def main():
    """Main function to run the Alerting Service."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize database
    await init_database()
    
    try:
        # Create and start server
        service = AlertingService()
        logger.info("Starting Alerting Service...")
        await service.start()
    finally:
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
