"""
Unit tests for MCP client functionality.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from shared.mcp_client.base import MCPClient, MCPMessage, MCPError, MessageType, ErrorSeverity


class TestMCPMessage:
    """Test MCP message functionality."""
    
    def test_message_creation(self):
        """Test creating an MCP message."""
        message = MCPMessage(
            request_id=str(uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="test_client",
            destination="test_server",
            message_type=MessageType.REQUEST,
            content={"test": "data"}
        )
        
        assert message.source == "test_client"
        assert message.destination == "test_server"
        assert message.message_type == MessageType.REQUEST
        assert message.content["test"] == "data"
    
    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        message = MCPMessage(
            request_id="test-id",
            timestamp="2024-01-01T00:00:00Z",
            source="test_client",
            destination="test_server",
            message_type=MessageType.REQUEST,
            content={"test": "data"}
        )
        
        result = message.to_dict()
        
        assert result["request_id"] == "test-id"
        assert result["source"] == "test_client"
        assert result["message_type"] == "REQUEST"
        assert result["content"]["test"] == "data"
    
    def test_message_from_dict(self):
        """Test creating message from dictionary."""
        data = {
            "request_id": "test-id",
            "timestamp": "2024-01-01T00:00:00Z",
            "source": "test_client",
            "destination": "test_server",
            "message_type": "REQUEST",
            "content": {"test": "data"}
        }
        
        message = MCPMessage.from_dict(data)
        
        assert message.request_id == "test-id"
        assert message.source == "test_client"
        assert message.message_type == MessageType.REQUEST
        assert message.content["test"] == "data"
    
    def test_message_with_error(self):
        """Test message with error."""
        error = MCPError(
            error_code="TEST_ERROR",
            error_message="Test error message",
            severity=ErrorSeverity.ERROR,
            recoverable=True,
            retry_after=30
        )
        
        message = MCPMessage(
            request_id="test-id",
            timestamp="2024-01-01T00:00:00Z",
            source="test_client",
            destination="test_server",
            message_type=MessageType.ERROR,
            content={},
            error=error
        )
        
        result = message.to_dict()
        
        assert result["error"]["error_code"] == "TEST_ERROR"
        assert result["error"]["severity"] == "ERROR"
        assert result["error"]["recoverable"] is True
        assert result["error"]["retry_after"] == 30


class TestMCPClient:
    """Test MCP client functionality."""
    
    def test_client_creation(self):
        """Test creating an MCP client."""
        client = MCPClient("test_client")
        
        assert client.client_name == "test_client"
        assert client.timeout == 30
    
    def test_create_request_message(self):
        """Test creating a request message."""
        client = MCPClient("test_client")
        
        message = client._create_request_message(
            "test_server",
            {"test": "data"},
            {"metadata": "value"}
        )
        
        assert message.source == "test_client"
        assert message.destination == "test_server"
        assert message.message_type == MessageType.REQUEST
        assert message.content["test"] == "data"
        assert message.metadata["metadata"] == "value"
    
    def test_create_error_response(self):
        """Test creating an error response."""
        client = MCPClient("test_client")
        
        error = MCPError(
            error_code="TEST_ERROR",
            error_message="Test error",
            severity=ErrorSeverity.ERROR,
            recoverable=False
        )
        
        message = client._create_error_response(
            "test-request-id",
            "test_server",
            error
        )
        
        assert message.request_id == "test-request-id"
        assert message.source == "test_client"
        assert message.destination == "test_server"
        assert message.message_type == MessageType.ERROR
        assert message.error.error_code == "TEST_ERROR"


@pytest.mark.asyncio
class TestMCPClientAsync:
    """Test async MCP client functionality."""
    
    async def test_client_context_manager(self):
        """Test using client as context manager."""
        async with MCPClient("test_client") as client:
            assert client.session is not None
        
        # Session should be closed after exiting context
        assert client.session.closed


if __name__ == "__main__":
    pytest.main([__file__])
