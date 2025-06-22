"""
Base MCP (Model Context Protocol) client implementation for AWM system.
Provides standardized communication between all system components.
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import logging

logger = logging.getLogger(__name__)


class MessageType(Enum):
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    ERROR = "ERROR"
    EVENT = "EVENT"


class ErrorSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class MCPError:
    error_code: str
    error_message: str
    severity: ErrorSeverity
    recoverable: bool
    retry_after: Optional[int] = None


@dataclass
class MCPMessage:
    request_id: str
    timestamp: str
    source: str
    destination: str
    message_type: MessageType
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[MCPError] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for JSON serialization."""
        result = {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "destination": self.destination,
            "message_type": self.message_type.value,
            "content": self.content,
        }
        
        if self.metadata:
            result["metadata"] = self.metadata
            
        if self.error:
            result["error"] = {
                "error_code": self.error.error_code,
                "error_message": self.error.error_message,
                "severity": self.error.severity.value,
                "recoverable": self.error.recoverable,
                "retry_after": self.error.retry_after
            }
            
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessage':
        """Create message from dictionary."""
        error = None
        if "error" in data:
            error_data = data["error"]
            error = MCPError(
                error_code=error_data["error_code"],
                error_message=error_data["error_message"],
                severity=ErrorSeverity(error_data["severity"]),
                recoverable=error_data["recoverable"],
                retry_after=error_data.get("retry_after")
            )
        
        return cls(
            request_id=data["request_id"],
            timestamp=data["timestamp"],
            source=data["source"],
            destination=data["destination"],
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            metadata=data.get("metadata"),
            error=error
        )


class MCPClient:
    """Base MCP client for communicating with MCP servers."""
    
    def __init__(self, client_name: str, timeout: int = 30):
        self.client_name = client_name
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _create_request_message(
        self,
        destination: str,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> MCPMessage:
        """Create a standardized request message."""
        return MCPMessage(
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=self.client_name,
            destination=destination,
            message_type=MessageType.REQUEST,
            content=content,
            metadata=metadata
        )
    
    def _create_error_response(
        self,
        request_id: str,
        destination: str,
        error: MCPError
    ) -> MCPMessage:
        """Create a standardized error response."""
        return MCPMessage(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=self.client_name,
            destination=destination,
            message_type=MessageType.ERROR,
            content={},
            error=error
        )
    
    async def send_request(
        self,
        server_url: str,
        endpoint: str,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        retries: int = 3
    ) -> MCPMessage:
        """Send a request to an MCP server and return the response."""
        if not self.session:
            raise RuntimeError("MCPClient must be used as async context manager")
        
        destination = server_url.split("//")[-1].split("/")[0]  # Extract server name from URL
        request_msg = self._create_request_message(destination, content, metadata)
        
        url = f"{server_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        for attempt in range(retries + 1):
            try:
                async with self.session.post(
                    url,
                    json=request_msg.to_dict(),
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        return MCPMessage.from_dict(response_data)
                    else:
                        error_text = await response.text()
                        logger.error(f"HTTP {response.status} from {url}: {error_text}")
                        
                        if attempt == retries:
                            error = MCPError(
                                error_code="HTTP_ERROR",
                                error_message=f"HTTP {response.status}: {error_text}",
                                severity=ErrorSeverity.ERROR,
                                recoverable=response.status >= 500,
                                retry_after=60 if response.status >= 500 else None
                            )
                            return self._create_error_response(
                                request_msg.request_id, destination, error
                            )
                        
                        if response.status >= 500:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        else:
                            break
                            
            except asyncio.TimeoutError:
                logger.error(f"Timeout connecting to {url}")
                if attempt == retries:
                    error = MCPError(
                        error_code="TIMEOUT_ERROR",
                        error_message=f"Request to {url} timed out",
                        severity=ErrorSeverity.ERROR,
                        recoverable=True,
                        retry_after=30
                    )
                    return self._create_error_response(
                        request_msg.request_id, destination, error
                    )
                await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                logger.error(f"Error connecting to {url}: {str(e)}")
                if attempt == retries:
                    error = MCPError(
                        error_code="CONNECTION_ERROR",
                        error_message=f"Failed to connect to {url}: {str(e)}",
                        severity=ErrorSeverity.ERROR,
                        recoverable=True,
                        retry_after=60
                    )
                    return self._create_error_response(
                        request_msg.request_id, destination, error
                    )
                await asyncio.sleep(2 ** attempt)
        
        # Should not reach here, but just in case
        error = MCPError(
            error_code="UNKNOWN_ERROR",
            error_message="Unknown error occurred",
            severity=ErrorSeverity.ERROR,
            recoverable=False
        )
        return self._create_error_response(request_msg.request_id, destination, error)
    
    async def health_check(self, server_url: str) -> bool:
        """Check if an MCP server is healthy."""
        try:
            response = await self.send_request(
                server_url,
                "health",
                {"check": "ping"}
            )
            return response.message_type != MessageType.ERROR
        except Exception:
            return False
