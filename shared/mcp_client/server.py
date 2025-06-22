"""
Base MCP (Model Context Protocol) server implementation for AWM system.
Provides standardized server framework for all MCP services.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Callable, Awaitable
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from .base import MCPMessage, MCPError, MessageType, ErrorSeverity

logger = logging.getLogger(__name__)


class MCPServer:
    """Base MCP server implementation."""
    
    def __init__(self, server_name: str, host: str = "0.0.0.0", port: int = 8000):
        self.server_name = server_name
        self.host = host
        self.port = port
        self.app = FastAPI(title=f"{server_name} MCP Server")
        self.handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = {}
        
        # Setup default routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup default MCP server routes."""
        
        @self.app.post("/{endpoint}")
        async def handle_request(endpoint: str, request: Request):
            try:
                # Parse MCP message
                body = await request.json()
                mcp_request = MCPMessage.from_dict(body)
                
                # Validate request
                if mcp_request.message_type != MessageType.REQUEST:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid message type. Expected REQUEST."
                    )
                
                # Check if handler exists
                if endpoint not in self.handlers:
                    error = MCPError(
                        error_code="ENDPOINT_NOT_FOUND",
                        error_message=f"Endpoint '{endpoint}' not found",
                        severity=ErrorSeverity.ERROR,
                        recoverable=False
                    )
                    response = self._create_error_response(mcp_request, error)
                    return JSONResponse(content=response.to_dict(), status_code=404)
                
                # Execute handler
                try:
                    result = await self.handlers[endpoint](mcp_request.content)
                    response = self._create_success_response(mcp_request, result)
                    return JSONResponse(content=response.to_dict())
                    
                except Exception as e:
                    logger.error(f"Handler error for {endpoint}: {str(e)}")
                    error = MCPError(
                        error_code="HANDLER_ERROR",
                        error_message=f"Handler execution failed: {str(e)}",
                        severity=ErrorSeverity.ERROR,
                        recoverable=True,
                        retry_after=30
                    )
                    response = self._create_error_response(mcp_request, error)
                    return JSONResponse(content=response.to_dict(), status_code=500)
                    
            except json.JSONDecodeError:
                return JSONResponse(
                    content={"error": "Invalid JSON in request body"},
                    status_code=400
                )
            except Exception as e:
                logger.error(f"Request processing error: {str(e)}")
                return JSONResponse(
                    content={"error": "Internal server error"},
                    status_code=500
                )
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "server": self.server_name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        @self.app.get("/info")
        async def server_info():
            """Server information endpoint."""
            return {
                "server_name": self.server_name,
                "endpoints": list(self.handlers.keys()),
                "version": "1.0.0",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _create_success_response(
        self,
        request: MCPMessage,
        content: Dict[str, Any]
    ) -> MCPMessage:
        """Create a successful response message."""
        return MCPMessage(
            request_id=request.request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=self.server_name,
            destination=request.source,
            message_type=MessageType.RESPONSE,
            content=content
        )
    
    def _create_error_response(
        self,
        request: MCPMessage,
        error: MCPError
    ) -> MCPMessage:
        """Create an error response message."""
        return MCPMessage(
            request_id=request.request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=self.server_name,
            destination=request.source,
            message_type=MessageType.ERROR,
            content={},
            error=error
        )
    
    def register_handler(
        self,
        endpoint: str,
        handler: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
    ):
        """Register a handler for a specific endpoint."""
        self.handlers[endpoint] = handler
        logger.info(f"Registered handler for endpoint: {endpoint}")
    
    def handler(self, endpoint: str):
        """Decorator for registering handlers."""
        def decorator(func: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]):
            self.register_handler(endpoint, func)
            return func
        return decorator
    
    async def start(self):
        """Start the MCP server."""
        logger.info(f"Starting {self.server_name} MCP server on {self.host}:{self.port}")
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    def run(self):
        """Run the server (blocking)."""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )


# Utility functions for common MCP operations
async def validate_required_fields(content: Dict[str, Any], required_fields: list) -> None:
    """Validate that required fields are present in the request content."""
    missing_fields = [field for field in required_fields if field not in content]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")


async def sanitize_input(content: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize input content to prevent injection attacks."""
    # Basic sanitization - can be extended based on needs
    sanitized = {}
    for key, value in content.items():
        if isinstance(value, str):
            # Remove potentially dangerous characters
            sanitized[key] = value.replace(";", "").replace("--", "").strip()
        else:
            sanitized[key] = value
    return sanitized
