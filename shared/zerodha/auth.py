"""
Zerodha Authentication Service for AWM System.
Handles OAuth 2.0 flow, token management, and session persistence.
"""

import asyncio
import logging
import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from urllib.parse import urlencode
import aiohttp
import redis.asyncio as redis

from .client import ZerodhaClient, ZerodhaAPIError

logger = logging.getLogger(__name__)


class ZerodhaAuthService:
    """
    Handles Zerodha authentication flow and token management.
    Supports both manual and automated authentication flows.
    """
    
    def __init__(self, redis_host: str = "redis", redis_port: int = 6379):
        self.api_key = os.getenv("ZERODHA_API_KEY")
        self.api_secret = os.getenv("ZERODHA_API_SECRET")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Zerodha API key and secret are required")
        
        # Redis for token storage
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        
        # Token storage keys
        self.access_token_key = f"zerodha:access_token:{self.api_key}"
        self.session_data_key = f"zerodha:session_data:{self.api_key}"
        
        logger.info("Zerodha auth service initialized")
    
    async def _get_redis_client(self) -> redis.Redis:
        """Get Redis client connection."""
        if not self.redis_client:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True
            )
        return self.redis_client
    
    async def get_login_url(self) -> str:
        """
        Generate Zerodha login URL for OAuth flow.
        
        Returns:
            Login URL for user authentication
        """
        base_url = "https://kite.zerodha.com/connect/login"
        params = {
            "api_key": self.api_key,
            "v": "3"
        }
        
        login_url = f"{base_url}?{urlencode(params)}"
        logger.info("Generated Zerodha login URL")
        return login_url
    
    async def authenticate_with_request_token(self, request_token: str) -> Dict[str, Any]:
        """
        Complete authentication using request token from login flow.
        
        Args:
            request_token: Request token received after user login
            
        Returns:
            Authentication data including access token
        """
        try:
            # Create client and authenticate
            client = ZerodhaClient(self.api_key, self.api_secret)
            auth_data = await client.authenticate(request_token)
            
            # Store tokens in Redis
            await self._store_session_data(auth_data)
            
            logger.info("Successfully authenticated with Zerodha")
            return auth_data
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise ZerodhaAPIError(f"Authentication failed: {str(e)}", "AUTH_ERROR", e)
    
    async def get_stored_access_token(self) -> Optional[str]:
        """
        Retrieve stored access token from Redis.
        
        Returns:
            Access token if available and valid, None otherwise
        """
        try:
            redis_client = await self._get_redis_client()
            
            # Get stored session data
            session_data_str = await redis_client.get(self.session_data_key)
            if not session_data_str:
                return None
            
            session_data = json.loads(session_data_str)
            
            # Check if token is still valid (Zerodha tokens expire daily)
            stored_date = datetime.fromisoformat(session_data.get("stored_at", ""))
            if datetime.now() - stored_date > timedelta(hours=23):  # Expire after 23 hours
                logger.info("Stored access token has expired")
                await self._clear_stored_tokens()
                return None
            
            access_token = session_data.get("access_token")
            if access_token:
                logger.info("Retrieved valid access token from storage")
                return access_token
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve stored access token: {e}")
            return None
    
    async def _store_session_data(self, auth_data: Dict[str, Any]) -> None:
        """Store authentication data in Redis."""
        try:
            redis_client = await self._get_redis_client()
            
            # Prepare session data
            session_data = {
                "access_token": auth_data["access_token"],
                "user_id": auth_data.get("user_id"),
                "user_name": auth_data.get("user_name"),
                "user_shortname": auth_data.get("user_shortname"),
                "email": auth_data.get("email"),
                "user_type": auth_data.get("user_type"),
                "broker": auth_data.get("broker"),
                "exchanges": auth_data.get("exchanges", []),
                "products": auth_data.get("products", []),
                "order_types": auth_data.get("order_types", []),
                "stored_at": datetime.now().isoformat()
            }
            
            # Store with 24-hour expiration
            await redis_client.setex(
                self.session_data_key,
                86400,  # 24 hours
                json.dumps(session_data)
            )
            
            # Also store just the access token for quick access
            await redis_client.setex(
                self.access_token_key,
                86400,  # 24 hours
                auth_data["access_token"]
            )
            
            logger.info("Session data stored successfully")
            
        except Exception as e:
            logger.error(f"Failed to store session data: {e}")
            raise
    
    async def _clear_stored_tokens(self) -> None:
        """Clear stored tokens from Redis."""
        try:
            redis_client = await self._get_redis_client()
            await redis_client.delete(self.access_token_key, self.session_data_key)
            logger.info("Cleared stored tokens")
        except Exception as e:
            logger.error(f"Failed to clear stored tokens: {e}")
    
    async def get_authenticated_client(self) -> ZerodhaClient:
        """
        Get an authenticated Zerodha client.
        
        Returns:
            Authenticated ZerodhaClient instance
            
        Raises:
            ZerodhaAPIError: If no valid authentication is available
        """
        # Try to get stored access token
        access_token = await self.get_stored_access_token()
        
        if not access_token:
            raise ZerodhaAPIError(
                "No valid access token available. Please authenticate first.",
                "NO_AUTH_TOKEN"
            )
        
        # Create client with stored token
        client = ZerodhaClient(self.api_key, self.api_secret, access_token)
        
        # Verify the token is still valid
        try:
            await client.health_check()
            return client
        except Exception as e:
            logger.warning(f"Stored token is invalid: {e}")
            await self._clear_stored_tokens()
            raise ZerodhaAPIError(
                "Stored access token is invalid. Please re-authenticate.",
                "INVALID_TOKEN"
            )
    
    async def is_authenticated(self) -> bool:
        """
        Check if we have valid authentication.
        
        Returns:
            True if authenticated, False otherwise
        """
        try:
            client = await self.get_authenticated_client()
            return await client.health_check()
        except:
            return False
    
    async def get_session_info(self) -> Optional[Dict[str, Any]]:
        """
        Get stored session information.
        
        Returns:
            Session data if available, None otherwise
        """
        try:
            redis_client = await self._get_redis_client()
            session_data_str = await redis_client.get(self.session_data_key)
            
            if session_data_str:
                return json.loads(session_data_str)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session info: {e}")
            return None
    
    async def logout(self) -> None:
        """Logout and clear all stored authentication data."""
        try:
            await self._clear_stored_tokens()
            logger.info("Successfully logged out")
        except Exception as e:
            logger.error(f"Error during logout: {e}")
            raise
    
    async def refresh_authentication(self, request_token: str) -> Dict[str, Any]:
        """
        Refresh authentication with a new request token.
        This is typically called daily as Zerodha tokens expire.
        
        Args:
            request_token: New request token from login flow
            
        Returns:
            New authentication data
        """
        try:
            # Clear old tokens
            await self._clear_stored_tokens()
            
            # Authenticate with new token
            auth_data = await self.authenticate_with_request_token(request_token)
            
            logger.info("Authentication refreshed successfully")
            return auth_data
            
        except Exception as e:
            logger.error(f"Failed to refresh authentication: {e}")
            raise
    
    async def get_auth_status(self) -> Dict[str, Any]:
        """
        Get comprehensive authentication status.
        
        Returns:
            Authentication status information
        """
        try:
            session_info = await self.get_session_info()
            is_auth = await self.is_authenticated()
            
            if session_info:
                stored_at = datetime.fromisoformat(session_info["stored_at"])
                expires_at = stored_at + timedelta(hours=24)
                time_remaining = expires_at - datetime.now()
                
                return {
                    "is_authenticated": is_auth,
                    "user_id": session_info.get("user_id"),
                    "user_name": session_info.get("user_name"),
                    "broker": session_info.get("broker"),
                    "stored_at": session_info["stored_at"],
                    "expires_at": expires_at.isoformat(),
                    "time_remaining_hours": max(0, time_remaining.total_seconds() / 3600),
                    "exchanges": session_info.get("exchanges", []),
                    "products": session_info.get("products", [])
                }
            else:
                return {
                    "is_authenticated": False,
                    "message": "No authentication data found"
                }
                
        except Exception as e:
            logger.error(f"Failed to get auth status: {e}")
            return {
                "is_authenticated": False,
                "error": str(e)
            }
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
