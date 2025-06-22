"""
Zerodha Kite Connect API Client for AWM System.
Provides a unified interface for all Zerodha API operations with proper error handling,
rate limiting, and authentication management.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from decimal import Decimal
import json

from kiteconnect import KiteConnect
from kiteconnect.exceptions import (
    KiteException, 
    NetworkException, 
    TokenException, 
    PermissionException,
    OrderException,
    InputException
)

logger = logging.getLogger(__name__)


class ZerodhaRateLimiter:
    """Rate limiter for Zerodha API calls (3 requests per second)."""
    
    def __init__(self, max_requests: int = 3, time_window: int = 1):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make an API call."""
        async with self._lock:
            now = time.time()
            # Remove old requests outside the time window
            self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
            
            if len(self.requests) >= self.max_requests:
                # Calculate wait time
                oldest_request = min(self.requests)
                wait_time = self.time_window - (now - oldest_request)
                if wait_time > 0:
                    logger.debug(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)
                    return await self.acquire()
            
            self.requests.append(now)


class ZerodhaAPIError(Exception):
    """Custom exception for Zerodha API errors."""
    
    def __init__(self, message: str, error_code: str = None, original_exception: Exception = None):
        super().__init__(message)
        self.error_code = error_code
        self.original_exception = original_exception


class ZerodhaClient:
    """
    Unified Zerodha Kite Connect API client with enhanced error handling,
    rate limiting, and authentication management.
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, access_token: str = None):
        self.api_key = api_key or os.getenv("ZERODHA_API_KEY")
        self.api_secret = api_secret or os.getenv("ZERODHA_API_SECRET")
        self.access_token = access_token or os.getenv("ZERODHA_ACCESS_TOKEN")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Zerodha API key and secret are required")
        
        # Initialize KiteConnect
        self.kite = KiteConnect(api_key=self.api_key)
        if self.access_token:
            self.kite.set_access_token(self.access_token)
        
        # Rate limiter
        self.rate_limiter = ZerodhaRateLimiter()
        
        # Connection status
        self.is_connected = False
        self.last_heartbeat = None
        
        logger.info("Zerodha client initialized")
    
    async def authenticate(self, request_token: str = None) -> Dict[str, Any]:
        """
        Authenticate with Zerodha using request token.
        
        Args:
            request_token: Request token from Zerodha login flow
            
        Returns:
            Authentication response with access token
        """
        try:
            if not request_token:
                raise ValueError("Request token is required for authentication")
            
            await self.rate_limiter.acquire()
            
            # Generate session
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            self.is_connected = True
            self.last_heartbeat = datetime.now()
            
            logger.info("Successfully authenticated with Zerodha")
            return data
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise ZerodhaAPIError(f"Authentication failed: {str(e)}", "AUTH_ERROR", e)
    
    async def get_profile(self) -> Dict[str, Any]:
        """Get user profile information."""
        try:
            await self.rate_limiter.acquire()
            profile = self.kite.profile()
            self.last_heartbeat = datetime.now()
            return profile
        except Exception as e:
            logger.error(f"Failed to get profile: {e}")
            raise ZerodhaAPIError(f"Failed to get profile: {str(e)}", "PROFILE_ERROR", e)
    
    async def get_margins(self) -> Dict[str, Any]:
        """Get account margins."""
        try:
            await self.rate_limiter.acquire()
            margins = self.kite.margins()
            return margins
        except Exception as e:
            logger.error(f"Failed to get margins: {e}")
            raise ZerodhaAPIError(f"Failed to get margins: {str(e)}", "MARGINS_ERROR", e)
    
    async def get_positions(self) -> Dict[str, Any]:
        """Get current positions."""
        try:
            await self.rate_limiter.acquire()
            positions = self.kite.positions()
            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise ZerodhaAPIError(f"Failed to get positions: {str(e)}", "POSITIONS_ERROR", e)
    
    async def get_holdings(self) -> List[Dict[str, Any]]:
        """Get current holdings."""
        try:
            await self.rate_limiter.acquire()
            holdings = self.kite.holdings()
            return holdings
        except Exception as e:
            logger.error(f"Failed to get holdings: {e}")
            raise ZerodhaAPIError(f"Failed to get holdings: {str(e)}", "HOLDINGS_ERROR", e)
    
    async def get_instruments(self, exchange: str = None) -> List[Dict[str, Any]]:
        """
        Get instruments list.
        
        Args:
            exchange: Exchange name (NSE, BSE, etc.)
            
        Returns:
            List of instruments
        """
        try:
            await self.rate_limiter.acquire()
            if exchange:
                instruments = self.kite.instruments(exchange)
            else:
                instruments = self.kite.instruments()
            return instruments
        except Exception as e:
            logger.error(f"Failed to get instruments: {e}")
            raise ZerodhaAPIError(f"Failed to get instruments: {str(e)}", "INSTRUMENTS_ERROR", e)
    
    async def get_quote(self, instruments: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Get market quotes for instruments.
        
        Args:
            instruments: Single instrument or list of instruments
            
        Returns:
            Quote data
        """
        try:
            await self.rate_limiter.acquire()
            if isinstance(instruments, str):
                instruments = [instruments]
            quotes = self.kite.quote(instruments)
            return quotes
        except Exception as e:
            logger.error(f"Failed to get quote: {e}")
            raise ZerodhaAPIError(f"Failed to get quote: {str(e)}", "QUOTE_ERROR", e)
    
    async def get_ltp(self, instruments: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Get Last Traded Price for instruments.
        
        Args:
            instruments: Single instrument or list of instruments
            
        Returns:
            LTP data
        """
        try:
            await self.rate_limiter.acquire()
            if isinstance(instruments, str):
                instruments = [instruments]
            ltp = self.kite.ltp(instruments)
            return ltp
        except Exception as e:
            logger.error(f"Failed to get LTP: {e}")
            raise ZerodhaAPIError(f"Failed to get LTP: {str(e)}", "LTP_ERROR", e)
    
    async def place_order(
        self,
        variety: str,
        exchange: str,
        tradingsymbol: str,
        transaction_type: str,
        quantity: int,
        product: str,
        order_type: str,
        price: float = None,
        trigger_price: float = None,
        validity: str = "DAY",
        disclosed_quantity: int = None,
        squareoff: float = None,
        stoploss: float = None,
        trailing_stoploss: float = None,
        tag: str = None
    ) -> str:
        """
        Place an order.
        
        Returns:
            Order ID
        """
        try:
            await self.rate_limiter.acquire()
            
            order_params = {
                "variety": variety,
                "exchange": exchange,
                "tradingsymbol": tradingsymbol,
                "transaction_type": transaction_type,
                "quantity": quantity,
                "product": product,
                "order_type": order_type,
                "validity": validity
            }
            
            # Add optional parameters
            if price is not None:
                order_params["price"] = price
            if trigger_price is not None:
                order_params["trigger_price"] = trigger_price
            if disclosed_quantity is not None:
                order_params["disclosed_quantity"] = disclosed_quantity
            if squareoff is not None:
                order_params["squareoff"] = squareoff
            if stoploss is not None:
                order_params["stoploss"] = stoploss
            if trailing_stoploss is not None:
                order_params["trailing_stoploss"] = trailing_stoploss
            if tag is not None:
                order_params["tag"] = tag
            
            order_id = self.kite.place_order(**order_params)
            logger.info(f"Order placed successfully: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise ZerodhaAPIError(f"Failed to place order: {str(e)}", "ORDER_PLACE_ERROR", e)
    
    async def modify_order(
        self,
        variety: str,
        order_id: str,
        quantity: int = None,
        price: float = None,
        order_type: str = None,
        trigger_price: float = None,
        validity: str = None,
        disclosed_quantity: int = None
    ) -> str:
        """Modify an existing order."""
        try:
            await self.rate_limiter.acquire()
            
            modify_params = {"variety": variety, "order_id": order_id}
            
            # Add parameters to modify
            if quantity is not None:
                modify_params["quantity"] = quantity
            if price is not None:
                modify_params["price"] = price
            if order_type is not None:
                modify_params["order_type"] = order_type
            if trigger_price is not None:
                modify_params["trigger_price"] = trigger_price
            if validity is not None:
                modify_params["validity"] = validity
            if disclosed_quantity is not None:
                modify_params["disclosed_quantity"] = disclosed_quantity
            
            result = self.kite.modify_order(**modify_params)
            logger.info(f"Order modified successfully: {order_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to modify order {order_id}: {e}")
            raise ZerodhaAPIError(f"Failed to modify order: {str(e)}", "ORDER_MODIFY_ERROR", e)
    
    async def cancel_order(self, variety: str, order_id: str) -> str:
        """Cancel an order."""
        try:
            await self.rate_limiter.acquire()
            result = self.kite.cancel_order(variety=variety, order_id=order_id)
            logger.info(f"Order cancelled successfully: {order_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise ZerodhaAPIError(f"Failed to cancel order: {str(e)}", "ORDER_CANCEL_ERROR", e)
    
    async def get_orders(self) -> List[Dict[str, Any]]:
        """Get all orders for the day."""
        try:
            await self.rate_limiter.acquire()
            orders = self.kite.orders()
            return orders
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            raise ZerodhaAPIError(f"Failed to get orders: {str(e)}", "ORDERS_ERROR", e)
    
    async def get_order_history(self, order_id: str) -> List[Dict[str, Any]]:
        """Get order history for a specific order."""
        try:
            await self.rate_limiter.acquire()
            history = self.kite.order_history(order_id=order_id)
            return history
        except Exception as e:
            logger.error(f"Failed to get order history for {order_id}: {e}")
            raise ZerodhaAPIError(f"Failed to get order history: {str(e)}", "ORDER_HISTORY_ERROR", e)
    
    async def get_trades(self) -> List[Dict[str, Any]]:
        """Get all trades for the day."""
        try:
            await self.rate_limiter.acquire()
            trades = self.kite.trades()
            return trades
        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            raise ZerodhaAPIError(f"Failed to get trades: {str(e)}", "TRADES_ERROR", e)
    
    async def health_check(self) -> bool:
        """Check if the connection to Zerodha is healthy."""
        try:
            await self.get_profile()
            self.is_connected = True
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            self.is_connected = False
            return False
    
    def is_market_open(self) -> bool:
        """Check if the Indian market is currently open."""
        now = datetime.now()
        
        # Check if it's a weekday (Monday = 0, Sunday = 6)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Market hours: 9:15 AM to 3:30 PM IST
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close
