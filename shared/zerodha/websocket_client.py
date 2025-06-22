"""
Zerodha WebSocket Client for real-time market data feeds.
Handles WebSocket connections, subscriptions, and data processing.
"""

import asyncio
import logging
import json
import struct
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from .client import ZerodhaClient, ZerodhaAPIError
from .auth import ZerodhaAuthService
from .config import config_manager

logger = logging.getLogger(__name__)


class ZerodhaWebSocketClient:
    """
    WebSocket client for Zerodha Kite Connect real-time data feeds.
    Handles tick data, order updates, and other real-time events.
    """
    
    def __init__(self, auth_service: ZerodhaAuthService = None):
        self.auth_service = auth_service or ZerodhaAuthService()
        self.config = config_manager.get_zerodha_config()
        
        # WebSocket connection
        self.websocket = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = self.config.websocket_reconnect_attempts
        
        # Subscriptions
        self.subscribed_tokens = set()
        self.subscription_modes = {}  # token -> mode mapping
        
        # Event handlers
        self.tick_handlers: List[Callable] = []
        self.order_update_handlers: List[Callable] = []
        self.error_handlers: List[Callable] = []
        
        # Connection state
        self.last_heartbeat = None
        self.connection_stats = {
            "total_ticks": 0,
            "total_reconnects": 0,
            "last_error": None,
            "connected_at": None
        }
        
        # Kite WebSocket modes
        self.MODE_LTP = "ltp"
        self.MODE_QUOTE = "quote"
        self.MODE_FULL = "full"
    
    async def connect(self) -> bool:
        """
        Connect to Zerodha WebSocket feed.
        
        Returns:
            True if connected successfully, False otherwise
        """
        try:
            # Get authenticated client
            client = await self.auth_service.get_authenticated_client()
            
            # Get WebSocket URL and access token
            access_token = client.access_token
            api_key = client.api_key
            
            # Construct WebSocket URL
            ws_url = f"{self.config.websocket_url}?api_key={api_key}&access_token={access_token}"
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(
                ws_url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.is_connected = True
            self.reconnect_attempts = 0
            self.connection_stats["connected_at"] = datetime.now().isoformat()
            
            logger.info("Connected to Zerodha WebSocket")
            
            # Start message handling
            asyncio.create_task(self._handle_messages())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Zerodha WebSocket: {e}")
            self.connection_stats["last_error"] = str(e)
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket."""
        try:
            self.is_connected = False
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
            logger.info("Disconnected from Zerodha WebSocket")
        except Exception as e:
            logger.error(f"Error disconnecting from WebSocket: {e}")
    
    async def subscribe(self, tokens: List[int], mode: str = "ltp") -> bool:
        """
        Subscribe to instruments for real-time data.
        
        Args:
            tokens: List of instrument tokens
            mode: Subscription mode ("ltp", "quote", "full")
            
        Returns:
            True if subscription successful, False otherwise
        """
        if not self.is_connected:
            logger.error("WebSocket not connected")
            return False
        
        try:
            # Prepare subscription message
            message = {
                "a": "subscribe",
                "v": tokens
            }
            
            # Send subscription
            await self._send_message(message)
            
            # Set mode for tokens
            mode_message = {
                "a": "mode",
                "v": [mode, tokens]
            }
            
            await self._send_message(mode_message)
            
            # Update local state
            self.subscribed_tokens.update(tokens)
            for token in tokens:
                self.subscription_modes[token] = mode
            
            logger.info(f"Subscribed to {len(tokens)} instruments in {mode} mode")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to instruments: {e}")
            return False
    
    async def unsubscribe(self, tokens: List[int]) -> bool:
        """
        Unsubscribe from instruments.
        
        Args:
            tokens: List of instrument tokens to unsubscribe
            
        Returns:
            True if unsubscription successful, False otherwise
        """
        if not self.is_connected:
            logger.error("WebSocket not connected")
            return False
        
        try:
            # Prepare unsubscription message
            message = {
                "a": "unsubscribe",
                "v": tokens
            }
            
            # Send unsubscription
            await self._send_message(message)
            
            # Update local state
            self.subscribed_tokens.difference_update(tokens)
            for token in tokens:
                self.subscription_modes.pop(token, None)
            
            logger.info(f"Unsubscribed from {len(tokens)} instruments")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from instruments: {e}")
            return False
    
    async def _send_message(self, message: Dict[str, Any]):
        """Send message to WebSocket."""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")
        
        message_str = json.dumps(message)
        await self.websocket.send(message_str)
    
    async def _handle_messages(self):
        """Handle incoming WebSocket messages."""
        try:
            async for message in self.websocket:
                try:
                    if isinstance(message, bytes):
                        # Binary tick data
                        await self._process_tick_data(message)
                    else:
                        # Text message (usually confirmations or errors)
                        await self._process_text_message(message)
                        
                    self.last_heartbeat = datetime.now()
                    
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    await self._handle_error(e)
                    
        except ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.is_connected = False
            await self._handle_reconnect()
            
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self.is_connected = False
            await self._handle_error(e)
            await self._handle_reconnect()
    
    async def _process_tick_data(self, data: bytes):
        """Process binary tick data from WebSocket."""
        try:
            # Parse binary tick data according to Kite Connect protocol
            ticks = self._parse_binary_data(data)
            
            for tick in ticks:
                self.connection_stats["total_ticks"] += 1
                
                # Call tick handlers
                for handler in self.tick_handlers:
                    try:
                        await handler(tick)
                    except Exception as e:
                        logger.error(f"Error in tick handler: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing tick data: {e}")
    
    async def _process_text_message(self, message: str):
        """Process text messages from WebSocket."""
        try:
            data = json.loads(message)
            
            # Handle different message types
            if data.get("type") == "order":
                # Order update
                for handler in self.order_update_handlers:
                    try:
                        await handler(data)
                    except Exception as e:
                        logger.error(f"Error in order update handler: {e}")
            else:
                logger.debug(f"Received WebSocket message: {data}")
                
        except Exception as e:
            logger.error(f"Error processing text message: {e}")
    
    def _parse_binary_data(self, data: bytes) -> List[Dict[str, Any]]:
        """
        Parse binary tick data according to Kite Connect protocol.
        This is a simplified implementation - full implementation would handle
        all tick formats (LTP, Quote, Full).
        """
        ticks = []
        offset = 0
        
        try:
            while offset < len(data):
                # Read packet length (2 bytes)
                if offset + 2 > len(data):
                    break
                    
                packet_length = struct.unpack(">H", data[offset:offset+2])[0]
                offset += 2
                
                if offset + packet_length > len(data):
                    break
                
                # Read packet data
                packet_data = data[offset:offset+packet_length]
                offset += packet_length
                
                # Parse tick from packet
                tick = self._parse_tick_packet(packet_data)
                if tick:
                    ticks.append(tick)
                    
        except Exception as e:
            logger.error(f"Error parsing binary data: {e}")
        
        return ticks
    
    def _parse_tick_packet(self, packet: bytes) -> Optional[Dict[str, Any]]:
        """Parse individual tick packet."""
        try:
            if len(packet) < 8:
                return None
            
            # Read instrument token (4 bytes)
            instrument_token = struct.unpack(">I", packet[0:4])[0]
            
            # Read LTP (4 bytes)
            ltp = struct.unpack(">I", packet[4:8])[0] / 100.0
            
            tick = {
                "instrument_token": instrument_token,
                "last_price": ltp,
                "timestamp": datetime.now().isoformat(),
                "mode": self.subscription_modes.get(instrument_token, "ltp")
            }
            
            # Parse additional fields based on packet length and mode
            if len(packet) >= 12:
                # Volume (4 bytes)
                volume = struct.unpack(">I", packet[8:12])[0]
                tick["volume"] = volume
            
            if len(packet) >= 20:
                # OHLC data
                tick["ohlc"] = {
                    "open": struct.unpack(">I", packet[12:16])[0] / 100.0,
                    "high": struct.unpack(">I", packet[16:20])[0] / 100.0,
                    "low": struct.unpack(">I", packet[20:24])[0] / 100.0,
                    "close": struct.unpack(">I", packet[24:28])[0] / 100.0
                }
            
            return tick
            
        except Exception as e:
            logger.error(f"Error parsing tick packet: {e}")
            return None
    
    async def _handle_error(self, error: Exception):
        """Handle WebSocket errors."""
        self.connection_stats["last_error"] = str(error)
        
        for handler in self.error_handlers:
            try:
                await handler(error)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")
    
    async def _handle_reconnect(self):
        """Handle WebSocket reconnection."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return
        
        self.reconnect_attempts += 1
        self.connection_stats["total_reconnects"] += 1
        
        logger.info(f"Attempting to reconnect (attempt {self.reconnect_attempts})")
        
        # Wait before reconnecting
        await asyncio.sleep(self.config.websocket_reconnect_delay)
        
        # Attempt reconnection
        if await self.connect():
            # Re-subscribe to previous subscriptions
            if self.subscribed_tokens:
                tokens_by_mode = {}
                for token, mode in self.subscription_modes.items():
                    if mode not in tokens_by_mode:
                        tokens_by_mode[mode] = []
                    tokens_by_mode[mode].append(token)
                
                for mode, tokens in tokens_by_mode.items():
                    await self.subscribe(tokens, mode)
    
    def add_tick_handler(self, handler: Callable):
        """Add tick data handler."""
        self.tick_handlers.append(handler)
    
    def add_order_update_handler(self, handler: Callable):
        """Add order update handler."""
        self.order_update_handlers.append(handler)
    
    def add_error_handler(self, handler: Callable):
        """Add error handler."""
        self.error_handlers.append(handler)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            **self.connection_stats,
            "is_connected": self.is_connected,
            "subscribed_tokens": len(self.subscribed_tokens),
            "reconnect_attempts": self.reconnect_attempts,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None
        }
