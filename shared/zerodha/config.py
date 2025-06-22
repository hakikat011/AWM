"""
Zerodha Configuration Management for AWM System.
Handles environment-specific settings, API configurations, and trading parameters.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from decimal import Decimal


@dataclass
class ZerodhaConfig:
    """Zerodha API configuration."""
    
    # API Credentials
    api_key: str
    api_secret: str
    access_token: Optional[str] = None
    
    # Environment Settings
    environment: str = "production"  # "sandbox" or "production"
    base_url: str = "https://api.kite.trade"
    login_url: str = "https://kite.zerodha.com/connect/login"
    
    # Rate Limiting
    max_requests_per_second: int = 3
    request_timeout: int = 30
    
    # Trading Settings
    default_exchange: str = "NSE"
    default_product: str = "CNC"  # CNC, MIS, NRML
    default_validity: str = "DAY"  # DAY, IOC, GTT
    
    # Risk Management
    max_order_value: Decimal = Decimal("1000000")  # ₹10 lakh
    max_position_size: Decimal = Decimal("500000")  # ₹5 lakh
    max_daily_trades: int = 100
    
    # Market Data Settings
    websocket_url: str = "wss://ws.kite.trade"
    enable_websocket: bool = True
    websocket_reconnect_attempts: int = 5
    websocket_reconnect_delay: int = 5
    
    # Logging and Monitoring
    enable_audit_logging: bool = True
    log_level: str = "INFO"
    
    @classmethod
    def from_environment(cls) -> "ZerodhaConfig":
        """Create configuration from environment variables."""
        
        api_key = os.getenv("ZERODHA_API_KEY")
        api_secret = os.getenv("ZERODHA_API_SECRET")
        
        if not api_key or not api_secret:
            raise ValueError("ZERODHA_API_KEY and ZERODHA_API_SECRET must be set")
        
        return cls(
            api_key=api_key,
            api_secret=api_secret,
            access_token=os.getenv("ZERODHA_ACCESS_TOKEN"),
            environment=os.getenv("ZERODHA_ENVIRONMENT", "production"),
            max_requests_per_second=int(os.getenv("ZERODHA_MAX_RPS", "3")),
            request_timeout=int(os.getenv("ZERODHA_TIMEOUT", "30")),
            default_exchange=os.getenv("ZERODHA_DEFAULT_EXCHANGE", "NSE"),
            default_product=os.getenv("ZERODHA_DEFAULT_PRODUCT", "CNC"),
            max_order_value=Decimal(os.getenv("ZERODHA_MAX_ORDER_VALUE", "1000000")),
            max_position_size=Decimal(os.getenv("ZERODHA_MAX_POSITION_SIZE", "500000")),
            max_daily_trades=int(os.getenv("ZERODHA_MAX_DAILY_TRADES", "100")),
            enable_websocket=os.getenv("ZERODHA_ENABLE_WEBSOCKET", "true").lower() == "true",
            enable_audit_logging=os.getenv("ZERODHA_AUDIT_LOGGING", "true").lower() == "true",
            log_level=os.getenv("ZERODHA_LOG_LEVEL", "INFO")
        )
    
    def is_sandbox(self) -> bool:
        """Check if running in sandbox mode."""
        return self.environment.lower() == "sandbox"
    
    def get_base_url(self) -> str:
        """Get appropriate base URL for the environment."""
        if self.is_sandbox():
            return "https://api.kite.trade"  # Zerodha doesn't have separate sandbox URL
        return self.base_url
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.api_key:
            raise ValueError("API key is required")
        
        if not self.api_secret:
            raise ValueError("API secret is required")
        
        if self.max_requests_per_second <= 0:
            raise ValueError("Max requests per second must be positive")
        
        if self.request_timeout <= 0:
            raise ValueError("Request timeout must be positive")
        
        if self.max_order_value <= 0:
            raise ValueError("Max order value must be positive")
        
        if self.max_position_size <= 0:
            raise ValueError("Max position size must be positive")
        
        if self.default_exchange not in ["NSE", "BSE", "NFO", "BFO", "CDS", "MCX"]:
            raise ValueError(f"Invalid default exchange: {self.default_exchange}")
        
        if self.default_product not in ["CNC", "MIS", "NRML"]:
            raise ValueError(f"Invalid default product: {self.default_product}")
        
        if self.default_validity not in ["DAY", "IOC", "GTT"]:
            raise ValueError(f"Invalid default validity: {self.default_validity}")


@dataclass
class TradingConfig:
    """Trading-specific configuration."""
    
    # Trading Hours (IST)
    market_open_time: str = "09:15"
    market_close_time: str = "15:30"
    pre_market_open_time: str = "09:00"
    post_market_close_time: str = "16:00"
    
    # Order Management
    order_retry_attempts: int = 3
    order_retry_delay: int = 1  # seconds
    order_timeout: int = 300  # 5 minutes
    
    # Position Management
    enable_stop_loss: bool = True
    default_stop_loss_pct: Decimal = Decimal("0.05")  # 5%
    enable_take_profit: bool = True
    default_take_profit_pct: Decimal = Decimal("0.10")  # 10%
    
    # Risk Controls
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: Decimal = Decimal("0.20")  # 20% price movement
    enable_position_limits: bool = True
    max_positions_per_symbol: int = 1
    
    # Execution Settings
    enable_smart_routing: bool = True
    preferred_execution_venue: str = "NSE"
    enable_dark_pool: bool = False
    
    # Mid-Frequency Trading Settings
    target_trades_per_day: int = 50
    max_trades_per_day: int = 100
    min_trade_interval_seconds: int = 60  # 1 minute between trades
    max_order_execution_latency_ms: int = 500  # 500ms target
    
    @classmethod
    def from_environment(cls) -> "TradingConfig":
        """Create trading configuration from environment variables."""
        return cls(
            market_open_time=os.getenv("TRADING_START_TIME", "09:15"),
            market_close_time=os.getenv("TRADING_END_TIME", "15:30"),
            order_retry_attempts=int(os.getenv("ORDER_RETRY_ATTEMPTS", "3")),
            order_retry_delay=int(os.getenv("ORDER_RETRY_DELAY", "1")),
            order_timeout=int(os.getenv("ORDER_TIMEOUT", "300")),
            enable_stop_loss=os.getenv("ENABLE_STOP_LOSS", "true").lower() == "true",
            default_stop_loss_pct=Decimal(os.getenv("DEFAULT_STOP_LOSS", "0.05")),
            enable_take_profit=os.getenv("ENABLE_TAKE_PROFIT", "true").lower() == "true",
            default_take_profit_pct=Decimal(os.getenv("DEFAULT_TAKE_PROFIT", "0.10")),
            enable_circuit_breaker=os.getenv("ENABLE_CIRCUIT_BREAKER", "true").lower() == "true",
            circuit_breaker_threshold=Decimal(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "0.20")),
            max_positions_per_symbol=int(os.getenv("MAX_POSITIONS_PER_SYMBOL", "1")),
            target_trades_per_day=int(os.getenv("TARGET_TRADES_PER_DAY", "50")),
            max_trades_per_day=int(os.getenv("MAX_TRADES_PER_DAY", "100")),
            min_trade_interval_seconds=int(os.getenv("MIN_TRADE_INTERVAL", "60")),
            max_order_execution_latency_ms=int(os.getenv("MAX_EXECUTION_LATENCY_MS", "500"))
        )


@dataclass
class ComplianceConfig:
    """SEBI compliance and regulatory configuration."""
    
    # Audit and Logging
    enable_audit_trail: bool = True
    audit_retention_days: int = 365
    enable_trade_reporting: bool = True
    
    # Risk Management
    enable_position_limits: bool = True
    enable_exposure_limits: bool = True
    enable_concentration_limits: bool = True
    
    # Order Controls
    enable_price_checks: bool = True
    max_price_deviation_pct: Decimal = Decimal("0.20")  # 20%
    enable_quantity_checks: bool = True
    max_order_quantity_multiplier: int = 10  # 10x average volume
    
    # Market Making Controls
    enable_market_making_controls: bool = False
    max_market_making_exposure: Decimal = Decimal("1000000")  # ₹10 lakh
    
    # Algorithmic Trading
    algo_trading_enabled: bool = True
    require_algo_approval: bool = True
    max_algo_orders_per_second: int = 1
    
    @classmethod
    def from_environment(cls) -> "ComplianceConfig":
        """Create compliance configuration from environment variables."""
        return cls(
            enable_audit_trail=os.getenv("COMPLIANCE_AUDIT_TRAIL", "true").lower() == "true",
            audit_retention_days=int(os.getenv("AUDIT_RETENTION_DAYS", "365")),
            enable_trade_reporting=os.getenv("COMPLIANCE_TRADE_REPORTING", "true").lower() == "true",
            enable_position_limits=os.getenv("COMPLIANCE_POSITION_LIMITS", "true").lower() == "true",
            enable_exposure_limits=os.getenv("COMPLIANCE_EXPOSURE_LIMITS", "true").lower() == "true",
            max_price_deviation_pct=Decimal(os.getenv("MAX_PRICE_DEVIATION", "0.20")),
            max_order_quantity_multiplier=int(os.getenv("MAX_ORDER_QUANTITY_MULTIPLIER", "10")),
            algo_trading_enabled=os.getenv("ALGO_TRADING_ENABLED", "true").lower() == "true",
            require_algo_approval=os.getenv("REQUIRE_ALGO_APPROVAL", "true").lower() == "true",
            max_algo_orders_per_second=int(os.getenv("MAX_ALGO_ORDERS_PER_SECOND", "1"))
        )


class ConfigManager:
    """Centralized configuration manager for Zerodha integration."""
    
    def __init__(self):
        self.zerodha_config = ZerodhaConfig.from_environment()
        self.trading_config = TradingConfig.from_environment()
        self.compliance_config = ComplianceConfig.from_environment()
        
        # Validate all configurations
        self.validate_all()
    
    def validate_all(self) -> None:
        """Validate all configuration settings."""
        self.zerodha_config.validate()
        # Add validation for other configs as needed
    
    def get_zerodha_config(self) -> ZerodhaConfig:
        """Get Zerodha API configuration."""
        return self.zerodha_config
    
    def get_trading_config(self) -> TradingConfig:
        """Get trading configuration."""
        return self.trading_config
    
    def get_compliance_config(self) -> ComplianceConfig:
        """Get compliance configuration."""
        return self.compliance_config
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.zerodha_config.is_sandbox()
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as dictionary."""
        return {
            "zerodha": self.zerodha_config.__dict__,
            "trading": self.trading_config.__dict__,
            "compliance": self.compliance_config.__dict__
        }


# Global configuration instance
config_manager = ConfigManager()
