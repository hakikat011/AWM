"""
Utility functions for Zerodha integration and Indian market operations.
"""

import re
from datetime import datetime, time
from typing import Dict, Any, Optional, Tuple
from decimal import Decimal
import pytz

# Indian timezone
IST = pytz.timezone('Asia/Kolkata')

# Indian market holidays for 2024 (can be updated annually)
INDIAN_MARKET_HOLIDAYS_2024 = [
    "2024-01-26",  # Republic Day
    "2024-03-08",  # Holi
    "2024-03-25",  # Holi (Second day)
    "2024-04-11",  # Eid ul-Fitr
    "2024-04-17",  # Ram Navami
    "2024-05-01",  # Maharashtra Day
    "2024-08-15",  # Independence Day
    "2024-10-02",  # Gandhi Jayanti
    "2024-11-01",  # Diwali
    "2024-11-15",  # Guru Nanak Jayanti
    "2024-12-25",  # Christmas
]


def format_indian_symbol(symbol: str, exchange: str = "NSE") -> str:
    """
    Format symbol for Indian market trading.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE", "TCS")
        exchange: Exchange name (NSE, BSE)
        
    Returns:
        Formatted symbol for Zerodha (e.g., "NSE:RELIANCE")
    """
    if ":" in symbol:
        return symbol  # Already formatted
    
    return f"{exchange.upper()}:{symbol.upper()}"


def parse_zerodha_symbol(zerodha_symbol: str) -> Tuple[str, str]:
    """
    Parse Zerodha symbol format.
    
    Args:
        zerodha_symbol: Symbol in Zerodha format (e.g., "NSE:RELIANCE")
        
    Returns:
        Tuple of (exchange, symbol)
    """
    if ":" in zerodha_symbol:
        exchange, symbol = zerodha_symbol.split(":", 1)
        return exchange.upper(), symbol.upper()
    else:
        return "NSE", zerodha_symbol.upper()


def convert_to_indian_format(symbol: str, exchange: str = None) -> str:
    """
    Convert various symbol formats to Indian market format.
    
    Args:
        symbol: Symbol in any format
        exchange: Target exchange (NSE, BSE)
        
    Returns:
        Symbol in Indian format
    """
    # Remove common suffixes
    symbol = symbol.upper()
    symbol = re.sub(r'\.(NS|BO|EQ)$', '', symbol)
    
    # Default to NSE if no exchange specified
    if not exchange:
        exchange = "NSE"
    
    return format_indian_symbol(symbol, exchange)


def validate_trading_hours(dt: datetime = None) -> Dict[str, Any]:
    """
    Validate if the given time is within Indian market trading hours.
    
    Args:
        dt: Datetime to check (defaults to current time)
        
    Returns:
        Dictionary with trading status information
    """
    if dt is None:
        dt = datetime.now(IST)
    elif dt.tzinfo is None:
        dt = IST.localize(dt)
    else:
        dt = dt.astimezone(IST)
    
    # Check if it's a weekend
    if dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return {
            "is_trading_hours": False,
            "reason": "Weekend",
            "next_trading_day": _get_next_trading_day(dt)
        }
    
    # Check if it's a holiday
    date_str = dt.strftime("%Y-%m-%d")
    if date_str in INDIAN_MARKET_HOLIDAYS_2024:
        return {
            "is_trading_hours": False,
            "reason": "Market Holiday",
            "next_trading_day": _get_next_trading_day(dt)
        }
    
    # Regular market hours: 9:15 AM to 3:30 PM IST
    market_open = time(9, 15)
    market_close = time(15, 30)
    current_time = dt.time()
    
    # Pre-market session: 9:00 AM to 9:15 AM
    pre_market_open = time(9, 0)
    
    # Post-market session: 3:40 PM to 4:00 PM
    post_market_open = time(15, 40)
    post_market_close = time(16, 0)
    
    if market_open <= current_time <= market_close:
        return {
            "is_trading_hours": True,
            "session": "regular",
            "time_to_close": _time_until(dt, market_close)
        }
    elif pre_market_open <= current_time < market_open:
        return {
            "is_trading_hours": False,
            "session": "pre_market",
            "reason": "Pre-market session",
            "time_to_open": _time_until(dt, market_open)
        }
    elif post_market_open <= current_time <= post_market_close:
        return {
            "is_trading_hours": False,
            "session": "post_market", 
            "reason": "Post-market session",
            "next_trading_day": _get_next_trading_day(dt)
        }
    else:
        return {
            "is_trading_hours": False,
            "session": "closed",
            "reason": "Market closed",
            "next_trading_day": _get_next_trading_day(dt)
        }


def _time_until(current_dt: datetime, target_time: time) -> str:
    """Calculate time until target time."""
    target_dt = current_dt.replace(
        hour=target_time.hour,
        minute=target_time.minute,
        second=target_time.second,
        microsecond=0
    )
    
    if target_dt <= current_dt:
        # Target time is tomorrow
        target_dt = target_dt.replace(day=target_dt.day + 1)
    
    time_diff = target_dt - current_dt
    hours, remainder = divmod(time_diff.total_seconds(), 3600)
    minutes, _ = divmod(remainder, 60)
    
    return f"{int(hours):02d}:{int(minutes):02d}"


def _get_next_trading_day(current_dt: datetime) -> str:
    """Get the next trading day."""
    next_day = current_dt
    
    while True:
        next_day = next_day.replace(day=next_day.day + 1)
        
        # Skip weekends
        if next_day.weekday() >= 5:
            continue
        
        # Skip holidays
        if next_day.strftime("%Y-%m-%d") in INDIAN_MARKET_HOLIDAYS_2024:
            continue
        
        return next_day.strftime("%Y-%m-%d")


def calculate_indian_taxes(
    transaction_value: Decimal,
    transaction_type: str,
    holding_period_days: int = 0,
    exchange: str = "NSE"
) -> Dict[str, Decimal]:
    """
    Calculate Indian market taxes and charges.
    
    Args:
        transaction_value: Transaction value in INR
        transaction_type: "BUY" or "SELL"
        holding_period_days: Days held (for capital gains calculation)
        exchange: Exchange (NSE, BSE)
        
    Returns:
        Dictionary with tax breakdown
    """
    taxes = {}
    
    # STT (Securities Transaction Tax)
    if transaction_type.upper() == "BUY":
        # STT on purchase: 0.1% for delivery, 0.025% for intraday
        if holding_period_days > 0:  # Delivery
            stt_rate = Decimal("0.001")  # 0.1%
        else:  # Intraday
            stt_rate = Decimal("0.00025")  # 0.025%
    else:  # SELL
        # STT on sale: 0.1% for delivery, 0.025% for intraday
        if holding_period_days > 0:  # Delivery
            stt_rate = Decimal("0.001")  # 0.1%
        else:  # Intraday
            stt_rate = Decimal("0.00025")  # 0.025%
    
    taxes["stt"] = transaction_value * stt_rate
    
    # Exchange Transaction Charges
    if exchange.upper() == "NSE":
        # NSE: 0.00345% for equity delivery, 0.00173% for intraday
        if holding_period_days > 0:
            exchange_rate = Decimal("0.0000345")  # 0.00345%
        else:
            exchange_rate = Decimal("0.0000173")  # 0.00173%
    else:  # BSE
        # BSE: 0.00375% for equity delivery, 0.00188% for intraday
        if holding_period_days > 0:
            exchange_rate = Decimal("0.0000375")  # 0.00375%
        else:
            exchange_rate = Decimal("0.0000188")  # 0.00188%
    
    taxes["exchange_charges"] = transaction_value * exchange_rate
    
    # SEBI Charges: ₹10 per crore
    sebi_rate = Decimal("0.000001")  # ₹10 per ₹1 crore
    taxes["sebi_charges"] = transaction_value * sebi_rate
    
    # GST on (Brokerage + Exchange charges + SEBI charges)
    # Assuming zero brokerage for calculation
    taxable_amount = taxes["exchange_charges"] + taxes["sebi_charges"]
    taxes["gst"] = taxable_amount * Decimal("0.18")  # 18% GST
    
    # Stamp Duty (only on purchase)
    if transaction_type.upper() == "BUY":
        # 0.015% or ₹1500 per crore, whichever is lower
        stamp_duty_rate = Decimal("0.00015")  # 0.015%
        stamp_duty_max = transaction_value * Decimal("0.00015")  # ₹1500 per crore
        taxes["stamp_duty"] = min(transaction_value * stamp_duty_rate, stamp_duty_max)
    else:
        taxes["stamp_duty"] = Decimal("0")
    
    # Total charges
    taxes["total_charges"] = sum(taxes.values())
    
    # Capital Gains Tax (only on sell transactions with profit)
    if transaction_type.upper() == "SELL" and holding_period_days > 0:
        # This would require purchase price to calculate actual gains
        # For now, just indicate the rates
        if holding_period_days <= 365:
            taxes["stcg_rate"] = Decimal("0.15")  # 15% STCG
        else:
            taxes["ltcg_rate"] = Decimal("0.10")  # 10% LTCG above ₹1 lakh
    
    return taxes


def get_lot_size(symbol: str, exchange: str = "NSE") -> int:
    """
    Get lot size for a symbol (for F&O contracts).
    This is a simplified version - in production, this should be fetched from Zerodha instruments.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        
    Returns:
        Lot size (1 for equity, specific values for F&O)
    """
    # Common F&O lot sizes (as of 2024)
    fo_lot_sizes = {
        "NIFTY": 50,
        "BANKNIFTY": 15,
        "RELIANCE": 250,
        "TCS": 150,
        "INFY": 300,
        "HDFCBANK": 550,
        "ICICIBANK": 1375,
        "SBIN": 3000,
        "ITC": 3200,
        "HINDUNILVR": 300,
        "BAJFINANCE": 125,
        "ASIANPAINT": 400,
        "MARUTI": 100,
        "KOTAKBANK": 400,
        "LT": 700,
        "AXISBANK": 1200,
        "WIPRO": 1200,
        "ULTRACEMCO": 300,
        "SUNPHARMA": 1000,
        "TITAN": 2000
    }
    
    # Extract base symbol
    base_symbol = symbol.split("-")[0].upper()  # Remove expiry/strike info
    
    return fo_lot_sizes.get(base_symbol, 1)  # Default to 1 for equity


def get_tick_size(symbol: str, price: float) -> float:
    """
    Get tick size for a symbol based on price.
    
    Args:
        symbol: Trading symbol
        price: Current price
        
    Returns:
        Tick size
    """
    # Indian market tick sizes based on price ranges
    if price <= 0:
        return 0.01
    elif price < 1:
        return 0.0025
    elif price < 10:
        return 0.01
    elif price < 20:
        return 0.01
    elif price < 50:
        return 0.01
    elif price < 100:
        return 0.05
    elif price < 200:
        return 0.05
    elif price < 500:
        return 0.05
    elif price < 1000:
        return 0.05
    elif price < 2000:
        return 0.10
    elif price < 5000:
        return 0.10
    elif price < 10000:
        return 0.10
    else:
        return 0.10


def format_indian_currency(amount: float, include_symbol: bool = True) -> str:
    """
    Format amount in Indian currency format.
    
    Args:
        amount: Amount to format
        include_symbol: Whether to include ₹ symbol
        
    Returns:
        Formatted currency string
    """
    # Indian number formatting (lakhs and crores)
    if abs(amount) >= 10000000:  # 1 crore
        formatted = f"{amount/10000000:.2f} Cr"
    elif abs(amount) >= 100000:  # 1 lakh
        formatted = f"{amount/100000:.2f} L"
    elif abs(amount) >= 1000:  # 1 thousand
        formatted = f"{amount/1000:.2f} K"
    else:
        formatted = f"{amount:.2f}"
    
    if include_symbol:
        return f"₹{formatted}"
    else:
        return formatted
