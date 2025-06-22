"""
Technical Analysis MCP Server for AWM system.
Provides technical indicators and pattern recognition through MCP protocol.
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from decimal import Decimal
import json

# Add the project root to Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from shared.mcp_client.server import MCPServer, validate_required_fields, sanitize_input

logger = logging.getLogger(__name__)

# Try to import TA-Lib, fall back to pandas_ta if not available
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    logger.warning("TA-Lib not available, using pandas_ta as fallback")

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    logger.warning("pandas_ta not available")


class TechnicalAnalysisServer(MCPServer):
    """Technical Analysis MCP Server implementation."""
    
    def __init__(self):
        host = os.getenv("TECHNICAL_ANALYSIS_SERVER_HOST", "0.0.0.0")
        port = int(os.getenv("TECHNICAL_ANALYSIS_SERVER_PORT", "8002"))
        super().__init__("technical_analysis_server", host, port)
        
        # Register handlers
        self.register_handlers()
    
    def register_handlers(self):
        """Register all MCP handlers."""
        
        @self.handler("calculate_indicator")
        async def calculate_indicator(content: Dict[str, Any]) -> Dict[str, Any]:
            """Calculate technical indicators."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["indicator", "data"])
            
            indicator = content["indicator"].upper()
            price_data = content["data"]
            params = content.get("params", {})
            
            # Convert data to pandas DataFrame
            df = pd.DataFrame(price_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    return {"error": f"Missing required column: {col}"}
            
            try:
                result = await self._calculate_indicator(df, indicator, params)
                return {
                    "indicator": indicator,
                    "data": result,
                    "params": params
                }
            except Exception as e:
                logger.error(f"Error calculating {indicator}: {str(e)}")
                return {"error": f"Failed to calculate {indicator}: {str(e)}"}
        
        @self.handler("detect_patterns")
        async def detect_patterns(content: Dict[str, Any]) -> Dict[str, Any]:
            """Detect chart patterns."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["data"])
            
            price_data = content["data"]
            pattern_types = content.get("patterns", ["all"])
            
            # Convert data to pandas DataFrame
            df = pd.DataFrame(price_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            try:
                patterns = await self._detect_patterns(df, pattern_types)
                return {
                    "patterns": patterns,
                    "count": len(patterns)
                }
            except Exception as e:
                logger.error(f"Error detecting patterns: {str(e)}")
                return {"error": f"Failed to detect patterns: {str(e)}"}
        
        @self.handler("run_backtest")
        async def run_backtest(content: Dict[str, Any]) -> Dict[str, Any]:
            """Run strategy backtest."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["strategy", "data"])
            
            strategy = content["strategy"]
            price_data = content["data"]
            params = content.get("params", {})
            
            # Convert data to pandas DataFrame
            df = pd.DataFrame(price_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            try:
                results = await self._run_backtest(df, strategy, params)
                return {
                    "strategy": strategy,
                    "results": results,
                    "params": params
                }
            except Exception as e:
                logger.error(f"Error running backtest: {str(e)}")
                return {"error": f"Failed to run backtest: {str(e)}"}
        
        @self.handler("get_support_resistance")
        async def get_support_resistance(content: Dict[str, Any]) -> Dict[str, Any]:
            """Calculate support and resistance levels."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["data"])
            
            price_data = content["data"]
            lookback = content.get("lookback", 20)
            
            # Convert data to pandas DataFrame
            df = pd.DataFrame(price_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            try:
                levels = await self._calculate_support_resistance(df, lookback)
                return {
                    "support_levels": levels["support"],
                    "resistance_levels": levels["resistance"],
                    "current_price": float(df['close'].iloc[-1])
                }
            except Exception as e:
                logger.error(f"Error calculating support/resistance: {str(e)}")
                return {"error": f"Failed to calculate support/resistance: {str(e)}"}
    
    async def _calculate_indicator(self, df: pd.DataFrame, indicator: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate specific technical indicator."""
        
        if indicator == "RSI":
            period = params.get("period", 14)
            if HAS_TALIB:
                rsi = talib.RSI(df['close'].values, timeperiod=period)
            else:
                rsi = df['close'].rolling(window=period).apply(
                    lambda x: 100 - (100 / (1 + x.diff().clip(lower=0).mean() / x.diff().clip(upper=0).abs().mean()))
                )
            
            result = []
            for i, (timestamp, value) in enumerate(zip(df.index, rsi)):
                if not pd.isna(value):
                    result.append({
                        "timestamp": timestamp.isoformat(),
                        "value": float(value)
                    })
            return result
        
        elif indicator == "MACD":
            fast = params.get("fast", 12)
            slow = params.get("slow", 26)
            signal = params.get("signal", 9)
            
            if HAS_TALIB:
                macd, macd_signal, macd_hist = talib.MACD(df['close'].values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            else:
                ema_fast = df['close'].ewm(span=fast).mean()
                ema_slow = df['close'].ewm(span=slow).mean()
                macd = ema_fast - ema_slow
                macd_signal = macd.ewm(span=signal).mean()
                macd_hist = macd - macd_signal
            
            result = []
            for i, timestamp in enumerate(df.index):
                if not pd.isna(macd.iloc[i]):
                    result.append({
                        "timestamp": timestamp.isoformat(),
                        "macd": float(macd.iloc[i]),
                        "signal": float(macd_signal.iloc[i]) if not pd.isna(macd_signal.iloc[i]) else None,
                        "histogram": float(macd_hist.iloc[i]) if not pd.isna(macd_hist.iloc[i]) else None
                    })
            return result
        
        elif indicator == "BOLLINGER_BANDS":
            period = params.get("period", 20)
            std_dev = params.get("std_dev", 2)
            
            if HAS_TALIB:
                upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
            else:
                middle = df['close'].rolling(window=period).mean()
                std = df['close'].rolling(window=period).std()
                upper = middle + (std * std_dev)
                lower = middle - (std * std_dev)
            
            result = []
            for i, timestamp in enumerate(df.index):
                if not pd.isna(middle.iloc[i]):
                    result.append({
                        "timestamp": timestamp.isoformat(),
                        "upper": float(upper.iloc[i]),
                        "middle": float(middle.iloc[i]),
                        "lower": float(lower.iloc[i])
                    })
            return result
        
        elif indicator == "SMA":
            period = params.get("period", 20)
            if HAS_TALIB:
                sma = talib.SMA(df['close'].values, timeperiod=period)
            else:
                sma = df['close'].rolling(window=period).mean()
            
            result = []
            for i, (timestamp, value) in enumerate(zip(df.index, sma)):
                if not pd.isna(value):
                    result.append({
                        "timestamp": timestamp.isoformat(),
                        "value": float(value)
                    })
            return result
        
        elif indicator == "EMA":
            period = params.get("period", 20)
            if HAS_TALIB:
                ema = talib.EMA(df['close'].values, timeperiod=period)
            else:
                ema = df['close'].ewm(span=period).mean()
            
            result = []
            for i, (timestamp, value) in enumerate(zip(df.index, ema)):
                if not pd.isna(value):
                    result.append({
                        "timestamp": timestamp.isoformat(),
                        "value": float(value)
                    })
            return result
        
        else:
            raise ValueError(f"Unsupported indicator: {indicator}")
    
    async def _detect_patterns(self, df: pd.DataFrame, pattern_types: List[str]) -> List[Dict[str, Any]]:
        """Detect chart patterns."""
        patterns = []
        
        if "all" in pattern_types or "doji" in pattern_types:
            # Simple Doji detection
            doji_threshold = 0.1  # 0.1% of close price
            for i in range(len(df)):
                open_price = df['open'].iloc[i]
                close_price = df['close'].iloc[i]
                high_price = df['high'].iloc[i]
                low_price = df['low'].iloc[i]
                
                body_size = abs(close_price - open_price)
                price_range = high_price - low_price
                
                if body_size <= (close_price * doji_threshold / 100) and price_range > 0:
                    patterns.append({
                        "pattern": "doji",
                        "timestamp": df.index[i].isoformat(),
                        "confidence": 0.8,
                        "description": "Doji candlestick pattern detected"
                    })
        
        if "all" in pattern_types or "hammer" in pattern_types:
            # Simple Hammer detection
            for i in range(len(df)):
                open_price = df['open'].iloc[i]
                close_price = df['close'].iloc[i]
                high_price = df['high'].iloc[i]
                low_price = df['low'].iloc[i]
                
                body_size = abs(close_price - open_price)
                lower_shadow = min(open_price, close_price) - low_price
                upper_shadow = high_price - max(open_price, close_price)
                
                if lower_shadow > 2 * body_size and upper_shadow < body_size:
                    patterns.append({
                        "pattern": "hammer",
                        "timestamp": df.index[i].isoformat(),
                        "confidence": 0.7,
                        "description": "Hammer candlestick pattern detected"
                    })
        
        return patterns
    
    async def _run_backtest(self, df: pd.DataFrame, strategy: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run strategy backtest."""
        
        if strategy == "sma_crossover":
            fast_period = params.get("fast_period", 10)
            slow_period = params.get("slow_period", 20)
            initial_capital = params.get("initial_capital", 100000)
            
            # Calculate SMAs
            df['sma_fast'] = df['close'].rolling(window=fast_period).mean()
            df['sma_slow'] = df['close'].rolling(window=slow_period).mean()
            
            # Generate signals
            df['signal'] = 0
            df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1
            df.loc[df['sma_fast'] <= df['sma_slow'], 'signal'] = -1
            
            # Calculate returns
            df['position'] = df['signal'].shift(1)
            df['returns'] = df['close'].pct_change()
            df['strategy_returns'] = df['position'] * df['returns']
            
            # Calculate performance metrics
            total_return = (1 + df['strategy_returns']).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(df)) - 1
            volatility = df['strategy_returns'].std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            max_drawdown = (df['strategy_returns'].cumsum() - df['strategy_returns'].cumsum().expanding().max()).min()
            
            return {
                "total_return": float(total_return),
                "annual_return": float(annual_return),
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "trades": int(df['signal'].diff().abs().sum() / 2)
            }
        
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
    
    async def _calculate_support_resistance(self, df: pd.DataFrame, lookback: int) -> Dict[str, List[float]]:
        """Calculate support and resistance levels."""
        highs = df['high'].rolling(window=lookback, center=True).max()
        lows = df['low'].rolling(window=lookback, center=True).min()
        
        # Find local maxima and minima
        resistance_levels = []
        support_levels = []
        
        for i in range(lookback, len(df) - lookback):
            if df['high'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(float(df['high'].iloc[i]))
            
            if df['low'].iloc[i] == lows.iloc[i]:
                support_levels.append(float(df['low'].iloc[i]))
        
        # Remove duplicates and sort
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)[:5]
        support_levels = sorted(list(set(support_levels)))[:5]
        
        return {
            "resistance": resistance_levels,
            "support": support_levels
        }


async def main():
    """Main function to run the Technical Analysis MCP Server."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create and start server
        server = TechnicalAnalysisServer()
        logger.info("Starting Technical Analysis MCP Server...")
        await server.start()
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
