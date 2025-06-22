"""
Quantitative Analysis MCP Server for AWM system.
Provides advanced quantitative analysis, signal generation, and strategy implementation.
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from decimal import Decimal
import json
from datetime import datetime, timedelta
import statistics

# Add the project root to Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from shared.mcp_client.server import MCPServer, validate_required_fields, sanitize_input

logger = logging.getLogger(__name__)

# Try to import TA-Lib and other analysis libraries
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    logger.warning("TA-Lib not available")

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    logger.warning("pandas_ta not available")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not available")


class QuantitativeAnalysisServer(MCPServer):
    """Quantitative Analysis MCP Server implementation."""
    
    def __init__(self):
        host = os.getenv("QUANTITATIVE_ANALYSIS_SERVER_HOST", "0.0.0.0")
        port = int(os.getenv("QUANTITATIVE_ANALYSIS_SERVER_PORT", "8003"))
        super().__init__("quantitative_analysis_server", host, port)
        
        # Initialize strategy configurations
        self.strategy_configs = {
            "sma_crossover": {
                "short_period": 20,
                "long_period": 50,
                "signal_threshold": 0.02
            },
            "rsi_mean_reversion": {
                "rsi_period": 14,
                "oversold_threshold": 30,
                "overbought_threshold": 70,
                "exit_threshold": 50
            },
            "bollinger_bands": {
                "period": 20,
                "std_dev": 2,
                "squeeze_threshold": 0.1
            },
            "momentum": {
                "lookback_period": 10,
                "momentum_threshold": 0.05
            }
        }
        
        # Register handlers
        self.register_handlers()
    
    def register_handlers(self):
        """Register all MCP handlers."""
        
        @self.handler("generate_signals")
        async def generate_signals(content: Dict[str, Any]) -> Dict[str, Any]:
            """Generate trading signals for given market data."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["symbol", "data"])
            
            symbol = content["symbol"]
            price_data = content["data"]
            strategies = content.get("strategies", ["sma_crossover", "rsi_mean_reversion"])
            
            # Convert data to pandas DataFrame
            df = self._prepare_dataframe(price_data)
            
            try:
                signals = await self._generate_trading_signals(df, symbol, strategies)
                return {
                    "symbol": symbol,
                    "signals": signals,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error generating signals for {symbol}: {str(e)}")
                return {"error": f"Failed to generate signals: {str(e)}"}
        
        @self.handler("analyze_strategy")
        async def analyze_strategy(content: Dict[str, Any]) -> Dict[str, Any]:
            """Analyze a specific quantitative strategy."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["strategy", "data"])
            
            strategy = content["strategy"]
            price_data = content["data"]
            params = content.get("params", {})
            
            df = self._prepare_dataframe(price_data)
            
            try:
                analysis = await self._analyze_strategy(df, strategy, params)
                return {
                    "strategy": strategy,
                    "analysis": analysis,
                    "params": params
                }
            except Exception as e:
                logger.error(f"Error analyzing strategy {strategy}: {str(e)}")
                return {"error": f"Failed to analyze strategy: {str(e)}"}
        
        @self.handler("backtest_strategy")
        async def backtest_strategy(content: Dict[str, Any]) -> Dict[str, Any]:
            """Run comprehensive backtest for a strategy."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["strategy", "data"])
            
            strategy = content["strategy"]
            price_data = content["data"]
            params = content.get("params", {})
            initial_capital = content.get("initial_capital", 100000)
            
            df = self._prepare_dataframe(price_data)
            
            try:
                backtest_results = await self._run_comprehensive_backtest(
                    df, strategy, params, initial_capital
                )
                return {
                    "strategy": strategy,
                    "backtest_results": backtest_results,
                    "params": params
                }
            except Exception as e:
                logger.error(f"Error backtesting strategy {strategy}: {str(e)}")
                return {"error": f"Failed to backtest strategy: {str(e)}"}
        
        @self.handler("calculate_risk_metrics")
        async def calculate_risk_metrics(content: Dict[str, Any]) -> Dict[str, Any]:
            """Calculate comprehensive risk metrics."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["returns"])
            
            returns = content["returns"]
            benchmark_returns = content.get("benchmark_returns", [])
            
            try:
                risk_metrics = await self._calculate_risk_metrics(returns, benchmark_returns)
                return {
                    "risk_metrics": risk_metrics,
                    "calculation_timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error calculating risk metrics: {str(e)}")
                return {"error": f"Failed to calculate risk metrics: {str(e)}"}
        
        @self.handler("optimize_portfolio")
        async def optimize_portfolio(content: Dict[str, Any]) -> Dict[str, Any]:
            """Optimize portfolio allocation using quantitative methods."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["assets", "returns_data"])
            
            assets = content["assets"]
            returns_data = content["returns_data"]
            method = content.get("method", "mean_variance")
            constraints = content.get("constraints", {})
            
            try:
                optimization_results = await self._optimize_portfolio(
                    assets, returns_data, method, constraints
                )
                return {
                    "optimization_results": optimization_results,
                    "method": method,
                    "assets": assets
                }
            except Exception as e:
                logger.error(f"Error optimizing portfolio: {str(e)}")
                return {"error": f"Failed to optimize portfolio: {str(e)}"}
        
        @self.handler("detect_market_regime")
        async def detect_market_regime(content: Dict[str, Any]) -> Dict[str, Any]:
            """Detect current market regime (trending, ranging, volatile)."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["data"])
            
            price_data = content["data"]
            lookback_period = content.get("lookback_period", 50)
            
            df = self._prepare_dataframe(price_data)
            
            try:
                regime_analysis = await self._detect_market_regime(df, lookback_period)
                return {
                    "market_regime": regime_analysis,
                    "lookback_period": lookback_period
                }
            except Exception as e:
                logger.error(f"Error detecting market regime: {str(e)}")
                return {"error": f"Failed to detect market regime: {str(e)}"}

    def _prepare_dataframe(self, price_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert price data to pandas DataFrame with proper formatting."""
        df = pd.DataFrame(price_data)
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                if col == 'volume':
                    df[col] = 0  # Default volume if not provided
                else:
                    raise ValueError(f"Required column '{col}' not found in data")
        
        # Convert timestamp and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Convert price columns to float
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        
        # Sort by timestamp
        df = df.sort_index()
        
        return df

    async def _generate_trading_signals(self, df: pd.DataFrame, symbol: str, strategies: List[str]) -> List[Dict[str, Any]]:
        """Generate trading signals using multiple quantitative strategies."""
        signals = []

        for strategy in strategies:
            try:
                if strategy == "sma_crossover":
                    strategy_signals = await self._sma_crossover_signals(df)
                elif strategy == "rsi_mean_reversion":
                    strategy_signals = await self._rsi_mean_reversion_signals(df)
                elif strategy == "bollinger_bands":
                    strategy_signals = await self._bollinger_bands_signals(df)
                elif strategy == "momentum":
                    strategy_signals = await self._momentum_signals(df)
                else:
                    logger.warning(f"Unknown strategy: {strategy}")
                    continue

                # Add strategy name to each signal
                for signal in strategy_signals:
                    signal["strategy"] = strategy
                    signal["symbol"] = symbol

                signals.extend(strategy_signals)

            except Exception as e:
                logger.error(f"Error generating {strategy} signals: {e}")

        # Sort signals by timestamp
        signals.sort(key=lambda x: x.get("timestamp", ""))

        return signals

    async def _sma_crossover_signals(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate signals based on Simple Moving Average crossover."""
        config = self.strategy_configs["sma_crossover"]
        short_period = config["short_period"]
        long_period = config["long_period"]

        # Calculate SMAs
        df['sma_short'] = df['close'].rolling(window=short_period).mean()
        df['sma_long'] = df['close'].rolling(window=long_period).mean()

        # Generate signals
        signals = []
        position = 0  # 0: no position, 1: long, -1: short

        for i in range(long_period, len(df)):
            current_short = df['sma_short'].iloc[i]
            current_long = df['sma_long'].iloc[i]
            prev_short = df['sma_short'].iloc[i-1]
            prev_long = df['sma_long'].iloc[i-1]

            # Check for crossover
            if prev_short <= prev_long and current_short > current_long and position != 1:
                # Golden cross - buy signal
                signals.append({
                    "timestamp": df.index[i].isoformat(),
                    "signal": "BUY",
                    "price": df['close'].iloc[i],
                    "confidence": 0.7,
                    "reason": f"SMA crossover: {short_period}-period above {long_period}-period",
                    "indicators": {
                        "sma_short": current_short,
                        "sma_long": current_long
                    }
                })
                position = 1

            elif prev_short >= prev_long and current_short < current_long and position != -1:
                # Death cross - sell signal
                signals.append({
                    "timestamp": df.index[i].isoformat(),
                    "signal": "SELL",
                    "price": df['close'].iloc[i],
                    "confidence": 0.7,
                    "reason": f"SMA crossover: {short_period}-period below {long_period}-period",
                    "indicators": {
                        "sma_short": current_short,
                        "sma_long": current_long
                    }
                })
                position = -1

        return signals

    async def _rsi_mean_reversion_signals(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate signals based on RSI mean reversion strategy."""
        config = self.strategy_configs["rsi_mean_reversion"]
        rsi_period = config["rsi_period"]
        oversold = config["oversold_threshold"]
        overbought = config["overbought_threshold"]
        exit_threshold = config["exit_threshold"]

        # Calculate RSI
        if HAS_TALIB:
            df['rsi'] = talib.RSI(df['close'].values, timeperiod=rsi_period)
        else:
            # Simple RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

        signals = []
        position = 0  # 0: no position, 1: long, -1: short

        for i in range(rsi_period, len(df)):
            rsi_value = df['rsi'].iloc[i]
            price = df['close'].iloc[i]

            if rsi_value < oversold and position != 1:
                # Oversold - buy signal
                signals.append({
                    "timestamp": df.index[i].isoformat(),
                    "signal": "BUY",
                    "price": price,
                    "confidence": 0.8,
                    "reason": f"RSI oversold: {rsi_value:.2f} < {oversold}",
                    "indicators": {"rsi": rsi_value}
                })
                position = 1

            elif rsi_value > overbought and position != -1:
                # Overbought - sell signal
                signals.append({
                    "timestamp": df.index[i].isoformat(),
                    "signal": "SELL",
                    "price": price,
                    "confidence": 0.8,
                    "reason": f"RSI overbought: {rsi_value:.2f} > {overbought}",
                    "indicators": {"rsi": rsi_value}
                })
                position = -1

            elif abs(rsi_value - exit_threshold) < 5 and position != 0:
                # Exit signal when RSI returns to neutral
                signals.append({
                    "timestamp": df.index[i].isoformat(),
                    "signal": "EXIT",
                    "price": price,
                    "confidence": 0.6,
                    "reason": f"RSI returning to neutral: {rsi_value:.2f}",
                    "indicators": {"rsi": rsi_value}
                })
                position = 0

        return signals

    async def _bollinger_bands_signals(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate signals based on Bollinger Bands strategy."""
        config = self.strategy_configs["bollinger_bands"]
        period = config["period"]
        std_dev = config["std_dev"]

        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        df['bb_std'] = df['close'].rolling(window=period).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_dev)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_dev)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        signals = []
        position = 0

        for i in range(period, len(df)):
            price = df['close'].iloc[i]
            upper = df['bb_upper'].iloc[i]
            lower = df['bb_lower'].iloc[i]
            middle = df['bb_middle'].iloc[i]
            width = df['bb_width'].iloc[i]

            if price <= lower and position != 1:
                # Price touches lower band - buy signal
                signals.append({
                    "timestamp": df.index[i].isoformat(),
                    "signal": "BUY",
                    "price": price,
                    "confidence": 0.75,
                    "reason": f"Price at lower Bollinger Band: {price:.2f} <= {lower:.2f}",
                    "indicators": {
                        "bb_upper": upper,
                        "bb_lower": lower,
                        "bb_middle": middle,
                        "bb_width": width
                    }
                })
                position = 1

            elif price >= upper and position != -1:
                # Price touches upper band - sell signal
                signals.append({
                    "timestamp": df.index[i].isoformat(),
                    "signal": "SELL",
                    "price": price,
                    "confidence": 0.75,
                    "reason": f"Price at upper Bollinger Band: {price:.2f} >= {upper:.2f}",
                    "indicators": {
                        "bb_upper": upper,
                        "bb_lower": lower,
                        "bb_middle": middle,
                        "bb_width": width
                    }
                })
                position = -1

            elif abs(price - middle) / middle < 0.01 and position != 0:
                # Price returns to middle band - exit signal
                signals.append({
                    "timestamp": df.index[i].isoformat(),
                    "signal": "EXIT",
                    "price": price,
                    "confidence": 0.6,
                    "reason": f"Price returning to middle band: {price:.2f} ≈ {middle:.2f}",
                    "indicators": {
                        "bb_upper": upper,
                        "bb_lower": lower,
                        "bb_middle": middle,
                        "bb_width": width
                    }
                })
                position = 0

        return signals

    async def _momentum_signals(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate signals based on momentum strategy."""
        config = self.strategy_configs["momentum"]
        lookback_period = config["lookback_period"]
        momentum_threshold = config["momentum_threshold"]

        # Calculate momentum
        df['momentum'] = df['close'].pct_change(periods=lookback_period)
        df['momentum_ma'] = df['momentum'].rolling(window=5).mean()

        signals = []
        position = 0

        for i in range(lookback_period + 5, len(df)):
            momentum = df['momentum'].iloc[i]
            momentum_ma = df['momentum_ma'].iloc[i]
            price = df['close'].iloc[i]

            if momentum > momentum_threshold and momentum_ma > 0 and position != 1:
                # Strong positive momentum - buy signal
                signals.append({
                    "timestamp": df.index[i].isoformat(),
                    "signal": "BUY",
                    "price": price,
                    "confidence": 0.7,
                    "reason": f"Strong positive momentum: {momentum:.3f} > {momentum_threshold}",
                    "indicators": {
                        "momentum": momentum,
                        "momentum_ma": momentum_ma
                    }
                })
                position = 1

            elif momentum < -momentum_threshold and momentum_ma < 0 and position != -1:
                # Strong negative momentum - sell signal
                signals.append({
                    "timestamp": df.index[i].isoformat(),
                    "signal": "SELL",
                    "price": price,
                    "confidence": 0.7,
                    "reason": f"Strong negative momentum: {momentum:.3f} < {-momentum_threshold}",
                    "indicators": {
                        "momentum": momentum,
                        "momentum_ma": momentum_ma
                    }
                })
                position = -1

            elif abs(momentum) < momentum_threshold / 2 and position != 0:
                # Momentum weakening - exit signal
                signals.append({
                    "timestamp": df.index[i].isoformat(),
                    "signal": "EXIT",
                    "price": price,
                    "confidence": 0.6,
                    "reason": f"Momentum weakening: {momentum:.3f}",
                    "indicators": {
                        "momentum": momentum,
                        "momentum_ma": momentum_ma
                    }
                })
                position = 0

        return signals

    async def _analyze_strategy(self, df: pd.DataFrame, strategy: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a specific strategy's performance characteristics."""
        # Update strategy config with custom params
        if strategy in self.strategy_configs:
            config = self.strategy_configs[strategy].copy()
            config.update(params)
        else:
            config = params

        # Generate signals for the strategy
        signals = []
        if strategy == "sma_crossover":
            signals = await self._sma_crossover_signals(df)
        elif strategy == "rsi_mean_reversion":
            signals = await self._rsi_mean_reversion_signals(df)
        elif strategy == "bollinger_bands":
            signals = await self._bollinger_bands_signals(df)
        elif strategy == "momentum":
            signals = await self._momentum_signals(df)

        # Analyze signal characteristics
        analysis = {
            "total_signals": len(signals),
            "buy_signals": len([s for s in signals if s["signal"] == "BUY"]),
            "sell_signals": len([s for s in signals if s["signal"] == "SELL"]),
            "exit_signals": len([s for s in signals if s["signal"] == "EXIT"]),
            "average_confidence": statistics.mean([s["confidence"] for s in signals]) if signals else 0,
            "signal_frequency": len(signals) / len(df) if len(df) > 0 else 0,
            "config_used": config
        }

        # Calculate signal distribution over time
        if signals:
            signal_dates = [pd.to_datetime(s["timestamp"]) for s in signals]
            analysis["first_signal"] = min(signal_dates).isoformat()
            analysis["last_signal"] = max(signal_dates).isoformat()

            # Signal spacing analysis
            if len(signal_dates) > 1:
                signal_dates.sort()
                intervals = [(signal_dates[i] - signal_dates[i-1]).days for i in range(1, len(signal_dates))]
                analysis["average_signal_interval_days"] = statistics.mean(intervals)
                analysis["median_signal_interval_days"] = statistics.median(intervals)

        return analysis

    async def _run_comprehensive_backtest(self, df: pd.DataFrame, strategy: str, params: Dict[str, Any], initial_capital: float) -> Dict[str, Any]:
        """Run comprehensive backtest with performance metrics."""
        # Generate signals
        signals = []
        if strategy == "sma_crossover":
            signals = await self._sma_crossover_signals(df)
        elif strategy == "rsi_mean_reversion":
            signals = await self._rsi_mean_reversion_signals(df)
        elif strategy == "bollinger_bands":
            signals = await self._bollinger_bands_signals(df)
        elif strategy == "momentum":
            signals = await self._momentum_signals(df)

        if not signals:
            return {"error": "No signals generated for backtest"}

        # Simulate trading
        portfolio_value = initial_capital
        position = 0  # Number of shares held
        cash = initial_capital
        trades = []
        portfolio_history = []

        # Create a price lookup for signals
        signal_dict = {pd.to_datetime(s["timestamp"]): s for s in signals}

        for timestamp, row in df.iterrows():
            current_price = row['close']

            # Check if there's a signal for this timestamp
            if timestamp in signal_dict:
                signal = signal_dict[timestamp]

                if signal["signal"] == "BUY" and position <= 0:
                    # Buy signal - go long
                    shares_to_buy = int(cash / current_price)
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price
                        cash -= cost
                        position += shares_to_buy

                        trades.append({
                            "timestamp": timestamp.isoformat(),
                            "action": "BUY",
                            "price": current_price,
                            "quantity": shares_to_buy,
                            "value": cost,
                            "strategy": strategy
                        })

                elif signal["signal"] == "SELL" and position >= 0:
                    # Sell signal - go short or close long
                    if position > 0:
                        # Close long position
                        proceeds = position * current_price
                        cash += proceeds

                        trades.append({
                            "timestamp": timestamp.isoformat(),
                            "action": "SELL",
                            "price": current_price,
                            "quantity": position,
                            "value": proceeds,
                            "strategy": strategy
                        })

                        position = 0

                elif signal["signal"] == "EXIT" and position != 0:
                    # Exit current position
                    if position > 0:
                        proceeds = position * current_price
                        cash += proceeds

                        trades.append({
                            "timestamp": timestamp.isoformat(),
                            "action": "EXIT_LONG",
                            "price": current_price,
                            "quantity": position,
                            "value": proceeds,
                            "strategy": strategy
                        })

                        position = 0

            # Calculate current portfolio value
            current_portfolio_value = cash + (position * current_price)
            portfolio_history.append({
                "timestamp": timestamp.isoformat(),
                "portfolio_value": current_portfolio_value,
                "cash": cash,
                "position": position,
                "price": current_price
            })

        # Calculate performance metrics
        final_value = portfolio_history[-1]["portfolio_value"] if portfolio_history else initial_capital
        total_return = (final_value - initial_capital) / initial_capital

        # Calculate returns series for risk metrics
        portfolio_values = [p["portfolio_value"] for p in portfolio_history]
        returns = [
            (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            for i in range(1, len(portfolio_values))
        ]

        risk_metrics = await self._calculate_risk_metrics(returns, [])

        return {
            "initial_capital": initial_capital,
            "final_value": final_value,
            "total_return": total_return,
            "total_trades": len(trades),
            "portfolio_history": portfolio_history[-100:],  # Last 100 points for visualization
            "trades": trades,
            "risk_metrics": risk_metrics,
            "strategy": strategy
        }

    async def _calculate_risk_metrics(self, returns: List[float], benchmark_returns: List[float]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics."""
        if not returns:
            return {"error": "No returns data provided"}

        returns_array = np.array(returns)

        # Basic metrics
        mean_return = np.mean(returns_array)
        volatility = np.std(returns_array)

        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0

        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Value at Risk (VaR) at 95% confidence
        var_95 = np.percentile(returns_array, 5)

        # Expected Shortfall (Conditional VaR)
        expected_shortfall = np.mean(returns_array[returns_array <= var_95])

        # Sortino ratio (downside deviation)
        downside_returns = returns_array[returns_array < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0

        # Calmar ratio
        calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate
        positive_returns = len(returns_array[returns_array > 0])
        win_rate = positive_returns / len(returns_array)

        metrics = {
            "mean_return": float(mean_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "calmar_ratio": float(calmar_ratio),
            "max_drawdown": float(max_drawdown),
            "var_95": float(var_95),
            "expected_shortfall": float(expected_shortfall),
            "win_rate": float(win_rate),
            "total_periods": len(returns_array)
        }

        # Beta and alpha if benchmark provided
        if benchmark_returns and len(benchmark_returns) == len(returns):
            benchmark_array = np.array(benchmark_returns)
            covariance = np.cov(returns_array, benchmark_array)[0, 1]
            benchmark_variance = np.var(benchmark_array)

            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            alpha = mean_return - beta * np.mean(benchmark_array)

            metrics["beta"] = float(beta)
            metrics["alpha"] = float(alpha)

        return metrics

    async def _optimize_portfolio(self, assets: List[str], returns_data: Dict[str, List[float]],
                                method: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize portfolio allocation."""
        if not HAS_SKLEARN:
            return {"error": "scikit-learn not available for portfolio optimization"}

        # Convert returns data to matrix
        returns_matrix = []
        for asset in assets:
            if asset in returns_data:
                returns_matrix.append(returns_data[asset])
            else:
                return {"error": f"Returns data not found for asset: {asset}"}

        returns_matrix = np.array(returns_matrix).T  # Transpose to get time x assets

        if method == "mean_variance":
            # Simple equal-weight portfolio for now
            # In a full implementation, this would use optimization libraries like cvxpy
            n_assets = len(assets)
            weights = np.ones(n_assets) / n_assets

            # Calculate portfolio metrics
            portfolio_return = np.mean(np.sum(returns_matrix * weights, axis=1))
            portfolio_volatility = np.std(np.sum(returns_matrix * weights, axis=1))

            return {
                "method": method,
                "weights": {asset: float(weight) for asset, weight in zip(assets, weights)},
                "expected_return": float(portfolio_return),
                "expected_volatility": float(portfolio_volatility),
                "sharpe_ratio": float(portfolio_return / portfolio_volatility) if portfolio_volatility > 0 else 0
            }

        return {"error": f"Optimization method '{method}' not implemented"}

    async def _detect_market_regime(self, df: pd.DataFrame, lookback_period: int) -> Dict[str, Any]:
        """Detect current market regime."""
        if len(df) < lookback_period:
            return {"error": "Insufficient data for regime detection"}

        # Calculate recent price statistics
        recent_data = df.tail(lookback_period)
        returns = recent_data['close'].pct_change().dropna()

        # Trend detection
        price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
        trend_strength = abs(price_change)

        # Volatility analysis
        volatility = returns.std()
        mean_volatility = df['close'].pct_change().rolling(window=lookback_period).std().mean()
        volatility_ratio = volatility / mean_volatility if mean_volatility > 0 else 1

        # Range analysis
        high_low_range = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['close'].mean()

        # Determine regime
        if trend_strength > 0.1 and volatility_ratio < 1.5:
            if price_change > 0:
                regime = "BULL_TRENDING"
            else:
                regime = "BEAR_TRENDING"
        elif volatility_ratio > 2.0:
            regime = "HIGH_VOLATILITY"
        elif high_low_range < 0.05:
            regime = "LOW_VOLATILITY_RANGING"
        else:
            regime = "RANGING"

        return {
            "regime": regime,
            "trend_strength": float(trend_strength),
            "volatility_ratio": float(volatility_ratio),
            "price_change": float(price_change),
            "high_low_range": float(high_low_range),
            "confidence": 0.7,  # Simple confidence score
            "lookback_period": lookback_period
        }

if __name__ == "__main__":
    async def main():
        """Main function to run the Quantitative Analysis MCP Server."""
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        try:
            # Create and start server
            server = QuantitativeAnalysisServer()
            logger.info("Starting Quantitative Analysis MCP Server...")
            await server.start()
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise
    
    asyncio.run(main())
