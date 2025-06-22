"""
Signal Generation MCP Server for AWM system.
Generates trading signals by combining multiple quantitative strategies and market analysis.
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Add the project root to Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from shared.mcp_client.server import MCPServer, validate_required_fields, sanitize_input
from shared.mcp_client.base import MCPClient

logger = logging.getLogger(__name__)


class SignalGenerationServer(MCPServer):
    """Signal Generation MCP Server implementation."""
    
    def __init__(self):
        host = os.getenv("SIGNAL_GENERATION_SERVER_HOST", "0.0.0.0")
        port = int(os.getenv("SIGNAL_GENERATION_SERVER_PORT", "8004"))
        super().__init__("signal_generation_server", host, port)
        
        # MCP client for calling other servers
        self.mcp_client = MCPClient("signal_generation")
        
        # Server URLs
        self.server_urls = {
            "quantitative_analysis": os.getenv("QUANTITATIVE_ANALYSIS_SERVER_URL", "http://quantitative-analysis-server:8003"),
            "technical_analysis": os.getenv("TECHNICAL_ANALYSIS_SERVER_URL", "http://technical-analysis-server:8002"),
            "market_data": os.getenv("MARKET_DATA_SERVER_URL", "http://market-data-server:8001"),
            "news": os.getenv("NEWS_SERVER_URL", "http://news-server:8006"),
            "llm_market_intelligence": os.getenv("LLM_MARKET_INTELLIGENCE_SERVER_URL", "http://llm-market-intelligence-server:8007")
        }
        
        # Signal generation configurations
        self.signal_configs = {
            "default": {
                "strategies": ["sma_crossover", "rsi_mean_reversion", "bollinger_bands"],
                "min_confidence": 0.6,
                "consensus_threshold": 0.7,
                "lookback_days": 30
            },
            "conservative": {
                "strategies": ["sma_crossover", "bollinger_bands"],
                "min_confidence": 0.8,
                "consensus_threshold": 0.8,
                "lookback_days": 50
            },
            "aggressive": {
                "strategies": ["rsi_mean_reversion", "momentum", "bollinger_bands"],
                "min_confidence": 0.5,
                "consensus_threshold": 0.6,
                "lookback_days": 20
            }
        }
        
        # Register handlers
        self.register_handlers()
    
    def register_handlers(self):
        """Register all MCP handlers."""
        
        @self.handler("generate_signals")
        async def generate_signals(content: Dict[str, Any]) -> Dict[str, Any]:
            """Generate comprehensive trading signals for a symbol."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["symbol"])
            
            symbol = content["symbol"]
            config_name = content.get("config", "default")
            custom_strategies = content.get("strategies", None)
            
            try:
                signals = await self._generate_comprehensive_signals(symbol, config_name, custom_strategies)
                return {
                    "symbol": symbol,
                    "signals": signals,
                    "generation_timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error generating signals for {symbol}: {str(e)}")
                return {"error": f"Failed to generate signals: {str(e)}"}
        
        @self.handler("generate_watchlist_signals")
        async def generate_watchlist_signals(content: Dict[str, Any]) -> Dict[str, Any]:
            """Generate signals for multiple symbols in a watchlist."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["symbols"])
            
            symbols = content["symbols"]
            config_name = content.get("config", "default")
            
            try:
                watchlist_signals = await self._generate_watchlist_signals(symbols, config_name)
                return {
                    "watchlist_signals": watchlist_signals,
                    "generation_timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error generating watchlist signals: {str(e)}")
                return {"error": f"Failed to generate watchlist signals: {str(e)}"}
        
        @self.handler("get_signal_consensus")
        async def get_signal_consensus(content: Dict[str, Any]) -> Dict[str, Any]:
            """Get consensus signal from multiple strategies."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["signals"])
            
            signals = content["signals"]
            consensus_threshold = content.get("consensus_threshold", 0.7)
            
            try:
                consensus = await self._calculate_signal_consensus(signals, consensus_threshold)
                return {
                    "consensus": consensus,
                    "input_signals": len(signals)
                }
            except Exception as e:
                logger.error(f"Error calculating signal consensus: {str(e)}")
                return {"error": f"Failed to calculate consensus: {str(e)}"}
        
        @self.handler("rank_signals")
        async def rank_signals(content: Dict[str, Any]) -> Dict[str, Any]:
            """Rank signals by strength and confidence."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["signals"])
            
            signals = content["signals"]
            ranking_method = content.get("method", "confidence_weighted")
            
            try:
                ranked_signals = await self._rank_signals(signals, ranking_method)
                return {
                    "ranked_signals": ranked_signals,
                    "ranking_method": ranking_method
                }
            except Exception as e:
                logger.error(f"Error ranking signals: {str(e)}")
                return {"error": f"Failed to rank signals: {str(e)}"}
        
        @self.handler("filter_signals")
        async def filter_signals(content: Dict[str, Any]) -> Dict[str, Any]:
            """Filter signals based on various criteria."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["signals"])
            
            signals = content["signals"]
            filters = content.get("filters", {})
            
            try:
                filtered_signals = await self._filter_signals(signals, filters)
                return {
                    "filtered_signals": filtered_signals,
                    "original_count": len(signals),
                    "filtered_count": len(filtered_signals),
                    "filters_applied": filters
                }
            except Exception as e:
                logger.error(f"Error filtering signals: {str(e)}")
                return {"error": f"Failed to filter signals: {str(e)}"}

    async def _generate_comprehensive_signals(self, symbol: str, config_name: str, custom_strategies: Optional[List[str]]) -> Dict[str, Any]:
        """Generate comprehensive signals combining multiple analysis methods."""
        config = self.signal_configs.get(config_name, self.signal_configs["default"])
        strategies = custom_strategies or config["strategies"]
        
        # Get market data
        market_data = await self._get_market_data(symbol, config["lookback_days"])
        if not market_data:
            return {"error": "Failed to get market data"}
        
        # Generate quantitative signals
        quant_signals = await self._get_quantitative_signals(symbol, market_data, strategies)
        
        # Get technical analysis
        technical_analysis = await self._get_technical_analysis(symbol, market_data)
        
        # Get news sentiment
        news_sentiment = await self._get_news_sentiment(symbol)

        # Get LLM market intelligence
        llm_sentiment = await self._get_llm_sentiment(symbol, market_data)
        market_regime = await self._get_market_regime(market_data)

        # Combine all signals
        combined_signals = await self._combine_signals(
            quant_signals, technical_analysis, news_sentiment, llm_sentiment, market_regime, config
        )

        return {
            "symbol": symbol,
            "quantitative_signals": quant_signals,
            "technical_analysis": technical_analysis,
            "news_sentiment": news_sentiment,
            "llm_sentiment": llm_sentiment,
            "market_regime": market_regime,
            "combined_signals": combined_signals,
            "config_used": config_name,
            "strategies_used": strategies
        }
    
    async def _get_market_data(self, symbol: str, lookback_days: int) -> List[Dict[str, Any]]:
        """Get market data for the symbol."""
        try:
            async with self.mcp_client as client:
                response = await client.send_request(
                    self.server_urls["market_data"],
                    "get_historical_data",
                    {
                        "symbol": symbol,
                        "period": f"{lookback_days}d",
                        "interval": "1d"
                    }
                )
                return response.get("data", [])
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return []
    
    async def _get_quantitative_signals(self, symbol: str, market_data: List[Dict[str, Any]], strategies: List[str]) -> List[Dict[str, Any]]:
        """Get quantitative signals from the quantitative analysis server."""
        try:
            async with self.mcp_client as client:
                response = await client.send_request(
                    self.server_urls["quantitative_analysis"],
                    "generate_signals",
                    {
                        "symbol": symbol,
                        "data": market_data,
                        "strategies": strategies
                    }
                )
                return response.get("signals", [])
        except Exception as e:
            logger.error(f"Error getting quantitative signals for {symbol}: {e}")
            return []
    
    async def _get_technical_analysis(self, symbol: str, market_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get technical analysis indicators."""
        try:
            # Get RSI
            async with self.mcp_client as client:
                rsi_response = await client.send_request(
                    self.server_urls["technical_analysis"],
                    "calculate_indicator",
                    {
                        "indicator": "RSI",
                        "data": market_data,
                        "params": {"period": 14}
                    }
                )
            
            # Get MACD
            async with self.mcp_client as client:
                macd_response = await client.send_request(
                    self.server_urls["technical_analysis"],
                    "calculate_indicator",
                    {
                        "indicator": "MACD",
                        "data": market_data,
                        "params": {"fast": 12, "slow": 26, "signal": 9}
                    }
                )
            
            # Get Support/Resistance
            async with self.mcp_client as client:
                sr_response = await client.send_request(
                    self.server_urls["technical_analysis"],
                    "get_support_resistance",
                    {
                        "data": market_data,
                        "lookback": 20
                    }
                )
            
            return {
                "rsi": rsi_response.get("data", []),
                "macd": macd_response.get("data", []),
                "support_resistance": sr_response
            }
        except Exception as e:
            logger.error(f"Error getting technical analysis for {symbol}: {e}")
            return {}
    
    async def _get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get news sentiment for the symbol."""
        try:
            async with self.mcp_client as client:
                response = await client.send_request(
                    self.server_urls["news"],
                    "get_sentiment",
                    {
                        "symbol": symbol,
                        "lookback_hours": 24
                    }
                )
                return response
        except Exception as e:
            logger.error(f"Error getting news sentiment for {symbol}: {e}")
            return {"sentiment": "neutral", "confidence": 0.5}

    async def _get_llm_sentiment(self, symbol: str, market_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get LLM-enhanced sentiment analysis for the symbol."""
        try:
            # Get recent news data first
            news_data = await self._get_recent_news_data(symbol)

            async with self.mcp_client as client:
                response = await client.send_request(
                    self.server_urls["llm_market_intelligence"],
                    "analyze_market_sentiment",
                    {
                        "news_data": news_data,
                        "timeframe": "1d",
                        "symbol": symbol
                    }
                )
                return response
        except Exception as e:
            logger.error(f"Error getting LLM sentiment for {symbol}: {e}")
            return {"sentiment_score": 0.0, "confidence": 0.5, "market_impact": "neutral"}

    async def _get_market_regime(self, market_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get current market regime analysis."""
        try:
            # Prepare market data summary for regime detection
            market_summary = self._prepare_market_summary(market_data)

            async with self.mcp_client as client:
                response = await client.send_request(
                    self.server_urls["llm_market_intelligence"],
                    "detect_market_regime",
                    {
                        "market_data": market_summary,
                        "lookback_period": 30
                    }
                )
                return response
        except Exception as e:
            logger.error(f"Error getting market regime: {e}")
            return {"regime_type": "sideways", "confidence": 0.5, "risk_level": "medium"}

    async def _get_recent_news_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Get recent news data for sentiment analysis."""
        try:
            async with self.mcp_client as client:
                response = await client.send_request(
                    self.server_urls["news"],
                    "get_recent_news",
                    {
                        "symbol": symbol,
                        "limit": 10,
                        "hours": 24
                    }
                )
                return response.get("articles", [])
        except Exception as e:
            logger.error(f"Error getting recent news for {symbol}: {e}")
            return []

    def _prepare_market_summary(self, market_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare market data summary for regime detection."""
        if not market_data:
            return {}

        # Calculate basic statistics from market data
        prices = [float(d.get("close", 0)) for d in market_data if d.get("close")]
        volumes = [float(d.get("volume", 0)) for d in market_data if d.get("volume")]

        if not prices:
            return {}

        current_price = prices[-1]
        price_change_1d = ((current_price - prices[-2]) / prices[-2] * 100) if len(prices) > 1 else 0
        price_change_1w = ((current_price - prices[-7]) / prices[-7] * 100) if len(prices) > 7 else 0
        price_change_1m = ((current_price - prices[-30]) / prices[-30] * 100) if len(prices) > 30 else 0

        # Calculate volatility (simple standard deviation of returns)
        returns = []
        for i in range(1, len(prices)):
            returns.append((prices[i] - prices[i-1]) / prices[i-1])

        volatility = (sum([(r - sum(returns)/len(returns))**2 for r in returns]) / len(returns))**0.5 * 100 if returns else 0

        # Volume analysis
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        current_volume = volumes[-1] if volumes else 0
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        return {
            "current_price": current_price,
            "change_1d": price_change_1d,
            "change_1w": price_change_1w,
            "change_1m": price_change_1m,
            "volatility": volatility,
            "volume_ratio": volume_ratio,
            "data_points": len(market_data)
        }

    async def _combine_signals(self, quant_signals: List[Dict[str, Any]], technical_analysis: Dict[str, Any],
                             news_sentiment: Dict[str, Any], llm_sentiment: Dict[str, Any],
                             market_regime: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Combine signals from different sources into a consensus signal."""
        if not quant_signals:
            return {"signal": "HOLD", "confidence": 0.0, "reason": "No quantitative signals available"}

        # Analyze quantitative signals
        buy_signals = [s for s in quant_signals if s.get("signal") == "BUY"]
        sell_signals = [s for s in quant_signals if s.get("signal") == "SELL"]

        # Calculate signal strength
        total_signals = len(quant_signals)
        buy_strength = len(buy_signals) / total_signals if total_signals > 0 else 0
        sell_strength = len(sell_signals) / total_signals if total_signals > 0 else 0

        # Weight by confidence
        buy_confidence = sum(s.get("confidence", 0) for s in buy_signals) / len(buy_signals) if buy_signals else 0
        sell_confidence = sum(s.get("confidence", 0) for s in sell_signals) / len(sell_signals) if sell_signals else 0

        # Technical analysis influence
        tech_influence = 0.0
        if technical_analysis:
            rsi_data = technical_analysis.get("rsi", [])
            if rsi_data:
                latest_rsi = rsi_data[-1].get("value", 50) if rsi_data else 50
                if latest_rsi < 30:
                    tech_influence += 0.2  # Oversold - bullish
                elif latest_rsi > 70:
                    tech_influence -= 0.2  # Overbought - bearish

        # News sentiment influence
        sentiment_influence = 0.0
        sentiment = news_sentiment.get("sentiment", "neutral")
        sentiment_confidence = news_sentiment.get("confidence", 0.5)

        if sentiment == "positive":
            sentiment_influence = 0.1 * sentiment_confidence
        elif sentiment == "negative":
            sentiment_influence = -0.1 * sentiment_confidence

        # LLM sentiment influence (stronger weight due to advanced analysis)
        llm_sentiment_influence = 0.0
        llm_sentiment_score = llm_sentiment.get("sentiment_score", 0.0)
        llm_confidence = llm_sentiment.get("confidence", 0.5)

        # LLM sentiment has higher weight when confidence is high
        if llm_confidence > 0.7:
            llm_sentiment_influence = llm_sentiment_score * 0.15 * llm_confidence
        else:
            llm_sentiment_influence = llm_sentiment_score * 0.1 * llm_confidence

        # Market regime influence
        regime_influence = 0.0
        regime_type = market_regime.get("regime_type", "sideways")
        regime_confidence = market_regime.get("confidence", 0.5)
        risk_level = market_regime.get("risk_level", "medium")

        # Adjust signals based on market regime
        if regime_type == "bull_market" and regime_confidence > 0.6:
            regime_influence = 0.1 * regime_confidence  # Favor buy signals
        elif regime_type == "bear_market" and regime_confidence > 0.6:
            regime_influence = -0.1 * regime_confidence  # Favor sell signals
        elif regime_type == "high_volatility":
            # Reduce signal confidence in high volatility
            buy_confidence *= 0.8
            sell_confidence *= 0.8

        # Risk level adjustment
        risk_multiplier = 1.0
        if risk_level == "high":
            risk_multiplier = 0.7  # Reduce signal strength in high risk
        elif risk_level == "low":
            risk_multiplier = 1.2  # Increase signal strength in low risk

        # Calculate final signal with all influences
        final_buy_score = (buy_strength * buy_confidence + tech_influence + sentiment_influence +
                          llm_sentiment_influence + regime_influence) * risk_multiplier
        final_sell_score = (sell_strength * sell_confidence - tech_influence - sentiment_influence -
                           llm_sentiment_influence - regime_influence) * risk_multiplier

        consensus_threshold = config.get("consensus_threshold", 0.7)
        min_confidence = config.get("min_confidence", 0.6)

        if final_buy_score > consensus_threshold and final_buy_score > final_sell_score:
            signal = "BUY"
            confidence = min(final_buy_score, 1.0)
            reason = f"Strong buy consensus: {len(buy_signals)} buy signals with avg confidence {buy_confidence:.2f}"
        elif final_sell_score > consensus_threshold and final_sell_score > final_buy_score:
            signal = "SELL"
            confidence = min(final_sell_score, 1.0)
            reason = f"Strong sell consensus: {len(sell_signals)} sell signals with avg confidence {sell_confidence:.2f}"
        else:
            signal = "HOLD"
            confidence = 0.5
            reason = "No clear consensus or insufficient confidence"

        # Apply minimum confidence filter
        if confidence < min_confidence:
            signal = "HOLD"
            reason += f" (below min confidence {min_confidence})"

        return {
            "signal": signal,
            "confidence": confidence,
            "reason": reason,
            "analysis": {
                "buy_strength": buy_strength,
                "sell_strength": sell_strength,
                "buy_confidence": buy_confidence,
                "sell_confidence": sell_confidence,
                "technical_influence": tech_influence,
                "sentiment_influence": sentiment_influence,
                "llm_sentiment_influence": llm_sentiment_influence,
                "regime_influence": regime_influence,
                "risk_multiplier": risk_multiplier,
                "final_buy_score": final_buy_score,
                "final_sell_score": final_sell_score
            },
            "llm_analysis": {
                "sentiment_score": llm_sentiment.get("sentiment_score", 0.0),
                "sentiment_confidence": llm_sentiment.get("confidence", 0.5),
                "market_impact": llm_sentiment.get("market_impact", "neutral"),
                "regime_type": market_regime.get("regime_type", "sideways"),
                "regime_confidence": market_regime.get("confidence", 0.5),
                "risk_level": market_regime.get("risk_level", "medium")
            }
        }

    async def _generate_watchlist_signals(self, symbols: List[str], config_name: str) -> List[Dict[str, Any]]:
        """Generate signals for multiple symbols."""
        watchlist_signals = []

        for symbol in symbols:
            try:
                signal_data = await self._generate_comprehensive_signals(symbol, config_name, None)
                watchlist_signals.append({
                    "symbol": symbol,
                    "signal_data": signal_data,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error generating signals for {symbol}: {e}")
                watchlist_signals.append({
                    "symbol": symbol,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

        return watchlist_signals

    async def _calculate_signal_consensus(self, signals: List[Dict[str, Any]], consensus_threshold: float) -> Dict[str, Any]:
        """Calculate consensus from multiple signals."""
        if not signals:
            return {"consensus": "HOLD", "confidence": 0.0, "reason": "No signals provided"}

        # Group signals by type
        buy_signals = [s for s in signals if s.get("signal") == "BUY"]
        sell_signals = [s for s in signals if s.get("signal") == "SELL"]
        hold_signals = [s for s in signals if s.get("signal") in ["HOLD", "EXIT"]]

        total_signals = len(signals)

        # Calculate weighted scores
        buy_score = sum(s.get("confidence", 0) for s in buy_signals) / total_signals
        sell_score = sum(s.get("confidence", 0) for s in sell_signals) / total_signals
        hold_score = sum(s.get("confidence", 0) for s in hold_signals) / total_signals

        # Determine consensus
        max_score = max(buy_score, sell_score, hold_score)

        if max_score >= consensus_threshold:
            if max_score == buy_score:
                consensus = "BUY"
            elif max_score == sell_score:
                consensus = "SELL"
            else:
                consensus = "HOLD"
        else:
            consensus = "HOLD"  # No clear consensus

        return {
            "consensus": consensus,
            "confidence": max_score,
            "scores": {
                "buy_score": buy_score,
                "sell_score": sell_score,
                "hold_score": hold_score
            },
            "signal_counts": {
                "buy_signals": len(buy_signals),
                "sell_signals": len(sell_signals),
                "hold_signals": len(hold_signals)
            },
            "consensus_threshold": consensus_threshold
        }

    async def _rank_signals(self, signals: List[Dict[str, Any]], ranking_method: str) -> List[Dict[str, Any]]:
        """Rank signals by various criteria."""
        if not signals:
            return []

        if ranking_method == "confidence_weighted":
            # Rank by confidence score
            return sorted(signals, key=lambda x: x.get("confidence", 0), reverse=True)

        elif ranking_method == "signal_strength":
            # Rank by signal strength (BUY/SELL over HOLD)
            def signal_strength(signal):
                signal_type = signal.get("signal", "HOLD")
                confidence = signal.get("confidence", 0)
                if signal_type in ["BUY", "SELL"]:
                    return confidence
                else:
                    return confidence * 0.5  # Reduce weight for HOLD signals

            return sorted(signals, key=signal_strength, reverse=True)

        elif ranking_method == "timestamp":
            # Rank by most recent
            return sorted(signals, key=lambda x: x.get("timestamp", ""), reverse=True)

        else:
            # Default to confidence ranking
            return sorted(signals, key=lambda x: x.get("confidence", 0), reverse=True)

    async def _filter_signals(self, signals: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter signals based on criteria."""
        filtered_signals = signals.copy()

        # Filter by minimum confidence
        if "min_confidence" in filters:
            min_conf = filters["min_confidence"]
            filtered_signals = [s for s in filtered_signals if s.get("confidence", 0) >= min_conf]

        # Filter by signal type
        if "signal_types" in filters:
            allowed_types = filters["signal_types"]
            filtered_signals = [s for s in filtered_signals if s.get("signal") in allowed_types]

        # Filter by strategy
        if "strategies" in filters:
            allowed_strategies = filters["strategies"]
            filtered_signals = [s for s in filtered_signals if s.get("strategy") in allowed_strategies]

        # Filter by symbol
        if "symbols" in filters:
            allowed_symbols = filters["symbols"]
            filtered_signals = [s for s in filtered_signals if s.get("symbol") in allowed_symbols]

        # Filter by time range
        if "start_time" in filters or "end_time" in filters:
            start_time = filters.get("start_time")
            end_time = filters.get("end_time")

            def in_time_range(signal):
                timestamp = signal.get("timestamp")
                if not timestamp:
                    return False

                signal_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

                if start_time and signal_time < datetime.fromisoformat(start_time):
                    return False
                if end_time and signal_time > datetime.fromisoformat(end_time):
                    return False

                return True

            filtered_signals = [s for s in filtered_signals if in_time_range(s)]

        return filtered_signals

if __name__ == "__main__":
    async def main():
        """Main function to run the Signal Generation MCP Server."""
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        try:
            # Create and start server
            server = SignalGenerationServer()
            logger.info("Starting Signal Generation MCP Server...")
            await server.start()
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise
    
    asyncio.run(main())
