"""
Market Analysis Agent for AWM system.
Performs technical and fundamental analysis to generate trading signals.
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from decimal import Decimal
import openai

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from shared.agents.base_agent import BaseAgent
from shared.models.trading import TradeSignal

logger = logging.getLogger(__name__)


class MarketAnalysisAgent(BaseAgent):
    """Agent responsible for market analysis and signal generation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("market_analysis_agent", config)
        
        # OpenAI configuration
        self.openai_client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        
        # Analysis parameters
        self.lookback_days = self.config.get("lookback_days", 30)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)
        
        # Technical indicators to calculate
        self.indicators = [
            {"indicator": "RSI", "params": {"period": 14}},
            {"indicator": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
            {"indicator": "BOLLINGER_BANDS", "params": {"period": 20, "std_dev": 2}},
            {"indicator": "SMA", "params": {"period": 20}},
            {"indicator": "SMA", "params": {"period": 50}},
            {"indicator": "EMA", "params": {"period": 12}},
            {"indicator": "EMA", "params": {"period": 26}}
        ]
    
    async def initialize(self):
        """Initialize the Market Analysis Agent."""
        self.logger.info("Initializing Market Analysis Agent...")
        
        # Test connections to required MCP servers
        try:
            # Test market data server
            await self.call_mcp_server("market_data", "health", {})
            self.logger.info("✓ Market Data Server connection verified")
            
            # Test technical analysis server
            await self.call_mcp_server("technical_analysis", "health", {})
            self.logger.info("✓ Technical Analysis Server connection verified")
            
            # Test news server
            await self.call_mcp_server("news", "health", {})
            self.logger.info("✓ News Server connection verified")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to required servers: {e}")
            raise
        
        self.logger.info("Market Analysis Agent initialized successfully")
    
    async def cleanup(self):
        """Cleanup the Market Analysis Agent."""
        self.logger.info("Cleaning up Market Analysis Agent...")
        # No specific cleanup needed
    
    async def process_task(self, task_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process a market analysis task."""
        
        if task_type == "analyze_instrument":
            return await self._analyze_instrument(parameters)
        elif task_type == "scan_market":
            return await self._scan_market(parameters)
        elif task_type == "generate_signals":
            return await self._generate_signals(parameters)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _analyze_instrument(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis of a single instrument."""
        symbol = parameters["symbol"]
        analysis_type = parameters.get("analysis_type", "comprehensive")
        
        self.logger.info(f"Analyzing instrument: {symbol}")
        
        try:
            # Get market data
            market_data = await self._get_market_data(symbol)
            
            # Perform technical analysis
            technical_analysis = await self._perform_technical_analysis(symbol, market_data)
            
            # Get news sentiment
            news_sentiment = await self._get_news_sentiment(symbol)
            
            # Generate AI-powered analysis
            ai_analysis = await self._generate_ai_analysis(
                symbol, market_data, technical_analysis, news_sentiment
            )
            
            # Combine all analyses
            result = {
                "symbol": symbol,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "market_data_summary": self._summarize_market_data(market_data),
                "technical_analysis": technical_analysis,
                "news_sentiment": news_sentiment,
                "ai_analysis": ai_analysis,
                "overall_signal": ai_analysis.get("signal"),
                "confidence_score": ai_analysis.get("confidence"),
                "reasoning": ai_analysis.get("reasoning")
            }
            
            # Store analysis result
            await self._store_analysis_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            raise
    
    async def _get_market_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Get historical market data for analysis."""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=self.lookback_days)
        
        response = await self.call_mcp_server(
            "market_data",
            "get_price_history",
            {
                "symbol": symbol,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "limit": 1000
            }
        )
        
        return response.get("data", [])
    
    async def _perform_technical_analysis(self, symbol: str, market_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform technical analysis using various indicators."""
        if not market_data:
            return {"error": "No market data available"}
        
        technical_results = {}
        
        # Calculate each indicator
        for indicator_config in self.indicators:
            try:
                response = await self.call_mcp_server(
                    "technical_analysis",
                    "calculate_indicator",
                    {
                        "indicator": indicator_config["indicator"],
                        "data": market_data,
                        "params": indicator_config["params"]
                    }
                )
                
                indicator_name = f"{indicator_config['indicator']}_{indicator_config['params'].get('period', '')}"
                technical_results[indicator_name] = response.get("data", [])
                
            except Exception as e:
                self.logger.error(f"Error calculating {indicator_config['indicator']}: {e}")
                technical_results[indicator_config["indicator"]] = {"error": str(e)}
        
        # Get support and resistance levels
        try:
            sr_response = await self.call_mcp_server(
                "technical_analysis",
                "get_support_resistance",
                {
                    "data": market_data,
                    "lookback": 20
                }
            )
            technical_results["support_resistance"] = sr_response
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {e}")
        
        # Detect patterns
        try:
            pattern_response = await self.call_mcp_server(
                "technical_analysis",
                "detect_patterns",
                {
                    "data": market_data,
                    "patterns": ["all"]
                }
            )
            technical_results["patterns"] = pattern_response.get("patterns", [])
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
        
        return technical_results
    
    async def _get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get news sentiment for the instrument."""
        try:
            response = await self.call_mcp_server(
                "news",
                "get_recent_news",
                {
                    "symbol": symbol,
                    "limit": 10
                }
            )
            
            # Analyze sentiment
            sentiment_response = await self.call_mcp_server(
                "news",
                "analyze_sentiment",
                {
                    "articles": response.get("articles", [])
                }
            )
            
            return sentiment_response
            
        except Exception as e:
            self.logger.error(f"Error getting news sentiment for {symbol}: {e}")
            return {"sentiment": "neutral", "confidence": 0.5, "error": str(e)}
    
    async def _generate_ai_analysis(
        self,
        symbol: str,
        market_data: List[Dict[str, Any]],
        technical_analysis: Dict[str, Any],
        news_sentiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate AI-powered analysis using OpenAI."""
        
        # Prepare context for AI analysis
        context = self._prepare_analysis_context(symbol, market_data, technical_analysis, news_sentiment)
        
        prompt = f"""
        As an expert financial analyst, analyze the following data for {symbol} and provide a trading recommendation.
        
        Context:
        {json.dumps(context, indent=2)}
        
        Please provide your analysis in the following JSON format:
        {{
            "signal": "BUY" | "SELL" | "HOLD",
            "confidence": 0.0-1.0,
            "target_price": number,
            "stop_loss": number,
            "reasoning": "detailed explanation of your analysis",
            "key_factors": ["list", "of", "key", "factors"],
            "risk_assessment": "low" | "medium" | "high",
            "time_horizon": "short" | "medium" | "long"
        }}
        
        Consider:
        1. Technical indicators and their signals
        2. Support and resistance levels
        3. Chart patterns
        4. News sentiment
        5. Market conditions
        6. Risk-reward ratio
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst specializing in technical and fundamental analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse AI response
            ai_response = response.choices[0].message.content
            
            # Try to extract JSON from response
            try:
                # Find JSON in the response
                start_idx = ai_response.find('{')
                end_idx = ai_response.rfind('}') + 1
                json_str = ai_response[start_idx:end_idx]
                
                analysis = json.loads(json_str)
                
                # Validate and sanitize the response
                analysis["confidence"] = max(0.0, min(1.0, float(analysis.get("confidence", 0.5))))
                analysis["signal"] = analysis.get("signal", "HOLD").upper()
                
                if analysis["signal"] not in ["BUY", "SELL", "HOLD"]:
                    analysis["signal"] = "HOLD"
                
                return analysis
                
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Failed to parse AI response: {e}")
                return {
                    "signal": "HOLD",
                    "confidence": 0.5,
                    "reasoning": "Failed to parse AI analysis",
                    "error": str(e)
                }
                
        except Exception as e:
            self.logger.error(f"Error generating AI analysis: {e}")
            return {
                "signal": "HOLD",
                "confidence": 0.5,
                "reasoning": "AI analysis failed",
                "error": str(e)
            }
    
    def _prepare_analysis_context(
        self,
        symbol: str,
        market_data: List[Dict[str, Any]],
        technical_analysis: Dict[str, Any],
        news_sentiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare context for AI analysis."""
        
        # Get latest price data
        latest_data = market_data[0] if market_data else {}
        
        # Extract key technical indicators
        rsi_data = technical_analysis.get("RSI_14", [])
        latest_rsi = rsi_data[-1]["value"] if rsi_data else None
        
        macd_data = technical_analysis.get("MACD_", [])
        latest_macd = macd_data[-1] if macd_data else {}
        
        bb_data = technical_analysis.get("BOLLINGER_BANDS_20", [])
        latest_bb = bb_data[-1] if bb_data else {}
        
        return {
            "symbol": symbol,
            "current_price": latest_data.get("close"),
            "price_change_1d": self._calculate_price_change(market_data, 1),
            "price_change_5d": self._calculate_price_change(market_data, 5),
            "volume_trend": self._calculate_volume_trend(market_data),
            "technical_indicators": {
                "rsi": latest_rsi,
                "macd": latest_macd,
                "bollinger_bands": latest_bb,
                "support_levels": technical_analysis.get("support_resistance", {}).get("support_levels", []),
                "resistance_levels": technical_analysis.get("support_resistance", {}).get("resistance_levels", [])
            },
            "patterns": technical_analysis.get("patterns", []),
            "news_sentiment": news_sentiment,
            "market_data_points": len(market_data)
        }
    
    def _calculate_price_change(self, market_data: List[Dict[str, Any]], days: int) -> Optional[float]:
        """Calculate price change over specified days."""
        if len(market_data) < days + 1:
            return None
        
        current_price = float(market_data[0]["close"])
        past_price = float(market_data[days]["close"])
        
        return (current_price - past_price) / past_price
    
    def _calculate_volume_trend(self, market_data: List[Dict[str, Any]]) -> str:
        """Calculate volume trend."""
        if len(market_data) < 5:
            return "insufficient_data"
        
        recent_volume = sum(data["volume"] for data in market_data[:5]) / 5
        past_volume = sum(data["volume"] for data in market_data[5:10]) / 5
        
        if recent_volume > past_volume * 1.2:
            return "increasing"
        elif recent_volume < past_volume * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _summarize_market_data(self, market_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize market data."""
        if not market_data:
            return {}
        
        latest = market_data[0]
        prices = [float(data["close"]) for data in market_data]
        volumes = [data["volume"] for data in market_data]
        
        return {
            "latest_price": float(latest["close"]),
            "high_52w": max(prices),
            "low_52w": min(prices),
            "avg_volume": sum(volumes) / len(volumes),
            "data_points": len(market_data)
        }
    
    async def _store_analysis_result(self, analysis: Dict[str, Any]) -> None:
        """Store analysis result in database."""
        try:
            # Get instrument ID
            instrument_response = await self.call_mcp_server(
                "market_data",
                "get_instruments",
                {"limit": 1000}
            )
            
            instruments = instrument_response.get("instruments", [])
            instrument = next((i for i in instruments if i["symbol"] == analysis["symbol"]), None)
            
            if not instrument:
                self.logger.error(f"Instrument {analysis['symbol']} not found")
                return
            
            # Store in database
            query = """
                INSERT INTO analysis_results 
                (instrument_id, analysis_type, signal, confidence_score, analysis_data, created_by)
                VALUES ($1, $2, $3, $4, $5, $6)
            """
            
            await db_manager.execute_query(
                query,
                instrument["id"],
                "COMPREHENSIVE",
                analysis.get("overall_signal"),
                analysis.get("confidence_score"),
                json.dumps(analysis),
                self.agent_name
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store analysis result: {e}")
    
    async def _scan_market(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Scan market for trading opportunities."""
        # Implementation for market scanning
        # This would analyze multiple instruments and rank them
        pass
    
    async def _generate_signals(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals for multiple instruments."""
        # Implementation for signal generation
        # This would generate signals for a watchlist
        pass


async def main():
    """Main function to run the Market Analysis Agent."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start agent
    agent = MarketAnalysisAgent()
    
    try:
        await agent.start()
        
        # Keep the agent running
        while agent.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
