"""
LLM Market Intelligence MCP Server for AWM system.
Provides LLM-enhanced market analysis and contextual intelligence through MCP protocol.
"""

import os
import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import aioredis

# Add the project root to Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from shared.mcp_client.server import MCPServer, validate_required_fields, sanitize_input
from .config import config
from .llm_engine import llm_engine

logger = logging.getLogger(__name__)


class LLMMarketIntelligenceServer(MCPServer):
    """LLM Market Intelligence MCP Server implementation."""
    
    def __init__(self):
        host = config.server.host
        port = config.server.port
        super().__init__("llm_market_intelligence_server", host, port)
        
        # Redis client for caching
        self.redis_client = None
        
        # Register handlers
        self.register_handlers()
    
    async def startup(self):
        """Initialize server components."""
        try:
            # Initialize Redis connection
            self.redis_client = aioredis.from_url(
                f"redis://{config.cache.redis_host}:{config.cache.redis_port}/{config.cache.redis_db}"
            )
            logger.info("Redis connection established")
            
            # Initialize LLM engine
            await llm_engine.initialize()
            logger.info("LLM engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize server components: {e}")
            raise
    
    async def shutdown(self):
        """Clean up server resources."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            llm_engine.cleanup()
            logger.info("Server shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def register_handlers(self):
        """Register all MCP handlers."""
        
        @self.handler("analyze_market_sentiment")
        async def analyze_market_sentiment(content: Dict[str, Any]) -> Dict[str, Any]:
            """Analyze market sentiment from news data."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["news_data"])
            
            news_data = content["news_data"]
            timeframe = content.get("timeframe", "1d")
            symbol = content.get("symbol", None)
            
            try:
                # Check cache first
                cache_key = f"sentiment:{hash(str(news_data))}:{timeframe}"
                cached_result = await self._get_cached_result(cache_key)
                if cached_result:
                    return cached_result
                
                # Prepare text for analysis
                if isinstance(news_data, list):
                    text = " ".join([item.get("title", "") + " " + item.get("content", "") for item in news_data])
                else:
                    text = str(news_data)
                
                # Add market context
                context = f"Timeframe: {timeframe}"
                if symbol:
                    context += f", Symbol: {symbol}"
                
                # Analyze sentiment using LLM
                sentiment_result = await llm_engine.analyze_sentiment(text, context)
                
                # Add metadata
                result = {
                    "sentiment_score": self._convert_sentiment_to_score(sentiment_result.get("sentiment", "neutral")),
                    "confidence": sentiment_result.get("confidence", 0.5),
                    "reasoning": sentiment_result.get("reasoning", ""),
                    "market_impact": sentiment_result.get("market_impact", "neutral"),
                    "key_factors": sentiment_result.get("key_factors", []),
                    "timeframe": timeframe,
                    "symbol": symbol,
                    "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                    "inference_time": sentiment_result.get("inference_time", 0)
                }
                
                # Cache result
                await self._cache_result(cache_key, result, config.cache.sentiment_cache_ttl)
                
                return result
                
            except Exception as e:
                logger.error(f"Error analyzing market sentiment: {str(e)}")
                return {"error": f"Failed to analyze market sentiment: {str(e)}"}
        
        @self.handler("detect_market_regime")
        async def detect_market_regime(content: Dict[str, Any]) -> Dict[str, Any]:
            """Detect current market regime."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["market_data"])
            
            market_data = content["market_data"]
            lookback_period = content.get("lookback_period", 30)
            
            try:
                # Check cache first
                cache_key = f"regime:{hash(str(market_data))}:{lookback_period}"
                cached_result = await self._get_cached_result(cache_key)
                if cached_result:
                    return cached_result
                
                # Detect market regime using LLM
                regime_result = await llm_engine.detect_market_regime(market_data, lookback_period)
                
                # Add metadata
                result = {
                    "regime_type": regime_result.get("regime_type", "sideways"),
                    "confidence": regime_result.get("confidence", 0.5),
                    "explanation": regime_result.get("explanation", ""),
                    "key_indicators": regime_result.get("key_indicators", []),
                    "duration_estimate": regime_result.get("duration_estimate", 30),
                    "risk_level": regime_result.get("risk_level", "medium"),
                    "lookback_period": lookback_period,
                    "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                    "inference_time": regime_result.get("inference_time", 0)
                }
                
                # Cache result
                await self._cache_result(cache_key, result, config.cache.regime_cache_ttl)
                
                return result
                
            except Exception as e:
                logger.error(f"Error detecting market regime: {str(e)}")
                return {"error": f"Failed to detect market regime: {str(e)}"}
        
        @self.handler("assess_event_impact")
        async def assess_event_impact(content: Dict[str, Any]) -> Dict[str, Any]:
            """Assess impact of market events."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["event_data"])
            
            event_data = content["event_data"]
            affected_symbols = content.get("affected_symbols", [])
            
            try:
                # Prepare prompt for event impact assessment
                system_prompt = """You are an expert financial analyst specializing in Indian equity markets.
Assess the impact of the provided event on the specified symbols and broader market.

Respond with a JSON object containing:
- impact_score: float between -1.0 (very negative) and 1.0 (very positive)
- duration_estimate: estimated impact duration in days
- reasoning: detailed explanation of the impact assessment
- affected_sectors: list of sectors likely to be affected
- risk_factors: list of key risk factors
- opportunities: list of potential opportunities
- confidence: float between 0.0 and 1.0

Consider Indian market context, regulatory environment, and economic factors."""
                
                prompt = f"Event data: {json.dumps(event_data, indent=2)}"
                if affected_symbols:
                    prompt += f"\nAffected symbols: {', '.join(affected_symbols)}"
                
                result = await llm_engine.generate_response(prompt, system_prompt=system_prompt, max_tokens=1024)
                
                try:
                    response_text = result["response"]
                    if "```json" in response_text:
                        json_start = response_text.find("```json") + 7
                        json_end = response_text.find("```", json_start)
                        response_text = response_text[json_start:json_end].strip()
                    
                    parsed_response = json.loads(response_text)
                    
                    # Add metadata
                    parsed_response.update({
                        "affected_symbols": affected_symbols,
                        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                        "inference_time": result["inference_time"]
                    })
                    
                    return parsed_response
                    
                except json.JSONDecodeError:
                    return {
                        "impact_score": 0.0,
                        "duration_estimate": 7,
                        "reasoning": "Failed to parse LLM response",
                        "affected_sectors": [],
                        "risk_factors": [],
                        "opportunities": [],
                        "confidence": 0.5,
                        "affected_symbols": affected_symbols,
                        "raw_response": result["response"],
                        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                        "inference_time": result["inference_time"]
                    }
                
            except Exception as e:
                logger.error(f"Error assessing event impact: {str(e)}")
                return {"error": f"Failed to assess event impact: {str(e)}"}

        @self.handler("generate_market_insights")
        async def generate_market_insights(content: Dict[str, Any]) -> Dict[str, Any]:
            """Generate actionable market insights."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["market_context"])

            market_context = content["market_context"]
            focus_areas = content.get("focus_areas", ["general"])

            try:
                system_prompt = """You are an expert financial analyst and portfolio manager specializing in Indian equity markets.
Generate actionable market insights based on the provided market context.

Respond with a JSON object containing:
- actionable_insights: list of specific, actionable insights
- risk_factors: list of key risk factors to monitor
- opportunities: list of potential trading/investment opportunities
- market_outlook: "bullish", "bearish", or "neutral"
- time_horizon: "short_term" (1-7 days), "medium_term" (1-4 weeks), or "long_term" (1-3 months)
- confidence: float between 0.0 and 1.0
- key_levels: important price/index levels to watch
- sector_recommendations: sector-specific recommendations

Consider Indian market dynamics, regulatory environment, and economic indicators."""

                prompt = f"Market context: {json.dumps(market_context, indent=2)}"
                if focus_areas:
                    prompt += f"\nFocus areas: {', '.join(focus_areas)}"

                result = await llm_engine.generate_response(prompt, system_prompt=system_prompt, max_tokens=1536)

                try:
                    response_text = result["response"]
                    if "```json" in response_text:
                        json_start = response_text.find("```json") + 7
                        json_end = response_text.find("```", json_start)
                        response_text = response_text[json_start:json_end].strip()

                    parsed_response = json.loads(response_text)

                    # Add metadata
                    parsed_response.update({
                        "focus_areas": focus_areas,
                        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                        "inference_time": result["inference_time"]
                    })

                    return parsed_response

                except json.JSONDecodeError:
                    return {
                        "actionable_insights": ["Failed to parse LLM response"],
                        "risk_factors": ["Analysis unavailable"],
                        "opportunities": [],
                        "market_outlook": "neutral",
                        "time_horizon": "medium_term",
                        "confidence": 0.5,
                        "key_levels": [],
                        "sector_recommendations": [],
                        "focus_areas": focus_areas,
                        "raw_response": result["response"],
                        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                        "inference_time": result["inference_time"]
                    }

            except Exception as e:
                logger.error(f"Error generating market insights: {str(e)}")
                return {"error": f"Failed to generate market insights: {str(e)}"}

        @self.handler("explain_market_context")
        async def explain_market_context(content: Dict[str, Any]) -> Dict[str, Any]:
            """Provide natural language explanation of market context."""
            content = await sanitize_input(content)
            await validate_required_fields(content, ["current_conditions"])

            current_conditions = content["current_conditions"]
            detail_level = content.get("detail_level", "medium")  # basic, medium, detailed

            try:
                system_prompt = """You are an expert financial analyst and market commentator specializing in Indian equity markets.
Provide a clear, natural language explanation of the current market context for traders and investors.

Your explanation should:
- Be clear and accessible to both novice and experienced traders
- Focus on actionable information
- Consider Indian market specifics (NSE/BSE, INR, SEBI regulations)
- Highlight key factors driving current market conditions
- Provide context for decision-making

Respond with a JSON object containing:
- explanation: comprehensive natural language explanation
- key_points: list of bullet-point key takeaways
- market_summary: brief 2-3 sentence summary
- trader_focus: what traders should focus on today
- investor_focus: what long-term investors should consider
- risk_warning: any important risk warnings"""

                prompt = f"Current market conditions: {json.dumps(current_conditions, indent=2)}"
                prompt += f"\nDetail level requested: {detail_level}"

                max_tokens = {"basic": 512, "medium": 1024, "detailed": 1536}.get(detail_level, 1024)

                result = await llm_engine.generate_response(prompt, system_prompt=system_prompt, max_tokens=max_tokens)

                try:
                    response_text = result["response"]
                    if "```json" in response_text:
                        json_start = response_text.find("```json") + 7
                        json_end = response_text.find("```", json_start)
                        response_text = response_text[json_start:json_end].strip()

                    parsed_response = json.loads(response_text)

                    # Add metadata
                    parsed_response.update({
                        "detail_level": detail_level,
                        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                        "inference_time": result["inference_time"]
                    })

                    return parsed_response

                except json.JSONDecodeError:
                    return {
                        "explanation": result["response"],
                        "key_points": ["Analysis provided in explanation field"],
                        "market_summary": "Market analysis available in explanation field",
                        "trader_focus": "Review detailed explanation for trading guidance",
                        "investor_focus": "Review detailed explanation for investment guidance",
                        "risk_warning": "Always consider your risk tolerance and investment objectives",
                        "detail_level": detail_level,
                        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                        "inference_time": result["inference_time"]
                    }

            except Exception as e:
                logger.error(f"Error explaining market context: {str(e)}")
                return {"error": f"Failed to explain market context: {str(e)}"}
    
    def _convert_sentiment_to_score(self, sentiment: str) -> float:
        """Convert sentiment string to numerical score."""
        sentiment_map = {
            "positive": 0.7,
            "negative": -0.7,
            "neutral": 0.0,
            "bullish": 0.8,
            "bearish": -0.8
        }
        return sentiment_map.get(sentiment.lower(), 0.0)
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result from Redis."""
        try:
            if self.redis_client:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
        return None
    
    async def _cache_result(self, cache_key: str, result: Dict[str, Any], ttl: int):
        """Cache result in Redis."""
        try:
            if self.redis_client:
                await self.redis_client.setex(cache_key, ttl, json.dumps(result))
        except Exception as e:
            logger.warning(f"Cache storage error: {e}")


async def main():
    """Main function to run the LLM Market Intelligence MCP Server."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create and start server
        server = LLMMarketIntelligenceServer()
        await server.startup()
        
        logger.info("Starting LLM Market Intelligence MCP Server...")
        await server.start()
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
    finally:
        if 'server' in locals():
            await server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
