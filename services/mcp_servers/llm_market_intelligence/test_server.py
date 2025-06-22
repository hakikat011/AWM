"""
Test script for LLM Market Intelligence MCP Server.
"""

import asyncio
import json
import logging
from typing import Dict, Any
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from shared.mcp_client.base import MCPClient

logger = logging.getLogger(__name__)


async def test_llm_market_intelligence_server():
    """Test the LLM Market Intelligence MCP Server."""
    
    server_url = "http://localhost:8007"
    
    async with MCPClient("test_client") as client:
        
        # Test 1: Analyze Market Sentiment
        print("\n=== Testing Market Sentiment Analysis ===")
        try:
            news_data = [
                {
                    "title": "Reliance Industries reports strong Q3 earnings",
                    "content": "Reliance Industries posted a 25% increase in quarterly profits driven by strong performance in petrochemicals and retail segments."
                },
                {
                    "title": "RBI maintains repo rate at 6.5%",
                    "content": "The Reserve Bank of India kept the repo rate unchanged, citing inflation concerns and global economic uncertainty."
                }
            ]
            
            sentiment_request = {
                "news_data": news_data,
                "timeframe": "1d",
                "symbol": "RELIANCE"
            }
            
            response = await client.send_request(
                server_url, "analyze_market_sentiment", sentiment_request
            )
            
            print(f"Sentiment Analysis Response: {json.dumps(response.content, indent=2)}")
            
        except Exception as e:
            print(f"Error testing sentiment analysis: {e}")
        
        # Test 2: Detect Market Regime
        print("\n=== Testing Market Regime Detection ===")
        try:
            market_data = {
                "nifty_50": {
                    "current_price": 21500,
                    "change_1d": -0.5,
                    "change_1w": 2.3,
                    "change_1m": -1.8,
                    "volatility": 15.2,
                    "volume_ratio": 1.2
                },
                "vix": 14.5,
                "advance_decline_ratio": 0.8,
                "market_breadth": "negative"
            }
            
            regime_request = {
                "market_data": market_data,
                "lookback_period": 30
            }
            
            response = await client.send_request(
                server_url, "detect_market_regime", regime_request
            )
            
            print(f"Market Regime Response: {json.dumps(response.content, indent=2)}")
            
        except Exception as e:
            print(f"Error testing market regime detection: {e}")
        
        # Test 3: Assess Event Impact
        print("\n=== Testing Event Impact Assessment ===")
        try:
            event_data = {
                "event_type": "earnings_announcement",
                "company": "TCS",
                "details": "Tata Consultancy Services announces Q3 results with 15% YoY growth in revenue",
                "sector": "IT",
                "announcement_time": "2024-01-15T16:00:00Z"
            }
            
            impact_request = {
                "event_data": event_data,
                "affected_symbols": ["TCS", "INFY", "WIPRO", "HCLTECH"]
            }
            
            response = await client.send_request(
                server_url, "assess_event_impact", impact_request
            )
            
            print(f"Event Impact Response: {json.dumps(response.content, indent=2)}")
            
        except Exception as e:
            print(f"Error testing event impact assessment: {e}")
        
        # Test 4: Generate Market Insights
        print("\n=== Testing Market Insights Generation ===")
        try:
            market_context = {
                "market_conditions": {
                    "trend": "sideways",
                    "volatility": "medium",
                    "sentiment": "cautious"
                },
                "economic_indicators": {
                    "inflation": 5.2,
                    "gdp_growth": 6.8,
                    "interest_rates": 6.5
                },
                "sector_performance": {
                    "IT": 2.1,
                    "Banking": -1.5,
                    "Pharma": 0.8
                }
            }
            
            insights_request = {
                "market_context": market_context,
                "focus_areas": ["trading", "risk_management"]
            }
            
            response = await client.send_request(
                server_url, "generate_market_insights", insights_request
            )
            
            print(f"Market Insights Response: {json.dumps(response.content, indent=2)}")
            
        except Exception as e:
            print(f"Error testing market insights generation: {e}")
        
        # Test 5: Explain Market Context
        print("\n=== Testing Market Context Explanation ===")
        try:
            current_conditions = {
                "nifty_level": 21500,
                "market_trend": "consolidation",
                "key_events": ["RBI policy meeting", "Q3 earnings season"],
                "global_factors": ["Fed policy uncertainty", "China reopening"],
                "sector_rotation": "IT to Banking"
            }
            
            explanation_request = {
                "current_conditions": current_conditions,
                "detail_level": "medium"
            }
            
            response = await client.send_request(
                server_url, "explain_market_context", explanation_request
            )
            
            print(f"Market Context Explanation: {json.dumps(response.content, indent=2)}")
            
        except Exception as e:
            print(f"Error testing market context explanation: {e}")


async def test_server_health():
    """Test server health endpoint."""
    print("\n=== Testing Server Health ===")
    
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8007/health") as response:
                health_data = await response.json()
                print(f"Health Check Response: {json.dumps(health_data, indent=2)}")
                
    except Exception as e:
        print(f"Error testing server health: {e}")


async def main():
    """Main test function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing LLM Market Intelligence MCP Server...")
    print("Make sure the server is running on localhost:8007")
    
    # Test server health first
    await test_server_health()
    
    # Test MCP endpoints
    await test_llm_market_intelligence_server()
    
    print("\nTesting completed!")


if __name__ == "__main__":
    asyncio.run(main())
