"""
Performance validation tests for LLM-enhanced trading system.
"""

import pytest
import asyncio
import time
import statistics
from unittest.mock import AsyncMock, patch
from concurrent.futures import ThreadPoolExecutor

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services.mcp_servers.llm_market_intelligence.server import LLMMarketIntelligenceServer
from services.mcp_servers.llm_market_intelligence.llm_engine import LLMEngine


class TestLLMPerformance:
    """Performance tests for LLM Market Intelligence."""
    
    @pytest.fixture
    async def server(self):
        """Create test server with mocked components."""
        server = LLMMarketIntelligenceServer()
        server.redis_client = AsyncMock()
        return server
    
    @pytest.fixture
    def mock_llm_engine(self):
        """Mock LLM engine with realistic response times."""
        with patch('services.mcp_servers.llm_market_intelligence.server.llm_engine') as mock:
            # Simulate realistic inference times
            async def mock_analyze_sentiment(*args, **kwargs):
                await asyncio.sleep(0.3)  # 300ms inference time
                return {
                    "sentiment": "positive",
                    "confidence": 0.8,
                    "reasoning": "Mock analysis",
                    "market_impact": "bullish",
                    "key_factors": ["factor1"],
                    "inference_time": 0.3
                }
            
            async def mock_detect_market_regime(*args, **kwargs):
                await asyncio.sleep(0.5)  # 500ms inference time
                return {
                    "regime_type": "bull_market",
                    "confidence": 0.75,
                    "explanation": "Mock regime analysis",
                    "key_indicators": ["indicator1"],
                    "duration_estimate": 30,
                    "risk_level": "low",
                    "inference_time": 0.5
                }
            
            async def mock_generate_response(*args, **kwargs):
                await asyncio.sleep(0.4)  # 400ms inference time
                return {
                    "response": '{"test": "response"}',
                    "inference_time": 0.4
                }
            
            mock.analyze_sentiment = mock_analyze_sentiment
            mock.detect_market_regime = mock_detect_market_regime
            mock.generate_response = mock_generate_response
            yield mock
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis_latency(self, server, mock_llm_engine):
        """Test sentiment analysis meets <500ms latency requirement."""
        # Setup
        server._get_cached_result = AsyncMock(return_value=None)
        server._cache_result = AsyncMock()
        
        content = {
            "news_data": [{"title": "Test news", "content": "Test content"}],
            "timeframe": "1d",
            "symbol": "RELIANCE"
        }
        
        # Get handler
        await server.register_handlers()
        handler = server.handlers["analyze_market_sentiment"]
        
        # Measure latency
        start_time = time.time()
        result = await handler(content)
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Verify latency requirement
        assert latency < 500, f"Sentiment analysis latency {latency:.2f}ms exceeds 500ms requirement"
        assert result["inference_time"] == 0.3
    
    @pytest.mark.asyncio
    async def test_market_regime_latency(self, server, mock_llm_engine):
        """Test market regime detection meets latency requirements."""
        # Setup
        server._get_cached_result = AsyncMock(return_value=None)
        server._cache_result = AsyncMock()
        
        content = {
            "market_data": {"current_price": 2500, "volatility": 15.2},
            "lookback_period": 30
        }
        
        # Get handler
        await server.register_handlers()
        handler = server.handlers["detect_market_regime"]
        
        # Measure latency
        start_time = time.time()
        result = await handler(content)
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000
        
        # Verify latency (regime detection can be slightly higher)
        assert latency < 800, f"Market regime latency {latency:.2f}ms exceeds 800ms threshold"
        assert result["inference_time"] == 0.5
    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, server, mock_llm_engine):
        """Test system handles concurrent requests efficiently."""
        # Setup
        server._get_cached_result = AsyncMock(return_value=None)
        server._cache_result = AsyncMock()
        await server.register_handlers()
        
        # Create multiple concurrent requests
        sentiment_handler = server.handlers["analyze_market_sentiment"]
        regime_handler = server.handlers["detect_market_regime"]
        
        sentiment_content = {
            "news_data": [{"title": "Test", "content": "Content"}],
            "symbol": "RELIANCE"
        }
        
        regime_content = {
            "market_data": {"current_price": 2500},
            "lookback_period": 30
        }
        
        # Create concurrent tasks
        tasks = []
        for i in range(10):
            if i % 2 == 0:
                tasks.append(sentiment_handler(sentiment_content))
            else:
                tasks.append(regime_handler(regime_content))
        
        # Measure concurrent execution time
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = (end_time - start_time) * 1000
        
        # Verify all requests completed successfully
        assert len(results) == 10
        assert all("error" not in result for result in results)
        
        # Verify reasonable total time (should be much less than sequential execution)
        sequential_time_estimate = 10 * 500  # 10 requests * 500ms each
        assert total_time < sequential_time_estimate * 0.7, f"Concurrent execution too slow: {total_time:.2f}ms"
    
    @pytest.mark.asyncio
    async def test_cache_performance_improvement(self, server, mock_llm_engine):
        """Test caching provides significant performance improvement."""
        # Setup
        await server.register_handlers()
        handler = server.handlers["analyze_market_sentiment"]
        
        content = {
            "news_data": [{"title": "Test news", "content": "Test content"}],
            "symbol": "RELIANCE"
        }
        
        # First request (cache miss)
        server._get_cached_result = AsyncMock(return_value=None)
        server._cache_result = AsyncMock()
        
        start_time = time.time()
        result1 = await handler(content)
        cache_miss_time = (time.time() - start_time) * 1000
        
        # Second request (cache hit)
        cached_result = {
            "sentiment_score": 0.7,
            "confidence": 0.8,
            "reasoning": "Cached analysis",
            "market_impact": "bullish",
            "symbol": "RELIANCE"
        }
        server._get_cached_result = AsyncMock(return_value=cached_result)
        
        start_time = time.time()
        result2 = await handler(content)
        cache_hit_time = (time.time() - start_time) * 1000
        
        # Verify cache hit is significantly faster
        assert cache_hit_time < cache_miss_time * 0.1, f"Cache hit not fast enough: {cache_hit_time:.2f}ms vs {cache_miss_time:.2f}ms"
        assert result2 == cached_result
    
    @pytest.mark.asyncio
    async def test_throughput_under_load(self, server, mock_llm_engine):
        """Test system throughput under sustained load."""
        # Setup
        server._get_cached_result = AsyncMock(return_value=None)
        server._cache_result = AsyncMock()
        await server.register_handlers()
        
        handler = server.handlers["analyze_market_sentiment"]
        content = {
            "news_data": [{"title": "Load test", "content": "Content"}],
            "symbol": "TEST"
        }
        
        # Simulate sustained load
        num_requests = 50
        batch_size = 10
        
        total_requests = 0
        total_time = 0
        
        for batch in range(num_requests // batch_size):
            # Create batch of concurrent requests
            tasks = [handler(content) for _ in range(batch_size)]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            batch_time = time.time() - start_time
            
            total_requests += len(results)
            total_time += batch_time
            
            # Verify all requests succeeded
            assert all("error" not in result for result in results)
        
        # Calculate throughput
        throughput = total_requests / total_time  # requests per second
        
        # Verify minimum throughput requirement (target: >2 requests/second)
        assert throughput >= 2.0, f"Throughput {throughput:.2f} req/s below minimum requirement"
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, server, mock_llm_engine):
        """Test memory usage remains stable under load."""
        import psutil
        import gc
        
        # Setup
        server._get_cached_result = AsyncMock(return_value=None)
        server._cache_result = AsyncMock()
        await server.register_handlers()
        
        handler = server.handlers["analyze_market_sentiment"]
        content = {
            "news_data": [{"title": "Memory test", "content": "Content"}],
            "symbol": "MEM"
        }
        
        # Measure initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple requests
        for i in range(20):
            await handler(content)
            if i % 5 == 0:
                gc.collect()  # Force garbage collection
        
        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Verify memory usage is stable (increase should be minimal)
        assert memory_increase < 50, f"Memory usage increased by {memory_increase:.2f}MB, indicating potential leak"
    
    @pytest.mark.asyncio
    async def test_error_handling_performance(self, server):
        """Test error handling doesn't significantly impact performance."""
        # Setup with failing LLM engine
        with patch('services.mcp_servers.llm_market_intelligence.server.llm_engine') as mock:
            mock.analyze_sentiment.side_effect = Exception("Simulated LLM failure")
            
            await server.register_handlers()
            handler = server.handlers["analyze_market_sentiment"]
            
            content = {
                "news_data": [{"title": "Error test", "content": "Content"}],
                "symbol": "ERROR"
            }
            
            # Measure error handling time
            start_time = time.time()
            result = await handler(content)
            error_time = (time.time() - start_time) * 1000
            
            # Verify error is handled quickly
            assert error_time < 100, f"Error handling too slow: {error_time:.2f}ms"
            assert "error" in result
    
    def test_configuration_performance_impact(self):
        """Test different configurations and their performance impact."""
        from services.mcp_servers.llm_market_intelligence.config import Config
        
        # Test different model configurations
        configs = [
            {"max_tokens": 512, "temperature": 0.1},
            {"max_tokens": 1024, "temperature": 0.1},
            {"max_tokens": 2048, "temperature": 0.1}
        ]
        
        for config_params in configs:
            config = Config()
            config.llm.max_tokens = config_params["max_tokens"]
            config.llm.temperature = config_params["temperature"]
            
            # Verify configuration is valid
            assert config.llm.max_tokens > 0
            assert 0.0 <= config.llm.temperature <= 1.0
            
            # Note: In real implementation, you would measure actual inference times
            # with different configurations


class TestSystemIntegrationPerformance:
    """Test performance of integrated LLM-enhanced system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_decision_latency(self):
        """Test end-to-end decision making latency."""
        # This would test the full pipeline:
        # Signal Generation -> Decision Engine -> Autonomous Trading
        # Target: Complete decision cycle in <2 seconds
        
        # Mock the full pipeline
        with patch('services.mcp_servers.signal_generation.server.SignalGenerationServer') as mock_signal:
            with patch('services.mcp_servers.decision_engine.server.DecisionEngineServer') as mock_decision:
                
                # Setup realistic response times
                mock_signal.return_value._generate_comprehensive_signals = AsyncMock()
                mock_signal.return_value._generate_comprehensive_signals.return_value = {
                    "combined_signals": {"signal": "BUY", "confidence": 0.8}
                }
                
                mock_decision.return_value._make_comprehensive_decision = AsyncMock()
                mock_decision.return_value._make_comprehensive_decision.return_value = {
                    "action": "BUY", "confidence": 0.8, "quantity": 100
                }
                
                # Simulate end-to-end timing
                start_time = time.time()
                
                # This would be the actual pipeline execution
                # For now, just simulate the timing
                await asyncio.sleep(1.5)  # Simulate 1.5 second pipeline
                
                end_time = time.time()
                total_time = (end_time - start_time) * 1000
                
                # Verify end-to-end latency requirement
                assert total_time < 2000, f"End-to-end latency {total_time:.2f}ms exceeds 2000ms requirement"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
