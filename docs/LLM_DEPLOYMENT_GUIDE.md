# LLM Market Intelligence Deployment Guide

## Overview

This guide covers the deployment of the LLM-Enhanced Market Intelligence system for the AWM quantitative trading platform. The system integrates Mistral-7B-Instruct-v0.2 for advanced market analysis and contextual trading intelligence.

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with 12-16GB VRAM minimum (RTX 3080/4080, Tesla V100, A100)
- **RAM**: 32GB system RAM recommended
- **Storage**: 50GB free space for model cache and data
- **CPU**: 8+ cores recommended for concurrent processing

### Software Requirements

- **Docker**: Version 20.10+ with GPU support
- **NVIDIA Container Toolkit**: For GPU access in containers
- **Python**: 3.11+ (for development/testing)
- **CUDA**: 12.1+ compatible drivers

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd AWM

# Create environment file
cp .env.example .env

# Configure LLM-specific environment variables
echo "LLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2" >> .env
echo "LLM_GPU_MEMORY_FRACTION=0.8" >> .env
echo "LLM_MAX_TOKENS=2048" >> .env
echo "LLM_TEMPERATURE=0.1" >> .env
```

### 2. Build and Deploy

```bash
# Build all services including LLM Market Intelligence
docker-compose build

# Start the complete system
docker-compose up -d

# Verify LLM service is running
curl http://localhost:8007/health
```

### 3. Validate Deployment

```bash
# Run comprehensive tests
python scripts/run_comprehensive_tests.py

# Run paper trading validation
python scripts/paper_trading_validation.py
```

## Configuration

### LLM Model Configuration

The system supports various configuration options in `services/mcp_servers/llm_market_intelligence/config.py`:

```python
# Model Configuration
LLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2  # Model to use
LLM_MODEL_PATH=/app/models                          # Model cache directory
LLM_MAX_TOKENS=2048                                 # Maximum response tokens
LLM_TEMPERATURE=0.1                                 # Response consistency (0.0-1.0)
LLM_GPU_MEMORY_FRACTION=0.8                         # GPU memory allocation

# Performance Configuration
LLM_TENSOR_PARALLEL_SIZE=1                          # Multi-GPU parallelization
LLM_MAX_CONCURRENT_REQUESTS=10                      # Concurrent request limit
LLM_REQUEST_TIMEOUT=300                             # Request timeout (seconds)

# Market Configuration
SENTIMENT_CONFIDENCE_THRESHOLD=0.7                  # Minimum confidence for sentiment
MAX_NEWS_ARTICLES=50                                # News articles per analysis
CACHE_TTL=300                                       # Cache duration (seconds)
```

### Integration Configuration

Update existing services to use LLM Market Intelligence:

```yaml
# docker-compose.yml additions
environment:
  - LLM_MARKET_INTELLIGENCE_SERVER_URL=http://llm-market-intelligence-server:8007
```

## Performance Optimization

### GPU Optimization

1. **Memory Management**:
   ```python
   # Adjust GPU memory fraction based on available VRAM
   LLM_GPU_MEMORY_FRACTION=0.8  # Use 80% of available GPU memory
   ```

2. **Model Optimization**:
   ```python
   # Use optimized data types
   dtype: "float16"  # Reduces memory usage by 50%
   ```

3. **Caching Strategy**:
   ```python
   # Configure Redis caching for frequent queries
   SENTIMENT_CACHE_TTL=1800      # 30 minutes for sentiment analysis
   REGIME_CACHE_TTL=900          # 15 minutes for market regime
   ```

### Latency Optimization

Target performance metrics:
- **Sentiment Analysis**: <500ms
- **Market Regime Detection**: <800ms
- **Market Insights**: <1500ms
- **End-to-End Decision**: <2000ms

## Monitoring and Logging

### Health Checks

The LLM service provides comprehensive health monitoring:

```bash
# Basic health check
curl http://localhost:8007/health

# Detailed metrics
curl http://localhost:8007/metrics
```

### Logging Configuration

```python
# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Performance Monitoring

Key metrics to monitor:
- **Inference Latency**: Response time per request
- **GPU Utilization**: Memory and compute usage
- **Cache Hit Rate**: Percentage of cached responses
- **Error Rate**: Failed requests per minute
- **Throughput**: Requests processed per second

## Troubleshooting

### Common Issues

1. **GPU Memory Errors**:
   ```bash
   # Reduce memory fraction
   LLM_GPU_MEMORY_FRACTION=0.6
   
   # Or use CPU fallback
   CUDA_VISIBLE_DEVICES=""
   ```

2. **Model Loading Failures**:
   ```bash
   # Clear model cache
   rm -rf /app/models/*
   
   # Restart with fresh download
   docker-compose restart llm-market-intelligence-server
   ```

3. **High Latency**:
   ```bash
   # Check GPU utilization
   nvidia-smi
   
   # Verify cache configuration
   redis-cli ping
   ```

4. **Integration Issues**:
   ```bash
   # Verify service connectivity
   docker-compose logs signal-generation-server
   docker-compose logs decision-engine-server
   ```

### Performance Tuning

1. **Batch Processing**:
   ```python
   # Process multiple symbols together
   batch_size = 5
   ```

2. **Async Optimization**:
   ```python
   # Use asyncio for concurrent processing
   max_concurrent_requests = 10
   ```

3. **Model Quantization**:
   ```python
   # Use quantized models for faster inference
   load_in_8bit = True
   ```

## Security Considerations

### API Security

1. **Authentication**: Implement API key authentication for production
2. **Rate Limiting**: Configure request rate limits per client
3. **Input Validation**: Sanitize all input data
4. **Network Security**: Use VPN/firewall for production deployment

### Data Privacy

1. **Model Isolation**: LLM processing is containerized and isolated
2. **Data Retention**: Configure appropriate cache TTL values
3. **Audit Logging**: Log all trading decisions with LLM context

## Production Deployment

### Scaling Considerations

1. **Horizontal Scaling**:
   ```yaml
   # docker-compose.yml
   llm-market-intelligence-server:
     deploy:
       replicas: 3
   ```

2. **Load Balancing**:
   ```yaml
   # Use nginx or similar for load balancing
   nginx:
     image: nginx:alpine
     ports:
       - "80:80"
   ```

3. **Database Scaling**:
   ```yaml
   # Redis cluster for caching
   redis-cluster:
     image: redis:alpine
     deploy:
       replicas: 3
   ```

### Backup and Recovery

1. **Model Backup**: Regular backup of model cache directory
2. **Configuration Backup**: Version control for all configuration files
3. **Data Backup**: Regular backup of trading data and logs

## Testing and Validation

### Automated Testing

```bash
# Run full test suite
python scripts/run_comprehensive_tests.py

# Run specific test categories
pytest tests/test_llm_market_intelligence.py -v
pytest tests/test_llm_integration.py -v
pytest tests/test_llm_performance.py -v
```

### Paper Trading Validation

```bash
# Run A/B comparison
python scripts/paper_trading_validation.py

# Extended validation (30 days)
python scripts/paper_trading_validation.py --days 30
```

### Performance Benchmarking

```bash
# Latency benchmarking
python scripts/benchmark_latency.py

# Throughput testing
python scripts/benchmark_throughput.py
```

## Support and Maintenance

### Regular Maintenance

1. **Model Updates**: Quarterly evaluation of newer models
2. **Performance Review**: Monthly performance analysis
3. **Security Updates**: Regular security patches and updates
4. **Cache Cleanup**: Weekly cache cleanup and optimization

### Support Contacts

- **Technical Issues**: [Technical Support]
- **Performance Issues**: [Performance Team]
- **Security Issues**: [Security Team]

## Appendix

### Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL_NAME` | `mistralai/Mistral-7B-Instruct-v0.2` | HuggingFace model name |
| `LLM_MODEL_PATH` | `/app/models` | Model cache directory |
| `LLM_MAX_TOKENS` | `2048` | Maximum response tokens |
| `LLM_TEMPERATURE` | `0.1` | Response randomness (0.0-1.0) |
| `LLM_GPU_MEMORY_FRACTION` | `0.8` | GPU memory allocation |
| `SENTIMENT_CONFIDENCE_THRESHOLD` | `0.7` | Minimum sentiment confidence |
| `CACHE_TTL` | `300` | Default cache TTL (seconds) |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/metrics` | GET | Performance metrics |
| `/analyze_market_sentiment` | POST | Sentiment analysis |
| `/detect_market_regime` | POST | Market regime detection |
| `/assess_event_impact` | POST | Event impact assessment |
| `/generate_market_insights` | POST | Market insights generation |
| `/explain_market_context` | POST | Market context explanation |
