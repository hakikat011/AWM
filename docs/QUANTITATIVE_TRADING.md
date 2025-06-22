# AWM Quantitative Trading System

## Overview

The AWM Quantitative Trading System is a comprehensive, AI-powered trading platform that combines advanced quantitative analysis, signal generation, and autonomous decision-making to execute trades in the Indian stock market.

## Architecture

The system follows a layered architecture with specialized components:

### 1. Quantitative Analysis Engine (`services/mcp_servers/quantitative_analysis/`)
- **Purpose**: Advanced quantitative analysis and strategy implementation
- **Port**: 8003
- **Capabilities**:
  - Multiple trading strategies (SMA crossover, RSI mean reversion, Bollinger Bands, Momentum)
  - Comprehensive backtesting with performance metrics
  - Risk metrics calculation (Sharpe ratio, VaR, drawdown analysis)
  - Market regime detection
  - Portfolio optimization

### 2. Signal Generation System (`services/mcp_servers/signal_generation/`)
- **Purpose**: Combines multiple analysis sources into trading signals
- **Port**: 8004
- **Capabilities**:
  - Multi-strategy signal aggregation
  - Technical analysis integration
  - News sentiment incorporation
  - Signal consensus calculation
  - Signal ranking and filtering

### 3. Decision Engine (`services/mcp_servers/decision_engine/`)
- **Purpose**: Makes autonomous trading decisions with risk management
- **Port**: 8005
- **Capabilities**:
  - Risk-adjusted decision making
  - Position sizing based on portfolio constraints
  - Stop-loss and take-profit calculation
  - Trade proposal evaluation
  - Portfolio-level decision coordination

### 4. Autonomous Trading Agent (`services/agents/autonomous_trading/`)
- **Purpose**: Continuously monitors markets and executes trades
- **Capabilities**:
  - Automated market scanning
  - Real-time decision making
  - Trade execution and monitoring
  - Position management
  - Performance tracking

## Trading Strategies

### 1. Simple Moving Average (SMA) Crossover
- **Logic**: Buy when short-term SMA crosses above long-term SMA, sell when it crosses below
- **Parameters**: 
  - Short period: 20 days (default)
  - Long period: 50 days (default)
- **Best for**: Trending markets

### 2. RSI Mean Reversion
- **Logic**: Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought)
- **Parameters**:
  - RSI period: 14 days (default)
  - Oversold threshold: 30
  - Overbought threshold: 70
- **Best for**: Range-bound markets

### 3. Bollinger Bands
- **Logic**: Buy when price touches lower band, sell when price touches upper band
- **Parameters**:
  - Period: 20 days (default)
  - Standard deviations: 2 (default)
- **Best for**: Mean-reverting markets

### 4. Momentum Strategy
- **Logic**: Buy on strong positive momentum, sell on strong negative momentum
- **Parameters**:
  - Lookback period: 10 days (default)
  - Momentum threshold: 5% (default)
- **Best for**: Trending markets with strong momentum

## Configuration Profiles

### Conservative Profile
```json
{
  "min_signal_confidence": 0.8,
  "max_position_size_pct": 0.05,
  "max_portfolio_risk": 0.02,
  "stop_loss_pct": 0.03,
  "take_profit_pct": 0.06,
  "paper_trading": true
}
```

### Moderate Profile (Default)
```json
{
  "min_signal_confidence": 0.7,
  "max_position_size_pct": 0.08,
  "max_portfolio_risk": 0.03,
  "stop_loss_pct": 0.04,
  "take_profit_pct": 0.08,
  "paper_trading": true
}
```

### Aggressive Profile
```json
{
  "min_signal_confidence": 0.6,
  "max_position_size_pct": 0.12,
  "max_portfolio_risk": 0.05,
  "stop_loss_pct": 0.05,
  "take_profit_pct": 0.10,
  "paper_trading": true
}
```

## Getting Started

### 1. Start the Services

```bash
# Start all services
docker-compose up -d

# Or start specific quantitative trading services
docker-compose up -d quantitative-analysis-server signal-generation-server decision-engine-server
```

### 2. Run the Demo

```bash
# Run the demonstration script
python scripts/demo_quantitative_trading.py
```

### 3. Start Autonomous Trading

```bash
# Enable autonomous trading (paper trading mode)
export AUTONOMOUS_TRADING_ENABLED=true
export PAPER_TRADING_ONLY=true

# Start the autonomous trading agent
docker-compose up -d autonomous-trading-agent
```

## API Usage Examples

### Generate Quantitative Signals

```python
from shared.mcp_client.base import MCPClient

async def generate_signals():
    client = MCPClient("demo")
    async with client as c:
        response = await c.send_request(
            "http://localhost:8003",
            "generate_signals",
            {
                "symbol": "RELIANCE",
                "data": market_data,
                "strategies": ["sma_crossover", "rsi_mean_reversion"]
            }
        )
    return response["signals"]
```

### Make Trading Decision

```python
async def make_decision():
    client = MCPClient("demo")
    async with client as c:
        response = await c.send_request(
            "http://localhost:8005",
            "make_trading_decision",
            {
                "symbol": "RELIANCE",
                "portfolio_id": "my-portfolio",
                "config": "moderate"
            }
        )
    return response["decision"]
```

### Run Strategy Backtest

```python
async def backtest_strategy():
    client = MCPClient("demo")
    async with client as c:
        response = await c.send_request(
            "http://localhost:8003",
            "backtest_strategy",
            {
                "strategy": "sma_crossover",
                "data": market_data,
                "initial_capital": 100000
            }
        )
    return response["backtest_results"]
```

## Risk Management

The system includes comprehensive risk management:

1. **Position Sizing**: Limits position size as percentage of portfolio
2. **Portfolio Risk**: Monitors overall portfolio VaR
3. **Stop Loss**: Automatic stop-loss orders
4. **Take Profit**: Automatic profit-taking
5. **Daily Limits**: Maximum number of trades per day
6. **Paper Trading**: Safe testing mode (default)

## Performance Metrics

The system tracks comprehensive performance metrics:

- **Returns**: Total return, annualized return
- **Risk**: Volatility, Sharpe ratio, Sortino ratio, Calmar ratio
- **Drawdown**: Maximum drawdown, current drawdown
- **Win Rate**: Percentage of profitable trades
- **Risk Metrics**: VaR, Expected Shortfall

## Monitoring and Alerts

- Real-time performance tracking
- Risk violation alerts
- Trade execution notifications
- System health monitoring

## Safety Features

1. **Paper Trading Mode**: All trades are simulated by default
2. **Risk Limits**: Hard limits on position sizes and portfolio risk
3. **Emergency Stop**: Immediate halt of all trading activities
4. **Manual Override**: Human intervention capabilities
5. **Audit Trail**: Complete logging of all decisions and trades

## Testing

```bash
# Run integration tests
pytest tests/integration/test_quantitative_trading_pipeline.py -v

# Run specific test
pytest tests/integration/test_quantitative_trading_pipeline.py::TestQuantitativeTradingPipeline::test_end_to_end_pipeline -v
```

## Environment Variables

```bash
# Trading Configuration
AUTONOMOUS_TRADING_ENABLED=false
PAPER_TRADING_ONLY=true
SCAN_INTERVAL_SECONDS=300
DECISION_CONFIG=moderate
MAX_DAILY_TRADES=10
TRADING_WATCHLIST=RELIANCE,TCS,HDFCBANK,INFY,HINDUNILVR

# Service URLs
QUANTITATIVE_ANALYSIS_SERVER_URL=http://quantitative-analysis-server:8003
SIGNAL_GENERATION_SERVER_URL=http://signal-generation-server:8004
DECISION_ENGINE_SERVER_URL=http://decision-engine-server:8005
```

## Next Steps

1. **Enable Live Trading**: Set `PAPER_TRADING_ONLY=false` (only after thorough testing)
2. **Add Custom Strategies**: Implement additional quantitative strategies
3. **Enhance Risk Management**: Add more sophisticated risk models
4. **Optimize Parameters**: Use machine learning for parameter optimization
5. **Add More Assets**: Extend to options, futures, and other instruments

## Support

For questions or issues with the quantitative trading system, please refer to the main AWM documentation or contact the development team.
