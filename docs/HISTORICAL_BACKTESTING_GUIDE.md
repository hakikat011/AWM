# Historical Backtesting Framework for LLM-Enhanced AWM Trading System

## Overview

This comprehensive historical backtesting framework validates the LLM-enhanced AWM trading system against 12 months of historical NSE/BSE data. It provides A/B comparison between LLM-enhanced and quantitative-only strategies with statistical significance testing, risk analysis, and production readiness assessment.

## Features

### 🔍 **Comprehensive Analysis**
- **12-month historical backtesting** (January 2023 - December 2023)
- **A/B comparison** between LLM-enhanced and quantitative-only strategies
- **Statistical significance testing** with multiple statistical tests
- **Risk analysis** including drawdown, VaR, and tail risk assessment
- **LLM effectiveness analysis** with sentiment and regime accuracy tracking
- **Edge case testing** for high volatility and stress scenarios
- **Production readiness assessment** with deployment recommendations

### 📊 **Performance Metrics**
- Total and annualized returns
- Sharpe and Sortino ratios
- Maximum drawdown analysis
- Win rate and profit factor
- Value at Risk (95%, 99%)
- Monthly returns distribution
- Transaction cost and slippage impact

### 🧪 **Statistical Testing**
- Paired t-test for return differences
- Mann-Whitney U test (non-parametric)
- Kolmogorov-Smirnov test for distribution differences
- Bootstrap test for Sharpe ratio significance
- Correlation analysis

### 🎯 **Production Readiness**
- Performance criteria evaluation
- Risk criteria assessment
- LLM effectiveness validation
- Infrastructure readiness check
- Deployment plan generation
- Risk mitigation procedures

## Quick Start

### 1. Prerequisites

Ensure all AWM system components are running:
```bash
# Start the complete AWM system
docker-compose up -d

# Verify services are running
curl http://localhost:8001/health  # Market Data Server
curl http://localhost:8004/health  # Signal Generation Server
curl http://localhost:8005/health  # Decision Engine Server
curl http://localhost:8007/health  # LLM Market Intelligence Server
```

### 2. Basic Usage

Run backtesting with default configuration:
```bash
python scripts/run_historical_backtest.py
```

Run with custom configuration:
```bash
python scripts/run_historical_backtest.py --config config/custom_backtest.yaml
```

### 3. Command Line Options

```bash
python scripts/run_historical_backtest.py \
    --config config/backtesting_config.yaml \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --symbols RELIANCE TCS HDFCBANK \
    --initial-capital 1000000 \
    --dry-run
```

**Options:**
- `--config`: Configuration file path
- `--start-date`: Override start date (YYYY-MM-DD)
- `--end-date`: Override end date (YYYY-MM-DD)
- `--symbols`: Override symbols to test
- `--initial-capital`: Override initial capital
- `--dry-run`: Validate configuration without running

## Configuration

### Configuration File Structure

The backtesting framework uses YAML configuration files. See `config/backtesting_config.yaml` for the complete example.

#### Key Configuration Sections:

**Backtest Period:**
```yaml
backtest_period:
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  timezone: "Asia/Kolkata"
```

**Portfolio Settings:**
```yaml
portfolio:
  initial_capital: 1000000  # ₹10,00,000
  max_position_size_pct: 0.10  # 10% max position
  cash_reserve_pct: 0.20  # 20% cash reserve
```

**Transaction Costs (Indian Markets):**
```yaml
transaction_costs:
  brokerage_pct: 0.001      # 0.1% brokerage
  stt_pct: 0.001            # Securities Transaction Tax
  exchange_charges_pct: 0.0001
  gst_pct: 0.00018
  total_cost_pct: 0.00228   # Total ~0.228%
```

**Production Readiness Criteria:**
```yaml
production_readiness:
  performance_criteria:
    min_return_improvement: 0.01      # 1% minimum improvement
    min_sharpe_improvement: 0.05      # 0.05 Sharpe improvement
    max_drawdown_tolerance: 0.15      # 15% max drawdown
  
  llm_criteria:
    min_sentiment_accuracy: 0.65      # 65% sentiment accuracy
    max_latency_p95: 500              # 500ms max latency
```

## Understanding Results

### Performance Comparison Output

```
📊 PERFORMANCE COMPARISON:
Metric                   LLM-Enhanced    Quantitative    Improvement    
----------------------------------------------------------------------
Total Return             15.20%          12.80%          2.40%         
Sharpe Ratio             1.250           1.180           0.070         
Max Drawdown             8.50%           11.20%          2.70%         
Win Rate                 62.30%          58.70%          3.60%         
Total Trades             156             142             14            
```

### Statistical Significance

```
📈 STATISTICAL SIGNIFICANCE:
Tests Significant: 2/3
Conclusion: SIGNIFICANT
```

### Production Readiness

```
🎯 PRODUCTION READINESS:
Overall Score: 78.5%
Recommendation: READY_WITH_MONITORING

📋 DEPLOYMENT PLAN:
Initial Allocation: 10%
Monitoring Frequency: Daily for first month
```

## Interpreting Results

### Performance Metrics

- **Total Return**: Absolute return over the backtesting period
- **Sharpe Ratio**: Risk-adjusted return (>1.0 is good, >1.5 is excellent)
- **Maximum Drawdown**: Largest peak-to-trough decline (<15% preferred)
- **Win Rate**: Percentage of profitable trades (>55% is good)

### Statistical Tests

- **T-test**: Tests if return differences are statistically significant
- **Mann-Whitney U**: Non-parametric test for distribution differences
- **Sharpe Ratio Test**: Bootstrap test for Sharpe ratio significance

### Production Readiness Scores

- **80%+**: Ready for production deployment
- **70-79%**: Ready with enhanced monitoring
- **60-69%**: Pilot deployment recommended
- **<60%**: Requires improvements

## Advanced Usage

### Custom Configuration

Create a custom configuration file:

```yaml
# custom_backtest.yaml
backtest_period:
  start_date: "2023-06-01"
  end_date: "2023-12-31"

portfolio:
  initial_capital: 500000

symbols:
  - "RELIANCE"
  - "TCS"

# Override other settings as needed
```

Run with custom configuration:
```bash
python scripts/run_historical_backtest.py --config custom_backtest.yaml
```

### Testing Specific Scenarios

**High Volatility Period:**
```yaml
backtest_period:
  start_date: "2023-03-01"  # Banking sector stress
  end_date: "2023-03-31"
```

**Earnings Season:**
```yaml
backtest_period:
  start_date: "2023-10-01"  # Q2 earnings season
  end_date: "2023-11-30"
```

### Batch Testing

Run multiple configurations:
```bash
# Test different time periods
for period in Q1 Q2 Q3 Q4; do
    python scripts/run_historical_backtest.py --config config/backtest_${period}.yaml
done
```

## Output Files

### Generated Reports

All results are saved in the `test_results/` directory:

- **`historical_backtest_results_YYYYMMDD_HHMMSS.json`**: Complete results in JSON format
- **`historical_backtest_report_YYYYMMDD_HHMMSS.txt`**: Human-readable report
- **`backtest_config_YYYYMMDD_HHMMSS.yaml`**: Configuration used for the run

### Report Sections

1. **Executive Summary**: Key findings and recommendations
2. **Performance Analysis**: Detailed performance metrics
3. **Statistical Tests**: Significance testing results
4. **Risk Analysis**: Drawdown and tail risk assessment
5. **LLM Effectiveness**: Sentiment and regime accuracy
6. **Production Readiness**: Deployment recommendations

## Testing the Framework

Run unit tests:
```bash
pytest tests/test_historical_backtesting.py -v
```

Run integration tests:
```bash
python scripts/run_historical_backtest.py --dry-run
```

## Troubleshooting

### Common Issues

**1. Service Connection Errors:**
```bash
# Check if all services are running
docker-compose ps

# Restart services if needed
docker-compose restart
```

**2. Historical Data Issues:**
```bash
# Verify market data server
curl http://localhost:8001/health

# Check data availability
curl "http://localhost:8001/get_historical_data?symbol=RELIANCE&start_date=2023-01-01"
```

**3. Memory Issues:**
```bash
# Monitor memory usage
docker stats

# Reduce batch size in configuration
portfolio:
  max_concurrent_requests: 5
```

**4. LLM Service Issues:**
```bash
# Check LLM service
curl http://localhost:8007/health

# Monitor GPU usage
nvidia-smi
```

### Performance Optimization

**1. Reduce Data Range:**
```yaml
backtest_period:
  start_date: "2023-10-01"  # Test shorter period first
  end_date: "2023-12-31"
```

**2. Limit Symbols:**
```yaml
symbols:
  - "RELIANCE"  # Test with fewer symbols
  - "TCS"
```

**3. Adjust Batch Sizes:**
```yaml
servers:
  timeouts:
    market_data: 60      # Increase timeouts
    llm_intelligence: 180
```

## Best Practices

### 1. Validation Workflow

1. **Start Small**: Test with 1-2 symbols and 1-3 months
2. **Validate Configuration**: Use `--dry-run` to check settings
3. **Monitor Progress**: Watch logs for errors or warnings
4. **Review Results**: Analyze statistical significance
5. **Scale Up**: Gradually increase scope if results are positive

### 2. Configuration Management

- Use version control for configuration files
- Document configuration changes
- Test configuration changes in isolation
- Maintain separate configs for different scenarios

### 3. Result Analysis

- Focus on statistical significance, not just raw performance
- Consider transaction costs and slippage impact
- Analyze risk-adjusted returns (Sharpe ratio)
- Review drawdown characteristics
- Validate LLM effectiveness metrics

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review log files in `logs/historical_backtesting.log`
3. Examine detailed results in `test_results/` directory
4. Run unit tests to validate framework functionality

## Next Steps

After successful backtesting:

1. **Review Results**: Analyze comprehensive report
2. **Validate Significance**: Ensure statistical significance
3. **Plan Deployment**: Follow production readiness recommendations
4. **Monitor Performance**: Implement suggested monitoring
5. **Gradual Rollout**: Start with recommended allocation percentage
