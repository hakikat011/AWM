# Model Context Protocol (MCP)

## Overview
The Model Context Protocol (MCP) is the standardized communication format used between all components in the MFT-1 system. It enables modular development and seamless integration of new components.

## Message Structure
All MCP messages follow this JSON structure:

```json
{
  "request_id": "uuid-string",
  "timestamp": "ISO-8601-timestamp",
  "source": "component-name",
  "destination": "component-name",
  "message_type": "REQUEST|RESPONSE|ERROR|EVENT",
  "content": {
    // Message-specific payload
  },
  "metadata": {
    // Optional context information
  }
}
```

## Request Types
Each MCP server implements specific endpoints:

### MarketData_Server
- `get_price_history`: Retrieve OHLCV data for an instrument
- `get_current_quote`: Get latest price and volume
- `scan_market`: Find instruments matching criteria

### TechnicalAnalysis_Server
- `calculate_indicator`: Compute technical indicators (RSI, MACD, etc.)
- `detect_patterns`: Identify chart patterns
- `run_backtest`: Test a strategy against historical data

### News_Server
- `get_recent_news`: Retrieve news for specific instrument/sector
- `analyze_sentiment`: Get sentiment analysis of news

## Error Handling
Errors follow a standardized format:

```json
{
  "error_code": "ERROR_TYPE_CODE",
  "error_message": "Human-readable description",
  "severity": "INFO|WARNING|ERROR|CRITICAL",
  "recoverable": true|false,
  "retry_after": null|seconds
}
```