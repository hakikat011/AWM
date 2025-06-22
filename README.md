# AWM (Autonomous Wealth Management) System

A comprehensive, AI-driven trading system built with a layered architecture using the Model Context Protocol (MCP) for inter-service communication.

## Architecture Overview

The AWM system follows a 4-layer architecture:

### Layer 1: Data & Tooling (MCP Servers)
- **Market Data Server**: Real-time and historical market data
- **Technical Analysis Server**: Technical indicators and pattern recognition
- **Portfolio Management Server**: Position tracking and P&L calculation
- **Risk Assessment Server**: Risk metrics and scenario analysis
- **Trade Execution Server**: Order management and broker integration
- **News Server**: News data and sentiment analysis

### Layer 2: Intelligence (AI Agents)
- **Market Analysis Agent**: Technical and fundamental analysis
- **Risk Assessment Agent**: Portfolio risk evaluation
- **Strategy Optimization Agent**: Strategy backtesting and optimization
- **Trade Execution Agent**: Intelligent order routing

### Layer 3: Execution & Risk
- **Risk Management Engine**: Real-time risk monitoring and controls
- **Order Management System (OMS)**: Order lifecycle management
- **Portfolio Management System**: Real-time portfolio tracking

### Layer 4: Control & Oversight
- **Dashboard UI**: Real-time monitoring and trade approval
- **Alerting Service**: Notifications and alerts
- **Admin Interface**: System configuration

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd AWM
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Build and start services**
   ```bash
   docker-compose build
   docker-compose up -d
   ```

4. **Access the dashboard**
   Open http://localhost:8501 in your browser

### Environment Configuration

Key environment variables to configure in `.env`:

```bash
# Database
POSTGRES_USER=trading_user
POSTGRES_PASSWORD=your_strong_password
POSTGRES_DB=trading_db

# Broker API (Zerodha)
ZERODHA_API_KEY=your_api_key
ZERODHA_API_SECRET=your_api_secret
ZERODHA_ACCESS_TOKEN=  # Generated daily

# AI/LLM
OPENAI_API_KEY=your_openai_key

# Risk Management
MAX_POSITION_SIZE=100000
MAX_DAILY_LOSS=10000
MAX_PORTFOLIO_RISK=0.02

# Alerts
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## Project Structure

```
AWM/
├── services/
│   ├── mcp_servers/          # Layer 1: MCP Servers
│   │   ├── market_data/
│   │   ├── technical_analysis/
│   │   ├── portfolio_management/
│   │   ├── risk_assessment/
│   │   ├── trade_execution/
│   │   └── news/
│   ├── agents/               # Layer 2: AI Agents
│   │   ├── market_analysis/
│   │   ├── risk_assessment/
│   │   ├── strategy_optimization/
│   │   └── trade_execution/
│   ├── execution/            # Layer 3: Execution Systems
│   │   ├── risk_management/
│   │   ├── oms/
│   │   └── portfolio_management/
│   ├── dashboard/            # Layer 4: UI
│   └── alerting/
├── shared/                   # Shared libraries
│   ├── mcp_client/          # MCP communication
│   ├── database/            # Database utilities
│   ├── models/              # Data models
│   └── utils/               # Common utilities
├── database/
│   ├── init/                # Database initialization
│   └── migrations/          # Schema migrations
├── tests/                   # Test suites
├── docs/                    # Documentation
└── config/                  # Configuration files
```

## Development

### Running Individual Services

Each service can be run independently for development:

```bash
# Market Data Server
python services/mcp_servers/market_data/server.py

# Technical Analysis Server
python services/mcp_servers/technical_analysis/server.py
```

### Testing

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/unit/
pytest tests/integration/
```

### Adding New MCP Servers

1. Create new directory under `services/mcp_servers/`
2. Implement server using `shared.mcp_client.server.MCPServer`
3. Add service to `docker-compose.yml`
4. Update documentation

## Safety Features

- **Paper Trading Mode**: Test strategies without real money
- **Risk Management Engine**: Hard-coded safety limits
- **Kill Switch**: Emergency stop for all trading
- **Audit Logging**: Complete transaction history
- **Multi-layer Approval**: Human oversight for all trades

## Monitoring

- Health checks for all services
- Prometheus metrics collection
- Real-time dashboard monitoring
- Configurable alerts via Telegram/Slack

## Security

- Environment-based secrets management
- JWT authentication between services
- Rate limiting on API endpoints
- Audit logging for all actions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Add your license here]

## Support

For questions and support, please [add contact information].

---

**⚠️ Important**: This system handles real financial data and trading. Always test thoroughly in paper trading mode before using with real capital. Never commit API keys or secrets to version control.
