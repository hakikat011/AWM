# AWM Dashboard

The AWM Dashboard is a comprehensive web-based interface for monitoring and controlling the Autonomous Wealth Management system. Built with Streamlit, it provides real-time insights into portfolio performance, risk management, AI agents, and system operations.

## Features

### 🏠 Main Dashboard
- **Real-time Portfolio Overview**: Current value, P&L, cash position, and return metrics
- **Risk Monitoring**: Live risk level assessment with violation alerts
- **Position Tracking**: Current holdings with real-time market values
- **Active Orders**: Monitor pending and executing orders
- **Quick Trading**: Place orders directly from the dashboard
- **Emergency Controls**: System-wide emergency stop functionality

### 📊 Analytics Page
- **Performance Metrics**: Sharpe ratio, volatility, max drawdown analysis
- **Risk-Adjusted Returns**: Comprehensive performance attribution
- **Concentration Analysis**: Portfolio diversification metrics
- **Top/Worst Performers**: Position-level performance ranking
- **Sector Allocation**: Industry exposure analysis
- **Historical Charts**: Portfolio value and drawdown over time

### 🤖 AI Agents Page
- **Agent Status Monitoring**: Real-time health and performance metrics
- **Task Management**: View agent task history and trigger new tasks
- **Performance Analytics**: Success rates and processing times
- **Agent Controls**: Start, pause, and restart individual agents

### ⚠️ Risk Management Page
- **Risk Limits Configuration**: Update position, loss, and leverage limits
- **Violation Monitoring**: Real-time risk breach alerts
- **Risk Metrics Tracking**: VaR, volatility, and drawdown trends
- **Emergency Controls**: System-wide trading halt capabilities
- **Risk Distribution**: Position-level risk contribution analysis

### 📋 Orders Page
- **Active Orders**: Monitor all pending and executing orders
- **Order History**: Complete order lifecycle tracking
- **Order Analytics**: Performance metrics and success rates
- **Quick Order Placement**: Streamlined order entry
- **Order Reconciliation**: Sync with broker systems

## Architecture

### Frontend Components
```
services/dashboard/
├── main.py                 # Main dashboard page
├── pages/
│   ├── 1_📊_Analytics.py   # Analytics and performance
│   ├── 2_🤖_AI_Agents.py   # AI agent monitoring
│   ├── 3_⚠️_Risk_Management.py # Risk controls
│   └── 4_📋_Orders.py      # Order management
├── Dockerfile              # Container configuration
└── README.md              # This file
```

### Backend Integration
The dashboard communicates with backend services through MCP (Model Context Protocol):

- **Portfolio Management System** (Port 8012): Portfolio data and positions
- **Risk Management Engine** (Port 8010): Risk metrics and controls
- **Order Management System** (Port 8011): Order lifecycle management
- **Market Data Server** (Port 8001): Real-time market data
- **Alerting Service** (Port 8013): System notifications

### Data Flow
```
Dashboard → MCP Client → Backend Services → Database/External APIs
    ↑                                              ↓
    └── Real-time Updates ← WebSocket/Polling ←────┘
```

## Configuration

### Environment Variables
```bash
# Database Configuration
POSTGRES_HOST=timescaledb
POSTGRES_PORT=5432
POSTGRES_USER=trading_user
POSTGRES_PASSWORD=trading_password
POSTGRES_DB=trading_db

# Redis Configuration
REDIS_URL=redis://redis:6379

# Service URLs
PORTFOLIO_MANAGEMENT_SYSTEM_URL=http://portfolio-management-system:8012
RISK_MANAGEMENT_ENGINE_URL=http://risk-management-engine:8010
OMS_URL=http://order-management-system:8011
MARKET_DATA_SERVER_URL=http://market-data-server:8001
ALERTING_SERVICE_URL=http://alerting-service:8013
```

### Streamlit Configuration
The dashboard runs on port 8501 with the following settings:
- **Auto-refresh**: 30-second intervals for real-time updates
- **Wide layout**: Optimized for large screens
- **Multi-page**: Organized by functional areas
- **Responsive design**: Works on desktop and tablet devices

## Usage

### Starting the Dashboard
```bash
# Using Docker Compose (recommended)
docker-compose up dashboard

# Direct execution
streamlit run services/dashboard/main.py --server.port=8501 --server.address=0.0.0.0
```

### Accessing the Interface
- **URL**: http://localhost:8501
- **Main Dashboard**: Portfolio overview and controls
- **Analytics**: http://localhost:8501/📊_Analytics
- **AI Agents**: http://localhost:8501/🤖_AI_Agents
- **Risk Management**: http://localhost:8501/⚠️_Risk_Management
- **Orders**: http://localhost:8501/📋_Orders

### Key Operations

#### Portfolio Monitoring
1. Select portfolio from sidebar
2. View real-time metrics and positions
3. Monitor risk levels and violations
4. Track performance attribution

#### Order Management
1. Navigate to Orders page
2. View active orders and history
3. Place new orders using quick form
4. Cancel or modify existing orders

#### Risk Control
1. Access Risk Management page
2. Monitor current risk metrics
3. Update risk limits as needed
4. Use emergency stop if required

#### AI Agent Management
1. Go to AI Agents page
2. Monitor agent health and performance
3. Trigger specific agent tasks
4. Review task history and success rates

## Security Features

### Access Control
- **Environment-based configuration**: Sensitive data in environment variables
- **Service isolation**: Each service runs in separate containers
- **Network security**: Internal communication through Docker networks

### Risk Management
- **Emergency stop**: Immediate trading halt capability
- **Risk limits**: Configurable position and loss limits
- **Real-time monitoring**: Continuous risk assessment
- **Audit logging**: Complete activity tracking

### Data Protection
- **Encrypted communication**: HTTPS/TLS for external connections
- **Database security**: Encrypted storage and secure connections
- **API authentication**: Token-based service authentication

## Monitoring and Alerts

### Health Checks
- **Service health**: Real-time status of all components
- **Database connectivity**: Connection monitoring
- **API responsiveness**: Performance tracking

### Alert Integration
- **Risk violations**: Immediate notifications
- **System errors**: Component failure alerts
- **Performance issues**: Degradation warnings

### Metrics Dashboard
- **System performance**: Response times and throughput
- **Trading metrics**: Order success rates and execution times
- **Risk metrics**: Portfolio risk and exposure tracking

## Troubleshooting

### Common Issues

#### Dashboard Not Loading
```bash
# Check service status
docker-compose ps dashboard

# View logs
docker-compose logs dashboard

# Restart service
docker-compose restart dashboard
```

#### Backend Connection Errors
```bash
# Verify service URLs in environment
echo $PORTFOLIO_MANAGEMENT_SYSTEM_URL

# Check network connectivity
docker-compose exec dashboard curl -f http://portfolio-management-system:8012/health
```

#### Data Not Updating
```bash
# Check database connection
docker-compose logs timescaledb

# Verify Redis connectivity
docker-compose exec dashboard redis-cli -h redis ping
```

### Performance Optimization
- **Caching**: Enable Streamlit caching for expensive operations
- **Pagination**: Limit data queries for large datasets
- **Async operations**: Use async/await for backend calls
- **Connection pooling**: Reuse database connections

## Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export POSTGRES_HOST=localhost
export PORTFOLIO_MANAGEMENT_SYSTEM_URL=http://localhost:8012

# Run dashboard
streamlit run services/dashboard/main.py
```

### Adding New Pages
1. Create new file in `pages/` directory
2. Follow naming convention: `N_📊_PageName.py`
3. Import required dependencies
4. Implement page functionality
5. Test integration with backend services

### Customization
- **Styling**: Modify CSS in main.py
- **Layouts**: Adjust column configurations
- **Charts**: Customize Plotly visualizations
- **Metrics**: Add new KPIs and calculations

## API Reference

### Dashboard API Client
```python
from services.dashboard.main import DashboardAPI

api = DashboardAPI()

# Get portfolio data
portfolio = await api.get_portfolio_data("portfolio_id")

# Get positions
positions = await api.get_positions("portfolio_id")

# Get risk status
risk = await api.get_risk_status("portfolio_id")

# Get active orders
orders = await api.get_active_orders("portfolio_id")
```

### MCP Integration
The dashboard uses the shared MCP client for backend communication:
```python
from shared.mcp_client.base import MCPClient

async with MCPClient("dashboard") as client:
    response = await client.send_request(
        "http://service:port",
        "method_name",
        {"param": "value"}
    )
```

## Support

For issues, questions, or feature requests:
1. Check the troubleshooting section
2. Review service logs
3. Verify configuration
4. Test backend connectivity
5. Contact system administrators

The dashboard is designed to be the primary interface for AWM system operations, providing comprehensive visibility and control over all aspects of autonomous wealth management.
