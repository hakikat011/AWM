MFT-1: Modular Framework for Trading
Version: 1.0
Status: Scoping & Blueprint

This document outlines the architecture, features, and operational procedures for MFT-1, a semi-autonomous, AI-driven trading system.

1. Vision & Core Concepts
MFT-1 is an intelligent trading co-pilot. It automates the entire information-gathering and analysis pipeline, from scanning the market to generating specific trade proposals. The system leverages a suite of specialized AI agents that communicate using the Model Context Protocol (MCP), ensuring modularity and scalability.

The Operator (you) retains ultimate control, providing final approval for all capital-risking actions through a dedicated oversight dashboard. The system is designed for containerized cloud deployment to run reliably "in the shadows."

2. System Architecture
The framework is composed of four decoupled layers. This design allows for independent development, testing, and scaling of each component.

Layer 1: Data & Tooling Layer (The MCP Servers)
A collection of microservices that expose data and tools via the Model Context Protocol. This is the system's "toolbelt."


MarketData_Server: Exposes time-series data from a TimescaleDB.

TechnicalAnalysis_Server: A stateless function library for calculating indicators.

News_Server: Wraps a third-party news API.

ZerodhaAPI_Server: Secure read-only wrapper for account data.

TradeLog_Server: Exposes historical trade performance from PostgreSQL.

Layer 2: Intelligence Layer (The MCP Clients)
These are the AI agents that use the tools from Layer 1 to form conclusions.


Meta-Agent: The orchestrator that manages the analysis workflow.

Analysis Agent: A powerful agent (likely LLM-based) that performs a deep-dive on a single instrument by dynamically querying the MCP servers.

Insight & Leverage Agent: Translates the Analysis Agent's recommendation into a concrete trade proposal, calculating position size and risk parameters.

Layer 3: Execution & Risk Layer
The non-negotiable gateway to the live market.


Risk Management Engine: A critical safety component with hard-coded rules (e.g., max position size, daily loss limit). It can VETO any trade, even one approved by the Operator.

Order Management System (OMS): A robust service with direct, secure credentials to the Zerodha API's execution endpoints. It handles placing, tracking, and canceling orders.

Layer 4: Control & Oversight Layer
The human interface.


Dashboard UI: A web dashboard showing real-time P&L, system status, and the crucial Proposal Queue for approving/rejecting trades.

Alerting Service: Pushes notifications to Telegram/Slack for critical events.

KILL SWITCH: A single button to liquidate all positions and halt all agent activity.

3. Tech Stack

Backend & Agents: Python 3.10+

AI/LLM Framework: LangChain or LlamaIndex (for tool use)

Databases: PostgreSQL with TimescaleDB extension

Message Queue (Optional for v1): Redis Pub/Sub or RabbitMQ

Dashboard UI: Streamlit or Plotly Dash

Containerization: Docker & Docker Compose

Deployment: AWS (ECS/Fargate) or Railway

4. Environment Setup & Rules
Proper environment management is critical for security and reproducibility.

4.1. Prerequisites

Docker & Docker Compose

Python 3.10+

Git

4.2. Configuration Management
This project uses .env files for configuration. A template file, .env.example, should be maintained in the repository.

bash

# .env.example - DO NOT COMMIT REAL KEYS HERE

# Zerodha API Credentials (for OMS)
ZERODHA_API_KEY=your_api_key
ZERODHA_API_SECRET=your_api_secret
ZERODHA_ACCESS_TOKEN= # This will be generated dynamically

# News API Key
NEWS_API_KEY=your_newsapi_key

# PostgreSQL/TimescaleDB Credentials
POSTGRES_USER=trading_user
POSTGRES_PASSWORD=a_strong_password
POSTGRES_DB=trading_db

# Alerting Service (e.g., Telegram)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
4.3. Golden Rule for Secrets
NEVER commit your .env file or any file containing real secrets to Git. Add it to your .gitignore file immediately.

code

# .gitignore

# Environment files
.env

# Python specific
__pycache__/
*.pyc

# IDE specific
.idea/
.vscode/
For production deployment on AWS or Railway, these secrets should be injected as environment variables using their built-in secrets management tools (e.g., AWS Secrets Manager, Railway Environment Variables).

5. Developer & Operator Workflow
5.1. Initial Setup

git clone <repository_url>

cd <repository_name>

cp .env.example .env

Fill in the _API_KEY and other credentials in your local .env file. Do not fill in ZERODHA_ACCESS_TOKEN yet.

5.2. Running the System Locally

Build the containers: docker-compose build

Start all services: docker-compose up -d (The -d runs it in detached mode).

Generate Zerodha Access Token: The Kite Connect API requires a one-time login each day to generate an access token. You will need a simple script to perform this login flow. Once you obtain the daily access_token, paste it into your .env file and restart the relevant containers: docker-compose restart oms-service zerodha-api-server.

Access the Dashboard: Open your web browser to http://localhost:8501 (or the port you configured for Streamlit/Dash).

Monitor: Watch the logs using docker-compose logs -f.

5.3. Daily Operation

Start the services in the morning.

Generate and update the daily access token.

Monitor the dashboard for trade proposals.

Approve or reject proposals based on your judgment.

At the end of the day, shut down the system with docker-compose down.

6. Testing Protocol
Thorough testing is non-negotiable. Follow this multi-stage protocol before risking any capital.

6.1. Unit Tests

What: Test individual, pure functions in isolation.

Examples:
test_calculate_rsi(): Does the RSI function return the correct value for a known data series?
test_position_sizer(): Does the leverage agent correctly calculate position size based on inputs?

How: Use pytest. Run with pytest <service_name>. These should be fast and run frequently during development.

6.2. Integration Tests

What: Test the interaction between two or more services.

Examples:
Can the Analysis Agent successfully connect to and query the MarketData_Server via MCP?
Does a trade approved on the Dashboard correctly reach the Risk Management Engine?

How: These are more complex and often run in a dedicated test environment spun up via docker-compose.

6.3. End-to-End (E2E) Simulation (Backtesting)

What: The most critical test. This involves replaying historical data through the entire system from data ingestion to "paper" trade execution, to evaluate a strategy's performance over months or years.

How:
Create a dedicated "Backtest Mode" for the system.
When in this mode, the MarketData_Server serves historical data from the database instead of live data.
The OMS logs trades to the TradeLog_Server instead of sending them to the live Zerodha API.
Run the simulation and generate a performance report (e.g., P&L, Sharpe Ratio, Max Drawdown). No new strategy should go live without successful E2E simulation results.

7. Running Precautions & Safety Checklist
Financial markets are unforgiving. Treat this system with extreme caution.

PRE-FLIGHT CHECKLIST (Run this Every Time Before Going Live):

 Start with Paper Trading: Run the system for at least one week in a simulation mode where trades are logged but not executed. Verify that the system's decisions align with your expectations.

 Check Broker API Status: Visit the Zerodha developer dashboard. Is their API operational?

 Confirm Account Margin: Log into your Zerodha account manually. Is your available margin what you expect it to be?

 Verify Risk Engine Rules: Check the configuration for the Risk Management Engine. Are the MAX_POSITION_SIZE, MAX_DAILY_LOSS, and other safety limits set to conservative levels you are comfortable with?

 Know Your Kill Switch: Be 100% certain you know how the Kill Switch works and can activate it within seconds.

 Start Small: When first deploying real capital, do so with an amount you are fully prepared to lose. Operate with 5-10% of your planned capital for the first month.

 Monitor Actively: During the first weeks of live trading, do not let the system run unattended. Be present to monitor its behavior.

 Check for "Stuck" Orders: Manually check your Zerodha dashboard periodically to ensure that orders the system thinks are OPEN or CANCELLED match reality.
