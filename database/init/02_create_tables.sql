-- AWM Database Schema
-- Create all tables for the Autonomous Wealth Management system

-- Instruments table
CREATE TABLE instruments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(50) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    instrument_type instrument_type NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    segment VARCHAR(50),
    lot_size INTEGER DEFAULT 1,
    tick_size DECIMAL(10, 4) DEFAULT 0.01,
    instrument_token BIGINT,  -- Zerodha instrument token
    expiry DATE,  -- For F&O instruments
    strike DECIMAL(15, 4),  -- For options
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Market data table (TimescaleDB hypertable)
CREATE TABLE market_data (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    instrument_id UUID NOT NULL REFERENCES instruments(id),
    open_price DECIMAL(15, 4) NOT NULL,
    high_price DECIMAL(15, 4) NOT NULL,
    low_price DECIMAL(15, 4) NOT NULL,
    close_price DECIMAL(15, 4) NOT NULL,
    volume BIGINT NOT NULL DEFAULT 0,
    turnover DECIMAL(20, 4) DEFAULT 0,
    open_interest BIGINT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Convert market_data to hypertable
SELECT create_hypertable('market_data', 'time');

-- Create indexes for market_data
CREATE INDEX idx_market_data_instrument_time ON market_data (instrument_id, time DESC);
CREATE INDEX idx_market_data_time ON market_data (time DESC);

-- Portfolios table
CREATE TABLE portfolios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    initial_capital DECIMAL(20, 4) NOT NULL,
    current_value DECIMAL(20, 4) NOT NULL DEFAULT 0,
    available_cash DECIMAL(20, 4) NOT NULL DEFAULT 0,
    total_pnl DECIMAL(20, 4) NOT NULL DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Portfolio positions table
CREATE TABLE portfolio_positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id),
    instrument_id UUID NOT NULL REFERENCES instruments(id),
    quantity INTEGER NOT NULL DEFAULT 0,
    average_price DECIMAL(15, 4) NOT NULL DEFAULT 0,
    current_price DECIMAL(15, 4) NOT NULL DEFAULT 0,
    market_value DECIMAL(20, 4) NOT NULL DEFAULT 0,
    unrealized_pnl DECIMAL(20, 4) NOT NULL DEFAULT 0,
    realized_pnl DECIMAL(20, 4) NOT NULL DEFAULT 0,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(portfolio_id, instrument_id)
);

-- Orders table
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id),
    instrument_id UUID NOT NULL REFERENCES instruments(id),
    order_id VARCHAR(100) UNIQUE, -- Broker order ID
    parent_order_id UUID REFERENCES orders(id), -- For bracket orders
    order_type order_type NOT NULL,
    order_side order_side NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(15, 4),
    trigger_price DECIMAL(15, 4),
    filled_quantity INTEGER DEFAULT 0,
    average_price DECIMAL(15, 4) DEFAULT 0,
    status order_status NOT NULL DEFAULT 'PENDING',
    status_message TEXT,
    placed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trades table (filled orders)
CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID NOT NULL REFERENCES orders(id),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id),
    instrument_id UUID NOT NULL REFERENCES instruments(id),
    trade_id VARCHAR(100), -- Broker trade ID
    side order_side NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(15, 4) NOT NULL,
    value DECIMAL(20, 4) NOT NULL,
    commission DECIMAL(10, 4) DEFAULT 0,
    taxes DECIMAL(10, 4) DEFAULT 0,
    executed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Risk metrics table (TimescaleDB hypertable)
CREATE TABLE risk_metrics (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    portfolio_id UUID NOT NULL REFERENCES portfolios(id),
    var_1d DECIMAL(15, 4), -- Value at Risk (1 day)
    var_5d DECIMAL(15, 4), -- Value at Risk (5 days)
    expected_shortfall DECIMAL(15, 4),
    beta DECIMAL(8, 4),
    sharpe_ratio DECIMAL(8, 4),
    max_drawdown DECIMAL(8, 4),
    volatility DECIMAL(8, 4),
    correlation_to_market DECIMAL(8, 4),
    risk_level risk_level,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Convert risk_metrics to hypertable
SELECT create_hypertable('risk_metrics', 'time');

-- Analysis results table
CREATE TABLE analysis_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    instrument_id UUID NOT NULL REFERENCES instruments(id),
    analysis_type VARCHAR(100) NOT NULL,
    signal trade_signal,
    confidence_score DECIMAL(5, 4), -- 0.0 to 1.0
    target_price DECIMAL(15, 4),
    stop_loss DECIMAL(15, 4),
    analysis_data JSONB,
    created_by VARCHAR(100), -- Agent name
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Trade proposals table
CREATE TABLE trade_proposals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id),
    instrument_id UUID NOT NULL REFERENCES instruments(id),
    analysis_result_id UUID REFERENCES analysis_results(id),
    proposal_type order_type NOT NULL,
    side order_side NOT NULL,
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(15, 4),
    stop_loss DECIMAL(15, 4),
    take_profit DECIMAL(15, 4),
    risk_amount DECIMAL(20, 4),
    expected_return DECIMAL(20, 4),
    risk_reward_ratio DECIMAL(8, 4),
    status VARCHAR(50) DEFAULT 'PENDING', -- PENDING, APPROVED, REJECTED, EXECUTED
    approved_by VARCHAR(100),
    approved_at TIMESTAMP WITH TIME ZONE,
    executed_at TIMESTAMP WITH TIME ZONE,
    rejection_reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System alerts table
CREATE TABLE system_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_type VARCHAR(100) NOT NULL,
    severity alert_severity NOT NULL,
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    data JSONB,
    is_acknowledged BOOLEAN DEFAULT false,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System configuration table
CREATE TABLE system_config (
    key VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    updated_by VARCHAR(100),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Audit log table (TimescaleDB hypertable)
CREATE TABLE audit_log (
    time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    user_id VARCHAR(100),
    action VARCHAR(100) NOT NULL,
    table_name VARCHAR(100),
    record_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT
);

-- Convert audit_log to hypertable
SELECT create_hypertable('audit_log', 'time');

-- Agent tasks table
CREATE TABLE agent_tasks (
    id UUID PRIMARY KEY,
    agent_name VARCHAR(100) NOT NULL,
    task_type VARCHAR(100) NOT NULL,
    parameters JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'PENDING',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    result JSONB,
    error_message TEXT
);

-- Risk evaluations table
CREATE TABLE risk_evaluations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id),
    trade_proposal JSONB NOT NULL,
    evaluation_result JSONB NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_portfolio_positions_portfolio ON portfolio_positions(portfolio_id);
CREATE INDEX idx_orders_portfolio ON orders(portfolio_id);
CREATE INDEX idx_orders_instrument ON orders(instrument_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_trades_portfolio ON trades(portfolio_id);
CREATE INDEX idx_trades_instrument ON trades(instrument_id);
CREATE INDEX idx_trades_executed_at ON trades(executed_at);
CREATE INDEX idx_risk_metrics_portfolio_time ON risk_metrics(portfolio_id, time DESC);
CREATE INDEX idx_analysis_results_instrument ON analysis_results(instrument_id);
CREATE INDEX idx_analysis_results_created_at ON analysis_results(created_at DESC);
CREATE INDEX idx_trade_proposals_portfolio ON trade_proposals(portfolio_id);
CREATE INDEX idx_trade_proposals_status ON trade_proposals(status);
CREATE INDEX idx_system_alerts_severity ON system_alerts(severity);
CREATE INDEX idx_system_alerts_created_at ON system_alerts(created_at DESC);
CREATE INDEX idx_audit_log_action ON audit_log(action);
CREATE INDEX idx_audit_log_table_name ON audit_log(table_name);
CREATE INDEX idx_agent_tasks_agent_name ON agent_tasks(agent_name);
CREATE INDEX idx_agent_tasks_status ON agent_tasks(status);
CREATE INDEX idx_agent_tasks_created_at ON agent_tasks(created_at DESC);
CREATE INDEX idx_risk_evaluations_portfolio ON risk_evaluations(portfolio_id);
CREATE INDEX idx_risk_evaluations_timestamp ON risk_evaluations(timestamp DESC);

-- Add audit triggers to key tables
CREATE TRIGGER audit_portfolios BEFORE INSERT OR UPDATE ON portfolios
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_portfolio_positions BEFORE INSERT OR UPDATE ON portfolio_positions
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_orders BEFORE INSERT OR UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_trade_proposals BEFORE INSERT OR UPDATE ON trade_proposals
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
