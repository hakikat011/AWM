-- AWM Initial Data Setup
-- Insert initial configuration and sample data

-- Insert system configuration
INSERT INTO system_config (key, value, description) VALUES
('system_version', '1.0.0', 'Current system version'),
('trading_enabled', 'false', 'Enable/disable trading functionality'),
('paper_trading_mode', 'true', 'Enable paper trading mode'),
('max_position_size', '100000', 'Maximum position size in INR'),
('max_daily_loss', '10000', 'Maximum daily loss limit in INR'),
('max_portfolio_risk', '0.02', 'Maximum portfolio risk (2%)'),
('default_stop_loss', '0.05', 'Default stop loss percentage (5%)'),
('default_take_profit', '0.10', 'Default take profit percentage (10%)'),
('trading_start_time', '09:15', 'Market opening time'),
('trading_end_time', '15:30', 'Market closing time'),
('risk_check_interval', '60', 'Risk check interval in seconds'),
('max_orders_per_day', '50', 'Maximum orders per day'),
('min_order_value', '1000', 'Minimum order value in INR'),
('alert_email_enabled', 'true', 'Enable email alerts'),
('alert_telegram_enabled', 'true', 'Enable Telegram alerts'),
('backup_enabled', 'true', 'Enable automatic backups'),
('audit_logging_enabled', 'true', 'Enable audit logging');

-- Insert sample instruments (NSE stocks)
INSERT INTO instruments (symbol, name, instrument_type, exchange, segment, lot_size, tick_size) VALUES
('RELIANCE', 'Reliance Industries Limited', 'EQUITY', 'NSE', 'EQ', 1, 0.05),
('TCS', 'Tata Consultancy Services Limited', 'EQUITY', 'NSE', 'EQ', 1, 0.05),
('HDFCBANK', 'HDFC Bank Limited', 'EQUITY', 'NSE', 'EQ', 1, 0.05),
('INFY', 'Infosys Limited', 'EQUITY', 'NSE', 'EQ', 1, 0.05),
('HINDUNILVR', 'Hindustan Unilever Limited', 'EQUITY', 'NSE', 'EQ', 1, 0.05),
('ICICIBANK', 'ICICI Bank Limited', 'EQUITY', 'NSE', 'EQ', 1, 0.05),
('KOTAKBANK', 'Kotak Mahindra Bank Limited', 'EQUITY', 'NSE', 'EQ', 1, 0.05),
('BHARTIARTL', 'Bharti Airtel Limited', 'EQUITY', 'NSE', 'EQ', 1, 0.05),
('ITC', 'ITC Limited', 'EQUITY', 'NSE', 'EQ', 1, 0.05),
('SBIN', 'State Bank of India', 'EQUITY', 'NSE', 'EQ', 1, 0.05),
('LT', 'Larsen & Toubro Limited', 'EQUITY', 'NSE', 'EQ', 1, 0.05),
('ASIANPAINT', 'Asian Paints Limited', 'EQUITY', 'NSE', 'EQ', 1, 0.05),
('MARUTI', 'Maruti Suzuki India Limited', 'EQUITY', 'NSE', 'EQ', 1, 0.05),
('BAJFINANCE', 'Bajaj Finance Limited', 'EQUITY', 'NSE', 'EQ', 1, 0.05),
('HCLTECH', 'HCL Technologies Limited', 'EQUITY', 'NSE', 'EQ', 1, 0.05),
('WIPRO', 'Wipro Limited', 'EQUITY', 'NSE', 'EQ', 1, 0.05),
('ULTRACEMCO', 'UltraTech Cement Limited', 'EQUITY', 'NSE', 'EQ', 1, 0.05),
('TITAN', 'Titan Company Limited', 'EQUITY', 'NSE', 'EQ', 1, 0.05),
('NESTLEIND', 'Nestle India Limited', 'EQUITY', 'NSE', 'EQ', 1, 0.05),
('POWERGRID', 'Power Grid Corporation of India Limited', 'EQUITY', 'NSE', 'EQ', 1, 0.05);

-- Insert sample Nifty 50 futures
INSERT INTO instruments (symbol, name, instrument_type, exchange, segment, lot_size, tick_size) VALUES
('NIFTY24DECFUT', 'Nifty 50 December 2024 Future', 'FUTURES', 'NSE', 'NFO', 50, 0.05),
('BANKNIFTY24DECFUT', 'Bank Nifty December 2024 Future', 'FUTURES', 'NSE', 'NFO', 25, 0.05);

-- Create a default portfolio
INSERT INTO portfolios (name, description, initial_capital, current_value, available_cash) VALUES
('Default Portfolio', 'Main trading portfolio for AWM system', 1000000.00, 1000000.00, 1000000.00);

-- Insert sample market data for the last few days (using RELIANCE as example)
DO $$
DECLARE
    reliance_id UUID;
    current_date TIMESTAMP WITH TIME ZONE;
    i INTEGER;
    base_price DECIMAL(15,4) := 2500.00;
    price_variation DECIMAL(15,4);
BEGIN
    -- Get RELIANCE instrument ID
    SELECT id INTO reliance_id FROM instruments WHERE symbol = 'RELIANCE';
    
    -- Insert market data for the last 30 days
    FOR i IN 0..29 LOOP
        current_date := NOW() - INTERVAL '1 day' * i;
        
        -- Skip weekends (Saturday = 6, Sunday = 0)
        IF EXTRACT(DOW FROM current_date) NOT IN (0, 6) THEN
            -- Generate some realistic price variations
            price_variation := (RANDOM() - 0.5) * 100; -- +/- 50 INR variation
            
            INSERT INTO market_data (time, instrument_id, open_price, high_price, low_price, close_price, volume)
            VALUES (
                current_date,
                reliance_id,
                base_price + price_variation,
                base_price + price_variation + (RANDOM() * 50),
                base_price + price_variation - (RANDOM() * 50),
                base_price + price_variation + (RANDOM() - 0.5) * 30,
                FLOOR(RANDOM() * 1000000 + 100000)::BIGINT
            );
        END IF;
    END LOOP;
END $$;

-- Insert sample analysis results
DO $$
DECLARE
    reliance_id UUID;
    tcs_id UUID;
BEGIN
    SELECT id INTO reliance_id FROM instruments WHERE symbol = 'RELIANCE';
    SELECT id INTO tcs_id FROM instruments WHERE symbol = 'TCS';
    
    INSERT INTO analysis_results (instrument_id, analysis_type, signal, confidence_score, target_price, stop_loss, analysis_data, created_by)
    VALUES 
    (reliance_id, 'TECHNICAL_ANALYSIS', 'BUY', 0.75, 2600.00, 2400.00, 
     '{"rsi": 45.2, "macd": "bullish", "support": 2450, "resistance": 2580}', 'market_analysis_agent'),
    (tcs_id, 'FUNDAMENTAL_ANALYSIS', 'HOLD', 0.65, 3800.00, 3500.00,
     '{"pe_ratio": 28.5, "revenue_growth": 0.12, "profit_margin": 0.24}', 'market_analysis_agent');
END $$;

-- Insert sample system alerts
INSERT INTO system_alerts (alert_type, severity, title, message, data) VALUES
('SYSTEM_STARTUP', 'INFO', 'System Started', 'AWM system has been successfully started', '{"timestamp": "2024-01-01T09:00:00Z"}'),
('RISK_WARNING', 'WARNING', 'Portfolio Risk Elevated', 'Portfolio risk has exceeded 1.5% threshold', '{"current_risk": 0.018, "threshold": 0.015}'),
('TRADE_EXECUTED', 'INFO', 'Trade Executed Successfully', 'Buy order for RELIANCE executed at 2550.00', '{"symbol": "RELIANCE", "quantity": 100, "price": 2550.00}');

-- Create data retention policies for TimescaleDB
-- Keep detailed market data for 2 years, then compress
SELECT add_retention_policy('market_data', INTERVAL '2 years');

-- Keep risk metrics for 1 year
SELECT add_retention_policy('risk_metrics', INTERVAL '1 year');

-- Keep audit logs for 5 years (compliance requirement)
SELECT add_retention_policy('audit_log', INTERVAL '5 years');

-- Create compression policies to save storage
SELECT add_compression_policy('market_data', INTERVAL '7 days');
SELECT add_compression_policy('risk_metrics', INTERVAL '30 days');
SELECT add_compression_policy('audit_log', INTERVAL '90 days');
