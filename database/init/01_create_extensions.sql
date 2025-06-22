-- AWM Database Initialization Script
-- Create necessary extensions for TimescaleDB and other features

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable cryptographic functions
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Enable additional data types
CREATE EXTENSION IF NOT EXISTS hstore;

-- Enable full-text search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create custom types
CREATE TYPE order_status AS ENUM (
    'PENDING',
    'OPEN',
    'COMPLETE',
    'CANCELLED',
    'REJECTED',
    'EXPIRED'
);

CREATE TYPE order_type AS ENUM (
    'MARKET',
    'LIMIT',
    'STOP_LOSS',
    'STOP_LOSS_MARKET',
    'BRACKET'
);

CREATE TYPE order_side AS ENUM (
    'BUY',
    'SELL'
);

CREATE TYPE instrument_type AS ENUM (
    'EQUITY',
    'FUTURES',
    'OPTIONS',
    'CURRENCY',
    'COMMODITY'
);

CREATE TYPE risk_level AS ENUM (
    'LOW',
    'MEDIUM',
    'HIGH',
    'CRITICAL'
);

CREATE TYPE alert_severity AS ENUM (
    'INFO',
    'WARNING',
    'ERROR',
    'CRITICAL'
);

CREATE TYPE trade_signal AS ENUM (
    'BUY',
    'SELL',
    'HOLD'
);

-- Create audit trigger function
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        NEW.created_at = COALESCE(NEW.created_at, NOW());
        NEW.updated_at = NOW();
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        NEW.updated_at = NOW();
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;
