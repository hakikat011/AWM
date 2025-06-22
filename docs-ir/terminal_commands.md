# MFT-1 Terminal Commands Reference

## System Management

### Starting the System
```bash
# Build all containers
docker-compose build

# Start all services in detached mode
docker-compose up -d

# Start specific service
docker-compose up -d <service-name>
```

### Monitoring
```bash
# View logs from all services
docker-compose logs -f

# View logs from specific service
docker-compose logs -f <service-name>

# Check system status
docker-compose ps
```

### Maintenance
```bash
# Generate new Zerodha access token
python scripts/generate_zerodha_token.py

# Restart services that use Zerodha API
docker-compose restart oms-service zerodha-api-server

# Backup database
docker-compose exec postgres pg_dump -U trading_user trading_db > backup_$(date +%Y%m%d).sql
```

## Emergency Commands

### Kill Switch
```bash
# Activate kill switch (liquidate all positions and halt system)
curl -X POST http://localhost:8080/api/kill-switch/activate

# Or use the Dashboard UI emergency button
```

### Manual Database Operations
```bash
# Connect to database CLI
docker-compose exec postgres psql -U trading_user -d trading_db

# Reset trade log (use with caution!)
docker-compose exec postgres psql -U trading_user -d trading_db -c "TRUNCATE TABLE trade_log;"
```