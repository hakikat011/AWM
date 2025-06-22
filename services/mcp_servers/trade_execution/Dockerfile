FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules
COPY shared/ ./shared/

# Copy trade execution server
COPY services/mcp_servers/trade_execution/ ./services/mcp_servers/trade_execution/

# Set environment variables
ENV PYTHONPATH=/app
ENV TRADE_EXECUTION_SERVER_HOST=0.0.0.0
ENV TRADE_EXECUTION_SERVER_PORT=8005

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import asyncio; import aiohttp; asyncio.run(aiohttp.ClientSession().get('http://localhost:8005/health').close())" || exit 1

# Run the server
CMD ["python", "services/mcp_servers/trade_execution/server.py"]
