# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/python3.11 /usr/bin/python3

# Set working directory
WORKDIR /app

# Upgrade pip and install wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY services/mcp_servers/llm_market_intelligence/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional GPU-optimized packages
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy the entire project
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Create directory for model cache
RUN mkdir -p /app/models && chmod 777 /app/models
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models

# Expose port
EXPOSE 8007

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8007/health || exit 1

# Run the server
CMD ["python", "services/mcp_servers/llm_market_intelligence/server.py"]
