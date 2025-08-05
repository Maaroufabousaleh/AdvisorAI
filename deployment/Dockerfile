###############################
# 1) ─── Python builder ───
###############################
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ git curl wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/app/wheels -r requirements.txt

###############################
# 2) ─── Runtime image ───
###############################
FROM python:3.11-slim
WORKDIR /app

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# OS runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY --from=builder /app/wheels /wheels
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard] python-multipart aiofiles psutil 
    # Install Playwright system dependencies and browsers
    # && python -m playwright install-deps \
    # && python -m playwright install chromium firefox webkit

# Create necessary directories and set permissions
RUN mkdir -p /app/data/{advisorai-data,alpaca,crypto-bubbles,finnhub,finviz,marketaux,merged}/{archive,features,temp,train} \
    && mkdir -p /app/logs \
    && mkdir -p /var/log/nginx \
    && mkdir -p /var/lib/nginx \
    && mkdir -p /run/nginx \
    && chown -R appuser:appuser /app/data \
    && chown -R appuser:appuser /app/logs \
    && chown -R appuser:appuser /var/log/nginx \
    && chown -R appuser:appuser /var/lib/nginx \
    && chown -R appuser:appuser /run/nginx

# ─── Application code ───
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p /app/data/{advisorai-data,alpaca,crypto-bubbles,finnhub,finviz,marketaux,merged}/{archive,features,temp,train} \
    && mkdir -p /var/log/nginx \
    && mkdir -p /var/lib/nginx \
    && mkdir -p /run/nginx \
    && chown -R appuser:appuser /app/data \
    && chown -R appuser:appuser /var/log/nginx \
    && chown -R appuser:appuser /var/lib/nginx \
    && chown -R appuser:appuser /run/nginx

# PYTHONPATH for FastAPI
ENV PYTHONPATH=/app:/app/src:/app/src/api:/app/src/data_cloud:/app/src/fetchers:/app/src/merge

# Nginx config
RUN rm -f /etc/nginx/conf.d/default.conf
COPY deployment/nginx.conf /etc/nginx/conf.d/app.conf

# Set resource limits for free plan
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV MAX_MEMORY_MB=512

# Supervisord config
COPY deployment/supervisord.conf /etc/supervisord.conf

COPY deployment/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set resource limits
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV MAX_MEMORY_MB=1024

ENTRYPOINT ["/entrypoint.sh"]

# Ports
EXPOSE 80 10000

CMD ["supervisord", "-c", "/etc/supervisord.conf"]