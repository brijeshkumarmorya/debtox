# DebtOx Backend Dockerfile
FROM python:3.11-slim

LABEL maintainer="DebtOx Research Team"
LABEL description="DebtOx Backend API & ML Engine"

WORKDIR /app

# Install system dependencies: Git for repository mining and OpenMP for XGBoost/LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and pre-trained models
COPY analyzer/ /app/analyzer/
COPY backend/ /app/backend/
COPY cli/ /app/cli/
COPY data/ /app/data/
COPY ml/ /app/ml/
COPY models/ /app/models/
COPY experiments/ /app/experiments/
COPY debt-ox /app/debt-ox
RUN chmod +x /app/debt-ox

# Create storage directory
RUN mkdir -p /app/storage/analyses /app/storage/reports

ENV PYTHONPATH=/app
ENV DEBTOX_ENV=production
ENV DEBTOX_STORAGE_DIR=/app/storage

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
