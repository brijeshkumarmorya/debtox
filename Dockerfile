# Unified DebtOx All-in-One Container (Frontend + Backend + ML Models)
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Python Backend & Engine Stage
FROM python:3.11-slim

LABEL maintainer="DebtOx Research Team"
LABEL description="DebtOx — Code Smell Prediction and Technical Debt Estimation Platform"

WORKDIR /app

# Install Git and OpenMP (required for LightGBM / XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source tree
COPY analyzer/ /app/analyzer/
COPY backend/ /app/backend/
COPY cli/ /app/cli/
COPY data/ /app/data/
COPY ml/ /app/ml/
COPY models/ /app/models/
COPY experiments/ /app/experiments/
COPY debt-ox /app/debt-ox
RUN chmod +x /app/debt-ox

# Copy built frontend assets from builder stage
COPY --from=frontend-builder /frontend/dist /app/frontend/dist

# Initialize persistence dirs
RUN mkdir -p /app/storage/analyses /app/storage/reports

ENV PYTHONPATH=/app
ENV DEBTOX_ENV=production
ENV DEBTOX_STORAGE_DIR=/app/storage

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
