FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Enable bytecode compilation and set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

# Copy dependency definition files
COPY pyproject.toml uv.lock ./

# Install project dependencies into system environment without installing the project in editable mode
RUN uv sync --frozen --no-install-project --no-dev
ENV PATH="/app/.venv/bin:$PATH"

# Copy project source code
COPY . .

# Expose Streamlit and MLflow ports
EXPOSE 8501 5000
