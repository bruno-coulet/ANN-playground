# ==============================================================================
# Multi-stage Dockerfile for ANN Playground
# Optimized with BuildKit cache mounts and explicit wheels
# ==============================================================================

# Stage 1: Build virtual environment
FROM python:3.12-slim AS builder

# Install uv from official binary image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Leverage BuildKit cache for uv download cache
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Stage 2: Runtime image
FROM python:3.12-slim AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"\
    PYTHONPATH="/app/src"



# Copy pre-built virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy project definition and source code
COPY pyproject.toml ./
COPY src/ ./src/

EXPOSE 8501 5000
