# ── Base image ────────────────────────────────────────────────────────────────
# Python 3.11 slim — matches runtime.txt, keeps image small.
FROM python:3.11-slim

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Copy requirements first (layer caching) ───────────────────────────────────
# Docker caches this layer. If only code changes (not requirements),
# pip install is skipped on rebuild — much faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────────────────────
COPY src/    ./src/
COPY api/    ./api/
COPY models/ ./models/

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start command ─────────────────────────────────────────────────────────────
# 0.0.0.0 makes the server accessible outside the container.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]