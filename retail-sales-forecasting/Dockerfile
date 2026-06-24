FROM python:3.11-slim

LABEL maintainer="abhiramsoni1@gmail.com"
LABEL description="Retail Sales Forecasting – Streamlit Dashboard"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project ──────────────────────────────────────────────────────────────
COPY . .

# ── Expose Streamlit port ─────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "streamlit_app/Home.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
