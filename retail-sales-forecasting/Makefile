.PHONY: help install data train test lint format app docker clean

# ── Variables ─────────────────────────────────────────────────────────────────
PYTHON = python
PIP    = pip

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Retail Sales Forecasting – Makefile"
	@echo "  ─────────────────────────────────────────────────"
	@echo "  make install   Install all Python dependencies"
	@echo "  make data      Generate synthetic raw dataset"
	@echo "  make train     Train all forecasting models"
	@echo "  make test      Run pytest unit tests"
	@echo "  make lint      Lint with Ruff"
	@echo "  make format    Auto-format with Black + isort"
	@echo "  make app       Launch Streamlit dashboard"
	@echo "  make docker    Build and run with Docker Compose"
	@echo "  make clean     Remove generated files"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt

# ── Data ──────────────────────────────────────────────────────────────────────
data:
	$(PYTHON) -c "from src.data_loader import generate_synthetic_data; generate_synthetic_data()"
	@echo "✅ Synthetic data generated at data/raw/retail_sales_raw.csv"

# ── Train ─────────────────────────────────────────────────────────────────────
train:
	$(PYTHON) -m src.models.trainer
	@echo "✅ All models trained and saved to models/"

# ── Test ──────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# ── Code Quality ──────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/

format:
	black src/ tests/ streamlit_app/
	isort src/ tests/ streamlit_app/

# ── App ───────────────────────────────────────────────────────────────────────
app:
	streamlit run streamlit_app/Home.py

# ── Docker ────────────────────────────────────────────────────────────────────
docker:
	docker-compose up --build

docker-stop:
	docker-compose down

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
	@echo "✅ Cleaned up generated files."
