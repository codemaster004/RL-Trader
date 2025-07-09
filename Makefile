VENV_DIR := .venv
PYTHON_BASE := python3
PYTHON := .venv/bin/python


download_data:
	$(PYTHON) scripts/download_stock_data.py
