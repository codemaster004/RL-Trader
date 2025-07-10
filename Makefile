VENV_DIR := .venv
PYTHON_BASE := python3
PYTHON := .venv/bin/python


train_vae:
	$(PYTHON) -m lab.train.train_vae

download_data:
	$(PYTHON) scripts/download_stock_data.py

prep_data:
	$(PYTHON) scripts/collect_stock_img_data.py
