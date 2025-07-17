VENV_DIR := .venv
PYTHON_BASE := python3
PYTHON := .venv/bin/python


run_mlflow:
	source $(VENV_DIR)/bin/activate && mlflow ui

eval_agent:
	$(PYTHON) -m lab.experiments.eval_agent

train_agent:
	$(PYTHON) -m lab.experiments.train_agent

train_vae:
	$(PYTHON) -m lab.train.train_vae

run_download_data:
	$(PYTHON) scripts/download_stock_data.py

run_prep_data:
	$(PYTHON) scripts/collect_stock_img_data.py
