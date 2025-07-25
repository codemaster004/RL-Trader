import hydra
import mlflow
from omegaconf import DictConfig

import importlib
import logging as log
import os


mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))


@hydra.main(config_path="../../config/", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
	# mlflow.set_experiment("")
	
	# # Dynamically loading, based on config.
	# env_path, env_name = cfg.env.type, cfg.env.name
	# env_cls = getattr(importlib.import_module(env_path), env_name)  # dynamic load a given class from a library
	# env = env_cls(**cfg.env.params)  # create an instance, with params from config
	
	# Dynamic load of Agent class
	agent = getattr(importlib.import_module(cfg.agent.type), cfg.agent.name)(**cfg.agent.params)
	with mlflow.start_run(run_name=cfg.train.run_name):
		# ML Flow log training params
		mlflow.log_params(cfg.train.params)

		agent.train(env_id=cfg.env.register_id, env_options=cfg.env.options, **cfg.train.params, **cfg.agent.train_params)
		agent.save(path='saves/')
	

if __name__ == "__main__":
	main()
