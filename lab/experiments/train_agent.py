import hydra
import mlflow
from omegaconf import DictConfig

import importlib
import logging as log


@hydra.main(config_path="../../config/", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
	# # Dynamically loading, based on config.
	# env_path, env_name = cfg.env.type, cfg.env.name
	# env_cls = getattr(importlib.import_module(env_path), env_name)  # dynamic load a given class from a library
	# env = env_cls(**cfg.env.params)  # create an instance, with params from config
	
	# Dynamic load of Agent class
	agent = getattr(importlib.import_module(cfg.agent.type), cfg.agent.name)(**cfg.agent.params)
	print(agent.q_table)
	with mlflow.start_run(run_name=cfg.train.run_name):
		# ML Flow log training params
		mlflow.log_params(cfg.train.params)

		agent.train(env_id=cfg.env.register_id, **cfg.train.params, **cfg.agent.train_params)
		agent.save(path='saves/')
	

if __name__ == "__main__":
	main()
