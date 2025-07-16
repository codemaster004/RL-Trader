import hydra
from omegaconf import DictConfig, OmegaConf

import importlib


@hydra.main(config_path="../../config/", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
	# Dynamically loading, based on config.
	env_path, env_name = cfg.env.type, cfg.env.name
	env_cls = getattr(importlib.import_module(env_path), env_name)  # dynamic load a given class from a library
	env = env_cls(**cfg.env.params)  # create an instance, with params from config
	
	# Dynamic load of Agent class
	agent = getattr(importlib.import_module(cfg.agent.type), cfg.agent.name)(**cfg.agent.params)
	
	agent.train(env=env, **cfg.train.params, **cfg.agent.train_params)
	agent.save(path='saves/')
	print(agent.q_table)


if __name__ == "__main__":
	main()
