import importlib
import os

import mlflow
import optuna
import hydra
from omegaconf import OmegaConf
import gymnasium as gym
import logging as log

from lab.experiments.eval_agent import eval_agent


mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))


@hydra.main(config_path="../../config/", config_name="config", version_base="1.3")
def sweep(cfg):
	log.info(f"[Sweeping]: Start experiment {cfg.experiment.name}...")
	mlflow.set_experiment(f"{cfg.env.name}-{cfg.agent.name}-{cfg.experiment.name}")
	
	def objective(trial):
		# Modify cfg dynamically
		for key, options in cfg.tuning.search_space.items():
			if options.type == "float":
				value = trial.suggest_float(name=key.split(".")[-1], low=options.min, high=options.max)
			elif options.type == "int":
				value = trial.suggest_int(name=key.split(".")[-1], low=options.min, high=options.max)
			else:
				raise NotImplementedError("Tuning parameter of unknown type")
			OmegaConf.update(cfg, key, value)

		env = gym.make(cfg.env.register_id)
		agent = getattr(importlib.import_module(cfg.agent.type), cfg.agent.name)(**cfg.agent.params)

		with mlflow.start_run(run_name=cfg.train.run_name):
			# ML Flow log training params
			mlflow.log_params(cfg.train.params)
			mlflow.log_params(cfg.agent.train_params)
			mlflow.log_params(cfg.agent.params)
			mlflow.log_params(cfg.env.options)

			agent.train(env=env, env_options=cfg.env.options, seed=cfg.seed, **cfg.train.params, **cfg.agent.train_params)
			
			final_return = eval_agent(env, agent, cfg)
			mlflow.log_metric("final_return", final_return)

		return final_return

	study = optuna.create_study(direction=cfg.tuning.direction)
	study.optimize(objective, n_trials=cfg.tuning.n_trials)


if __name__ == "__main__":
	sweep()
