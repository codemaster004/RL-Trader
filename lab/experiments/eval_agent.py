import importlib

import hydra
import numpy as np
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from lab.envs import Actions

import logging as log


def eval_agent(env, agent):
	state, info = env.reset()
	mask = info['action_mask']

	log.info("Starting Evaluation")
	log.info("Agent Strategy:")
	agent.q_table[agent.q_table == 0.0] = float('-inf')
	print(agent.q_table)
	print(f"""
	TrendDown & NoShares: {Actions(np.argmax(agent.q_table[0, 0])).name}
	TrendDown & HasShares: {Actions(np.argmax(agent.q_table[0, 1])).name}
	TrendUp & NoShares: {Actions(np.argmax(agent.q_table[1, 0])).name}
	TrendUp & HasShares: {Actions(np.argmax(agent.q_table[1, 1])).name}
	""")

	agent_buy_actions = []
	agent_sell_actions = []

	done = False
	while not done:
		action = agent.select_action(state, mask, epsilon=0)
		if action == Actions.BUY.value:
			agent_buy_actions.append(env.current_step)
			# log.info(f'BUY at {round(env.price, 2)}')
		elif action == Actions.SELL.value:
			agent_sell_actions.append(env.current_step)
			# log.info(f'SELL at {round(env.price, 2)}')

		state, reward, terminated, truncated, info = env.step(action)
		mask = info['action_mask']
		done = terminated or truncated

	env.sell(env.shares_count)

	fig, ax = plt.subplots(figsize=(10, 5))
	env.plot(ax=ax)

	ax.scatter(agent_buy_actions, env.get_prices()[agent_buy_actions] - 0.3, marker="^", color="seagreen", label="Buy",
	           zorder=5)
	ax.scatter(agent_sell_actions, env.get_prices()[agent_sell_actions] + 0.3, marker="v", color="firebrick",
	           label="Sell", zorder=5)

	plt.title(f"Single Agent Run: {int(env.funds)}")
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.show()
	plt.show()


@hydra.main(config_path="../../config/", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
	# Dynamic load of Env and Agent classes
	env = getattr(importlib.import_module(cfg.env.type), cfg.env.name)(**cfg.env.params)
	agent = getattr(importlib.import_module(cfg.agent.type), cfg.agent.name)(**cfg.agent.params)

	agent.load(path='saves')
	eval_agent(env, agent)


if __name__ == '__main__':
	main()
