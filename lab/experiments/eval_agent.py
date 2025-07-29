import importlib
import logging as log

import hydra
import numpy as np
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from lab.envs import Actions

import scipy.stats as stats


def eval_agent(env, agent, cfg, show=False):
	log.info("Starting Evaluation")
	if show:
		print("Agent Strategy:")
		agent.q_table[agent.q_table == 0.0] = float('-inf')
		print(agent.q_table)
		print(f"""
	TrendDown & NoShares: {Actions(np.argmax(agent.q_table[0, 0])).name}
	TrendDown & HasShares: {Actions(np.argmax(agent.q_table[0, 1])).name}
	TrendUp & NoShares: {Actions(np.argmax(agent.q_table[1, 0])).name}
	TrendUp & HasShares: {Actions(np.argmax(agent.q_table[1, 1])).name}
		""")

	returns = []
	agent_buy_actions = []
	agent_sell_actions = []
	for i in range(1_000):
		state, info = env.reset(options=cfg.env.options)
		mask = info['action_mask']
		
		done = False
		while not done:
			action = agent.select_action(state, mask, epsilon=0)
			if show:
				if action == Actions.BUY.value:
					agent_buy_actions.append(env.unwrapped.current_step)
					# log.info(f'BUY at {round(env.price, 2)}')
				elif action == Actions.SELL.value:
					agent_sell_actions.append(env.unwrapped.current_step)
					# log.info(f'SELL at {round(env.price, 2)}')
	
			state, reward, terminated, truncated, info = env.step(action)
			mask = info['action_mask']
			done = terminated or truncated
	
		env.unwrapped.sell(env.unwrapped.shares_count)
		returns.append(env.unwrapped.funds - 10_000)
	
	if show:
		fig, ax = plt.subplots(figsize=(10, 5))
		env.plot(ax=ax)
	
		ax.scatter(agent_buy_actions, env.get_prices()[agent_buy_actions] - 1, s=50.0,
		           marker="^", color="seagreen", label="Buy", zorder=5)
		ax.scatter(agent_sell_actions, env.get_prices()[agent_sell_actions] + 1, s=30,
		           marker="v", color="firebrick", label="Sell", zorder=5)
	
		plt.title(f"Single Agent Run: {int(env.funds)}")
		plt.grid(True)
		plt.legend()
		plt.tight_layout()
		plt.show()
		plt.show()

	returns = np.array(returns) / 10_000 * 100
	mean = float(np.mean(returns))
	std = float(np.std(returns))
	if show:
		print(f"{mean=:.2f} +/- {std=:.2f}")
		print(f"lost money: {stats.norm.cdf((0 - mean) / std) * 100:.2f}% of time")
	
	if show:
		plt.hist(returns, bins=50, label='Returns')
		plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean = {mean:.2f}%')
		plt.title("Return from 2y of trading a single random stock")
		plt.legend()
		plt.show()
	
	return mean


@hydra.main(config_path="../../config/", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
	# Dynamic load of Env and Agent classes
	env = getattr(importlib.import_module(cfg.env.type), cfg.env.name)()
	agent = getattr(importlib.import_module(cfg.agent.type), cfg.agent.name)(**cfg.agent.params)

	agent.load(path='saves', filename='MC-Agent.npy')
	eval_agent(env, agent, cfg=cfg, show=True)


if __name__ == '__main__':
	main()
