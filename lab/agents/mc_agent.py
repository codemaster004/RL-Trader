from multiprocessing import Pool
from os.path import join as pjoin

import mlflow
import numpy as np
import logging as log
import gymnasium as gym

from lab.agents.base_agent import BaseAgent


class MCBuffer:
	def __init__(self):
		self.states = None
		self.actions = None
		self.rewards = None

		self.reset()

	def reset(self):
		self.states = []
		self.actions = []
		self.rewards = []

	def add(self, state, action, reward):
		self.states.append(state)
		self.actions.append(action)
		self.rewards.append(reward)

	def get(self):
		return self.states, self.actions, self.rewards


class MonteCarloAgent(BaseAgent):
	def __init__(self, state_dim, action_dim, options=None):
		super(MonteCarloAgent, self).__init__(state_dim, action_dim, options)

		self.q_table = np.zeros((*self.state_dim, *self.action_dim))
		self.buffer = MCBuffer()

		self.env_options = None

	def select_action(self, state, mask, **kwargs):
		epsilon = kwargs.get('epsilon', 0.1)
		allowed_indices = np.where(mask)[0]
		if np.random.random() < epsilon:
			action = np.random.choice(allowed_indices)
		else:
			masked_q_values = self.q_table[(*state,)][mask]
			action = allowed_indices[np.argmax(masked_q_values)]

		return action

	def update(self, *args, **kwargs):
		states, actions, rewards = args

		alpha = kwargs.get('alpha', 0.01)

		for s, a, r in zip(states, actions, rewards):
			self.q_table[(*s, a)] += alpha * (r - self.q_table[(*s, a)])

	def train(self, env=None, env_id=None, env_options=None, *args, **kwargs):
		self.env_options = env_options
		(
			learning_rate,
			discounting,
			episodes,
			epsilon,
			trajectories_per_episode,
			num_threads,
			log_step
		) = self._extract_train_kwargs(**kwargs)
		
		log.info(f'[{self.__class__.__name__}]: Starting Training...')
		with Pool(num_threads) as pool:
			for episode in range(episodes):
				self.buffer.reset()

				args = [(env_id, epsilon) for _ in range(trajectories_per_episode)]
				results = pool.starmap(self._run_episode_worker, args)

				states, actions, rewards = [], [], []
				for s, a, r in results:
					states += s
					actions += a
					rewards += self._calc_cumsum_rewards(r, discounting)
				# Collect Experience
				# for _ in range(trajectories_per_episode):
				# 	self._run_episode(env, epsilon)

				# Update Q-table
				self.update(states, actions, rewards, alpha=learning_rate)

				# Logging
				episode_avg_reward = float(np.mean(rewards))
				mlflow.log_metric("avg_reward", episode_avg_reward, step=episode)
				if (episode + 1) % log_step == 0:
					log.info(f"Episode: {episode + 1}, avg reward: {episode_avg_reward}")

	def _run_episode(self, env, epsilon):
		state, info = env.reset(options=self.env_options)
		mask = info['action_mask']

		done = False
		while not done:
			action = self.select_action(state, mask, epsilon=epsilon)

			next_state, reward, terminated, truncated, info = env.step(action)
			mask = info['action_mask']
			self.buffer.add(state, action, reward)
			state = next_state

			done = terminated or truncated

		return self.buffer.get()

	def _run_episode_worker(self, env_id, epsilon):
		env = gym.make(env_id)
		return self._run_episode(env, epsilon)

	def _extract_train_kwargs(self, **kwargs):
		lr = kwargs.get('learning_rate', 0.001)
		disc = kwargs.get('discount', 0.99)
		epi = kwargs.get('episodes', 1000)
		eps = kwargs.get('epsilon', 0.2)
		traj_n_e = kwargs.get('trajectories_per_episode', 1)
		thr_n_e = kwargs.get('threads_per_episode', 10)
		log_step = kwargs.get('log_step', 100)

		return lr, disc, epi, eps, traj_n_e, thr_n_e, log_step

	def save(self, path='.', filename='MC-Agent.npy'):
		np.save(pjoin(path, filename), self.q_table)

	def load(self, path='.', filename='MC-Agent.npy'):
		self.q_table = np.load(pjoin(path, filename))

	@staticmethod
	def _calc_cumsum_rewards(rewards, lam):
		# Calculating rewards from final reward with discounting lam
		running_sum = 0.0
		for t in reversed(range(len(rewards))):
			running_sum = rewards[t] + lam * running_sum
			rewards[t] = running_sum
		return rewards
