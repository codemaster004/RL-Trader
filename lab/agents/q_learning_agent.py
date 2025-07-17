from lab.agents.base_agent import BaseAgent

import logging as log


class QLearningAgent(BaseAgent):

	def __init__(self, state_dim, action_dim, options=None):
		super(QLearningAgent, self).__init__(state_dim, action_dim, options=options)

	def select_action(self, state, mask, *args, **kwargs):
		pass

	def update(self, *args, **kwargs):
		pass

	def train(self, env=None, env_id=None, *args, **kwargs):
		if env is None and env_fn is None:
			raise ValueError('env or env_fn is required')

		(learning_rate,
		 discounting,
		 episodes,
		 epsilon,
		 trajectories_per_episode,
		 num_threads) = self._extract_train_kwargs(**kwargs)
		
		log.info(f"[{self.__class__.__name__}]: Training...")
		for episode in range(episodes):
			pass

	def _extract_train_kwargs(self, **kwargs):
		lr = kwargs.get('learning_rate', 0.001)
		disc = kwargs.get('discount', 0.99)
		epi = kwargs.get('episodes', 1000)
		eps = kwargs.get('epsilon', 0.2)
		n_th = kwargs.get('num_threads', 1)
		traj_n_epi = kwargs.get('trajectories_per_episode', 1)
		
		return lr, disc, epi, eps, n_th, traj_n_epi

	def save(self, path='.', filename='agent.npy'):
		pass

	def load(self, path='.', filename='agent.npy'):
		pass
