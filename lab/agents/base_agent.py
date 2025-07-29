from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor


class BaseAgent(ABC):
	def __init__(self, state_dim, action_dim, options=None):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.options = options if options is not None else {}

	@abstractmethod
	def select_action(self, state, mask, **kwargs):
		pass

	@abstractmethod
	def update(self, *args, **kwargs):
		pass

	@abstractmethod
	def train(self, env=None, env_options=None, seed=None, *args, **kwargs):
		pass
	
	@abstractmethod
	def _extract_train_kwargs(self, **kwargs):
		pass

	def save(self, path='.', filename='agent.npy'):
		raise NotImplementedError

	def load(self, path='.', filename='agent.npy'):
		raise NotImplementedError

