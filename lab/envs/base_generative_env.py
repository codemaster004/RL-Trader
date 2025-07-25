import gymnasium as gym
import numpy as np

from lab.envs import Actions


class GenerativeEnv(gym.Env):
	metadata = {"render_modes": ["human"]}
	
	def __init__(self):
		super(GenerativeEnv, self).__init__()

		# Custom attributes
		self.max_steps = 0
		self._current_step = 0
		
		self.funds = 0
		self.shares_count = 0
		self.init_options = {
			"simulations_length": 670,
			"start_price": 100,
			"funds": 10_000.0,
			"init_days": 60,
			"mu": 0.0005,
			"sigma": 0.01,
		}
		
		# Env RL attributes
		self.state = None
		self.reward = 0.0
		self.terminated = False
		self.truncated = False
		self.info = None
		
		# Protected
		self._prices = None
		
		self._buy_history = None
		self._sell_history = None
	
	def reset(self, seed=None, options=None):
		options = options or {}
		super().reset(seed=seed, options=None)  # Passing none, since the base method does not require any

		self.init_options.update(options)
		self.funds = self.init_options["funds"]
		
		self.max_steps = self.init_options["simulations_length"]
		self._current_step = 0
		
		# Generate historical data
		self._prices = self._init_gen_prices(**self.init_options)
		self._buy_history = {}
		self._sell_history = {}
	
	def step(self, action):
		self._prices = np.append(self._prices, self._gen_next_price(self._prices[-1], **self.init_options))
		self._current_step += 1

		self.terminated = self._is_terminated()
		self.truncated = self._is_truncated()
		self.reward = self._calc_reward(action)
	
	def buy(self, amount):
		self.shares_count += amount
		self.funds -= amount * self.price
		self.funds -= amount * self.price * 0.01
		self._sell_history = {}
		self._buy_history[self.price] = amount
	
	def sell(self, amount):
		# todo: improve
		self.shares_count -= amount
		self.funds += amount * self.price
		self.funds -= amount * self.price * 0.01
		self._buy_history = {}
		self._sell_history[self.price] = amount
	
	@property
	def price(self):
		return self._prices[-1]
	
	@property
	def current_step(self):
		return self._current_step
	
	def get_prices(self):
		return self._prices[self._get_plot_range()]
	
	def plot(self, ax):
		ax.plot(self._prices[self._get_plot_range()], label="price")
	
	def _determine_state(self):
		raise NotImplementedError  # Must be overwritten
	
	def _get_info(self):
		raise NotImplementedError  # Must be overwritten

	def _get_plot_range(self):
		return range(self.init_options["init_days"], self.init_options["init_days"] + self.current_step)

	def _calc_reward(self, action=None):
		# Stock price <= 0, bankrupt
		if self.truncated:
			return -100
		# # Small penalty for not being on the market
		# if self.shares_count <= 0:
		# 	return -1
		# At the end or run, reward is the summ of funds and value of all shares
		# if self.terminated:
		# 	return self.funds + self.shares_count * self.price
		
		reward = 0
		
		if action == Actions.BUY.value:
			reward -= self.price * 0.01
		if action == Actions.SELL.value:
			reward -= self.price * 0.01
		
		if self.shares_count > 0:
			# Reward is the difference between price when bought the stock and current price
			for acquisition_price, amount in self._buy_history.items():
				reward += self.price - acquisition_price
		else:
			# When not on the market reward for not loosing money and penalty for not gaining
			for sold_price, amount in self._sell_history.items():
				reward += sold_price - self.price
		
		return reward

	def _is_terminated(self):
		return self.current_step >= self.max_steps
	
	def _is_truncated(self):
		return self._prices[-1] <= 0

	@staticmethod
	def _init_gen_prices(start_price=100.0, init_days=5, mu=0.0005, sigma=0.01, df=5, *args, **kwargs):
		log_returns = GenerativeEnv._gen_returns_t(num_days=init_days, mu=mu, sigma=sigma, df=df)
		log_prices = np.cumsum(log_returns)
		return start_price * np.exp(log_prices)

	@staticmethod
	def _gen_returns_t(num_days=5, mu=0.0005, sigma=0.01, df=5):
		returns = np.random.standard_t(df=df, size=num_days)
		returns = mu + sigma * returns
		return returns

	@staticmethod
	def _gen_next_price(previous_price=100.0, mu=0.0005, sigma=0.01, df=5, *args, **kwargs):
		log_return_next = GenerativeEnv._gen_returns_t(num_days=1, mu=mu, sigma=sigma, df=df)
		next_price = previous_price * np.exp(log_return_next)
		return next_price[0]
