import pytest
import numpy as np
from lab.envs.base_generative_env import GenerativeEnv


@pytest.fixture
def env():
	env = GenerativeEnv(simulations_length=100)
	env.reset()
	return env


def test_reset_initializes_prices_and_funds(env):
	assert env._prices is not None
	assert len(env._prices) == env.init_options["init_days"]
	assert env.funds == env.init_options["funds"]
	assert env.current_step == 0


def test_gen_returns_t_shape_and_range():
	returns = GenerativeEnv._gen_returns_t(num_days=10)
	assert returns.shape == (10,)
	assert np.all(np.isfinite(returns))


def test_gen_next_price():
	price = GenerativeEnv._gen_next_price(previous_price=100.0)
	assert price > 0


def test_step_adds_price_and_increments_step(env):
	before_len = len(env._prices)
	env.step(action=0)  # dummy action
	assert len(env._prices) == before_len + 1
	assert env.current_step == 1


def test_buy_and_sell_affect_funds_and_shares(env):
	initial_funds = env.funds
	price = env.price
	env.buy(amount=2)
	assert env.shares_count == 2
	assert np.isclose(env.funds, initial_funds - 2 * price)

	env.sell(amount=2)
	assert env.shares_count == 0
	assert env.funds >= initial_funds - 2 * price  # could be more due to price change


def test_price_property(env):
	assert env.price == env._prices[-1]


def test_get_prices_returns_correct_slice(env):
	env.step(0)
	prices = env.get_prices()
	expected_length = len(env._get_plot_range())
	assert len(prices) == expected_length


def test_calc_reward_handles_no_shares(env):
	env._buy_history = {}
	env.shares_count = 0
	reward = env._calc_reward()
	assert reward == 0


def test_calc_reward_handles_truncated(env):
	env.truncated = True
	reward = env._calc_reward()
	assert reward == -100


def test_is_terminated(env):
	env.current_step = 99
	assert not env._is_terminated()
	env.current_step = 100
	assert env._is_terminated()


def test_is_truncated_when_price_is_zero(env):
	env._prices = np.append(env._prices, 0.0)
	assert env._is_truncated()


def test_plot_range_bounds(env):
	r = env._get_plot_range()
	assert r.start == env.init_options["init_days"]
	assert r.stop == env.init_options["init_days"] + env.current_step


@pytest.mark.parametrize("acquisition_price, current_price, amount, expected_reward", [
	(100.0, 105.0, 1, -5.0),  # loss because price > buy price
	(105.0, 100.0, 1, 5.0),  # gain because price < buy price
	(100.0, 100.0, 1, 0.0),  # no gain/loss
	(100.0, 90.0, 3, 30.0),  # larger gain (buy high, now lower)
	(90.0, 100.0, 2, -20.0),  # larger loss (buy low, now higher)
])
def test_calc_reward_with_holdings(acquisition_price, current_price, amount, expected_reward):
	env = GenerativeEnv()
	env.reset()

	# Manually override for test
	env._buy_history = {acquisition_price: amount}
	env.shares_count = amount
	env._prices = np.append(env._prices, current_price)  # set last price
	env.truncated = False

	reward = env._calc_reward()
	assert pytest.approx(reward, rel=1e-6) == expected_reward
