import pytest
import numpy as np
from lab.envs.base_generative_env import GenerativeEnv
from lab.envs import Actions


@pytest.fixture
def env():
	env = GenerativeEnv()
	env.reset(options={"simulations_length": 100})
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
	env.step(action=Actions.HOLD.value)
	assert len(env._prices) == before_len + 1
	assert env.current_step == 1


def test_buy_and_sell_affect_funds_and_shares(env):
	initial_funds = env.funds
	price = env.price
	env.buy(amount=2)
	assert env.shares_count == 2
	assert np.isclose(env.funds, initial_funds - 2 * price * 1.01)

	env.sell(amount=2)
	assert env.shares_count == 0
	assert np.isclose(env.funds, initial_funds, rtol=0.05)


def test_price_property(env):
	assert env.price == env._prices[-1]


def test_get_prices_returns_correct_slice(env):
	env.step(Actions.HOLD.value)
	prices = env.get_prices()
	expected_length = len(env._get_plot_range())
	assert len(prices) == expected_length


def test_calc_reward_handles_truncated(env):
	env.truncated = True
	reward = env._calc_reward()
	assert reward == -100


def test_is_terminated(env):
	env.current_step = env.max_steps - 1
	assert not env._is_terminated()
	env.current_step = env.max_steps
	assert env._is_terminated()


def test_is_truncated_when_price_is_zero(env):
	env._prices = np.append(env._prices, 0.0)
	assert env._is_truncated()


def test_plot_range_bounds(env):
	r = env._get_plot_range()
	assert r.start == env.init_options["init_days"]
	assert r.stop == env.init_options["init_days"] + env.current_step


@pytest.mark.parametrize("acquisition_price, current_price, amount, expected_reward", [
	(100.0, 105.0, 1, 5.0),  # gain
	(105.0, 100.0, 1, -5.0),  # loss
	(100.0, 100.0, 1, 0.0),  # break-even
])
def test_calc_reward_with_holdings(acquisition_price, current_price, amount, expected_reward):
	env = GenerativeEnv()
	env.reset()
	env._buy_history = {acquisition_price: amount}
	env.shares_count = amount
	env._prices = np.append(env._prices, current_price)
	env.truncated = False

	reward = env._calc_reward()
	assert pytest.approx(reward, rel=1e-6) == expected_reward


@pytest.mark.parametrize("sold_price, current_price, amount, expected_reward", [
	(100.0, 95.0, 1, 5.0),  # reward for exiting before price drops
	(95.0, 100.0, 1, -5.0),  # penalty for missing rise
])
def test_calc_reward_with_sell_history(sold_price, current_price, amount, expected_reward):
	env = GenerativeEnv()
	env.reset()
	env._sell_history = {sold_price: amount}
	env.shares_count = 0
	env._prices = np.append(env._prices, current_price)
	env.truncated = False

	reward = env._calc_reward()
	assert pytest.approx(reward, rel=1e-6) == expected_reward
