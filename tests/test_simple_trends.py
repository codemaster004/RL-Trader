import pytest
import numpy as np
from lab.envs.simple_trends_env import SimpleTrends
from lab.envs._common import Actions


@pytest.fixture
def env():
	env = SimpleTrends(simulations_length=100)
	env.reset()
	return env


def test_reset_initializes_state_and_sma(env):
	assert isinstance(env.state, np.ndarray)
	assert env.state.shape == (2,)
	assert env._short_sma is not None
	assert env._long_sma is not None
	assert len(env._short_sma) == len(env._prices)
	assert len(env._long_sma) == len(env._prices)


def test_step_hold_does_not_buy_or_sell(env):
	initial_funds = env.funds
	initial_shares = env.shares_count
	env.step(Actions.HOLD.value)
	assert env.funds == initial_funds
	assert env.shares_count == initial_shares


def test_step_buy_increases_shares(env):
	env.funds = 1000  # ensure sufficient funds
	shares_before = env.shares_count
	env.step(Actions.BUY.value)
	assert env.shares_count > shares_before


def test_step_sell_decreases_shares(env):
	env.funds = 1000
	env.step(Actions.BUY.value)
	shares_bought = env.shares_count
	env.step(Actions.SELL.value)
	assert env.shares_count == 0
	assert env.funds > 1000 - shares_bought * env.price


def test_step_updates_sma_lengths(env):
	len_before = len(env._short_sma)
	env.step(Actions.HOLD.value)
	assert len(env._short_sma) == len_before + 1
	assert len(env._long_sma) == len_before + 1


@pytest.mark.parametrize("short_sma, long_sma, expected_trend", [
	(np.array([1.0]), np.array([2.0]), 0),  # short < long => downtrend
	(np.array([3.0]), np.array([2.0]), 1),  # short > long => uptrend
])
def test_determine_state_trend_logic(short_sma, long_sma, expected_trend):
	env = SimpleTrends()
	env.reset()
	env._short_sma = short_sma
	env._long_sma = long_sma
	env.shares_count = 0
	state = env._determine_state()
	assert state[0] == expected_trend


def test_get_info_returns_correct_action_mask(env):
	env.funds = env.price * 2
	env.shares_count = 5
	info = env._get_info()
	assert isinstance(info, dict)
	assert "action_mask" in info
	assert info["action_mask"].tolist() == [True, True, True]


def test_plot_range_matches_sma(env):
	env.step(Actions.HOLD.value)
	plot_range = env._get_plot_range()
	sma_short = env._short_sma[plot_range]
	sma_long = env._long_sma[plot_range]
	assert len(sma_short) == len(plot_range)
	assert len(sma_long) == len(plot_range)


def test_render_does_not_crash(env, capsys):
	# This test verifies that render() prints something and does not raise
	env.render()
	captured = capsys.readouterr()
	assert "funds=" in captured.out
	assert "shares_count=" in captured.out
	assert "price=" in captured.out


def test_step_invalid_action_raises(env):
	with pytest.raises(ValueError, match="Invalid action"):
		env.step(999)  # Not part of Actions enum


def test_step_zero_price_raises(env):
	env._prices = np.append(env._prices, 0.0)  # override last price
	with pytest.raises(ValueError, match="Invalid price"):
		env.step(Actions.BUY.value)


def test_step_negative_price_raises(env):
	env._prices = np.append(env._prices, -5.0)
	with pytest.raises(ValueError, match="Invalid price"):
		env.step(Actions.BUY.value)


def test_step_buy_with_insufficient_funds_does_nothing(env):
	env.funds = 0.5  # less than price
	shares_before = env.shares_count
	env.step(Actions.BUY.value)
	assert env.shares_count == shares_before


def test_step_sell_with_no_shares_does_nothing(env):
	env.shares_count = 0
	funds_before = env.funds
	env.step(Actions.SELL.value)
	assert env.funds == funds_before
