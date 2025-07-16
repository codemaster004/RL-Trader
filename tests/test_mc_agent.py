import numpy as np
import pytest
from lab.agents.mc_agent import MonteCarloAgent, MCBuffer


# === MCBuffer ===

def test_mc_buffer_add_and_get():
    buffer = MCBuffer()
    buffer.add([0, 1], 2, 1.0)
    buffer.add([1, 0], 1, 0.5)

    states, actions, rewards = buffer.get()

    assert states == [[0, 1], [1, 0]]
    assert actions == [2, 1]
    assert rewards == [1.0, 0.5]


def test_mc_buffer_reset():
    buffer = MCBuffer()
    buffer.add([0, 1], 2, 1.0)
    buffer.reset()

    states, actions, rewards = buffer.get()

    assert states == []
    assert actions == []
    assert rewards == []


# === MonteCarloAgent ===

@pytest.fixture
def mc_agent():
    return MonteCarloAgent(state_dim=(3,), action_dim=(3,))


def test_q_table_initialized_correctly(mc_agent):
    assert mc_agent.q_table.shape == (3, 3)
    assert np.all(mc_agent.q_table == 0)


def test_select_action_epsilon_greedy(mc_agent):
    state = [0]
    mask = np.array([True, False, True])

    np.random.seed(42)  # ensure deterministic
    action = mc_agent.select_action(state, mask, epsilon=0.0)  # greedy
    assert action in [0, 2]


def test_select_action_random_when_exploring(mc_agent):
    state = [1]
    mask = np.array([False, True, True])

    np.random.seed(1)
    action = mc_agent.select_action(state, mask, epsilon=1.0)  # always random
    assert action in [1, 2]


def test_update_q_table(mc_agent):
    states = [[0], [1]]
    actions = [0, 1]
    rewards = [1.0, 0.5]

    mc_agent.update(states, actions, rewards, alpha=0.1)

    assert mc_agent.q_table[0, 0] == pytest.approx(0.1)
    assert mc_agent.q_table[1, 1] == pytest.approx(0.05)


def test_calc_cumsum_rewards_discounted():
    rewards = [1.0, 0.0, 1.0]
    discounted = MonteCarloAgent._calc_cumsum_rewards(rewards.copy(), lam=0.9)
    assert discounted == pytest.approx([1.81, 0.9, 1.0], rel=1e-4)


def test_save_and_load_q_table(tmp_path, mc_agent):
    mc_agent.q_table[1, 1] = 42.0
    mc_agent.save(path=tmp_path, filename='agent.npy')

    # Load into new agent
    new_agent = MonteCarloAgent(state_dim=(3,), action_dim=(3,))
    new_agent.load(path=tmp_path, filename='agent.npy')

    assert new_agent.q_table[1, 1] == 42.0
