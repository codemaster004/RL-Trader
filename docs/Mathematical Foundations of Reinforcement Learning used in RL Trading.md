
---
**Reinforcement Learning (RL)** is a method of training an **agent** to take **actions** in an **environment** by observing its **state**, with the goal of **maximising cumulative rewards** over time.

## Reinforcement Learning - definition
Imagine an agent (a trading bot) placed in a predefined environment - the **stock market**. At every time step - day, the agent:
1. **Observes** the state of the environment (e.g., stock prices, trends).
2. **Decides** which action to take (e.g., Buy, Sell, or Hold).
3. **Receives a reward** from the environment based on whether that decision was good or bad.

Over time, the agent learns which actions lead to higher rewards and adjusts its behaviour accordingly.

<img src="images/RL-img.png" alt="Description" width="300"/>

## Key Terminology dictionary
Don't be threatened by scary math, it will be clearly explained below.

| Symbol        | Name                  | Definition                                                                                   |
| ------------- | --------------------- | -------------------------------------------------------------------------------------------- |
| $s$           | state                 | What the agent sees by observing the enviroment.                                             |
| $a$           | action                | What the agent does in the enviroment.                                                       |
| $r$           | reward                | Feedback from the environment indicating if the action was good or bad.                      |
| $\pi$         | policy                | A strategy that defines the probability of taking each action in a state.                    |
| $R_t$         | return                | The total accumulated reward from time step $t$ onward.                                      |
| $\gamma$      | discount factor       | A value in $[0, 1]$ that reduces the impact of future rewards.                               |
| $G_t$         | (sample) return       | The observed return following time $t$.                                                      |
| $Q^\pi(s, a)$ | action-value function | Expected return when being in state $s$, taking action $a$, and then following policy $\pi$. |
# Terminology clearly explained
---
## State - $s$
In the context of trading on the stock market, the simplest way to think about a **state** might be the current price. However, upon reflection, it’s difficult to define robust trading rules based on price alone. For example, a rule like “if the price is 20, sell” is not very versatile or reliable.

Instead, it’s more effective to define the state using **market trends** and the **bot’s current portfolio**. For example, state can be represented by two components:
- **Market Trend**: `TrendUp` or `TrendDown`
- **Portfolio Status**: `NoShares` or `HasShares`

An example of the agent's perceived state of the environment could then be:
$$
s_t = [TrendUp, NoShares]
$$
where:
- $s_t$ represents the agent’s current state at time step $t$ (some day).
- $t$ refers to the current step in the episode (e.g., trading day number 32).

## Action - $a$
Now that we’ve defined what a **state** is, the next key concept is an **action**. In a stock trading scenario, actions are decisions that the agent (our trading bot) can take at each time step.

Let’s keep things simple and define three basic actions:
- `Buy` - purchase a shares of stock
- `Sell` - sell a shares of stock
- `Hold` - do nothing, wait and see

So, for example, if considering the state from above, bot's action could be:
$$
a_t =Buy
$$
where:
- $a_t$ represents the agent’s action at time step $t$.

## Reward - $r$
Once the agent takes an action, the environment needs to respond. This response is called a **reward**, and it's how the agent learns whether the action was good or bad.

In trading, reward could be defined in many ways, what I picked is:
- When agent bought shares, reward at each time step $t$ is the difference between `CurrentPrice` and `PriceWhenBought` - this way, when the price goes up agent gets a positive reward, or negative when the price drops.
- When agent does not have any shares, reward at each time step id difference between `LastSoldPrice` and `CurrentPrice` - this way when the bot sold the stock and price went down, it will get a positive reward, and negative when he sold but the price wend up.
- Additionally each action of action of `Buy` and `Sell` has a constant negative reward, to show a penalty of selling or buying shares. 

For example:
- If we buy at $100 and the price later goes to $110, the reward is +10.
- If we sell at $90 after selling price went to $100, the reward is -10.

## Episode
In reinforcement learning, an **episode** is a single run where the agent interacts with the environment from a starting point to an endpoint.

In stock trading, one episode could be the agent trading through a **year of historical market data**, making decisions each day based on the current state, taking actions, receiving rewards, and moving to the next state.

## Trajectory
A **trajectory** is the complete sequence of everything that happens during an episode. It includes the **states**, **actions**, and **rewards** the agent experiences from start to finish.

Formally, a trajectory can be written as:
$$
\tau = ((s_0,a_0,r_1),(s_1,a_1,r_2),…,(s_{T-1},a_{T-1},s_T))
$$

In the context of stock trading, a trajectory could represent one full trading year with:
- $s_t$ - observes daily market conditions (**states**),
- $a_t$ - decides what to do (**actions**),
- $r_t$ - received feedback (**rewards**) after each decision.

The trajectory captures the **full experience** of the agent in one episode and is used to evaluate or improve during training.

## Policy – $\pi$
The **policy** is like the brain of the agent - it’s the strategy that decides which action to take in a given state.

Formally, a policy is a function (probability of choosing action $a$ given state $s$):
$$
\pi(a|s)
$$

Notice that this means that the probabilities change depending on the state given to the function.

Example:
- state: [`TrendUp`, `NoShares`], action probabilities: {`Buy`: 0.9, `Hold`: 0.09, `Sell`: 0.01}
- state: [`TrendUp`, `HasShares`], action probabilities: {`Buy`: 0.01, `Hold`: 0.98, `Sell`: 0.01}

Commonly used simple policies:
- greedy: pick the action with the highest expected reward.
- $\epsilon$-greedy: sometimes picks the best action sometimes random.
- softmax: select actions with higher mean reward, according to softmax function.

## Discount Factor - $\gamma$
The **discount factor**, written as $\gamma$, is a number between 0 and 1 that controls **how much the agent values future rewards** compared to immediate ones.

In simple terms:
- A **low** $\gamma$ (close to 0) means the agent cares **mostly about immediate rewards**.
- A **high** $\gamma$ (close to 1) means the agent cares **more about long-term rewards**.

## Grand Total Return - $G_t$
The **grand total return**, written as $G_t$, is the **total reward the agent expects to receive starting from time step** $t$ until the end of the episode.
It combines **immediate and future rewards**, possibly discounted, into a single number.

$$
G_t = \sum^{T-t-1}_{k=0}\gamma^k r_{t+k+1}
$$
$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ... + \gamma^{T-t-1}r_T
$$
Where:
- $r_t+1$ is the reward received after taking action at time $t$,
- $\gamma$ is the **discount factor** (see above).
- $T$ is the final step of the episode (e.g., final day of trading data)

## Action-Value Function - $Q^π(s,a)$
The **action-value function** tells us **how good is it to take a specific action $a$ in a specific state $s$**, assuming the agent will follow policy $\pi$ afterward.

Formal definition:
$$
Q^\pi(s,a)= \mathbb{E}_\pi \left[ G_t | s_t =s,a_t =a \right]
$$

As we can see above, the **Action-Value Function** 

Where:
- **$\pi$** - The **policy** the agent follows after taking the first action. (see above)
- **$\mathbb{E}_\pi$** - The **expected value**, or average, over all possible future outcomes under policy $\pi$.
- **$G_t$** - The **total return** from time step $t$ onward (immediate + future rewards, possibly discounted):
- **$s_t=s, a_t=a$** - We assume that **$a_t$ time step $t$**, the agent is in state $s$ and takes action $a$.

Example:
If agent is in a state $s = [TrendUp, NoShares]$ and takes the action $a = Buy$ then:
$$
Q^\pi([TrendUp,NoShares], Buy)
$$

Would give us the average reward of buying shares when the trend is up and bot don't have any shares.

# Iterative Monte Carlo
---

To train an agent, one of the simplest and most intuitive approaches in Reinforcement Learning is the **Monte Carlo (MC) method**.

The Interactive Monte Carlo method is all about **averaging experience from episodes over time**. The agent plays out an **episode**, collects all the rewards, and then uses that information to **update it's action-value function and policy**.

## 1. Setup
With our previous setup for stock trading we need first to create the representation of the Action-Value Function $Q^\pi(s, a)$, our State has 2 dimensions each dimension with 2 discrete values, and our action has 1 dimension with 3 discrete values, to represent all combination of states and actions we need a 3 dimensional table, this table is called a **Q-Table**.
```python
import numpy as np

table_shape = (
	2,  # 2 values for trend: Down (0), Up (1)
	2,  # 2 values for shares: No (0), Has (1)
	3,  # 3 values for action: Hold (0), Buy (1), Sell (2)
)

q_table = np.zeros(table_shape)
print(q_table)
"""
[[[0. 0. 0.]
  [0. 0. 0.]]

 [[0. 0. 0.]
  [0. 0. 0.]]]
"""
```

## 2. Single episode run

To learn from experience, the agent needs to **interact with the environment**, record what happens, and analyse the results afterward. This process is done one **episode** at a time.
### What Happens in One Episode?

Here’s a breakdown of what takes place during a single episode:
1. **Initialise the Environment**
2. **Single day of trading - step $t$**    
	- Observe the **current state** $s_t$.
	- Choose an **action** $a_t$ according to the **policy** $\pi(a|s_t)$.
	- Execute the action and receive:
		- A **reward** $r_{t+1}$.
		- A **new state** $s_{t+1}$.
3. **Repeat**
    - Continue until the episode ends (e.g., all trading days are finished).
4. **Store the Trajectory**
    - Collect the full sequence:

> In our trading bot example, this could mean simulating an entire year of trades, capturing which days it bought/sold/held and what rewards those choices led to.

```python
def run_episode(env, policy):
	trajectory = []
    state = env.reset()
    
    done = False
    while not done:
        action_probs = policy[state]
        action = pick_action(len(action_probs), p=action_probs)

        next_state, reward, done = env.step(action)

        trajectory.append((state, action, reward))
        state = next_state
    return trajectory
```

### 3. Update
After the episode ends, we go through the trajectory and compute the **return** $G_t$ for each time step $t$. This gives us the total discounted reward the agent received after taking action $a_t$ in state $s_t$.

We then use the following **update formula** to update our Action-Value function for each time step $t$:
$$
Q(s_t​,a_t​) = Q(s_t​,a_t​) + \alpha [G_t​−Q(s_t​,a_t​)]
$$

Where:
- $G_t$ is the **return** from time step $t$ onward.
- $\alpha$ is a **learning rate** (e.g., 0.1) - how quickly the updated value changes.
- $Q(s_t, a_t)$ is the current estimate of the value of taking action $a_t$ in state $s_t$.

> This adjusts the Q-value toward the **average of observed returns**, making it more accurate over time.

```python
def update(q_table, states, actions, rewards, alpha=0.1):
	grand_totals = calc_grand_totals(rewards)
	for s, a, g in zip(states, actions, grand_totals):
		q_table[s[0], s[1], a] += alpha * (g - q_table[s[0], s[1], a])
```
### 4. Repeat
Repeat steps 1-3 for many episodes
