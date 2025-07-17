from gymnasium.envs.registration import register


register(
	id="SimpleTrends-v0",
	entry_point="lab.envs.simple_trends_env:SimpleTrends",
)
