from gymnasium.envs.registration import register


register(
	id="SimpleTrends-v1",
	entry_point="lab.envs.simple_trends_env:SimpleTrends",
)
