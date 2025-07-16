class ExperienceBuffer:
	def __init__(self):
		self.reset()
		self.data = None

	def reset(self):
		self.data = {
			"states": [],
			"actions": [],
			"rewards": [],
			"next_states": [],
			"dones": [],
		}

	def add(self, state, action, reward, next_state=None, done=False):
		self.data["states"].append(state)
		self.data["actions"].append(action)
		self.data["rewards"].append(reward)
		self.data["next_states"].append(next_state)
		self.data["dones"].append(done)

	def get(self, *keys):
		if keys:
			return tuple(self.data[k] for k in keys)
		return tuple(self.data[k] for k in self.data)

	def size(self):
		return len(self.data["states"])
