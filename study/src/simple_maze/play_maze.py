import random
import numpy as np
from environment import Environment

class Agent():

	def __init__(self, env):
		self.actions = env.actions

	def select_action_by_policy(self, state):
		return random.choice(self.actions)

def main():
	grid = np.array(
		[
			[0, 0, 0, 0],
			[0, 9, 0, -1],
			[0, 0, 0, 1]
		]
	)

	env = Environment(grid)
	agent = Agent(env)

	for i in range(10):
		state = env.reset_agent_state()
		total_reward = 0
		is_terminal = False

		while not is_terminal:
			action = agent.select_action_by_policy(state)
			next_state, reward, is_terminal = env.do_step(action)
			total_reward += reward
			state = next_state

		print("Episode {}: Agent gets {} reward.".format(i, total_reward))

if __name__ == "__main__":
	main()