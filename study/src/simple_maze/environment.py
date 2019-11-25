import numpy as np
from enum import Enum


class State:

	def __init__(self, row=-1, column=-1):
		self._row = row
		self._column = column

	@property
	def row(self):
		return self._row

	@property
	def column(self):
		return self._column

	def __get_state_representation(self):
		return "State:: [{}, {}]".format(self.row, self.column)

	def clone(self):
		return State(self.row, self.column)

	# make class State hashable
	def __hash__(self):
		return hash((self.row, self.column))

	def __eq__(self, other):
		return self.row == other.row and self.column == other.column

class Action(Enum):
	UP = 1
	DOWN = -1
	LEFT = 2
	RIGHT = -2

class Environment:

	def __init__(self, grid, move_prob=0.8):
		# initialize instance variables
		self.grid = grid
		self._row_length = grid.shape[0]
		self._column_length = grid.shape[1]
		self.agent_state = State()
		self.default_reward = -0.04
		self.move_prob = move_prob

		# locate the agent at top left corner
		self.reset_agent_state()

	@property
	def row_length(self):
		return self._row_length

	@property
	def column_length(self):
		return self._column_length

	@property
	def actions(self):
		return list(Action)

	@property
	def agent_states(self):
		agent_states = []

		for row in range(self.row_length):
			for column in range(self.column_length):
				if not self.is_possible_state(self.grid[row][column]):
					agent_states.append(State(row, column))

		return agent_states

	def calc_transition_probs(self, state, action):
		transition_probs = {}

		if not self.can_act(state):
			# already on the terminal cell
			return transition_probs

		opposite_action = Action(action.value * -1)

		for possible_action in self.actions:
			prob = 0

			if possible_action == action:
				prob = self.move_prob
			elif possible_action != opposite_action:
				prob = (1 - self.move_prob) / 2

			next_state = self._move(state, possible_action)

			if next_state in transition_probs:
				transition_probs[next_state] += prob
			else:
				transition_probs[next_state] = prob

		return transition_probs

	def calc_reward(self, state):
		reward = self.default_reward
		grid_attribute = self.grid[state.row][state.column]

		# if terminal cell, get reward
		if grid_attribute in {-1, 1}:
			reward = grid_attribute

		return reward

	def is_terminal_state(self, state):
		is_terminal = False
		grid_attribute = self.grid[state.row][state.column]

		if grid_attribute in {-1, 1}:
			is_terminal = True

		return is_terminal

	def _move(self, state, action):
		if not self.can_act(state):
			raise Exception("Cannot move from here")

		if action == action.UP:
			next_state = State(state.row - 1, state.column)
		elif action == action.DOWN:
			next_state = State(state.row + 1, state.column)
		elif action == action.LEFT:
			next_state = State(state.row, state.column - 1)
		elif action == action.RIGHT:
			next_state = State(state.row, state.column + 1)

		# if state will be out of grid, return current state
		if next_state.row < 0 or next_state.row >= self.row_length:
			next_state = state
		if next_state.column < 0 or next_state.column >= self.column_length:
			next_state = state

		# if impossible state, return current state
		if not self.is_possible_state(self.grid[next_state.row][next_state.column]):
			next_state = state

		return next_state

	def is_possible_state(self, grid_attribute):
		return grid_attribute in {1, 0, -1}

	def can_act(self, state):
		return self.grid[state.row][state.column] == 0

	def reset_agent_state(self):
		self.agent_state = State(0, 0)

	def do_step(self, action):
		next_state, reward, is_terminal = self.transit(self.agent_state, action)

		if next_state is not None:
			self.agent_state = next_state

		return next_state, reward, is_terminal

	def transit(self, state, action):
		transition_probs = self.calc_transition_probs(state, action)

		if len(transition_probs) == 0:
			return None, None, True

		next_states = []
		probs = []

		for next_state, prob in transition_probs.items():
			next_states.append(next_state)
			probs.append(prob)

		next_state = np.random.choice(next_states, p=probs)
		reward = self.calc_reward(next_state)
		is_terminal = self.is_terminal_state(next_state)

		return next_state, reward, is_terminal
