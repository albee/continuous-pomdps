"""
A node of a belief, action, or observation.
"""
import numpy as np
import matplotlib.pyplot as plt

class HistoryNode:
	def __init__(self, data, parent, is_action_node):
		self.parent = parent
		self.data = data  # either belief (root), action or observation
		self.children = []
		self.num_visits = 1.0  # float, should it be 0?
		self.is_action_node = is_action_node  # either an action node, or a belief node
		self.Q = 0
		self.num_gens = 0.0
		self.associated_states = []

	def get_children(self):
		return self.children()

	def num_children(self):
		return len(self.children)

	def add_child(self, data, is_action_node):
		parent = self
		new_child = HistoryNode(data, self, is_action_node)
		self.children.append(new_child)
		return new_child

	# If an action_node, choose an observation from among children
	def choose_obs(self):
		if not self.is_action_node:
			print "This should be an action node!"
			return None
		M_count = []
		for obs_node in self.children:
			M_count.append(obs_node.num_gens)
		M_total = sum(M_count)
		probs = np.array(M_count)/M_total
		o_idx = np.random.choice(len(self.children), 1, p=probs)[0]  # choose an observation with probability by occurence rate
		obs_node = self.children[o_idx]
		return obs_node

	# If an obs_nose, choose a next_state from possible beliefs (found via particle filter)
	def choose_simulated_state(self):
		if self.is_action_node:
			print "This should be an obs node!"
			return None
		probs = np.ones(len(self.associated_states))/len(self.associated_states)
		state_idx = np.random.choice(len(self.associated_states), 1, p=probs)[0]  # choose an observation with probability by occurence rate
		return self.associated_states[state_idx]

	def plot_tree(self, fig):
		plt.figure(fig.number)
		points = []
		for child in self.children:
			if child.is_action_node:
				child.plot_tree(fig)
			else:
				for state in child.associated_states:
					# print "plotting", state[0], state[1]
					last = plt.plot(state[0], state[1], marker='o', markersize=3, color="red")
				child.plot_tree(fig)
		return fig