"""
This module contains common procedures shared by both POMCPOW and PFT-DPW
"""

from HistoryNode import HistoryNode
import pomcp_dpw
import pomcpow
from Belief import Belief
import numpy as np
import math

# Entrypoint to the solver. n is the number of particles to draw. Returns the best action.
def plan(belief, sim, n):
	max_depth = 20
	root = HistoryNode(belief, None, False)

	for i in range(n):
		state = belief.sample()
		# total = pomcp_dpw.simulate(state, root, max_depth, sim)
		total = pomcpow.simulate(state, root, max_depth, sim)

	# select best Q return from action children
	best_Q = -np.inf
	best_action = None
	for action_node in root.children:
		Q = action_node.Q
		if Q > best_Q:
			best_Q = Q
			best_action = action_node.data
	return best_action, root

# A UCB-based action selection. Returns the best action action, with an exploration weighting.
def action_prog_widen(history_node, sim):
	# DPW limit
	k_actions = 3.0
	alpha_actions = 0.2/30.0
	N = history_node.num_visits
	child_action_limit = k_actions*(N**alpha_actions)

	if history_node.num_children() <= child_action_limit:  # okay to generate new action
		action = sim.generate_action()  # select a random action to use from the continuous action space
		history_node.add_child(action, True)

	best_UCB = -np.inf
	best_action_node = None
	c = 30.0  # exploration weighting
	for action_node in history_node.children:  # select from the action pool
		UCB = action_node.Q + c*math.sqrt(math.log(history_node.num_visits)/action_node.num_visits)
		if UCB > best_UCB:
			best_UCB = UCB
			best_action_node = action_node
	return best_action_node