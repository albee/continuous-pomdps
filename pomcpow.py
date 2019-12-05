"""
Performs POMCPOW on a continuous POMDP (continuous state, action, and observation spaces).
"""

import common

"""
state: sampled state
belief: the initial belief
max_depth: max permitted exploration depth
history: a history from the root belief (b, a, o, ...)
depth: depth to explore
max_depth:  max dpeth to explore
children: a list of children of a node
num_gens: number of times history has been generated
total: total accumulated reward along a history
"""
def simulate(state, belief_node, depth, sim):
	if depth == 0:  # max_depth reached via decrements
		return 0

	# choose an action
	action_node = common.action_prog_widen(belief_node, sim)

	# DPW limit
	k_obs = 1
	alpha_obs = 1
	N = action_node.num_visits
	child_obs_limit = k_obs*(N**alpha_obs)

	if action_node.num_children() <= child_obs_limit:  # okay to simulate
		state_next, obs, reward = sim.set_state_and_simulate(state, action_node.data)
		obs_node = action_node.add_child(obs, False)  # add an observation node, need to check if it already exists
		obs_node.num_gens += 1
		obs_node.associated_states.append(state_next)

		if obs_node.num_gens == 1:  # first time encountering this hao
			total = reward + sim.gamma*simulate(state_next, obs_node, depth-1, sim)  # THIS SHOULD BE ROLLOUT!!!
		else:
			total = reward + sim.gamma*simulate(state_next, obs_node, depth-1, sim)

	else:  # new observation is rejected, too many children: use an observation-state_next simulation from the tree
		obs_node = action_node.choose_obs()
		obs = obs_node.data
		state_next = obs_node.choose_simulated_state()
		reward = sim.set_state_and_get_reward(state, action_node.data, state_next)
		total = reward + sim.gamma*simulate(state_next, obs_node, depth-1)

	belief_node.num_visits += 1
	action_node.num_visits += 1
	action_node.Q += ((total-action_node.Q)/action_node.num_visits)
	return total

# Runs a simulation
def rollout(state, belief_node, depth, sim):
	return None