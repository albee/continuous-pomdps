"""
Performs POMCPOW on a continuous POMDP (continuous state, action, and observation spaces).
"""

import common
import numpy as np
import math

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
def simulate(state, belief_node, depth, sim, covar_factor):
	if depth == 0:  # max_depth reached via decrements
		return 0

	# choose an action
	action_node = common.action_prog_widen(belief_node, sim)
	state_next, obs, reward, likelihood, covar_factor = sim.set_state_and_simulate(state, action_node.data, covar_factor)  # vast majority of time is spent here

	# DPW limit
	k_obs = 5.0
	alpha_obs = 1.0/100
	N = action_node.num_visits
	child_obs_limit = k_obs*(N**alpha_obs)

	if action_node.num_children() <= child_obs_limit:  # simulated
		obs_node = action_node.add_child(obs, False)  # add an observation node
		obs_node.num_gens += 1
		simulated = True
	else:  # do not simulate, choose an existing observation
		obs_node = action_node.choose_obs()
		obs = obs_node.data
		likelihood = sim.likelihood_given_obs_and_state(obs, state_next)
		simulated = False

	# add to the weighted particle collection
	obs_node.associated_states.append(state_next)
	obs_node.weightings.append(likelihood)  # likelihood of the state observation we obtained, given s,a,s'

	if simulated:  # was simulated
		total = reward + sim.gamma*random_rollout(state_next, obs_node, depth-1, sim, covar_factor)  # THIS SHOULD BE ROLLOUT!!!			
	else:  # was not simulated, was chosen from existing observations
		state_next = obs_node.choose_simulated_state_by_weight()
		reward = sim.set_state_and_get_reward(state, action_node.data, state_next)
		total = reward + sim.gamma*simulate(state_next, obs_node, depth-1, sim, covar_factor)

	belief_node.num_visits += 1
	action_node.num_visits += 1
	action_node.Q += ((total-action_node.Q)/action_node.num_visits)
	return total

# Runs a simulation with a defined rollout policy
def rollout(state, belief_node, depth, sim, covar_factor):
	best_reward = -np.inf
	init_state = state
	for i in range(6):
		accumulated_reward = 0
		state = init_state
		action = sim.generate_action()
		for j in range(int(math.ceil(depth/1.0))):  # play out a single action
			state_next, obs_node, reward, _ , covar_factor= sim.set_state_and_simulate(state, action, covar_factor)
			accumulated_reward += sim.gamma*reward
		if accumulated_reward >= best_reward:
			best_reward = accumulated_reward
	return best_reward

# Runs a simulation with a defined rollout policy
def random_rollout(state, belief_node, depth, sim, covar_factor):
	accumulated_reward = 0
	for i in range(depth):
		action = sim.generate_action()
		state_next, obs_node, reward, _ , covar_factor= sim.set_state_and_simulate(state, action, covar_factor)
		accumulated_reward += sim.gamma*reward
	return accumulated_reward