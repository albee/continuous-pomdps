"""
A Gaussian representation of Belief
"""
import numpy as np

class Belief:
	def __init__(self, mean, covar):
		self.mean = mean
		self.covar = covar

	# Sample a Belief value
	def sample(self):
		sample = np.random.multivariate_normal(self.mean.flatten(), self.covar).reshape(-1, 1)
		return sample