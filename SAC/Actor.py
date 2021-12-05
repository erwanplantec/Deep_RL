import torch as T
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal

class Actor_Network(nn.Module):
	#--------------------------------------------------------------------------
	def __init__(self, state_dims, action_dims, hidden_dims, 
						nonlinearity = nn.ReLU, lr = 1e-2):

		super().__init__()

		self.state_dims = state_dims
		self.action_dims = action_dims
		self.hidden_dims = hidden_dims

		dims = [state_dims] + hidden_dims
		layers = []

		for i in range(len(dims) - 1):
			layers.append(nn.Linear(dims[i], dims[i + 1]))
			layers.append(nonlinearity())

		self.network = nn.Sequential(*layers)

		self.mu = nn.Linear(dims[-1], self.action_dims)
		self.sigma = nn.Linear(dims[-1], self.action_dims)

		self.optimizer = T.optim.Adam(self.parameters(), lr = lr)
	#--------------------------------------------------------------------------
	def forward(self, state):
		common = self.network(state)
		mu = self.mu(common)
		sigma = self.sigma(common)
		sigma = T.clamp(sigma, 1e-6, 1)

		return mu, sigma
	#--------------------------------------------------------------------------
	def save(self, filename):
		T.save(self.state_dict(), filename + ".pt")

