import torch as T
import torch.nn as nn
import numpy as np

class Critic_Network(nn.Module):
	#--------------------------------------------------------------------------
	def __init__(self, state_dims, action_dims, hidden_dims, lr = 1e-2,
		nonlinearity = nn.ReLU):
		
		super().__init__()

		self.state_dims = state_dims
		self.action_dims = action_dims
		self.hidden_dims = hidden_dims

		dims = [state_dims + action_dims] + hidden_dims
		layers = []

		for i in range(len(dims) - 1):
			layers.append(nn.Linear(dims[i], dims[i + 1]))
			layers.append(nonlinearity())
		layers.append(nn.Linear(dims[-1], 1))

		self.network = nn.Sequential( *layers )

		self.optimizer = T.optim.Adam(self.parameters(), lr = lr)
	#--------------------------------------------------------------------------
	def forward(self, state, action):
		state_action = T.cat((state, action), dim = 1)

		Q_sa = self.network(state_action)

		return Q_sa
	#--------------------------------------------------------------------------
	def save(self, filename):
		T.save(self.state_dict(), filename + ".pt")

class Value_Network(nn.Module):
	#--------------------------------------------------------------------------
	def __init__(self, state_dims, hidden_dims, lr = 1e-2, 
		nonlinearity = nn.ReLU):

		super().__init__()

		self.state_dims = state_dims
		self.hidden_dims = hidden_dims

		dims = [state_dims] + hidden_dims
		layers = []

		for i in range(len(dims) - 1):
			layers.append(nn.Linear(dims[i], dims[i + 1]))
			layers.append(nonlinearity())
		layers.append(nn.Linear(dims[-1], 1))

		self.network = nn.Sequential(*layers)

		self.optimizer = T.optim.Adam(self.parameters(), lr = lr)
	#--------------------------------------------------------------------------
	def forward(self, state):
		V_s = self.network(state)
		return V_s
	#--------------------------------------------------------------------------
	def save(self, filename):
		T.save(self.state_dict(), filename + ".pt")