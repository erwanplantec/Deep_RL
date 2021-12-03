import torch as T
import torch.nn as nn
from torch.functional import F


class Actor_Network(nn.Module):
	#--------------------------------------------------------------------------
	def __init__(self, state_dims, n_actions, hidden_dims, 
		learning_rate = 1e-2, act_fn = nn.ReLU):

		super().__init__()

		self.state_dims = state_dims
		self.n_actions = n_actions

		dims = [state_dims] + hidden_dims
		layers = []
		for i in range(len(dims) - 1):
			layers.append(nn.Linear(dims[i], dims[i + 1]))
			layers.append(act_fn())
		layers.append(nn.Linear(dims[-1], n_actions))

		self.network = nn.Sequential(*layers)

		self.optimizer = T.optim.Adam(self.parameters(), 
			lr = learning_rate)
	#--------------------------------------------------------------------------
	def forward(self, state):
		return self.network(state)
	#--------------------------------------------------------------------------
	def save(self, filename):
		self.save(self.state_dict(), filename)
	#--------------------------------------------------------------------------
	def load(self, filename):
		self.load_state_dict(T.load(filename))