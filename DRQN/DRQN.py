import numpy as np
import torch as T
import torch.nn as nn
from torch.functional import F

class DRQN(nn.Module):
	#--------------------------------------------------------------------------
	def __init__(self, state_dims, n_actions, gru_dims : int, gru_layers : int, 
		hidden_dims = [], lr = 1e-2, act_fn = nn.ReLU):
		"""
		gru_dims : size of gru hidden layers
		gru_layers : number of layers in gru 
		hidden_dims : list containing dims of subsequent hidden Linear layers (default [])
		"""

		super().__init__()

		self.state_dims = state_dims
		self.n_actions = n_actions
		self.gru_dims = gru_dims
		self.gru_layers = gru_layers
		self.hidden_dims = hidden_dims

		self.rnn = nn.GRU(input_size = state_dims,
			hidden_size = gru_dims,
			num_layers = gru_layers,
			batch_first = True)

		fc_layers = []
		for i in range(len(hidden_dims) - 1):
			fc_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
			fc_layers.append(nn.ReLU())

		self.fc_model = nn.Sequential(
			*fc_layers,
			nn.Linear(hidden_dims[-1], n_actions)
			)
		

		self.optimizer = T.optim.Adam(self.parameters(), lr = lr)
	#--------------------------------------------------------------------------
	def forward(self, state, h = None):
		"""
		hidden state h must be of shape (L, N, H)
		with :
			N the batch size
			L the number of layers
			H the dimension of GRU's hidden layers
		state must be of shape (N, SL, state_dims)
		with :
			SL : the sequence length
		"""
		if h is None :
			h = T.zeros(
				(
					self.gru_layers, #L
					state.shape[0],     #N
					self.gru_dims    #H
					)
				)
		x, h = self.rnn(state, h)
		q = self.fc_model(x)

		return q, h
