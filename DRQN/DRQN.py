import numpy as np
import torch as T
import torch.nn as nn
from torch.functional import F

class DRQN(nn.Module):
	#--------------------------------------------------------------------------
	def __init__(self, state_dims, n_actions, hidden_dims : int, 
		hidden_layers : int, lr = 1e-2):
		"""
		hidden_dims (int): dimension of GRU hidden layers
		hidden_layers (int): number of hidden layers of GRU unit
		"""

		super().__init__()

		self.state_dims = state_dims
		self.n_actions = n_actions
		self.hidden_dims = hidden_dims
		self.hidden_layers = hidden_layers

		self.rnn = nn.GRU(input_size = state_dims,
			hidden_size = hidden_dims,
			num_layers = hidden_layers,
			batch_first = True)
		
		self.q = nn.Linear(hidden_dims, n_actions)

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
					self.hidden_layers, #L
					state.shape[0],     #N
					self.hidden_dims    #H
					)
				)
		x, h = self.rnn(state, h)
		q = self.q(x)

		return q, h
