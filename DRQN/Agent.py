import numpy as np
from DRQN import DRQN
from EpisodicMemory import EpisodicMemory
import torch as T
from torch.functional import F
from torch.distributions import Categorical

import gym

class Agent :
	#--------------------------------------------------------------------------
	def __init__(self, env, gru_dims, gru_layers, hidden_dims, 
		batch_size, mem_size, gamma = .99, tau = .003):
		
		self.state_dims = env.observation_space.shape[0]
		self.n_actions = env.action_space.n
		self.gamma = gamma
		self.tau = tau

		self.memory = EpisodicMemory(batch_size, mem_size)

		self.drqn = DRQN(self.state_dims, self.n_actions,
			gru_dims, gru_layers, hidden_dims)
		
		self.target_drqn = DRQN(self.state_dims, self.n_actions,
			gru_dims, gru_layers, hidden_dims)
		
		self.h = None

		self.update_networks(1)
	#--------------------------------------------------------------------------
	def choose_action(self, state, testing = False):
		q, h = self.drqn.forward(T.tensor(state.reshape(1, 1, -1), 
			dtype = T.float32), self.h)
		self.h = h
		q = q.view(-1)

		if testing :
			q = q.detach().numpy()
			a = np.argmax(q)
		else :
			dist = Categorical(logits = q)
			a = dist.sample().view(-1)
			a = a.detach().numpy()[0]
		return a
	#--------------------------------------------------------------------------
	def learn(self):
		
		batches = self.memory.get_batch()
		if batches is None :
			return

		states = T.tensor(np.array([b[0] for b in batches]), dtype = T.float32)
		actions = T.tensor(np.array([b[1] for b in batches]), dtype = T.int64).unsqueeze(2)
		states_ = T.tensor(np.array([b[2] for b in batches]), dtype = T.float32)
		rewards = T.tensor(np.array([b[3] for b in batches]), dtype = T.float32).unsqueeze(2)
		dones = T.tensor(np.array([b[4] for b in batches]), dtype = T.float32).unsqueeze(2)

		q_preds, _ = self.drqn.forward(states)
		q_preds = T.gather(q_preds, 2, actions)
		
		q_ns, _ = self.target_drqn.forward(states_)
		q_ns = T.max(q_ns, dim = 2, keepdim = True)[0]

		q_targets = rewards + self.gamma * (1 - dones) * q_ns

		loss = F.mse_loss(q_preds, q_targets)

		self.drqn.optimizer.zero_grad()
		loss.backward()
		self.drqn.optimizer.step()

		self.update_networks()

	#--------------------------------------------------------------------------
	def store_transition(self, state, action, new_state, reward, done):
		self.memory.store_transition(state, action, new_state, 
			reward, done)
	#--------------------------------------------------------------------------
	def reset_hidden(self):
		self.h = None
	#--------------------------------------------------------------------------
	def update_networks(self, tau = None):
		if tau is None:
			tau = self.tau

		for target_param, param in zip(self.target_drqn.parameters(),
		 							   self.drqn.parameters()):
			target_param.data.copy_(
				target_param.data * (1. - tau) + param.data * tau
				)
	#--------------------------------------------------------------------------
	def new_episode(self):
		self.memory.add_episode()



if __name__ == "__main__":
	pass


