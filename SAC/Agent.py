import numpy as np

import torch as T
from torch.distributions.normal import Normal
from torch.functional import F

from Critic import Critic_Network, Value_Network
from Actor import Actor_Network
from ReplayBuffer import Replay_Buffer


class Agent :
	#--------------------------------------------------------------------------
	def __init__(self, env, actor_hidden_dims, critic_hidden_dims,
		mem_size, batch_size, lr_critic = 1e-2, lr_actor = 1e-2, 
		gamma = .99, alpha = .2, tau = .005, r_scale = 2):
		
		self.state_dims = env.observation_space.shape[0]
		self.action_dims = env.action_space.shape[0]
		self.max_actions = env.action_space.high
		self.gamma = gamma
		self.alpha = alpha
		self.tau = tau
		self.r_scale = r_scale

		self.replay_buffer = Replay_Buffer(self.state_dims, self.action_dims,
			mem_size, batch_size)

		self.Q_1 = Critic_Network(self.state_dims, self.action_dims,
			critic_hidden_dims, lr = lr_critic)

		self.Q_2 = Critic_Network(self.state_dims, self.action_dims,
			critic_hidden_dims, lr = lr_critic)

		self.V = Value_Network(self.state_dims, critic_hidden_dims,
			lr = lr_critic)

		self.target_V = Value_Network(self.state_dims, critic_hidden_dims,
			lr = lr_critic)

		self.actor = Actor_Network(self.state_dims, self.action_dims,
			actor_hidden_dims, lr = lr_actor)

		self.update_networks(1)
	#--------------------------------------------------------------------------
	def choose_action(self, state):
		state = T.tensor([state], dtype = T.float)

		action, lp = self.sample_action(state, reparameterize = False)

		return action.detach().numpy()[0]
	#--------------------------------------------------------------------------
	def sample_action(self, state, reparameterize = False):
		"""Return an action and associated log_probs"""
		
		mu, sigma = self.actor.forward(state)
		
		dist = Normal(mu, sigma)

		u = dist.rsample() if reparameterize else dist.sample()

		a = T.tanh(u) * T.tensor(self.max_actions)

		log_prob = dist.log_prob(u) # See paper appendix

		log_prob -= T.log(1 - a.pow(2) + 1e-6)

		log_prob = log_prob.sum(1, keepdim=True)

		return a, log_prob
	#--------------------------------------------------------------------------
	def store_transition(self, state, action, new_state, reward, done):
		self.replay_buffer.store_transition(state, action, new_state,
			reward, done)
	#--------------------------------------------------------------------------
	def learn(self):
		if self.replay_buffer.counter < self.replay_buffer.batch_size :
			return

		states, new_states, actions, rewards, dones = \
									self.replay_buffer.get_batch()
		states = T.tensor(states, dtype = T.float32)
		actions = T.tensor(actions, dtype = T.float32)
		new_states = T.tensor(new_states, dtype = T.float32)
		rewards = T.tensor(rewards, dtype = T.float32)
		dones = T.tensor(dones, dtype = T.bool)

		values = self.V.forward(states).view(-1)
		values_ns = self.target_V.forward(new_states).view(-1)
		values_ns[dones] = .0

		#===================Update V net==================
		actions_pi, log_probs = self.sample_action(states, 
			reparameterize = False)
		log_probs = log_probs.view(-1)

		Q_values_npi = T.min(	
			self.Q_1.forward(states, actions_pi), 
			self.Q_2.forward(states, actions_pi)
			) 
		Q_values_npi = Q_values_npi.view(-1)
		
		V_target = Q_values_npi - log_probs
		V_loss = .5 * F.mse_loss(values, V_target)
		
		self.V.optimizer.zero_grad()
		
		V_loss.backward(retain_graph = True)
		
		self.V.optimizer.step()

		#==================Update Q_nets==================
		self.Q_1.optimizer.zero_grad()
		self.Q_2.optimizer.zero_grad()

		Q_1_opi = self.Q_1.forward(states, actions).view(-1)
		Q_2_opi = self.Q_2.forward(states, actions).view(-1)

		Q_target = self.r_scale * rewards + self.gamma * values_ns

		Q1_loss = .5 * F.mse_loss(Q_1_opi, Q_target)
		Q2_loss = .5 * F.mse_loss(Q_2_opi, Q_target)

		Q_loss = Q1_loss + Q2_loss

		Q_loss.backward()

		self.Q_1.optimizer.step()
		self.Q_2.optimizer.step()

		#==================Update policy==================
		actions_pi, log_probs = self.sample_action(states, 
										reparameterize = True)
		log_probs = log_probs.view(-1)

		Q_values = T.min(
			self.Q_1.forward(states, actions_pi),
			self.Q_2.forward(states, actions_pi)
			)
		Q_values = Q_values.view(-1)

		pi_loss = T.mean( log_probs - Q_values )

		self.actor.optimizer.zero_grad()
		
		pi_loss.backward(retain_graph = True)
		
		self.actor.optimizer.step()

		#================Update target nets==============
		self.update_networks()

	#--------------------------------------------------------------------------
	def update_networks(self, tau = None):
		if tau is None:
			tau = self.tau

		for target_param, param in zip(self.target_V.parameters(),
		 							   self.V.parameters()):
			target_param.data.copy_(
				target_param.data * (1. - tau) + param.data * tau
				)
	#--------------------------------------------------------------------------
	def save(self, filename):
		self.Q_1.save(filename + "_Q1")
		self.Q_2.save(filename + "_Q2")
		self.V.save(filename + "_V")
		self.target_V.save(filename + "_V_target")
		self.actor.save(filename + "_actor")