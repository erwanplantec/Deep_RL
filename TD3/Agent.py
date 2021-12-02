import numpy as np

import torch as T
import torch.nn as nn
from torch.functional import F
from torch.optim import Adam

from Actor import Actor_Network
from Critic import Critic_Network
from ReplayBuffer import Replay_Buffer

class Agent :
	#--------------------------------------------------------------------------
	def __init__(self, env, batch_size, mem_size,
                 actor_hidden_dims, critic_hidden_dims, gamma = .98, 
                 lr_critic = .001, lr_actor = .001, tau = .003, 
                 action_noise = .2, target_noise = .2):

		self.state_dims = env.observation_space.shape[0]
		self.action_dims = env.action_space.shape[0]
		self.hidden_dims_actor = actor_hidden_dims
		self.hidden_dims_critic = critic_hidden_dims
		self.tau = tau
		self.gamma = gamma
		self.action_noise = action_noise
		self.target_noise = target_noise
		self.batch_size = batch_size
		self.max_actions = env.action_space.high
		self.min_actions = env.action_space.low
		self.timesteps = 0

		self.update_step = 0

		# Initialize replay buffer
		self.replay_buffer = Replay_Buffer(self.state_dims, self.action_dims, mem_size,
										   batch_size)

		# Initialize networks
		self.critic_1 = Critic_Network(self.state_dims, self.action_dims, critic_hidden_dims,
									  learning_rate = lr_critic)
		self.target_critic_1 = Critic_Network(self.state_dims, self.action_dims, 
									  critic_hidden_dims, learning_rate = lr_critic)
		
		self.critic_2 = Critic_Network(self.state_dims, self.action_dims, critic_hidden_dims,
									   learning_rate = lr_critic)
		self.target_critic_2 = Critic_Network(self.state_dims, self.action_dims, 
									   critic_hidden_dims, learning_rate = lr_critic)

		self.actor = Actor_Network(self.state_dims, self.action_dims, actor_hidden_dims,
			                       learning_rate = lr_actor)
		self.target_actor = Actor_Network(self.state_dims, self.action_dims, actor_hidden_dims,
			                       learning_rate = lr_actor)

		self.update_networks(1)
	#--------------------------------------------------------------------------
	def choose_action(self, state, noise = True):
		
		self.timesteps += 1
		
		if self.timesteps < 1000:
			action = T.randn((self.action_dims,)) * self.action_noise
		else :
			state = T.tensor(state, dtype = T.float32)
			action = self.actor.forward(state)
		
		if noise :
			noise = T.clamp(T.randn(action.shape) * self.action_noise,
							-.5, .5)
			action += noise

		return T.clamp(action, self.min_actions[0], 
			   		   self.max_actions[0]).detach().numpy()
	#--------------------------------------------------------------------------
	def learn(self):

		T.autograd.set_detect_anomaly(True)
		
		if self.replay_buffer.counter < self.batch_size :
			return None
        
		self.update_step += 1
		#=================get batch==================
		states, new_states, actions, rewards, dones = \
		    self.replay_buffer.get_batch()            
		states = T.tensor(states, dtype = T.float32)
		new_states = T.tensor(new_states, dtype = T.float32)
		actions = T.tensor(actions, dtype = T.float32)
		rewards = T.tensor(rewards, dtype = T.float32).unsqueeze(1)
		dones = T.tensor(dones, dtype = T.float32).unsqueeze(1)

		#===========Compute critic predictions=======
		target_actions = self.target_actor.forward(new_states)
		# Do target policy smoothing
		target_actions = target_actions + T.clamp(T.randn(target_actions.shape)
											* self.target_noise, -.5, .5) 
		target_actions = T.clamp(target_actions, self.min_actions[0], self.max_actions[0])

		# get new states values from both target critics : shape = (N, 1)
		critic_1_ns = self.target_critic_1.forward(new_states, target_actions)
		critic_2_ns = self.target_critic_2.forward(new_states, target_actions)
		# get min of 2 predictions : shape = (N, 1)
		ns_values = T.min(critic_1_ns, critic_2_ns)

		critic_1_preds = self.critic_1.forward(states, actions)
		critic_2_preds = self.critic_2.forward(states, actions)
		#===========Compute critic targets===========
		targets = rewards + (1 - dones) * (self.gamma * ns_values) 
		#===============Update critics===============

		self.critic_1.optimizer.zero_grad()
		self.critic_2.optimizer.zero_grad()
		
		critic_1_loss = F.mse_loss(critic_1_preds, targets)
		critic_2_loss = F.mse_loss(critic_2_preds, targets)
		
		loss = critic_1_loss + critic_2_loss

		loss.backward()

		self.critic_1.optimizer.step()
		self.critic_2.optimizer.step()

		#==============Update actor===================
		self.actor.optimizer.zero_grad()
		actor_loss = - self.critic_1.forward(states, 
			self.actor.forward(states))
		actor_loss = T.mean(actor_loss)
		actor_loss.backward()
		self.actor.optimizer.step()

		#=========Update target networks=============
		self.update_networks()
	#--------------------------------------------------------------------------
	def update_networks(self, tau = None):
		if tau is None :
			tau = self.tau

		for net in ("critic_1", "critic_2", "actor"):
			network = getattr(self, net)
			target = getattr(self, "target_" + net)

			target_params = dict(target.named_parameters())
			network_params = dict(network.named_parameters())

			for key in network_params.keys():
				target_params[key] = tau * network_params[key].clone() \
							+ (1 - tau) * target_params[key].clone()
			target.load_state_dict(target_params.copy())
	#--------------------------------------------------------------------------
	def store_transition(self, state, action, new_state, reward, done):
		self.replay_buffer.store_transition(state, action, new_state,
											reward, done)
	#--------------------------------------------------------------------------
	def save(self, filename):
		for net in ("critic_1", "critic_2", "actor"):
			net, targ = getattr(self, net), getattr(self, "target_" + net)
			net.save(filename + '_' + net)
			targ.save(filename + '_' + net + "_target")
	#--------------------------------------------------------------------------
	def load(self, filenames):
		for net in ("critic_1", "critic_2", "actor"):
			net, targ = getattr(self, net), getattr(self, "target_" + net)
			net.load(filename + '_' + net)
			targ.load(filename + '_' + net + "_target")

