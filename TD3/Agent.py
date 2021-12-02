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
	def __init__(self, state_dims, action_dims, batch_size, mem_size,
                 actor_hidden_dims, critic_hidden_dims, gamma = .98, 
                 lr_critic = .001, lr_actor = .001, tau = .003, 
                 action_noise = .2, target_noise = .2):

		self.state_dims = state_dims
		self.action_dims = action_dims
		self.hidden_dims = hidden_dims
		self.tau = tau
		self.gamma = gamma
		self.action_noise = action_noise
		self.target_noise = target_noise

		self.update_step = 0

		# Initialize replay buffer
		self.replay_buffer = Replay_Buffer(state_dims, action_dims, mem_size,
										   batch_size)

		# Initialize networks
		self.citic_1 = Critic_Network(state_dims, action_dims, critic_hidden_dims,
									  learning_rate = lr_critic)
		self.target_critic_1 = Critic_Network(state_dims, action_dims, 
									  critic_hidden_dims, learning_rate = lr_critic)
		
		self.critic_2 = Critic_Network(state_dims, action_dims, critic_hidden_dims,
									   learning_rate = lr_critic)
		self.target_critic_2 = Critic_Network(state_dims, action_dims, 
									   critic_hidden_dims, learning_rate = lr_critic)

		self.actor = Actor_Network(state_dims, action_dims, actor_hidden_dims,
			                       learning_rate = lr_actor)
		self.target_actor = Actor_Network(state_dims, action_dims, actor_hidden_dims,
			                       learning_rate = lr_actor)

		self.update_networks(1)
	#--------------------------------------------------------------------------
	def choose_action(self, state):

		state = T.tensor(state, dtype = T.float32)
		action = self.actor.forward(state)
		action += T.randn(action.shape) * self.action_noise

		return action.detach().numpy()
	#--------------------------------------------------------------------------
	def learn(self):
		if self.replay_buffer.counter < self.batch_size:
            return

        self.update_step += 1
        #=================get batch==================
        states, new_states, actions, rewards, dones = \
            self.replay_buffer.get_batch()
            
        states = T.tensor(states, dtype = T.float32)
        new_states = T.tensor(new_states, dtype = T.float32)
        actions = T.tensor(actions, dtype = T.float32)
        
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        self.target_actor.eval()
        self.target_critic_1.eval()
        self.target_critic_2.eval()

        #===========Compute critic predictions=======
        target_actions = self.target_actor.forward(new_states)
        #TODO : add clip to noise
        target_actions += T.randn(target_actions.shape) * self.target_noise

        # get predictions from both target critics : shape = (N, 1)
        critic_1_ns = self.target_critic_1.forward(new_states, target_actions)
        critic_2_ns = self.target_critic_2.forward(new_states, target_actions)
        # get min of 2 predictions : shape = (N, 1)
        ns_values = T.cat( (critic_1_ns, critic_2_ns), 1).min(dim = 1)

        critic_1_preds = self.critic_1.forward(states, actions)
        critic_2_preds = self.critic_2.forward(states, actions)
        #===========Compute critic targets===========
        targets = []
        for i in range(self.batch_size) :
            targets.append([ rewards[i] +  self.gamma * critic_ns_values[i] * 
                            (1 - dones[i])])
        targets = T.tensor(targets, dtype = T.float32)
        #===============Update critics===============
        self.critic_1.train()
        self.critic_1.optimizer.zero_grad()
        critic_1_loss = F.mse_loss(critic_1_preds, targets)
        critic_1_loss.backward()
        self.critic_1.optimizer.step()

        self.critic_2.train()
        self.critic_2.optimizer.zero_grad()
        critic_2_loss = F.mse_loss(critic_2_preds, targets)
        critic_2_loss.backward()
        self.critic_2.optimizer.step()

        if not self.update_step % 2 :
        	return
        #==============Update actor===================
        self.critic_1.eval()
        actor_vals = self.actor.forward(states)
        self.actor.train()
        self.actor.optimizer.zero_grad()
        actor_loss = - self.critic_1.forward(states, actor_vals)
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
			target.load_state_dict(target_params)

