# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 17:56:15 2021

@author: eplan
"""

import numpy as np
import torch as T
import torch.nn as nn
from torch.functional import F

from Critic import Critic_Network
from Actor import Actor_Network
from ReplayBuffer import Replay_Buffer
from OUNoise import OUActionNoise

#==============================================================================
#==============================================================================
#=====================Deep Deterministic Policy Gradient=======================
#==============================================================================
#==============================================================================

class Agent :
    #--------------------------------------------------------------------------
    def __init__(self, env, batch_size, mem_size,
                 actor_hidden_dims, critic_hidden_dims, gamma = .98, 
                 lr_critic = .001, lr_actor = .001, tau = .003):
        
        self.state_dims = env.observation_space.shape[0]
        self.action_dims = env.action_space.shape[0]
        self.max_actions = T.tensor(env.action_space.high)
        self.min_actions = T.tensor(env.action_space.low)

        self.critic = Critic_Network(self.state_dims, self.action_dims, 
                                     critic_hidden_dims, 
                                     learning_rate = lr_critic)
        
        self.target_critic = Critic_Network(self.state_dims, self.action_dims, 
                                            critic_hidden_dims)
        
        self.actor = Actor_Network(self.state_dims, self.action_dims, 
                                   actor_hidden_dims, 
                                   learning_rate = lr_actor)
        
        self.target_actor = Actor_Network(self.state_dims, self.action_dims, 
                                          actor_hidden_dims)
        
        self.replay_buffer = Replay_Buffer(self.state_dims, self.action_dims, 
                                           mem_size, batch_size)

        self.ou_noise = OUActionNoise(mu = np.zeros(self.action_dims))
        
        self.batch_size = batch_size
        
        self.gamma = gamma
        
        self.tau = tau
        
        self.update_networks(1)
    #--------------------------------------------------------------------------
    def choose_action(self, state, noise = False):
        
        state = T.tensor( state , dtype = T.float32 ).to(self.actor.device)
        action = self.actor.forward(state)

        if noise :
        	noise = T.tensor(self.ou_noise(), dtype = T.float32)
        	action += noise
        
        return action.detach().numpy()
    #--------------------------------------------------------------------------
    def store_transition(self, state, action, new_state, reward, done):
        
        self.replay_buffer.store_transition(state, action, new_state, 
                                            reward, done)
    #--------------------------------------------------------------------------
    def learn(self):
        if self.replay_buffer.counter < self.batch_size:
            return
        #=================get batch==================
        states, new_states, actions, rewards, dones = \
            self.replay_buffer.get_batch()
            
        states = T.tensor(states, dtype = T.float32)
        new_states = T.tensor(new_states, dtype = T.float32)
        actions = T.tensor(actions, dtype = T.float32)
        dones = T.tensor(dones, dtype = T.float32).view(-1, 1)
        rewards = T.tensor(rewards, dtype = T.float32).view(-1, 1)
        
        #===========Compute critic predictions=======
        target_actions = self.target_actor.forward(new_states)
        critic_ns_values = self.target_critic.forward(new_states, target_actions)
        critic_pred = self.critic.forward(states, actions)
        
        #===========Compute critic targets===========
        targets = rewards + self.gamma * (1 - dones) * critic_ns_values
        
        #==========Train the critic==================
        critic_loss = F.mse_loss(critic_pred, targets)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        
        #===========Train the actor==================
        actor_values = self.actor.forward(states)
        actor_loss = - self.critic.forward(states, actor_values) 
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        
        #=========Update target networks=============
        self.update_networks()
    #--------------------------------------------------------------------------
    def update_networks(self, tau = None):
        
        if tau is None : 
            tau = self.tau
        
        actor_params = dict(self.actor.named_parameters())
        critic_params = dict(self.critic.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())
        target_critic_params = dict(self.target_critic.named_parameters())
        
        for key in critic_params.keys():
            critic_params[key] = tau * critic_params[key].clone() + \
                (1 - tau) * target_critic_params[key].clone()
                
        self.target_critic.load_state_dict(critic_params)
        
        for key in actor_params.keys():
            actor_params[key] = tau * actor_params[key].clone() + \
                (1 - tau) * target_actor_params[key].clone()
        
        self.target_actor.load_state_dict(actor_params)
    #--------------------------------------------------------------------------
    def save(self, filename):
        self.actor.save(filename + "_actor")
        self.critic.save(filename + "_critic")
        
        self.target_actor.save(filename + "_actor_t")
        self.target_critic.save(filename + "_critic_t")
    #--------------------------------------------------------------------------
    def load(self, filename):
        self.actor.load(filename + "_actor")
        self.critic.load(filename + "_critic")
        
        self.target_actor.load(filename + "_actor_t")
        self.target_critic.load(filename + "_critic_t")
        
