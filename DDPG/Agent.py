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

#==============================================================================
#==============================================================================
#=====================Deep Deterministic Policy Gradient=======================
#==============================================================================
#==============================================================================

class DDPG_Agent :
    #--------------------------------------------------------------------------
    def __init__(self, state_dims, action_dims, batch_size, mem_size,
                 fc1_actor_dims, fc2_actor_dims, fc1_critic_dims, 
                 fc2_critic_dims, gamma = .98, lr_critic = .001, 
                 lr_actor = .001, tau = .003):
        
        self.critic = Critic_Network(state_dims, action_dims, fc1_critic_dims, 
                                     fc2_critic_dims)
        self.target_critic = Critic_Network(state_dims, action_dims, 
                                            fc1_critic_dims, fc2_critic_dims)
        
        self.actor = Actor_Network(state_dims, action_dims, fc1_actor_dims, 
                                   fc2_actor_dims)
        self.target_actor = Actor_Network(state_dims, action_dims, fc1_actor_dims, 
                                   fc2_actor_dims)
        
        self.replay_buffer = Replay_Buffer(state_dims, action_dims, mem_size, 
                                           batch_size)
        
        self.batch_size = batch_size
        
        self.gamma = gamma
        
        self.tau = tau
        
        self.update_networks(1)
    #--------------------------------------------------------------------------
    def choose_action(self, state):
        
        state = T.tensor( state , dtype = T.float32 ).to(self.actor.device)
        action = self.actor.forward(state)
        
        return action.detach().numpy()
    #--------------------------------------------------------------------------
    def store_transition(self, state, action, new_state, reward, done):
        
        self.replay_buffer.store_transition(state, action, new_state, 
                                            reward, done)
    #--------------------------------------------------------------------------
    def learn(self):
        #=================get batch==================
        states, new_states, actions, rewards, dones = \
            self.replay_buffer.get_batch()
            
        states = T.tensor(states, dtype = T.float32)
        new_states = T.tensor(new_states, dtype = T.float32)
        actions = T.tensor(actions, dtype = T.float32)
        
        self.actor.eval()
        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()
        
        #===========Compute critic predictions=======
        target_actions = self.target_actor.forward(new_states)
        critic_ns_values = self.target_critic.forward(new_states, target_actions)
        critic_pred = self.critic.forward(states, actions)
        
        #===========Compute critic targets===========
        targets = []
        for i in range(self.batch_size) :
            targets.append([ rewards[i] +  self.gamma * critic_ns_values[i] * 
                            (1 - dones[i])])
        targets = T.tensor(targets, dtype = T.float32)
        
        #==========Train the critic==================
        self.critic.train()
        self.critic.zero_grad()
        critic_loss = F.mse_loss(critic_pred, targets)
        critic_loss.backward()
        self.critic.optimizer.step()
        
        #===========Train the actor==================
        self.critic.eval()
        actor_values = self.actor.forward(states)
        self.actor.train()
        self.actor.optimizer.zero_grad()
        actor_loss = - self.critic.forward(states, actor_values) 
        actor_loss = T.mean(actor_loss)
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
    def noise(self, action):
        pass
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
        


        

        
        

        
        
    
        
        
            
            
        
        
        
        
        
        
        
        
        
        
        
    