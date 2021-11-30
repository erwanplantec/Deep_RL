# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 17:20:25 2021

@author: eplan
"""

import numpy as np
import torch as T
import torch.nn as nn
from torch.functional import F
from torch.distributions import Categorical

from PolicyGradient.PolicyNetwork import PolicyNetwork

class Agent :
    #--------------------------------------------------------------------------
    def __init__(self, state_dims, n_actions, hidden_sizes, lr = 1e-2,
                 weights = "vanilla"):
        
        if weights == "vanilla":
            self._get_weights = self._traj_returns
        elif weights == "rtg":
            self._get_weights = self._rtg
        else :
            raise NameError(f"{weights} is not an available weight method")
        
        self.state_dims = state_dims
        self.n_actions = n_actions
        
        self.policy_net = PolicyNetwork(state_dims, n_actions, hidden_sizes,
                                        lr = lr)
        
        self.reset_memory()
    #--------------------------------------------------------------------------
    def choose_action(self, state):
        
        state = T.as_tensor(state, dtype = T.float32)
        
        pi = self.get_policy(state)
        
        action = pi.sample()
        
        return action.detach().numpy()
    #--------------------------------------------------------------------------
    def get_policy(self, state):
        return Categorical(
            logits = self.policy_net.forward(state)
            )
    #--------------------------------------------------------------------------
    def learn(self):
        states = T.as_tensor(np.array(self.states_mem), dtype = T.float32)
        actions = T.tensor(np.array(self.action_mem))
        
        log_probs = self.get_policy(states).log_prob(actions)
        
        self.policy_net.optimizer.zero_grad()
        
        weights = T.tensor(self._get_weights())
        
        loss = - ( log_probs * weights ).mean()
        
        loss.backward()
        self.policy_net.optimizer.step()
    #--------------------------------------------------------------------------
    def remember(self, state, reward, action):
        self.states_mem.append(state)
        self.rews_mem.append(reward)
        self.action_mem.append(action)
    #--------------------------------------------------------------------------
    def reset_memory(self):
        self.states_mem = []
        self.rews_mem = []
        self.action_mem = []
        self.episode_indexes = []
    #--------------------------------------------------------------------------
    def add_episode(self):
        self.episode_indexes.append(len(self.states_mem))
    #--------------------------------------------------------------------------
    def _traj_returns(self):
        returns = []
        for i in range(len(self.episode_indexes) - 1):
            start, end = self.episode_indexes[i], self.episode_indexes[i + 1]
            episode_return = np.sum(self.rews_mem[start : end])
            returns += [episode_return] * ( end - start )
        return returns
    #--------------------------------------------------------------------------
    def _rtg(self):
        rtgs = []
        for i in range(len(self.episode_indexes) - 1):
            start, end = self.episode_indexes[i], self.episode_indexes[i + 1]
            for i in range(end - start):
                # Compute sum of rewards until end of trajectory
                rtg = np.sum(self.rews_mem[start + i : end])
                rtgs.append(rtg)
                
        return rtgs
    #--------------------------------------------------------------------------
    def save(self, filename):
        self.policy_net.save(filename)
        
        
        