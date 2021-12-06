# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 21:17:19 2021

@author: eplan
"""

import numpy as np
import torch as T
import torch.nn as nn
from torch.functional import F

class Critic_Network(nn.Module):
    #--------------------------------------------------------------------------
    def __init__(self, state_dims, action_dims, hidden_dims, act_fn = nn.ReLU,
                 learning_rate = .001, device = "cpu"):
        
        super().__init__()
        
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.hidden_dims = hidden_dims
        
        dims = [state_dims + action_dims] + self.hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(act_fn())
        layers.append(nn.Linear(dims[-1], 1))
        
        self.network = nn.Sequential(*layers)
        
        self.optimizer = T.optim.Adam(self.parameters(), lr = learning_rate)
        
        self.device = device
        self.to(device)
    #--------------------------------------------------------------------------
    def forward(self, state, action):
        state_action = T.cat((state, action), 
                                dim = 1)
        
        return self.network(state_action)
    #--------------------------------------------------------------------------
    def save(self, filename):
        T.save(self.state_dict(), filename)
    #--------------------------------------------------------------------------
    def load(self, filename):
        self.load_state_dict(T.load(filename))
    
    
    
    
    
    
    
    
    
    
        
        
        