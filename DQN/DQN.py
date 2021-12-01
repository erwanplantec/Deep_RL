# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 22:55:28 2021

@author: eplan
"""

import numpy as np
import torch as T
import torch.nn as nn
from torch.functional import F

class DQN(nn.Module):
    #--------------------------------------------------------------------------
    def __init__(self, state_dims, n_actions, hidden_dims, act_fn = nn.ReLU,
                 learning_rate = .0001, device = "cpu"):
        
        super().__init__()
        
        self.state_dims = state_dims
        self.n_actions = n_actions
        self.hidden_dims = hidden_dims
        
        dims = [state_dims] + hidden_dims 
        layers = []
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act_fn())
        layers.append(nn.Linear(dims[-1], n_actions))
        
        self.net = nn.Sequential(*layers)
        
        self.optimizer = T.optim.Adam(self.parameters(), lr = learning_rate)
        
        self.device = device
        self.to(device)
    #--------------------------------------------------------------------------
    def forward(self, state):
        state = state.to(self.device)
        return self.net(state)
    #--------------------------------------------------------------------------
    def save(self, filename):
        T.save(self.state_dict(), filename)
    #--------------------------------------------------------------------------
    def load(self, filename):
        self.load_state_dict(T.load(filename))
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    