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
    def __init__(self, state_dims, action_dims, fc1_dims, fc2_dims, 
                 learning_rate = .001, device = "cpu"):
        
        super().__init__()
        
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.fc1_size = fc1_dims
        self.fc2_size = fc2_dims
        
        self.fc1 = nn.Linear(state_dims + action_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.output = nn.Linear(fc2_dims, 1)
        
        self.optimizer = T.optim.Adam(self.parameters(), lr = learning_rate)
        
        self.device = device
        self.to(device)
    #--------------------------------------------------------------------------
    def forward(self, state, action):
        state_action = T.concat((state, action), 
                                dim = len(state.shape) - 1)
        q = self.fc1(state_action)
        q = F.relu(q)
        q = self.fc2(q)
        q = F.relu(q)
        q = self.output(q)
        
        return q
    #--------------------------------------------------------------------------
    def save(self, filename):
        T.save(self.state_dict(), filename)
    #--------------------------------------------------------------------------
    def load(self, filename):
        self.load_state_dict(T.load(filename))
    
    
    
    
    
    
    
    
    
    
        
        
        