# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 21:17:14 2021

@author: eplan
"""

import numpy as np
import torch as T
import torch.nn as nn
from torch.functional import F

class Actor_Network(nn.Module):
    #--------------------------------------------------------------------------
    def __init__(self, state_dims, action_dims, fc1_dims, fc2_dims, 
                 learning_rate = .001, device = "cpu"):
        
        super().__init__()
        
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        
        self.fc1 = nn.Linear(state_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.output = nn.Linear(fc2_dims, action_dims)
        
        self.optimizer = T.optim.Adam(self.parameters(), lr = learning_rate)
        
        self.device = device
        self.to(self.device)
    #--------------------------------------------------------------------------
    def forward(self, state):
        action = self.fc1(state)
        action = F.relu(action)
        action = self.fc2(action)
        action = F.relu(action)
        action = self.output(action)
        action = T.tanh(action)
        
        return action
    #--------------------------------------------------------------------------
    def save(self, filename):
        T.save(self.state_dict(), filename)
    #--------------------------------------------------------------------------
    def load(self, filename):
        self.load_state_dict(T.load(filename))
        
    