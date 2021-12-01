# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 17:07:49 2021

@author: eplan
"""


import numpy as np
import torch as T
import torch.nn as nn
from torch.functional import F


class PolicyNetwork(nn.Module):
    #--------------------------------------------------------------------------
    def __init__(self, state_dims, n_actions, hidden_sizes, act_fn = nn.ReLU,
                 lr = 1e-2):
        
        super().__init__()
        
        self.state_dims = state_dims
        self.n_actions = n_actions
        self.hidden_sizes = hidden_sizes
        self.act_fn = act_fn
        
        sizes = [state_dims] + hidden_sizes + [n_actions]
        
        layers = []
        for i in range(len(sizes) - 1):
            #add fully connected layer
            layers.append(
                nn.Linear(sizes[i], sizes[i + 1])
                )
            #add activation_function
            layers.append(act_fn())
            
        #create sequential neural net
        self.network = nn.Sequential( *layers )
        
        self.optimizer = T.optim.Adam(self.parameters(), lr = lr)
    #--------------------------------------------------------------------------
    def forward(self, state):
        return self.network(state)
    #--------------------------------------------------------------------------
    def save(self, filename):
        T.save(self.state_dict(), filename)
    #--------------------------------------------------------------------------
    def load(self, filename):
        self.load_state_dict(T.load(filename))
    