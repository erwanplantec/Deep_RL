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
    def __init__(self, state_dims, n_actions, fc1_dims, fc2_dims, 
                 learning_rate = .0001, device = "cpu"):
        
        super().__init__()
        
        self.state_dims = state_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        
        self.fc1 = nn.Linear(state_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.output = nn.Linear(fc2_dims, n_actions)
        
        self.optimizer = T.optim.Adam(self.parameters(), lr = learning_rate)
        
        self.device = device
        self.to(device)
    #--------------------------------------------------------------------------
    def forward(self, state):
        qs = self.fc1(state)
        qs = F.relu(qs)
        qs = self.fc2(qs)
        qs = F.relu(qs)
        qs = self.output(qs)
        
        return qs
    #--------------------------------------------------------------------------
    def save(self, filename):
        T.save(self.state_dict(), filename)
    #--------------------------------------------------------------------------
    def load(self, filename):
        self.load_state_dict(T.load(filename))
        
class DQN_FromPixels(nn.Module):
    #--------------------------------------------------------------------------
    def __init__(self, input_shape : tuple, n_actions, learning_rate = .001,
                 device = "cpu"):
        super().__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.device = device
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 5, stride = 2),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = 5, stride = 2),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )
        
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        fc1_input = 32 * conv2d_size_out(conv2d_size_out(input_shape[0])) \
            * conv2d_size_out(conv2d_size_out(input_shape[1]))
        
        self.fc1 = nn.Linear(fc1_input, self.n_actions)
    #--------------------------------------------------------------------------
    def forward(self, state):
        q_values = self.conv1(state)
        q_values = self.conv2(q_values)
        q_values = self.fc1(q_values.view(q_values.size()[0], -1))
        
        return q_values
        
        
        
class DRQN(nn.Module):
    #--------------------------------------------------------------------------
    def __init__(self, state_dims, n_actions, hidden_dims,
                 learning_rate = .001, device = "cpu"):
        
        super().__init__()
        
        self.state_dims = state_dims
        self.n_actions = n_actions
        self.hidden_dims = hidden_dims
        
        self.rnn = nn.RNN(state_dims, hidden_dims, batch_first = True)
        self.fc1 = nn.Linear(hidden_dims, n_actions)
        
        self.reset_hidden_states()
        
        self.optimizer = T.optim.Adam(self.parameters(), lr = learning_rate)
        
        self.device = device
    #--------------------------------------------------------------------------
    def forward(self, state, memorize_hidden = True):
        state = state.unsqueeze(0).unsqueeze(0).float()
        x, h = self.rnn(state, self.h)
        
        if memorize_hidden :
            self.h = h
        
        x = self.fc1(x)
        
        return x.squeeze(0).squeeze(0), h.squeeze(0).squeeze(0)
    #--------------------------------------------------------------------------
    def reset_hidden_states(self):
        #hidden state shape : (N, L, R) 
        self.h = T.zeros((1, 1, self.hidden_dims))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    