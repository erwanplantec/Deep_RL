# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 17:56:04 2021

@author: eplan
"""

import numpy as np
import random

class Replay_Buffer :
    #--------------------------------------------------------------------------
    def __init__(self, state_dims, action_dims, mem_size, batch_size):
        
        self.mem_size = mem_size
        self.counter = 0
        
        self.states = np.zeros( ( mem_size, state_dims ) )
        self.actions = np.zeros( ( mem_size, action_dims ) )
        self.new_states = np.zeros( ( mem_size, state_dims ) )
        self.rewards = np.zeros( mem_size )
        self.dones = np.zeros( mem_size )
        
        self.batch_size = batch_size
    #--------------------------------------------------------------------------
    def store_transition(self, state, action, new_state, reward, done):
        
        idx = self.counter % self.mem_size
        
        self.states[idx] = state
        self.actions[idx] = action
        self.new_states[idx] = new_state
        self.rewards[idx] = reward
        self.dones[idx] = done
        
        self.counter += 1
    #--------------------------------------------------------------------------
    def get_batch(self):
        
        max_size = min(self.counter, self.mem_size)
        
        idxs = np.random.choice( max_size, self.batch_size )
        
        states_batch = self.states[idxs]
        new_states_batch = self.new_states[idxs]
        actions_batch = self.actions[idxs]
        rewards_batch = self.rewards[idxs]
        dones_batch = self.dones[idxs]
        
        return states_batch, new_states_batch, actions_batch, rewards_batch, \
            dones_batch
            
            

        
        
        