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
            
            
#==============================================================================
#==============================================================================
#==============================================================================

class Recurrent_Replay_Buffer:
    #--------------------------------------------------------------------------
    def __init__(self, state_dims, action_dims, mem_size):
        
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.mem_size = mem_size
        
        self.episode_counter = -1
        
        self.memory = []
    #--------------------------------------------------------------------------
    def add_episode(self):
        """adds empty room for an episode in memory"""
        self.episode_counter += 1
        if self.episode_counter > self.mem_size :
            # if mem_size is exceeded then replace older sequences
            # first element are states, second actions and third rewards
            self.memory[self.get_memory_idx()] = ([], [], [])  
        else :
            # else add list to store new sequence in memory
            self.memory.append(([], [], []))
    #--------------------------------------------------------------------------
    def store_transition(self, state, action, new_state, reward, done):
        idx = self.get_memory_idx()
        self.memory[idx][0].append(state)
        self.memory[idx][1].append(action)
        self.memory[idx][2].append(reward)
    #--------------------------------------------------------------------------
    def get_memory_idx(self):
        """return the index in memory of the current episode"""
        return self.episode_counter % self.mem_size
    #--------------------------------------------------------------------------
    def sample_batch(self, batch_size:int, seq_length:int):
        """return a list of batch_size sequences of length seq_length"""
        
        if batch_size > self.episode_counter + 1 :
            return None
        
        sequences = tuple(random.sample(self.memory, k = batch_size))
        
        batch = []
        for seq in sequences :
            # only take subparts of length seq_length of the sequences
            _len = len(seq[0]) #total len of the sequence
            start = random.randint(0, _len - seq_length)
            batch.append( 
                    ( seq[0][start : start + seq_length],
                    seq[1][start : start + seq_length],
                    seq[2][start : start + seq_length] )
                )
        
        return batch
            
        
        
        
        
        
        