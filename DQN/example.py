# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 20:39:36 2021

@author: eplan
"""

import numpy as np
import torch as T
import torch.nn as nn
from torch.functional import F

import matplotlib.pyplot as plt

import gym

from Agent import Agent

ENV_NAME = "CartPole-v1"
EPOCHS = 200
LR = 1e-3
FC1_DIMS = 64
FC2_DIMS = 64
EPSILON = 1
EPSILON_DECAY = .97
EPSILON_MIN = 1e-3
BATCH_SIZE = 64
MEM_SIZE = 10000
HIDDEN_DIMS = [64, 64]

def train(agent, env, epochs, epsilon = 1, epsilon_decay = .95, 
          epsilon_min = .0001, verbose = True):
    
    returns = []
    
    for epoch in range(epochs):
        
        epoch_return = 0
        
        obs, done = env.reset(), False
        
        while not done :
            
            if np.random.random() < epsilon :
                action = env.action_space.sample()
            else :
                action = agent.choose_action(obs)
            
            obs_, rew, done, info = env.step(action)
            
            agent.store_transition(obs, action, obs_, rew, done)
            
            epoch_return += rew
            
            obs = obs_
            
            agent.learn()
            
        returns.append(epoch_return)
        
        if verbose :
            print(f"Epoch : {epoch + 1} --> return : {returns[-1]}, eps : {epsilon}")
        
        if epsilon > epsilon_min :
            epsilon *= epsilon_decay
        else:
            epsilon = epsilon_min
    
    return returns

def test(agent, env, episodes = 10, render = True):
    returns = 0
    for ep in range(episodes):
        obs, done = env.reset(), False
        ep_ret = 0
        while not done :
            if render :
                env.render()
            action = agent.choose_action(obs)
            obs, rew, done, info = env.step(action)
            ep_ret += rew
        returns += ep_ret
    if render : 
        env.close()
    return returns / episodes
        
        

if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    
    agent = Agent(env.observation_space.shape[0], 
                      env.action_space.n, 
                      BATCH_SIZE, 
                      MEM_SIZE, 
                      HIDDEN_DIMS, 
                      LR)
    
    returns = train(agent, env, EPOCHS, EPSILON, EPSILON_DECAY,
                    EPSILON_MIN)
    
    plt.plot(returns)
    
    
    
    
    
    
    
    
    
    
    
    
    