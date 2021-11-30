# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 19:13:08 2021

@author: eplan
"""

import gym

from PolicyGradient.Agent import Agent

import numpy as np
import torch as T
import torch.nn as nn
from torch.functional import F

import matplotlib.pyplot as plt

def train(agent, env, epochs, verbose = True, 
          batch_size = 5000):
    
    returns = []
    
    for epoch in range(epochs):
        
        state = env.reset()
        
        agent.add_episode()
        
        while True:
            # Pick action
            action = agent.choose_action(state)
            # env step
            s_, r, done, _ = env.step(action)
            # remember
            agent.remember(state, r, action)
            
            state = s_
            
            if done :
                
                obs, done = env.reset(), False
                
                agent.add_episode()
                
                if len(agent.states_mem) > batch_size:
                    break
                
            
        
        # Training step
        agent.learn()
        
        returns.append(np.mean(agent._traj_returns()))
        
        if verbose :
            print(f"Epoch : {epoch + 1}, \
                  return : {returns[-1]}")
        
        agent.reset_memory()
            
    return returns

def test(agent, env, episodes = 10, render = True):
    
    returns = []
    
    for ep in range(episodes):
        
        obs, done = env.reset(), False
        episode_return = 0
        
        while not done :
            
            if render :
                env.render()
            
            action = agent.choose_action(obs)
            obs, rew, done, info = env.step(action)
            episode_return += rew
        
        returns.append(episode_return)
        
    if render : 
        env.close()
    
    return np.mean(returns)
            
            
            
        
if __name__ == '__main__':
    
    env = gym.make("CartPole-v1")
    
    agent = Agent(4, 2, [32], lr = 1e-2, weights = "rtg")
    
    ret = train(agent, env, 50)
    
    plt.plot(ret)
    
    