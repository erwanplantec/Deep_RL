# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 09:30:12 2021

@author: eplan
"""

import pybullet_envs
import gym
import numpy as np
from Agent import *
import matplotlib.pyplot as plt

from Agent import Agent
from training import train

#===============Training parameters=============
nb_seeds = 1
epochs = 1000

actor_hidden_dims = [256, 256]
critic_hidden_dims = [256, 256]
learning_rate_actor = 1e-3
learning_rate_critic = 1e-3

tau = .001
gamma = .99
mem_size = 1000000
batch_size = 64

verbose = True


perfs = []
agents = []

for seed in range(nb_seeds):
    print("seed number : ", seed + 1)
    #==============Setup environment================
    env = gym.make( "LunarLanderContinuous-v2" )
    
    #==============Setup Agent======================
    agent = Agent(env, batch_size, mem_size, actor_hidden_dims, 
               critic_hidden_dims, lr_actor = learning_rate_actor, 
               lr_critic = learning_rate_critic, gamma = gamma, 
               tau = tau)
    
    #===============Train===========================
    perfs.append(train(epochs, agent, env, warmup = 0))
    agents.append(agent)
        
    #==================Plots========================
    plt.plot(range(epochs), perfs[-1])
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    