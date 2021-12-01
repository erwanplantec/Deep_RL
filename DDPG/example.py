# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 09:30:12 2021

@author: eplan
"""

import gym
import numpy as np
from Agent import *
import matplotlib.pyplot as plt

from Agent import DDPG_Agent

def train(epochs, agent, env, epsilon = 1, delta_epsilon = -.005, 
          epsilon_min = .001):
    """
    Training procedure
    """
    
    perfs = []
    
    for epoch in range(epochs):
        
        epoch_rews = 0
        
        obs = env.reset()
        
        done = False
        #====================================================
        while not done:
            
            #==============Take an action====================
            if np.random.random() < epsilon :
                action = env.action_space.sample()
            else :
                action = agent.choose_action(obs)
            new_obs, reward, done, info = env.step(action) # take a random action
            
            #==============Store transition==================
            transition = (obs, action, new_obs, reward, done)
            agent.store_transition(*transition)
            agent.learn()
            
            obs = new_obs
            epoch_rews += reward
            
        
        if verbose :
            print("Epoch {}, rewards : {}, epslon {}".format(epoch + 1, epoch_rews, epsilon))
            
        perfs.append(epoch_rews)
        
        epsilon += delta_epsilon
        #====================================================
    
    return perfs

def test(env, agent, render = True):
    rewards = 0
    done = False
    obs = env.reset()
    while not done:
        if render :
            env.render()
        
        #==============Take an action====================
        action = agent.choose_action(obs)
        new_obs, reward, done, info = env.step(action) # take a random action
        
        #==============Store transition==================       
        obs = new_obs
        rewards += reward
        
    env.close()
    
    return rewards

#===============Training parameters=============
nb_seeds = 1
epochs = 1000

actor_hidden_dims = [400, 300]
critic_hidden_dims = [400, 300]
learning_rate_actor = .00025
learning_rate_critic = .000025

tau = .001
gamma = .99
mem_size = 1000000
batch_size = 64
epsilon = 1
delta_epsion = -.005

verbose = True


perfs = []
agents = []

for seed in range(nb_seeds):
    print("seed number : ", seed + 1)
    #==============Setup environment================
    env = gym.make('LunarLanderContinuous-v2')
    
    #==============Setup Agent======================
    agent = DDPG_Agent(8, 2,batch_size, mem_size, actor_hidden_dims, 
                       critic_hidden_dims, lr_actor = learning_rate_actor, 
                       lr_critic = learning_rate_critic, gamma = gamma, 
                       tau = tau)
    
    #===============Train===========================
    perfs.append(train(epochs, agent, env, epsilon))
    agents.append(agent)
        
    #==================Plots========================
    plt.plot(range(epochs), perfs[-1])
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    