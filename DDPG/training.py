import pybullet_envs
import gym
import numpy as np
from Agent import *
import matplotlib.pyplot as plt

def train(episodes, agent, env, warmup = 0):
    """
    Training procedure
    """
    
    perfs = []
    timesteps = 0
    
    for episode in range(episodes):
        
        ep_rews = 0
        
        obs = env.reset()
        
        done = False
        #====================================================
        while not done:
            
            #==============Take an action====================
            action = agent.choose_action(obs, noise = True)
            new_obs, reward, done, info = env.step(action) 
            
            #==============Store transition==================
            agent.store_transition(obs, action, new_obs, 
                reward, done)
            agent.learn()
            
            obs = new_obs
            ep_rews += reward
            timesteps += 1
            
        if verbose :
            print("Episode {}, timestep {}, rewards : {}".
                format(episode + 1, timesteps, ep_rews))
            
        perfs.append(ep_rews)
        
        #====================================================
    
    return perfs
