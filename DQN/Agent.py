import numpy as np
import torch as T
import torch.nn as nn
from torch.functional import F

from DQN import DQN
from ReplayBuffer import Replay_Buffer

class Agent :
    #--------------------------------------------------------------------------
    def __init__(self, state_dims, n_actions, batch_size, mem_size,
                 hidden_dims, learning_rate, act_fn = nn.ReLU, tau = .003,
                 gamma = .98):
        
        self.DQN = DQN(state_dims, n_actions, hidden_dims, act_fn,
                       learning_rate)
        self.target_DQN = DQN(state_dims, n_actions, hidden_dims, act_fn)
        
        self.replay_buffer = Replay_Buffer(state_dims, 1, mem_size, 
                                           batch_size)
        
        self.state_dims = state_dims
        self.n_actions = n_actions
        
        self.batch_size = batch_size
       
        self.gamma = gamma
        
        self.tau = tau
        
        self.update_networks(1)
    #--------------------------------------------------------------------------
    def choose_action(self, state):
        state = T.tensor(state, dtype = T.float32).to(self.DQN.device)
        
        action = np.argmax(self.DQN.forward(state).detach().numpy())
        
        return action
    #--------------------------------------------------------------------------
    def store_transition(self, state, action, new_state, reward, done):
        self.replay_buffer.store_transition(state, action, new_state, 
                                            reward, done)
    #--------------------------------------------------------------------------
    def learn(self):
        
        # if self.replay_buffer.counter < self.replay_buffer.mem_size:
        #     return
        
        states, new_states, actions, rewards, dones = \
            self.replay_buffer.get_batch()
            
        states = T.tensor(states, dtype = T.float32)
        new_states = T.tensor(new_states, dtype = T.float32)
        actions = T.tensor(actions, dtype = T.int64)
        dones = T.tensor(dones, dtype = T.float32)
        rewards = T.tensor(rewards, dtype = T.float32)
        
        targets = rewards + self.gamma * T.max(self.target_DQN.forward(new_states), 
                                                dim = 1)[0] * (1 - dones)
        targets = targets.unsqueeze(1)
       
        preds = self.DQN.forward(states).gather(1, actions)

        loss = F.mse_loss(preds, targets)
        
        self.DQN.optimizer.zero_grad()
        
        loss.backward()
        
        self.DQN.optimizer.step()
        
        self.update_networks()
    #--------------------------------------------------------------------------
    def update_networks(self, tau = None):
        if tau is None : 
            tau = self.tau 
        
        params = dict(self.DQN.named_parameters())
        target_params = dict(self.target_DQN.named_parameters())
        
        for key in params.keys():
            params[key] = tau * params[key].clone() + \
                (1 - tau) * target_params[key].clone()
        
        self.target_DQN.load_state_dict(params)
    #--------------------------------------------------------------------------
    def save(self, filename):
        self.DQN.save(filename)
        self.target_DQN.save(filename + "_target")
        
