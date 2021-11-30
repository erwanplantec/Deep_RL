class DQN_Agent :
    #--------------------------------------------------------------------------
    def __init__(self, state_dims, n_actions, batch_size, mem_size,
                 fc1_dims, fc2_dims, learning_rate, tau = .003,
                 gamma = .98):
        
        self.DQN = DQN(state_dims, n_actions, fc1_dims, fc2_dims,
                       learning_rate)
        self.target_DQN = DQN(state_dims, n_actions, fc1_dims, fc2_dims)
        
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
        
        # q_distribution = F.softmax(self.DQN.forward(state)).detach().numpy()
        
        # q_cumsum = np.cumsum(q_distribution)
        
        # action = np.argmax(q_cumsum > np.random.random())
        
        action = np.argmax(self.DQN.forward(state).detach().numpy())
        
        return action
    #--------------------------------------------------------------------------
    def store_transition(self, state, action, new_state, reward, done):
        self.replay_buffer.store_transition(state, action, new_state, 
                                            reward, done)
    #--------------------------------------------------------------------------
    def learn(self):
        
        if self.replay_buffer.counter < self.replay_buffer.mem_size:
            return
        
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
        
#==============================================================================
#==============================================================================
#==============================================================================
        
class DRQN_Agent:
    #--------------------------------------------------------------------------
    def __init__(self, state_dims, n_actions, hidden_dims, seq_length,
                 learning_rate = .001, batch_size = 64, mem_size = 1000,
                 gamma = .98):
        
        self.state_dims = state_dims
        self.n_actions = n_actions
        self.hidden_dims = hidden_dims
        
        self.gamma= gamma
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.seq_length = seq_length
        
        self.DRQN = DRQN(state_dims, n_actions, hidden_dims)
        self.replay_buffer = Recurrent_Replay_Buffer(state_dims, n_actions, 
                                                     mem_size)
        
    #--------------------------------------------------------------------------
    def choose_action(self, state):
        q_values, *_ = self.DRQN.forward(T.tensor(state, dtype = T.float64))[0].detach().numpy()
        return np.argmax(q_values)
    #--------------------------------------------------------------------------
    def store_transition(self, state, action, new_state, reward, done):
        pass
    #--------------------------------------------------------------------------
    def learn(self):
        sequences = self.replay_buffer.sample_batch(self.batch_size, 
                                                    self.seq_length)
        
        preds = T.zeros((self.batch_size * self.seq_length, 1)) #predicted q value for the actions taken
        targets = T.zeros((self.batch_size * self.seq_length, 1)) #target q values : r + gamma max(Q(s_, a))
        
        # iterate ove all sequences in batch
        for i, (states, actions, rewards) in enumerate(sequences):
            # each sequence contains 3 lists for states, actions and rewards respectively
            # reset hidden states to zero
            self.DRQN.reset_hidden_states()
            #iterate over sequence
            for t in range(self.seq_length - 1):
                
                s, a, s_, r = states[t], actions[t], states[t + 1], rewards[t]
                #Compute predicted Q_value : Q(s_t, a_t, h_t)
                pred, h_pred = self.DRQN.forward( T.tensor(s, dtype = T.float64) )
                pred = pred[a]
                # Compute target Q_value : r + gamma * max Q(s_t+1, a_, h_t+1)
                target = r + self.gamma *\
                    T.max(self.DRQN.forward( T.tensor(s_), 
                                            memorize_hidden=False)[0])[0]
                
                self.preds[i] = pred
                self.targets[i] = target
                
        loss = F.mse_loss(preds, targets)
                
        self.DRQN.optimizer.zero_grad()
        
        loss.backward()
        
        self.DRQN.optimizer.step()
    #--------------------------------------------------------------------------