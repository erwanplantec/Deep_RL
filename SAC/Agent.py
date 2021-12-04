import torch as T
from torch.distributions import Normal
from torch.functional import F

from Critic import Critic_Network, Value_Network
from Actor import Actor_Network
from ReplayBuffer import Replay_Buffer


class Agent :
	#--------------------------------------------------------------------------
	def __init__(self, env, actor_hidden_dims, critic_hidden_dims,
		mem_size, batch_size, lr_critic = 1e-2, lr_actor = 1e-2, 
		gamma = .99, alpha = .2, tau = .003, r_scale = 2):
		
		self.state_dims = env.observation_space.shape[0]
		self.action_dims = env.observation_space.shape[0]
		self.max_actions = env.action_space.high
		self.gamma = gamma
		self.alpha = alpha
		self.tau = tau
		self.r_scale = r_scale

		self.replay_buffer = Replay_Buffer(self.state_dims, self.action_dims,
			mem_size, batch_size)

		self.Q_1 = Critic_Network(self.state_dims, self.action_dims,
			critic_hidden_dims, lr = lr_critic)

		self.Q_2 = Critic_Network(self.state_dims, self.action_dims,
			critic_hidden_dims, lr = lr_critic)

		self.target_Q_1 = Critic_Network(self.state_dims, self.action_dims,
			critic_hidden_dims	, lr = lr_critic)

		self.target_Q_2 = Critic_Network(self.state_dims, self.action_dims,
			critic_hidden_dims, lr = lr_critic)

		self.actor = Actor_Network(self.state_dims, self.action_dims,
			actor_hidden_dims, lr = lr_actor)

		self.update_networks(1)
	#--------------------------------------------------------------------------
	def choose_action(self, state, reparameterize = False):
		state = T.tensor([state], dtype = T.float32)

		mu, sigma = self.actor.forward(state)
		
		dist = Normal(mu, T.clamp(sigma, 1e-6, 1))

		action = dist.rsample() if reparameterize else dist.sample()

		action = T.tanh(action) * self.max_actions[0]

		action = action.view(-1)

		return action.detach().numpy()
	#--------------------------------------------------------------------------
	def sample_action(self, state, reparameterize = False):
		"""Return an action and associated log_probs"""

		mu, sigma = self.actor.forward(state)
		
		dist = Normal(mu, T.clamp(sigma, 1e-6, 1))

		action = dist.rsample() if reparameterize else dist.sample()

		action = T.tanh(action) * self.max_actions[0]

		log_prob = dist.log_prob(action) # See paper appendix

		log_prob = log_prob - T.log(1 - (action ** 2))

		log_prob = log_prob.sum(1, keepdim=True)

		return action, log_prob
	#--------------------------------------------------------------------------
	def store_transition(self, state, action, new_state, reward, done):
		self.replay_buffer.store_transition(state, action, new_state,
			reward, done)
	#--------------------------------------------------------------------------
	def learn(self):
		if self.replay_buffer.counter < self.replay_buffer.batch_size :
			return

		states, actions, new_states, rewards, dones = \
									self.replay_buffer.get_batch()
		states = T.tensor(states, dtype = T.float32)
		actions = T.tensor(actions, dtype = T.float32)
		new_states = T.tensor(new_states, dtype = T.float32)
		rewards = T.tensor(rewards, dtype = T.float32).unsqueeze(1)
		dones = T.tensor(dones, dtype = T.float32).unsqueeze(1)

		#==================Update Q_nets==================
		on_pi_actions_ns, log_probs_ns = self.sample_action(new_states, 
									reparameterize = True)
		# Get next state target q_values with respect to new policy
		Q_values_ns = T.min(	
			self.target_Q_1.forward(new_states, on_pi_actions_ns), 
			self.target_Q_2.forward(states, on_pi_actions_ns)
			) 

		Q_targets = rewards + self.gamma * (1 - dones) *\
			( Q_values_ns - self.alpha * log_probs_ns )

		Q1_loss = F.mse_loss(self.Q_1.forward(states, actions), Q_targets)
		Q2_loss = F.mse_loss(self.Q_2.forward(states, actions), Q_targets)

		Q_loss = Q1_loss + Q2_loss

		self.Q_1.optimizer.zero_grad()
		self.Q_2.optimizer.zero_grad()

		Q_loss.backward()

		self.Q_1.optimizer.step()
		self.Q_2.optimizer.step()
		#==================Update policy==================
		on_pi_actions, log_probs = self.sample_action(states)

		Q_values = T.min(
			self.Q_1.forward(states, on_pi_actions),
			self.Q_2.forward(states, on_pi_actions)
			)

		pi_loss = - (Q_values - self.alpha * log_probs).mean()

		self.actor.optimizer.zero_grad()

		pi_loss.backward()

		self.actor.optimizer.step()

		self.update_networks()
	#--------------------------------------------------------------------------
	def update_networks(self, tau = None):
		if tau is None:
			tau = self.tau

		target_1 = dict(self.target_Q_1.named_parameters())
		target_2 = dict(self.target_Q_2.named_parameters())
		p_1 = dict(self.Q_1.named_parameters())
		p_2 = dict(self.Q_2.named_parameters())

		for key in p_1.keys():
			p_1[key] = tau * p_1[key].clone() + \
					(1 - tau) * target_1[key].clone()
		self.target_Q_1.load_state_dict(p_1)

		for key in p_2.keys():
			p_2[key] = tau * p_2[key].clone() + \
					(1 - tau) * target_2[key].clone()
		self.target_Q_2.load_state_dict(p_2)

