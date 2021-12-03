
import torch as T
from torch.distributions import Categorical
import numpy as np

from Actor import Actor_Network
from Critic import Critic_Network
from Memory import Memory

class Agent :
	#--------------------------------------------------------------------------
	def __init__(self, env, actor_hidden_dims, critic_hidden_dims, 
		batch_size, lr_actor = 1e-2, lr_critic = 1e-2, gamma = .99, 
		lambda_ = .95, epsilon = .2):
		
		self.lambda_ = lambda_
		self.gamma = gamma
		self.epsilon = epsilon
		self.batch_size = batch_size
		self.state_dims = env.observation_space.shape[0]
		self.n_actions = env.action_space.n

		self.actor = Actor_Network(self.state_dims, self.n_actions, 
			actor_hidden_dims, lr_actor)
		self.critic = Critic_Network(self.state_dims, critic_hidden_dims,
			lr_critic)
		self.memory = Memory()
	#--------------------------------------------------------------------------
	def choose_action(self, state):
		"""
		take state as input
		return an action, state value and log_prob of action
		"""
		state = T.tensor([state], dtype = T.float32)

		dist = Categorical(logits = self.actor.forward(state))

		action = dist.sample()

		log_prob = dist.log_prob(action).detach().numpy()

		value = self.critic.forward(state).detach().numpy()

		return action.detach().squeeze().numpy(), log_prob, value
	#--------------------------------------------------------------------------
	def store_transition(self, state, action, new_state, reward, done, 
						 log_prob, value):
		self.memory.store_transition(state, action, new_state, reward, done, 
						 log_prob, value)
	#--------------------------------------------------------------------------
	def learn(self):
		
		states, actions, new_states, rewards, dones, \
			log_probs, values, batches = self.memory.get_batch(self.batch_size)
		
		#=============Compute advantages===============
		advantages = []
		T_ = len(rewards)
		g, l = self.gamma, self.lambda_
		for t in range(T_ - 1):
			A_t = 0 
			for t_ in range(t, T_ - 1):
				delta_t = rewards[t_] + g * (1 - dones[t_]) * \
							values[t_ + 1] - values[t_]
				A_t += ((g * l) ** (t - t_)) * delta_t
				if dones[t_] :
					break
			advantages.append(A_t)
		advantages.append(np.array(0).reshape((1, 1)))
		advantages = T.tensor(advantages)

		for batch in batches:
			states_mb = T.tensor(states[batch], dtype = T.float32)
			pi_old = T.tensor(log_probs[batch], dtype = T.float32)
			actions_mb = T.tensor(actions[batch])
			
			#=============Compute policy loss==============
			dist = Categorical(logits = self.actor.forward(states_mb))
			pi_new = dist.log_prob(actions_mb).unsqueeze(1)

			r = (pi_new - pi_old).exp()

			w = r * advantages[batch]
			w_clip = T.clamp(r, 1 - self.epsilon, 1 + self.epsilon) * advantages[batch]

			policy_loss = - T.min(w, w_clip).mean()

			#=============Compute critic loss==============
			new_values = self.critic.forward(states_mb)
			returns = advantages[batch] + values[batch]
			critic_loss = (returns - new_values)**2
			critic_loss = critic_loss.mean()

			#=========Compute total loss and step==========
			loss = policy_loss + .5 * critic_loss

			self.critic.optimizer.zero_grad()
			self.actor.optimizer.zero_grad()

			loss.backward()

			self.critic.optimizer.step()
			self.actor.optimizer.step()

		self.memory.reset()

	#--------------------------------------------------------------------------
	def save(self, filename):
		self.critic.save(filename + "_critic")
		self.actor.save(filename + "_actor")




