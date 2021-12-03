import numpy as np

class Memory:
	#--------------------------------------------------------------------------
	def __init__(self):
		
		self.reset()

	#--------------------------------------------------------------------------
	def get_batch(self, batch_size : int):
		n = len(self.states)
		starts = np.arange(0, n, batch_size)
		idxs = np.arange(n)
		np.random.shuffle(idxs)
		batches = [idxs[i:i+batch_size] for i in starts]

		return np.array(self.states), np.array(self.actions),\
			np.array(self.new_states),np.array(self.rewards),\
			np.array(self.dones), np.array(self.log_probs),\
			np.array(self.values), batches
	#--------------------------------------------------------------------------
	def reset(self):
		self.states = []
		self.new_states = []
		self.actions = []
		self.log_probs = []
		self.dones = []
		self.rewards = []
		self.values = []
	#--------------------------------------------------------------------------
	def store_transition(self, state, action, new_state, reward, done, 
						 log_prob, value):
		self.states.append(state)
		self.actions.append(action)
		self.log_probs.append(log_prob)
		self.values.append(value)
		self.rewards.append(reward)
		self.new_states.append(new_state)
		self.dones.append(done)
