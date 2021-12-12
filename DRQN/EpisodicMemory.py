
import numpy as np
import random

class EpisodicMemory:
	#--------------------------------------------------------------------------
	def __init__(self, batch_size, mem_size):
		"""
		batch_size : number of episodes in a batch
		mem_size : max number of stored episodes
		"""
		self.batch_size = batch_size
		self.mem_size = mem_size

		self.states = []
		self.states_ = []
		self.actions = []
		self.rewards = []
		self.dones = []

		self.ep_idxs = []
	#--------------------------------------------------------------------------
	def add_episode(self):
		self.ep_idxs.append(len(self.states))
		
		if len(self.ep_idxs) > self.mem_size:
			i = self.ep_idxs[1]
			self.states = self.states[i:]
			self.states_ = self.states_[i:]
			self.actions = self.actions[i:]
			self.rewards = self.rewards[i:]
			self.dones = self.dones[i:]

			self.ep_idxs = [k - i for k in self.ep_idxs[1:]]
	#--------------------------------------------------------------------------
	def store_transition(self, state, action, state_, reward, done):
		self.states.append(state)
		self.states_.append(state_)
		self.actions.append(action)
		self.rewards.append(reward)
		self.dones.append(done)
	#--------------------------------------------------------------------------
	def get_batch(self):
		if len(self.ep_idxs) - 1 < self.batch_size:
			return None
		idxs = random.sample(self.ep_idxs[:-1], k = self.batch_size)
		batches = []

		start_ends = [(i , self.ep_idxs[self.ep_idxs.index(i) + 1] if i != self.ep_idxs[-1] else len(self.states)) 
			for i in idxs]

		min_length = min([e - s for s, e in start_ends])
		for i, (s, e) in enumerate(start_ends):
			if e - s > min_length:
				ns = random.randint(s, e - min_length)
				ne = ns + min_length

				start_ends[i] = (ns, ne)

		for start, end in start_ends :
			batches.append(
				(
					np.array(self.states[start:end]),
					np.array(self.actions[start:end]),
					np.array(self.states_[start:end]),
					np.array(self.rewards[start:end]),
					np.array(self.dones[start:end])
					)
				)
		return batches
