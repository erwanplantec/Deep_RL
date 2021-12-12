import numpy as np

def train(agent, env, episodes, verbose = True):

	ep_returns = []
	avg_returns = []

	for episode in range(episodes):
		
		s, done = env.reset(), False

		ep_return  = 0

		agent.new_episode()

		while not done:

			a = agent.choose_action(s)

			s_, r, done, _ = env.step(a)

			ep_return += r

			agent.store_transition(s, a, s_, r, done)

			agent.learn()

			s = s_

		ep_returns.append(ep_return)
		avg_returns.append(np.mean(ep_returns[-50:]))

		agent.reset_hidden()

		if verbose :
			print(f"Episode : {episode + 1} ==> return : {ep_return}, avg_return : {avg_returns[-1]}")

	return ep_returns, avg_returns

