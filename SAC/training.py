import numpy as np

def train(agent, env, episodes, verbose = True):

	returns = []
	avg_returns = []

	for episode in range(episodes):

		ep_rews = 0

		s, done = env.reset(), False

		while not done :

			a = agent.choose_action(s)

			s_, r, done, _ = env.step(a)

			ep_rews += r

			agent.store_transition(s, a, s_, r, done)

			agent.learn()

		returns.append(ep_rews)
		avg_returns.append(np.mean(returns[-100:]))

		if verbose :
			print(f"Episode {episode + 1} ==> return : {returns[-1]}, avg_return : {avg_returns[-1]}")

	return returns, avg_returns