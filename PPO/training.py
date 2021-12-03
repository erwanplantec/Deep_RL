import numpy as np

def train(agent, env, episodes, N, verbose = True):

	learning_steps = 0
	returns = []
	avg_returns = []
	
	for episode in range(episodes):

		n_steps = 0
		s_t, done = env.reset(), False
		ep_rews = 0

		while not done :

			a_t, log_prob, V = agent.choose_action(s_t) 

			s_tp1, r_t, done, info = env.step(a_t)

			n_steps += 1
			ep_rews += r_t

			agent.store_transition(s_t, a_t, s_tp1,
				r_t, done, log_prob, V)

			s_t = s_tp1

			if not n_steps % N :
				agent.learn()
				learning_steps += 1

		returns.append(ep_rews)
		avg_returns.append(np.mean(returns[-100:]))

		if verbose :
			print(f"Episode {episode + 1} ==> return : {ep_rews}, avg_return : {avg_returns[-1]}, learning_steps : {learning_steps}")

	return returns, avg_returns