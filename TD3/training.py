import numpy as np

def train(agent, env, episodes, verbose = True, max_steps = np.inf):

	returns = []

	for episode in range(episodes):

		obs, done = env.reset(), False

		ep_rews = 0

		steps = 0

		while not done and steps < max_steps:

			action = agent.choose_action(obs)

			obs_, rew, done, info = env.step(action)

			steps += 1

			ep_rews += rew

			agent.store_transition(obs, action, obs_, rew, done)

			agent.learn()

		returns.append(ep_rews)
		avg_score = np.mean(returns[-100 : ])

		if verbose :
			print(f"Episode {episode + 1} --> return {returns[-1]}, \
				avg_score : {avg_score}")

	return returns