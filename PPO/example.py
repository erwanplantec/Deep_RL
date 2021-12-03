import numpy as np
import matplotlib.pyplot as plt
from training import train
import gym

from Agent import Agent

batch_size = 5
N = 20
lr = 3e-3
gamma = .99 #from paper
lambda_ = .95 #from paper
epsilon = .2 #from paper
episodes = 400
env_name = "CartPole-v1"

critic_dims = [100, 100]
actor_dims = [100, 100]

if __name__ == '__main__':

	env = gym.make(env_name)

	agent = Agent(env, actor_dims, critic_dims, batch_size, 
		lr_actor = lr, lr_critic = lr, gamma = gamma, 
		lambda_ = lambda_, epsilon = epsilon)

	returns, avg_returns = train(agent, env, episodes, N)

	plt.plot(returns, label = "returns", alpha = .2)
	plt.plot(avg_returns, label = "avg_returns")
	plt.legend()
	plt.show()