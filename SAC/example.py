import pybullet_envs
from training import train
from Agent import Agent
import gym

import matplotlib.pyplot as plt

env_name = "InvertedPendulumBulletEnv-v0"
actor_hidden_dims = [256, 256]
critic_hidden_dims = [256, 256]

lr_actor = lr_critic = 3e-4

gamma = .99
alpha = .2
tau = .005

batch_size = 256
mem_size = int(1e6)

episodes = 300

if __name__ == "__main__":

	env = gym.make( env_name )

	agent = Agent(env, actor_hidden_dims, critic_hidden_dims,
		mem_size, batch_size, lr_critic, lr_actor, gamma,
		alpha, tau)

	returns, avg_returns = train(agent, env, episodes)

	plt.plot(returns, alpha = .2, label = "returns")
	plt.plot(avg_returns, label = "running average")
	plt.legend()
	plt.show()



