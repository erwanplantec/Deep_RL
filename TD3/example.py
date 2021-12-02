import numpy as np
import gym
from Agent import Agent
from training import train

batch_size = 100
mem_size = int(1e6)
critic_hidden_dims = [400, 300]
actor_hidden_dims = [400, 300]
lr_actor = 1e-2
lr_critic = 1e-2
gamma = .99
tau = 5e-3
action_noise = .1
target_noise = .1

episodes = 1000

env_name = "LunarLanderContinuous-v2"

if __name__ == "__main__":
	env = gym.make(env_name)

	agent = Agent(env,
				  batch_size, mem_size,
				  actor_hidden_dims, critic_hidden_dims,
				  gamma, lr_critic, lr_actor,
				  tau, action_noise, target_noise
				  )

	returns = train(agent, env, episodes)

	plt.plot(returns)
	plt.show()