from Agent import Agent
from train import train
import gym

env_name = "CartPole-v1"

hidden_dims = 64
hidden_layers = 2

batch_size = 16
mem_size = 10000
lr = 1e-2
gamma = .99
tau = .003

episodes = 1000




if __name__ == "__main__":
	
	env = gym.make(env_name)

	agent = Agent(env, hidden_dims, hidden_layers, batch_size,
		mem_size, gamma, tau)

	returns, avg_returns = train(agent, env, episodes)

