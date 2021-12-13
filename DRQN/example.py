from Agent import Agent
from train import train
import gym

env_name = "CartPole-v1"

gru_dims = 64
gru_layers = 1
hidden_dims = [64]

batch_size = 16
mem_size = 10000
lr = 1e-2
gamma = .99
tau = .003

episodes = 1000




if __name__ == "__main__":
	
	env = gym.make(env_name)

	agent = Agent(env, gru_dims, gru_layers, hidden_dims, batch_size,
		mem_size, gamma, tau)

	returns, avg_returns = train(agent, env, episodes)

