import numpy as np


# from https://github.com/MoritzTaylor/ddpg-pytorch/blob/master/utils/noise.py
class OUNoise:
    def __init__(self, action_dimension, dt=0.01, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.dt = dt
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.random.randn(len(x)) * np.sqrt(self.dt)
        self.state = x + dx
        return self.state