from torch.distributions import Uniform
import torch
import gym
from sac_torch import Agent

class Metra():
    def __init__(self, **kwargs ):
        self.z_dist = Uniform(low = kwargs.latent_low, high=kwargs.latent_high)
        self.latent_dim = kwargs.latent_dim
        self.n_epochs = kwargs.n_epochs
        self.batch_size = kwargs.batch_size
        self.env_name = kwargs.env_name
        self.env = gym.make(self.env_name)
        self.agent = Agent(input_dims=self.env.observation_space.shape, env=self.env,
            n_actions=self.env.action_space.shape[0])
        

    def sample_skill(self):
        skill_sample = self.z_dist.sample((self.latent_dim,))
        return skill_sample

    def train(self):
        for epoch in range(self.n_epochs):
            done = False
            observation = self.env.reset()
            z = self.sample_skill()
            while not done:
                action = self.agent.choose_action(observation, z)
                observation_, reward, done, info = self.env.step(action)
                
                self.agent.remember(observation, action, reward, observation_, done)







args = {
    "latent_low": -1,
    "latent_high": 1,
    "latent_dim": 2,
    "n_epochs": 500,
    "batch_size": 256,
    "env_name": "Ant-v4"
}