from torch.distributions import Uniform
import torch
import gym
from sac_torch import Agent
from networks import Phi
import torch.optim as optim
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
        self.lr = kwargs.lr
        self.state_dim = self.env.observation_space.shape[0]
        self.phi = Phi(self.state_dim, self.latent_dim, self.lr )
        self.lamb = torch.tensor(kwargs.lamb,requires_grad=True)
        
        self.grad_steps_per_epoch = kwargs.grad_steps_per_epoch
        self.minibatch_size = kwargs.minibatch_size

        #optimizers
        self.lambda_optimizer = optim.Adam([self.lambda_param], lr=self.lr)

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
                observation_, reward, done, _ = self.env.step(action)
                self.agent.remember(observation, action, reward, observation_, done, z)
            
            for _ in range(self.grad_steps_per_epoch):
                if self.agent.memory.mem_cntr < self.batch_size:
                    continue
                
                states, actions, rewards, next_states, dones, skills = self.replay_buffer.sample(self.minibatch_size)


    def phi_loss(self, s, s_prime, z, epsilon):
        diff = self.phi(s_prime) - self.phi(s) 
        term1 = torch.dot(diff, torch.tensor(z).float())
        norm_diff = torch.norm(diff)
        term2 = self.lamb * min(epsilon, 1 - norm_diff**2)
        return -(term1 + term2)

    def lambda_loss_fn(self, s, s_prime, lambda_param, epsilon):
        norm_diff = torch.norm(self.phi(s_prime) - self.phi(s))**2
        penalty_term = min(epsilon, 1 - norm_diff)
        return -lambda_param * penalty_term

    def reward(self, s, s_prime, z):
        diff = self.phi(s_prime) - self.phi(s)
        return torch.dot(diff, torch.tensor(z).float())


args = {
    "latent_low": -1,
    "latent_high": 1,
    "latent_dim": 2,
    "n_epochs": 500,
    "batch_size": 256,
    "env_name": "Ant-v4",
    "lamb": 30,
    "lr":0.0001,
    "grad_steps_per_epoch":50,
    "minibatch_size":256
}